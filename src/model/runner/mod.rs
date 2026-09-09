mod attention;
mod gdn;
mod mlp;
#[cfg(test)]
mod tests;

use crate::{
    backend::qsfi::{MoeBf16PlanConfig, MoePlan, RmsNormBf16, Workspace},
    engine::{AppendBatch, Commit, DecodeBatch, Engine, RequestId, Status},
    ext::{SafeVec, try_clone_slice},
    model::{
        ActiveRunKind, BatchRun, QwenConfig, QwenWeights,
        scratch::{DeviceBuffer, RunnerScratch},
        state::GdnState,
        validate_token_ids,
        weights::QwenLayerWeights,
    },
};

use std::mem;

#[derive(Clone, Copy, Debug)]
pub struct QwenRequest<'a> {
    pub request_id: RequestId,
    pub tokens: &'a [i32],
    pub max_new_tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenResult {
    pub request_id: RequestId,
    pub prompt_tokens: u32,
    pub generated_tokens: Vec<i32>,
    pub live_tokens: Vec<i32>,
    /// One retained prediction row, for the final token in `live_tokens`.
    pub logits_rows: u32,
    pub logits_vocab_size: u32,
}

pub struct ModelRunner {
    config: QwenConfig,
    weights: QwenWeights,
    engine: Engine,
    moe_plan: Option<MoePlan>,
    gdn_state: Option<GdnState>,
    scratch: RunnerScratch,
    qscb_workspace: DeviceBuffer<u8>,
    live_request_id: Option<RequestId>,
    live_tokens: Vec<i32>,
    last_next_tokens: Vec<i32>,
    last_logits_rows: u32,
    last_logits_vocab_size: u32,
}

impl ModelRunner {
    pub fn new(config: QwenConfig, weights: QwenWeights) -> Result<Self, Status> {
        config.validate()?;
        let config = config.resolved_device_config()?;
        weights.validate_for(&config)?;
        let mut engine = Engine::new(config.engine_config())?;
        let mut scratch = RunnerScratch::new(config.device_ordinal);
        let moe_plan = if let Some(moe) = config.moe_config() {
            let (plan, workspace_bytes) = {
                let mut ops = engine.operators();
                let plan = unsafe {
                    ops.qsfi().create_moe_bf16_plan(MoeBf16PlanConfig {
                        kernel: config.moe_bf16_kernel,
                        max_num_tokens: config.max_seq_len,
                        hidden_size: config.hidden_size,
                        intermediate_size: moe.moe_intermediate_size,
                        num_experts: moe.num_experts,
                        top_k: moe.num_experts_per_tok,
                    })?
                };
                let workspace_bytes =
                    unsafe { ops.qsfi().moe_workspace_size(&plan, config.max_seq_len)? };
                (plan, workspace_bytes)
            };
            scratch.ensure_moe_workspace(workspace_bytes)?;
            Some(plan)
        } else {
            None
        };
        let mut qscb_workspace = DeviceBuffer::empty(config.device_ordinal);
        qscb_workspace.ensure(config.qscb_workspace_bytes)?;
        let gdn_state = if config.has_gdn_layers() {
            Some(GdnState::new(&config)?)
        } else {
            None
        };
        Ok(Self {
            scratch,
            qscb_workspace,
            config,
            weights,
            engine,
            moe_plan,
            gdn_state,
            live_request_id: None,
            live_tokens: Vec::new(),
            last_next_tokens: Vec::new(),
            last_logits_rows: 0,
            last_logits_vocab_size: 0,
        })
    }

    pub fn random_bf16(config: QwenConfig, seed: u64) -> Result<Self, Status> {
        let weights = QwenWeights::random_bf16(&config, seed)?;
        Self::new(config, weights)
    }

    pub fn reset(&mut self) -> Result<(), Status> {
        self.engine.reset()?;
        if let Some(state) = self.gdn_state.as_mut() {
            state.reset(&self.config)?;
        }
        self.live_request_id = None;
        self.live_tokens.clear();
        self.last_next_tokens.clear();
        self.last_logits_rows = 0;
        self.last_logits_vocab_size = 0;
        Ok(())
    }

    pub fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        let release_live = self
            .live_request_id
            .is_some_and(|id| request_ids.contains(&id));
        self.engine.release_requests(request_ids)?;
        if release_live {
            if let Some(state) = self.gdn_state.as_mut() {
                state.reset(&self.config)?;
            }
            self.live_request_id = None;
            self.live_tokens.clear();
            self.last_next_tokens.clear();
            self.last_logits_rows = 0;
            self.last_logits_vocab_size = 0;
        }
        Ok(())
    }

    pub fn live_tokens(&self) -> &[i32] {
        &self.live_tokens
    }

    #[cfg(test)]
    pub(crate) fn last_logits_row_for_test(&self) -> Result<Vec<f32>, Status> {
        if self.last_logits_rows == 0 || self.last_logits_vocab_size != self.config.vocab_size {
            return Err(Status::InvalidArgument);
        }
        let vocab = self.config.vocab_size as usize;
        let offset = (self.last_logits_rows as usize - 1)
            .checked_mul(vocab)
            .ok_or(Status::InvalidArgument)?;
        let mut logits = Vec::safe_new(vocab)?;
        logits.resize(vocab, 0.0);
        self.scratch
            .logits
            .download_range(self.config.stream, offset, &mut logits)?;
        Ok(logits)
    }

    pub fn run(&mut self, request: QwenRequest<'_>) -> Result<QwenResult, Status> {
        self.config.validate()?;
        if request.tokens.is_empty() {
            return Err(Status::InvalidArgument);
        }
        let prompt_len =
            u32::try_from(request.tokens.len()).map_err(|_| Status::InvalidArgument)?;
        let total_tokens = prompt_len
            .checked_add(request.max_new_tokens)
            .ok_or(Status::InvalidArgument)?;
        if total_tokens > self.config.max_seq_len {
            return Err(Status::InvalidArgument);
        }

        let mut generated_tokens = Vec::safe_new(request.max_new_tokens as usize)?;
        self.live_tokens
            .safe_reserve((total_tokens as usize).saturating_sub(self.live_tokens.len()))?;

        self.sync_prefix(request.request_id, request.tokens, total_tokens)?;
        for _ in 0..request.max_new_tokens {
            let next = *self.last_next_tokens.last().ok_or(Status::InternalError)?;
            validate_token_ids(&[next], self.config.vocab_size)?;
            generated_tokens.push(next);
            self.decode_one(request.request_id, next)?;
        }

        let live_tokens = try_clone_slice(&self.live_tokens)?;
        Ok(QwenResult {
            request_id: request.request_id,
            prompt_tokens: u32::try_from(request.tokens.len())
                .map_err(|_| Status::InvalidArgument)?,
            generated_tokens,
            live_tokens,
            logits_rows: self.last_logits_rows,
            logits_vocab_size: self.last_logits_vocab_size,
        })
    }

    fn sync_prefix(
        &mut self,
        request_id: RequestId,
        tokens: &[i32],
        total_tokens: u32,
    ) -> Result<(), Status> {
        let extends_live = self.live_request_id == Some(request_id)
            && tokens.len() >= self.live_tokens.len()
            && tokens[..self.live_tokens.len()] == self.live_tokens;

        if extends_live {
            let suffix = &tokens[self.live_tokens.len()..];
            if !suffix.is_empty() {
                self.append_tokens(request_id, suffix)?;
            } else if self.last_next_tokens.is_empty() {
                return Err(Status::InternalError);
            }
            return Ok(());
        }

        self.rebuild_prefix(request_id, tokens, total_tokens)
    }

    fn rebuild_prefix(
        &mut self,
        request_id: RequestId,
        tokens: &[i32],
        total_tokens: u32,
    ) -> Result<(), Status> {
        validate_token_ids(tokens, self.config.vocab_size)?;
        let rows = u32::try_from(tokens.len()).map_err(|_| Status::InvalidArgument)?;
        self.scratch.ensure(&self.config, rows)?;

        let mut rebuilt_live_tokens = Vec::new();
        rebuilt_live_tokens.safe_reserve(total_tokens as usize)?;

        let fresh_prefix = self.engine.fresh_prefix_state()?;
        let fresh_gdn_state = if self.config.has_gdn_layers() {
            Some(GdnState::new(&self.config)?)
        } else {
            None
        };
        let old_prefix = self.engine.replace_prefix_state(fresh_prefix)?;
        let old_gdn_state = mem::replace(&mut self.gdn_state, fresh_gdn_state);
        let old_live_request_id = self.live_request_id.take();
        let old_live_tokens = mem::replace(&mut self.live_tokens, rebuilt_live_tokens);
        let old_last_next_tokens = mem::take(&mut self.last_next_tokens);
        let old_last_logits_rows = self.last_logits_rows;
        let old_last_logits_vocab_size = self.last_logits_vocab_size;
        self.last_logits_rows = 0;
        self.last_logits_vocab_size = 0;

        match self.append_tokens(request_id, tokens) {
            Ok(()) => Ok(()),
            Err(status) => {
                let failed_prefix = self
                    .engine
                    .replace_prefix_state(old_prefix)
                    .expect("prefix rebuild keeps the execution config unchanged");
                drop(failed_prefix);
                let failed_gdn_state = mem::replace(&mut self.gdn_state, old_gdn_state);
                drop(failed_gdn_state);
                self.live_request_id = old_live_request_id;
                self.live_tokens = old_live_tokens;
                self.last_next_tokens = old_last_next_tokens;
                self.last_logits_rows = old_last_logits_rows;
                self.last_logits_vocab_size = old_last_logits_vocab_size;
                Err(status)
            }
        }
    }

    fn append_tokens(&mut self, request_id: RequestId, tokens: &[i32]) -> Result<(), Status> {
        if tokens.is_empty() {
            return Err(Status::InvalidArgument);
        }
        validate_token_ids(tokens, self.config.vocab_size)?;
        let start_pos =
            u32::try_from(self.live_tokens.len()).map_err(|_| Status::InvalidArgument)?;
        let end_pos = start_pos
            .checked_add(u32::try_from(tokens.len()).map_err(|_| Status::InvalidArgument)?)
            .ok_or(Status::InvalidArgument)?;
        if end_pos > self.config.max_seq_len {
            return Err(Status::InvalidArgument);
        }
        self.live_tokens.safe_reserve(tokens.len())?;

        self.engine.begin_append(AppendBatch {
            request_ids: &[request_id],
            token_indptr: &[
                0,
                i32::try_from(tokens.len()).map_err(|_| Status::InvalidArgument)?,
            ],
            tokens,
        })?;

        let result = self.execute_active_batch(BatchRun {
            tokens,
            start_pos,
            kind: ActiveRunKind::Append,
        });
        match result {
            Ok(sampled) => {
                self.commit_attention_then_gdn(Commit {
                    accepted_token_counts: None,
                })?;
                self.live_tokens.extend_from_slice(tokens);
                self.live_request_id = Some(request_id);
                self.last_next_tokens = sampled;
                Ok(())
            }
            Err(status) => {
                self.abort_attention_batch();
                Err(status)
            }
        }
    }

    fn decode_one(&mut self, request_id: RequestId, token: i32) -> Result<(), Status> {
        let start_pos =
            u32::try_from(self.live_tokens.len()).map_err(|_| Status::InvalidArgument)?;
        if start_pos >= self.config.max_seq_len {
            return Err(Status::InvalidArgument);
        }
        self.live_tokens.safe_reserve(1)?;
        self.engine.begin_decode(DecodeBatch {
            request_ids: &[request_id],
            tokens: &[token],
        })?;

        let result = self.execute_active_batch(BatchRun {
            tokens: &[token],
            start_pos,
            kind: ActiveRunKind::Decode,
        });
        match result {
            Ok(sampled) => {
                self.commit_attention_then_gdn(Commit {
                    accepted_token_counts: None,
                })?;
                self.live_tokens.push(token);
                self.live_request_id = Some(request_id);
                self.last_next_tokens = sampled;
                Ok(())
            }
            Err(status) => {
                self.abort_attention_batch();
                Err(status)
            }
        }
    }

    fn execute_active_batch(&mut self, run: BatchRun<'_>) -> Result<Vec<i32>, Status> {
        let rows = u32::try_from(run.tokens.len()).map_err(|_| Status::InvalidArgument)?;
        if rows == 0 {
            return Err(Status::InvalidArgument);
        }
        validate_token_ids(run.tokens, self.config.vocab_size)?;
        self.scratch.ensure(&self.config, rows)?;
        self.upload_batch_inputs(run.tokens, run.start_pos)?;

        let (weights, mut execution) = self.execution()?;
        execution.run(weights, rows, run.kind)?;
        self.sample_logits(1)
    }

    fn execution(&mut self) -> Result<(&QwenWeights, BatchExecution<'_>), Status> {
        let linear_workspace = self
            .qscb_workspace
            .workspace(self.config.qscb_workspace_bytes)?;
        let moe_workspace = self
            .scratch
            .moe_workspace
            .workspace(self.scratch.moe_workspace.cap)?;
        Ok((
            &self.weights,
            BatchExecution {
                config: &self.config,
                engine: &mut self.engine,
                scratch: &mut self.scratch,
                gdn_state: self.gdn_state.as_ref(),
                moe_plan: self.moe_plan.as_ref(),
                linear_workspace,
                moe_workspace,
            },
        ))
    }

    fn upload_batch_inputs(&mut self, tokens: &[i32], start_pos: u32) -> Result<(), Status> {
        self.scratch.token_ids.upload(self.config.stream, tokens)?;
        let mut positions = Vec::safe_new(tokens.len())?;
        for idx in 0..tokens.len() {
            let pos = start_pos
                .checked_add(u32::try_from(idx).map_err(|_| Status::InvalidArgument)?)
                .ok_or(Status::InvalidArgument)?;
            positions.push(i32::try_from(pos).map_err(|_| Status::InvalidArgument)?);
        }
        self.scratch
            .positions
            .upload(self.config.stream, &positions)
    }

    fn commit_gdn_state(&mut self) {
        if let Some(state) = self.gdn_state.as_mut() {
            state.commit();
        }
    }

    fn commit_attention_then_gdn(&mut self, commit: Commit<'_>) -> Result<(), Status> {
        if let Err(status) = self.engine.commit_batch(commit) {
            self.abort_attention_batch();
            return Err(status);
        }
        self.commit_gdn_state();
        Ok(())
    }

    fn abort_attention_batch(&mut self) {
        let _ = self.engine.abort_batch();
    }

    fn sample_logits(&mut self, rows: u32) -> Result<Vec<i32>, Status> {
        let logits = self.scratch.logits.matrix(rows, self.config.vocab_size)?;
        if self.config.logits_soft_cap > 0.0 {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscu()
                    .logits_soft_cap_f32(logits, self.config.logits_soft_cap)?
            };
        }
        let next_token_ids = self.scratch.next_token_ids.vector(rows)?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.qscu().greedy_argmax_f32(logits, next_token_ids)? };
        }

        let row_count = rows as usize;
        let mut sampled = Vec::safe_new(row_count)?;
        sampled.resize(row_count, 0_i32);
        self.scratch
            .next_token_ids
            .download(self.config.stream, &mut sampled)?;
        for token in &sampled {
            validate_token_ids(&[*token], self.config.vocab_size)
                .map_err(|_| Status::InternalError)?;
        }
        self.last_logits_rows = rows;
        self.last_logits_vocab_size = self.config.vocab_size;
        Ok(sampled)
    }
}

// Borrow execution resources independently of immutable weights. Views are
// prepared after scratch allocation; the scope borrows their owners while enqueueing.
struct BatchExecution<'a> {
    config: &'a QwenConfig,
    engine: &'a mut Engine,
    scratch: &'a mut RunnerScratch,
    gdn_state: Option<&'a GdnState>,
    moe_plan: Option<&'a MoePlan>,
    linear_workspace: Workspace,
    moe_workspace: Workspace,
}

impl BatchExecution<'_> {
    fn run(&mut self, weights: &QwenWeights, rows: u32, kind: ActiveRunKind) -> Result<(), Status> {
        let hidden = self.config.hidden_size;
        let mut input = self.scratch.norm.matrix(rows, hidden)?;
        let layer0 = weights.layers.first().ok_or(Status::InternalError)?;
        let norm = RmsNormBf16::qwen_decoder_norm(
            self.scratch.residual.matrix(rows, hidden)?,
            layer0.input_norm().vector(hidden)?,
            input,
            self.config.rms_norm_eps,
        )?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscu().embedding_gather_bf16(
                    self.scratch.token_ids.vector(rows)?,
                    weights
                        .token_embedding
                        .matrix(self.config.vocab_size, hidden)?,
                    self.scratch.residual.matrix(rows, hidden)?,
                    None,
                    true,
                )?;
                ops.qsfi().rmsnorm_bf16(&norm)?;
            }
        }
        for (index, layer) in weights.layers.iter().enumerate() {
            match layer {
                QwenLayerWeights::AttentionMlp(layer) => self.execute_attention_layer(
                    self.config.attention_layer_index(index as u32)?,
                    rows,
                    input,
                    layer,
                    kind,
                )?,
                QwenLayerWeights::Gdn(layer) => self.execute_gdn_layer(
                    self.config.gdn_layer_index(index as u32)?,
                    rows,
                    input,
                    layer,
                    kind,
                )?,
            }
            let next_norm = weights
                .layers
                .get(index + 1)
                .map_or(&weights.final_norm, |next| next.input_norm());
            let (norm, mlp) = layer.post_attention_mlp();
            self.execute_post_attention_mlp(rows, norm, mlp, next_norm)?;
            input = self.scratch.mlp_out.matrix(rows, hidden)?;
        }
        let mut ops = self.engine.operators();
        unsafe {
            ops.qscb().linear(
                input.row(rows - 1)?,
                weights.lm_head.matrix(self.config.vocab_size, hidden)?,
                self.scratch.logits.matrix(1, self.config.vocab_size)?,
                self.linear_workspace,
            )
        }
    }
}
