#[cfg(test)]
use crate::dtype::F32;
use crate::dtype::{BF16, I32, U8};
mod attention;
mod gdn;
mod linear;
mod mlp;
use linear::{Projection, QuantizedScratch};
#[cfg(test)]
mod tests;

use crate::{
    backend::{
        gdn_prefill::GdnPrefill,
        qscute::{Fp8Decode, Nvfp4Linears},
        qsfi::{MoeBf16PlanConfig, MoePlan, RmsNormBf16, Workspace},
        qstriton::{Fp8Reduce, GdnQkv as Bf16GdnQkv, LmHead},
    },
    constants::gdn::PACKED_QKV_CHANNELS,
    engine::{AppendBatch, Commit, DecodeBatch, Engine, RequestId, Status},
    ext::{SafeVec, try_clone_slice},
    memory::{CudaCtx, DeviceBuffer, HostBuffer},
    model::{
        ActiveRunKind, BatchRun, QwenConfig, QwenWeights,
        sampling::{Sampler, SamplingParams},
        scratch::RunnerScratch,
        state::GdnState,
        validate_token_ids,
        weights::{QwenLayerWeights, QwenModel},
    },
};

use std::{mem, rc::Rc};

enum GdnQkv {
    Bf16(Bf16GdnQkv),
    Fp8 { gemm: Fp8Decode, reduce: Fp8Reduce },
}

const _: () = assert!(Fp8Decode::N == Fp8Reduce::N);

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
    ctx: Rc<CudaCtx>,
    config: QwenConfig,
    tokenizer_token_count: u32,
    weights: QwenWeights,
    quantized_scratch: Option<QuantizedScratch>,
    engine: Engine,
    moe_plan: Option<MoePlan>,
    gdn_state: Option<GdnState>,
    scratch: RunnerScratch,
    qscb_workspace: DeviceBuffer<U8>,
    lm_head: Option<LmHead>,
    gdn_qkv: Option<GdnQkv>,
    gdn_prefill: Option<GdnPrefill>,
    sampler: Option<Sampler>,
    live_request_id: Option<RequestId>,
    live_tokens: Vec<i32>,
    last_next_tokens: Vec<i32>,
    last_logits_rows: u32,
    last_logits_vocab_size: u32,
}

impl ModelRunner {
    /// `tokenizer_token_count` comes from the loaded tokenizer, excluding padded
    /// logit slots. Tokenizer ownership and text formatting remain with the caller.
    pub fn new(
        ctx: Rc<CudaCtx>,
        config: QwenConfig,
        weights: QwenWeights,
        tokenizer_token_count: usize,
    ) -> Result<Self, Status> {
        if matches!(weights, QwenWeights::MoeNvfp4(_)) {
            return Err(Status::Unsupported);
        }
        if config.nvfp4_activation_override.is_some()
            && !matches!(weights, QwenWeights::DenseNvfp4(_))
        {
            return Err(Status::InvalidArgument);
        }
        if let QwenWeights::DenseNvfp4(model) = &weights {
            let validate = |p: &crate::model::weights::Nvfp4Block| -> Result<(), Status> {
                if config.nvfp4_activation_override.unwrap_or(p.activation)
                    != super::Nvfp4Activation::A4
                {
                    return Err(Status::Unsupported);
                }
                crate::backend::qsfi::scale_count(p.shape[0], p.shape[1])?;
                if !p.shape[0].is_multiple_of(8) {
                    return Err(Status::InvalidArgument);
                }
                Ok(())
            };
            validate(&model.lm_head)?;
            for layer in &model.layers {
                let (_, mlp) = layer.post_attention_mlp();
                for p in [&mlp.gate_proj, &mlp.up_proj, &mlp.down_proj] {
                    validate(p)?;
                }
            }
        }
        config.validate()?;
        let tokenizer_token_count =
            u32::try_from(tokenizer_token_count).map_err(|_| Status::InvalidArgument)?;
        if tokenizer_token_count == 0 || tokenizer_token_count > config.vocab_size() {
            return Err(Status::InvalidArgument);
        }
        let mut engine = Engine::new(ctx.clone(), config.engine_config())?;
        let (moe_plan, moe_workspace_bytes) = if let Some(moe) = config.moe_config() {
            let (plan, workspace_bytes) = {
                let mut ops = engine.operators();
                let plan = unsafe {
                    ops.qsfi().create_moe_bf16_plan(MoeBf16PlanConfig {
                        kernel: crate::backend::qsfi::MoeBf16Kernel::COMPILED,
                        max_num_tokens: config.max_seq_len,
                        hidden_size: config.hidden_size(),
                        intermediate_size: moe.moe_intermediate_size,
                        num_experts: moe.num_experts,
                        top_k: moe.num_experts_per_tok,
                    })?
                };
                let workspace_bytes =
                    unsafe { ops.qsfi().moe_workspace_size(&plan, config.max_seq_len)? };
                (plan, workspace_bytes)
            };
            (Some(plan), workspace_bytes)
        } else {
            (None, 0)
        };
        let quantized_scratch = weights
            .is_quantized()
            .then(|| {
                let nvfp4 = Nvfp4Linears::supports_model(
                    config.hidden_size(),
                    config.intermediate_size(),
                    config.vocab_size(),
                )
                .then(|| unsafe { Nvfp4Linears::load() })
                .transpose()?;
                QuantizedScratch::new(ctx.clone(), config.vocab_size(), nvfp4)
            })
            .transpose()?;
        let scratch = RunnerScratch::new(ctx.clone(), &config, moe_workspace_bytes)?;
        let qscb_workspace = DeviceBuffer::with_capacity(ctx.clone(), config.qscb_workspace_bytes)?;
        let gdn_state = config
            .has_gdn_layers()
            .then(|| GdnState::new(ctx.clone(), &config))
            .transpose()?;
        let gdn_prefill = config
            .has_gdn_layers()
            .then(|| unsafe { GdnPrefill::load() })
            .transpose()?;
        let gdn_qkv = if gdn_state.is_none() {
            None
        } else if weights.is_quantized() {
            Fp8Decode::supports(config.hidden_size(), PACKED_QKV_CHANNELS)
                .then(|| unsafe {
                    Ok::<_, Status>(GdnQkv::Fp8 {
                        gemm: Fp8Decode::load()?,
                        reduce: Fp8Reduce::load()?,
                    })
                })
                .transpose()?
        } else {
            Bf16GdnQkv::supports(config.hidden_size(), PACKED_QKV_CHANNELS)
                .then(|| unsafe { Bf16GdnQkv::load().map(GdnQkv::Bf16) })
                .transpose()?
        };
        // Engine construction and allocations establish the primary context.
        let lm_head = (!weights.is_quantized()
            && LmHead::supports(config.hidden_size(), config.vocab_size()))
        .then(|| unsafe { LmHead::load() })
        .transpose()?;
        Ok(Self {
            ctx,
            sampler: None,
            lm_head,
            gdn_qkv,
            gdn_prefill,
            scratch,
            qscb_workspace,
            config,
            tokenizer_token_count,
            weights,
            quantized_scratch,
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

    #[cfg(test)]
    pub(crate) fn random_bf16(
        ctx: Rc<CudaCtx>,
        config: QwenConfig,
        seed: u64,
    ) -> Result<Self, Status> {
        let weights = QwenWeights::random_bf16(ctx.clone(), &config, seed)?;
        Self::new(ctx, config, weights, config.vocab_size() as usize)
    }

    /// Configure before starting a request, or after reset/release. Reset keeps
    /// this configuration and workspace; the same seed/position repeats draws.
    /// The default is greedy and does not allocate/load stochastic kernels.
    pub fn set_sampling(&mut self, params: SamplingParams) -> Result<(), Status> {
        params.validate(self.config.vocab_size())?;
        if self.live_request_id.is_some() {
            return Err(Status::InvalidArgument);
        }
        let sampler = if params.temperature == 0.0 {
            None
        } else {
            Some(Sampler::new(
                self.ctx.clone(),
                self.config.vocab_size(),
                params,
            )?)
        };
        self.sampler = sampler;
        Ok(())
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

    pub(crate) fn lm_head_provider(&self) -> &'static str {
        if self
            .quantized_scratch
            .as_ref()
            .is_some_and(|s| s.nvfp4.is_some())
        {
            "cute-nvfp4"
        } else if self.quantized_scratch.is_some() {
            "flashinfer-cutlass-nvfp4"
        } else if self.lm_head.is_some() {
            "triton"
        } else {
            "cublaslt"
        }
    }

    pub(crate) fn nvfp4_tactic(&self, rows: u32) -> &'static str {
        if self
            .quantized_scratch
            .as_ref()
            .is_some_and(|s| s.nvfp4.is_some())
            && (1..=Nvfp4Linears::MAX_ROWS).contains(&rows)
        {
            Nvfp4Linears::TACTIC
        } else {
            crate::backend::qsfi::Nvfp4Tactic::for_rows(rows).name()
        }
    }

    pub(crate) fn gdn_qkv_provider(&self) -> &'static str {
        match self.gdn_qkv {
            Some(GdnQkv::Fp8 { .. }) => "cute-fp8-split2",
            Some(GdnQkv::Bf16(_)) => "triton",
            None if self.quantized_scratch.is_some() => "cublaslt-fp8",
            None => "cublaslt",
        }
    }

    #[cfg(test)]
    pub(crate) fn last_logits_row_for_test(&self) -> Result<Vec<f32>, Status> {
        if self.last_logits_rows == 0 || self.last_logits_vocab_size != self.config.vocab_size() {
            return Err(Status::InvalidArgument);
        }
        let vocab = self.config.vocab_size() as usize;
        let offset = (self.last_logits_rows as usize - 1)
            .checked_mul(vocab)
            .ok_or(Status::InvalidArgument)?;
        let mut logits = HostBuffer::<F32>::new(vocab)?;
        let result = unsafe { self.scratch.logits.download_range(offset, &mut logits) };
        if let Err(status) = self.ctx.synchronize() {
            eprintln!(
                "CUDA logits download completion uncertain ({status:?}); leaking {} host bytes",
                logits.as_ref().len()
            );
            mem::forget(logits);
            return Err(status);
        }
        result?;
        Ok(logits
            .as_ref()
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect())
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
            validate_token_ids(&[next], self.config.vocab_size())?;
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
        validate_token_ids(tokens, self.config.vocab_size())?;
        let rows = u32::try_from(tokens.len()).map_err(|_| Status::InvalidArgument)?;
        self.scratch.reserve(rows)?;

        let mut rebuilt_live_tokens = Vec::new();
        rebuilt_live_tokens.safe_reserve(total_tokens as usize)?;

        let fresh_prefix = self.engine.fresh_prefix_state()?;
        let fresh_gdn_state = if self.config.has_gdn_layers() {
            Some(GdnState::new(self.ctx.clone(), &self.config)?)
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
        validate_token_ids(tokens, self.config.vocab_size())?;
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
        validate_token_ids(run.tokens, self.config.vocab_size())?;
        self.scratch.reserve(rows)?;
        self.upload_batch_inputs(run)?;
        if let Some(quantized_scratch) = self.quantized_scratch.as_mut() {
            quantized_scratch.prepare(&mut self.engine, &self.weights, rows, &self.config)?;
        }

        let ctx = self.ctx.clone();

        let (weights, mut execution) = self.execution()?;
        // SAFETY: weights, state and workspace remain owned by this runner,
        // including on error. Inputs are uploaded on the engine stream; scratch
        // reuse is ordered on that stream and successful sampling waits for it.
        unsafe { execution.run(&ctx, weights, rows, run.kind)? };
        self.sample_logits(rows - 1)
    }

    fn execution(&mut self) -> Result<(&QwenWeights, BatchExecution<'_>), Status> {
        let linear_workspace = self
            .qscb_workspace
            .workspace(self.config.qscb_workspace_bytes)?;
        Ok((
            &self.weights,
            BatchExecution {
                config: &self.config,
                engine: &mut self.engine,
                scratch: &mut self.scratch,
                gdn_state: self.gdn_state.as_ref(),
                moe_plan: self.moe_plan.as_ref(),
                lm_head: self.lm_head.as_ref(),
                gdn_qkv: self.gdn_qkv.as_ref(),
                gdn_prefill: self.gdn_prefill.as_ref(),
                linear_workspace,
                quantized_scratch: self.quantized_scratch.as_ref(),
            },
        ))
    }

    fn upload_batch_inputs(&mut self, run: BatchRun<'_>) -> Result<(), Status> {
        let mut tokens = HostBuffer::<I32>::new(run.tokens.len())?;
        for (bytes, token) in tokens.as_mut().chunks_exact_mut(4).zip(run.tokens) {
            bytes.copy_from_slice(&token.to_ne_bytes());
        }
        let mut positions = HostBuffer::<I32>::new(tokens.len())?;
        for (idx, bytes) in positions.as_mut().chunks_exact_mut(4).enumerate() {
            let pos = run
                .start_pos
                .checked_add(u32::try_from(idx).map_err(|_| Status::InvalidArgument)?)
                .ok_or(Status::InvalidArgument)?;
            bytes.copy_from_slice(
                &i32::try_from(pos)
                    .map_err(|_| Status::InvalidArgument)?
                    .to_ne_bytes(),
            );
        }
        let sequence = if self.scratch.gdn.is_some() && matches!(run.kind, ActiveRunKind::Append) {
            let mut sequence = HostBuffer::<I32>::new(2)?;
            sequence.as_mut()[4..].copy_from_slice(
                &i32::try_from(tokens.len())
                    .map_err(|_| Status::InvalidArgument)?
                    .to_ne_bytes(),
            );
            Some(sequence)
        } else {
            None
        };
        let result = (|| unsafe {
            self.scratch.token_ids.upload(&tokens)?;
            self.scratch.positions.upload(&positions)?;
            if let Some(sequence) = &sequence {
                self.scratch
                    .gdn
                    .as_mut()
                    .ok_or(Status::InternalError)?
                    .seq_indptr
                    .upload(sequence)?;
            }
            Ok(())
        })();
        // TODO(async-upload-lifetimes): retain staging until batch completion,
        // then remove this existing preparation wait.
        if let Err(status) = self.ctx.synchronize() {
            eprintln!(
                "CUDA input upload completion uncertain ({status:?}); leaking {} host bytes",
                tokens.as_ref().len()
                    + positions.as_ref().len()
                    + sequence.as_ref().map_or(0, |buf| buf.as_ref().len())
            );
            mem::forget((tokens, positions, sequence));
            return Err(status);
        }
        result
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

    // Safe within the runner: checked views address owned, separate buffers;
    // logits production, sampling and the synchronizing download share a stream.
    fn sample_logits(&mut self, position_index: u32) -> Result<Vec<i32>, Status> {
        let rows = 1;
        let logits = self.scratch.logits.matrix(rows, self.config.vocab_size())?;
        let next_token_ids = self.scratch.next_token_ids.vector(rows)?;
        if let Some(sampler) = self.sampler.as_mut() {
            sampler.launch(
                self.ctx.stream,
                &self.scratch.logits,
                &self.scratch.positions,
                position_index,
                &self.scratch.next_token_ids,
            )?;
        } else {
            let mut ops = self.engine.operators();
            unsafe { ops.qscu().greedy_argmax_f32(logits, next_token_ids)? };
        }

        let row_count = rows as usize;
        let mut sampled = HostBuffer::<I32>::new(row_count)?;
        let result = unsafe { self.scratch.next_token_ids.download(&mut sampled) };
        if let Err(status) = self.ctx.synchronize() {
            eprintln!(
                "CUDA token download completion uncertain ({status:?}); leaking {} host bytes",
                sampled.as_ref().len()
            );
            mem::forget(sampled);
            return Err(status);
        }
        result?;
        let sampled: Vec<i32> = sampled
            .as_ref()
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        for token in &sampled {
            validate_token_ids(&[*token], self.tokenizer_token_count)
                .map_err(|_| Status::InternalError)?;
        }
        self.last_logits_rows = rows;
        self.last_logits_vocab_size = self.config.vocab_size();
        Ok(sampled)
    }
}

// Borrow execution resources independently of immutable weights. Views are
// prepared after scratch allocation; the scope borrows their owners while enqueueing.
// Launch contract: inputs must be initialized on the execution device, weights
// must not alias writable scratch, and workspace/plan/state must match config.
// Callers retain allocations and prevent conflicting accesses until stream work
// completes, even on error. These borrows alone do not establish that lifetime.
struct BatchExecution<'a> {
    config: &'a QwenConfig,
    engine: &'a mut Engine,
    scratch: &'a mut RunnerScratch,
    gdn_state: Option<&'a GdnState>,
    moe_plan: Option<&'a MoePlan>,
    lm_head: Option<&'a LmHead>,
    gdn_qkv: Option<&'a GdnQkv>,
    gdn_prefill: Option<&'a GdnPrefill>,
    linear_workspace: Workspace,
    quantized_scratch: Option<&'a QuantizedScratch>,
}

impl BatchExecution<'_> {
    unsafe fn run(
        &mut self,
        ctx: &CudaCtx,
        weights: &QwenWeights,
        rows: u32,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        unsafe {
            match weights {
                QwenWeights::DenseBf16(model) => self.run_model(ctx, model, rows, kind),
                QwenWeights::MoeBf16(model) => self.run_model(ctx, model, rows, kind),
                QwenWeights::DenseNvfp4(model) => self.run_model(ctx, model, rows, kind),
                QwenWeights::MoeNvfp4(_) => Err(Status::Unsupported),
            }
        }
    }

    unsafe fn run_model<M: mlp::Mlp, A: Projection, H: Projection>(
        &mut self,
        ctx: &CudaCtx,
        weights: &QwenModel<M, A, H>,
        rows: u32,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size();
        let mut input = self.scratch.norm.matrix(rows, hidden)?;
        let layer0 = weights.layers.first().ok_or(Status::InternalError)?;
        let norm = RmsNormBf16::qwen_decoder_norm(
            self.scratch.residual.matrix(rows, hidden)?,
            layer0.input_norm().vector(hidden)?,
            input,
            self.config.rms_norm_eps(),
        )?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscu().embedding_gather_bf16(
                    self.scratch.token_ids.vector(rows)?,
                    weights
                        .token_embedding
                        .matrix(self.config.vocab_size(), hidden)?,
                    self.scratch.residual.matrix(rows, hidden)?,
                    None,
                    true,
                )?;
                ops.qsfi().rmsnorm_bf16(&norm)?;
            }
        }
        for (index, layer) in weights.layers.iter().enumerate() {
            unsafe {
                match layer {
                    QwenLayerWeights::AttentionMlp(layer) => self.execute_attention_layer(
                        self.config.attention_layer_index(index as u32)?,
                        rows,
                        input,
                        layer,
                        kind,
                    )?,
                    QwenLayerWeights::Gdn(layer) => self.execute_gdn_layer(
                        ctx,
                        self.config.gdn_layer_index(index as u32)?,
                        rows,
                        input,
                        layer,
                        kind,
                    )?,
                }
            }
            let next_norm = weights
                .layers
                .get(index + 1)
                .map_or::<&crate::memory::DeviceSpan<BF16>, _>(&weights.final_norm, |next| {
                    next.input_norm()
                });
            let (norm, mlp) = layer.post_attention_mlp();
            unsafe { self.execute_post_attention_mlp(rows, norm, mlp, next_norm)? };
            input = self.scratch.mlp_out.matrix(rows, hidden)?;
        }
        unsafe {
            let input = input.row(rows - 1)?;
            let output = self.scratch.logits.matrix(1, self.config.vocab_size())?;
            self.project_logits(
                ctx,
                input,
                weights.lm_head.view(self.config.vocab_size(), hidden)?,
                output,
            )
        }
    }
}
