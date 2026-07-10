mod attention;
mod gdn;
mod mlp;

use crate::model::*;

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
    pub logits_rows: u32,
    pub logits_vocab_size: u32,
}

pub struct ModelRunner {
    pub(in crate::model) config: QwenConfig,
    pub(in crate::model) weights: QwenWeights,
    pub(in crate::model) engine: Engine,
    pub(in crate::model) moe_plan: Option<MoePlan>,
    pub(in crate::model) gdn_state: Option<GdnState>,
    pub(in crate::model) scratch: RunnerScratch,
    pub(in crate::model) qscb_workspace: DeviceBuffer<u8>,
    pub(in crate::model) live_request_id: Option<RequestId>,
    pub(in crate::model) live_tokens: Vec<i32>,
    pub(in crate::model) last_next_tokens: Vec<i32>,
    pub(in crate::model) last_logits_rows: u32,
    pub(in crate::model) last_logits_vocab_size: u32,
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
                    ops.flashinfer().create_moe_bf16_plan(MoeBf16PlanConfig {
                        max_num_tokens: config.max_seq_len,
                        hidden_size: config.hidden_size,
                        intermediate_size: moe.moe_intermediate_size,
                        num_experts: moe.num_experts,
                        top_k: moe.num_experts_per_tok,
                    })?
                };
                let workspace_bytes = unsafe {
                    ops.flashinfer()
                        .moe_workspace_size(&plan, config.max_seq_len)?
                };
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

    pub(in crate::model) fn sync_prefix(
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

    pub(in crate::model) fn rebuild_prefix(
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

        let fresh_engine = Engine::new(self.config.engine_config())?;
        let fresh_gdn_state = if self.config.has_gdn_layers() {
            Some(GdnState::new(&self.config)?)
        } else {
            None
        };
        let old_engine = mem::replace(&mut self.engine, fresh_engine);
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
                let failed_engine = mem::replace(&mut self.engine, old_engine);
                drop(failed_engine);
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

    pub(in crate::model) fn append_tokens(
        &mut self,
        request_id: RequestId,
        tokens: &[i32],
    ) -> Result<(), Status> {
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

    pub(in crate::model) fn decode_one(
        &mut self,
        request_id: RequestId,
        token: i32,
    ) -> Result<(), Status> {
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

    pub(in crate::model) fn execute_active_batch(
        &mut self,
        run: BatchRun<'_>,
    ) -> Result<Vec<i32>, Status> {
        let rows = u32::try_from(run.tokens.len()).map_err(|_| Status::InvalidArgument)?;
        if rows == 0 {
            return Err(Status::InvalidArgument);
        }
        validate_token_ids(run.tokens, self.config.vocab_size)?;
        self.scratch.ensure(&self.config, rows)?;
        self.upload_batch_inputs(run.tokens, run.start_pos)?;

        self.embedding_gather(rows)?;

        let hidden = self.config.hidden_size;
        let q_hidden = self.config.q_hidden_size()?;
        let intermediate = self.config.intermediate_size;
        let mut layer_input = self.scratch.norm.as_device_ptr();
        let layer0 = self
            .weights
            .layers
            .first()
            .ok_or(Status::InternalError)?
            .ptrs();
        self.rmsnorm(
            self.scratch.residual.as_device_ptr(),
            layer0.input_norm(),
            self.scratch.norm.as_device_ptr(),
            rows,
        )?;

        for layer_idx in 0..self.config.num_layers {
            let layer = self.weights.layers[layer_idx as usize].ptrs();
            let next_weight = if layer_idx + 1 == self.config.num_layers {
                self.weights.final_norm.as_device_ptr()
            } else {
                self.weights.layers[(layer_idx + 1) as usize]
                    .ptrs()
                    .input_norm()
            };
            match layer {
                QwenLayerPtrs::AttentionMlp(layer) => {
                    let attention_layer_idx = self.config.attention_layer_index(layer_idx)?;
                    let kv_hidden = self.config.kv_hidden_size()?;
                    self.execute_attention_layer(
                        attention_layer_idx,
                        rows,
                        hidden,
                        q_hidden,
                        kv_hidden,
                        layer_input,
                        layer,
                        run.kind,
                    )?;
                }
                QwenLayerPtrs::Gdn(layer) => {
                    let gdn_layer_idx = self.config.gdn_layer_index(layer_idx)?;
                    self.execute_gdn_layer(
                        gdn_layer_idx,
                        rows,
                        hidden,
                        layer_input,
                        layer,
                        run.kind,
                    )?;
                }
            }
            let post_mlp = layer.post_attention_mlp();
            self.execute_post_attention_mlp(
                rows,
                hidden,
                intermediate,
                post_mlp.norm,
                post_mlp.mlp,
                next_weight,
            )?;
            layer_input = self.scratch.mlp_out.as_device_ptr();
        }

        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            self.weights.lm_head.as_device_ptr(),
            self.scratch.logits.as_device_ptr(),
            self.config.vocab_size,
            GemmOut::F32,
        )?;
        self.sample_logits(rows)
    }

    pub(in crate::model) fn upload_batch_inputs(
        &mut self,
        tokens: &[i32],
        start_pos: u32,
    ) -> Result<(), Status> {
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

    pub(in crate::model) fn attention_heads(
        &self,
        data: ffi::DevicePtr,
        rows: u32,
        heads: u32,
    ) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, heads, self.config.head_dim)
    }

    pub(in crate::model) fn gdn_q_heads(
        &self,
        data: ffi::DevicePtr,
        rows: u32,
    ) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_KEY_DIM)
    }

    pub(in crate::model) fn gdn_k_heads(
        &self,
        data: ffi::DevicePtr,
        rows: u32,
    ) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_KEY_DIM)
    }

    pub(in crate::model) fn gdn_v_heads(
        &self,
        data: ffi::DevicePtr,
        rows: u32,
    ) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)
    }

    pub(in crate::model) fn gdn_state_views(
        &self,
        gdn_layer_idx: u32,
    ) -> Result<(u32, u32, u32, GdnConvState, GdnRecurrentState), Status> {
        let state = self.gdn_state.as_ref().ok_or(Status::InternalError)?;
        let slots = state.layer_slots(gdn_layer_idx)?;
        let conv_state = GdnConvState::contiguous(
            state.conv.as_device_ptr(),
            FloatStorage::Bf16,
            state.slots.state_pool,
        )?;
        let recurrent_state = GdnRecurrentState::contiguous(
            state.recurrent.as_device_ptr(),
            FloatStorage::Bf16,
            state.slots.state_pool,
        )?;
        Ok((
            state.slots.state_pool,
            slots.live_slot,
            slots.staged_slot,
            conv_state,
            recurrent_state,
        ))
    }

    pub(in crate::model) fn commit_gdn_state(&mut self) {
        if let Some(state) = self.gdn_state.as_mut() {
            state.commit();
        }
    }

    pub(in crate::model) fn commit_attention_then_gdn(
        &mut self,
        commit: Commit<'_>,
    ) -> Result<(), Status> {
        if let Err(status) = self.engine.commit_batch(commit) {
            self.abort_attention_batch();
            return Err(status);
        }
        self.commit_gdn_state();
        Ok(())
    }

    pub(in crate::model) fn abort_attention_batch(&mut self) {
        let _ = self.engine.abort_batch();
    }

    pub(in crate::model) fn embedding_gather(&mut self, rows: u32) -> Result<(), Status> {
        let desc = EmbeddingGatherBf16::with_options(
            DVec::contiguous(self.scratch.token_ids.as_device_ptr(), rows)?,
            DMat::contiguous(
                self.weights.token_embedding.as_device_ptr(),
                self.config.vocab_size,
                self.config.hidden_size,
            )?,
            DMat::contiguous(
                self.scratch.residual.as_device_ptr(),
                rows,
                self.config.hidden_size,
            )?,
            None,
            true,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.cuda().embedding_gather_bf16(&desc) }
    }

    pub(in crate::model) fn rmsnorm(
        &mut self,
        x: ffi::DevicePtr,
        weight: ffi::DevicePtr,
        out: ffi::DevicePtr,
        rows: u32,
    ) -> Result<(), Status> {
        let desc = RmsNormBf16::qwen_decoder_norm(
            DMat::contiguous(x, rows, self.config.hidden_size)?,
            DVec::contiguous(weight, self.config.hidden_size)?,
            DMat::contiguous(out, rows, self.config.hidden_size)?,
            self.config.rms_norm_eps,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.flashinfer().rmsnorm_bf16(&desc) }
    }

    pub(in crate::model) fn fused_add_rmsnorm(
        &mut self,
        x: ffi::DevicePtr,
        residual: ffi::DevicePtr,
        weight: ffi::DevicePtr,
        rows: u32,
    ) -> Result<(), Status> {
        let desc = FusedAddRmsNormBf16::qwen_decoder_norm(
            DMat::contiguous(x, rows, self.config.hidden_size)?,
            DMat::contiguous(residual, rows, self.config.hidden_size)?,
            DVec::contiguous(weight, self.config.hidden_size)?,
            self.config.rms_norm_eps,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.flashinfer().fused_add_rmsnorm_bf16(&desc) }
    }

    pub(in crate::model) fn gemm_bf16(
        &mut self,
        x: ffi::DevicePtr,
        rows: u32,
        in_features: u32,
        weight: ffi::DevicePtr,
        out: ffi::DevicePtr,
        out_features: u32,
        out_kind: GemmOut,
    ) -> Result<(), Status> {
        let out = match out_kind {
            GemmOut::Bf16 => Bf16OrF32Mat::Bf16(DMat::contiguous(out, rows, out_features)?),
            GemmOut::F32 => Bf16OrF32Mat::F32(DMat::contiguous(out, rows, out_features)?),
        };
        let workspace = if self.config.qscb_workspace_bytes == 0 {
            Workspace::none()
        } else {
            Workspace::new(
                self.qscb_workspace.as_device_ptr(),
                self.config.qscb_workspace_bytes,
            )?
        };
        let desc = Bf16Gemm::new(
            DMat::contiguous(x, rows, in_features)?,
            DMat::contiguous(weight, out_features, in_features)?,
            out,
            workspace,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.cublas().gemm_bf16(&desc) }
    }

    pub(in crate::model) fn silu_and_mul(
        &mut self,
        rows: u32,
        intermediate: u32,
        gate: ffi::DevicePtr,
        up: ffi::DevicePtr,
        out: ffi::DevicePtr,
    ) -> Result<(), Status> {
        let desc = SiluAndMulBf16::new(
            DMat::contiguous(gate, rows, intermediate)?,
            DMat::contiguous(up, rows, intermediate)?,
            DMat::contiguous(out, rows, intermediate)?,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.cuda().silu_and_mul_bf16(&desc) }
    }

    pub(in crate::model) fn shared_expert_gate_add(
        &mut self,
        rows: u32,
        hidden: u32,
    ) -> Result<(), Status> {
        let desc = Qwen36SharedExpertGateAddBf16::new(
            Bf16OrF32Mat::F32(DMat::contiguous(
                self.scratch.shared_gate_logits.as_device_ptr(),
                rows,
                1,
            )?),
            DMat::contiguous(self.scratch.shared_out.as_device_ptr(), rows, hidden)?,
            DMat::contiguous(self.scratch.mlp_out.as_device_ptr(), rows, hidden)?,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.cuda().qwen36_shared_expert_gate_add_bf16(&desc) }
    }

    pub(in crate::model) fn sample_logits(&mut self, rows: u32) -> Result<Vec<i32>, Status> {
        let logits = DMat::contiguous(
            self.scratch.logits.as_device_ptr(),
            rows,
            self.config.vocab_size,
        )?;
        if self.config.logits_soft_cap > 0.0 {
            let desc = LogitsSoftCapF32::new(logits, self.config.logits_soft_cap)?;
            let mut ops = self.engine.operators();
            unsafe { ops.cuda().logits_soft_cap_f32(&desc)? };
        }
        let desc = GreedyArgmaxF32::new(
            logits,
            DVec::contiguous(self.scratch.next_token_ids.as_device_ptr(), rows)?,
        )?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.cuda().greedy_argmax_f32(&desc)? };
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
