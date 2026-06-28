use crate::engine::{
    AppendBatch, Commit, DType, DecodeBatch, Engine, EngineConfig, EngineLayer, EngineTrait,
    KvLayout, RequestId, Status, validate_supported_attention_grouping,
    validate_supported_attention_head_dim,
};
use crate::ffi::{self, cuda};
use crate::runtime::kernels::{
    Bf16Gemm, Bf16Heads, Bf16OrF32Mat, DMat, DVec, EmbeddingGatherBf16, FusedAddRmsNormBf16,
    GreedyArgmaxF32, LogitsSoftCapF32, RmsNormBf16, SiluAndMulBf16, Workspace,
};

use std::ffi::c_void;
use std::{mem, ptr};

#[derive(Clone, Copy, Debug)]
pub struct QwenConfig {
    pub device_ordinal: i32,
    pub stream: *mut c_void,
    pub num_layers: u32,
    pub max_live_requests: u32,
    pub max_batch_size: u32,
    pub max_seq_len: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub logits_soft_cap: f32,
    pub qsfi_float_workspace_bytes: usize,
    pub qsfi_int_workspace_bytes: usize,
    pub qsfi_host_int_workspace_bytes: usize,
    pub qscb_workspace_bytes: usize,
}

impl QwenConfig {
    pub fn tiny_random_test(device_ordinal: i32) -> Self {
        Self {
            device_ordinal,
            stream: ptr::null_mut(),
            num_layers: 2,
            max_live_requests: 1,
            max_batch_size: 1,
            max_seq_len: 16,
            max_pages: 8,
            page_size: 4,
            hidden_size: 128,
            intermediate_size: 256,
            vocab_size: 64,
            num_q_heads: 2,
            num_kv_heads: 2,
            head_dim: 64,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            logits_soft_cap: 0.0,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
            qscb_workspace_bytes: 16 << 20,
        }
    }

    pub fn validate(&self) -> Result<(), Status> {
        if self.num_layers == 0
            || self.max_live_requests == 0
            || self.max_batch_size == 0
            || self.max_seq_len == 0
            || self.max_pages == 0
            || self.page_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.num_q_heads == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(Status::InvalidArgument);
        }
        if self.max_live_requests != 1 || self.max_batch_size != 1 {
            return Err(Status::Unsupported);
        }
        if self.vocab_size > i32::MAX as u32 {
            return Err(Status::Unsupported);
        }
        validate_supported_attention_grouping(self.num_q_heads, self.num_kv_heads)?;
        if checked_u32_product(&[self.num_q_heads, self.head_dim])? != self.hidden_size {
            return Err(Status::InvalidArgument);
        }
        let _ = self.kv_hidden_size()?;
        validate_supported_attention_head_dim(self.head_dim)?;
        let capacity = self
            .max_pages
            .checked_mul(self.page_size)
            .ok_or(Status::InvalidArgument)?;
        if self.max_seq_len > capacity {
            return Err(Status::InvalidArgument);
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(Status::InvalidArgument);
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rope_scale.is_finite()
            || self.rope_scale <= 0.0
            || !self.logits_soft_cap.is_finite()
            || self.logits_soft_cap < 0.0
        {
            return Err(Status::InvalidArgument);
        }
        if self.qsfi_float_workspace_bytes == 0
            || self.qsfi_int_workspace_bytes == 0
            || self.qsfi_host_int_workspace_bytes == 0
        {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }

    fn resolved_device_config(&self) -> Result<Self, Status> {
        let mut config = *self;
        config.device_ordinal = resolve_device_ordinal(config.device_ordinal)?;
        Ok(config)
    }

    fn kv_hidden_size(&self) -> Result<u32, Status> {
        checked_u32_product(&[self.num_kv_heads, self.head_dim])
    }

    fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            device_ordinal: self.device_ordinal,
            stream: self.stream,
            num_layers: self.num_layers,
            max_live_requests: self.max_live_requests,
            max_batch_size: self.max_batch_size,
            max_seq_len: self.max_seq_len,
            max_pages: self.max_pages,
            page_size: self.page_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            activation_dtype: DType::BF16,
            kv_dtype: DType::BF16,
            kv_layout: KvLayout::NHD,
            rope_theta: self.rope_theta,
            rope_scale: self.rope_scale,
            logits_soft_cap: self.logits_soft_cap,
            qsfi_float_workspace_bytes: self.qsfi_float_workspace_bytes,
            qsfi_int_workspace_bytes: self.qsfi_int_workspace_bytes,
            qsfi_host_int_workspace_bytes: self.qsfi_host_int_workspace_bytes,
        }
    }

    fn same_model_shape(&self, other: &Self) -> bool {
        self.num_layers == other.num_layers
            && self.hidden_size == other.hidden_size
            && self.intermediate_size == other.intermediate_size
            && self.vocab_size == other.vocab_size
            && self.num_q_heads == other.num_q_heads
            && self.num_kv_heads == other.num_kv_heads
            && self.head_dim == other.head_dim
    }
}

pub struct QwenWeights {
    config: QwenConfig,
    token_embedding: DeviceBuffer<u16>,
    final_norm: DeviceBuffer<u16>,
    lm_head: DeviceBuffer<u16>,
    layers: Vec<QwenLayerWeights>,
}

impl QwenWeights {
    pub fn random_bf16(config: &QwenConfig, seed: u64) -> Result<Self, Status> {
        config.validate()?;
        let config = config.resolved_device_config()?;
        let mut rng = DeterministicRng::new(seed);
        let device = config.device_ordinal;
        let stream = config.stream;
        let hidden = config.hidden_size;
        let kv_hidden = config.kv_hidden_size()?;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;

        let token_embedding = DeviceBuffer::from_slice(
            device,
            stream,
            &random_bf16_values(&mut rng, checked_usize_product(&[vocab, hidden])?, 0.08)?,
        )?;
        let final_norm =
            DeviceBuffer::from_slice(device, stream, &constant_bf16_values(hidden as usize, 1.0)?)?;
        let lm_head = DeviceBuffer::from_slice(
            device,
            stream,
            &random_bf16_values(&mut rng, checked_usize_product(&[vocab, hidden])?, 0.04)?,
        )?;

        let mut layers = try_vec_with_capacity(config.num_layers as usize)?;
        for _ in 0..config.num_layers {
            layers.push(QwenLayerWeights {
                attn_norm: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &constant_bf16_values(hidden as usize, 1.0)?,
                )?,
                q_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(&mut rng, checked_usize_product(&[hidden, hidden])?, 0.04)?,
                )?,
                k_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[kv_hidden, hidden])?,
                        0.04,
                    )?,
                )?,
                v_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[kv_hidden, hidden])?,
                        0.04,
                    )?,
                )?,
                o_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(&mut rng, checked_usize_product(&[hidden, hidden])?, 0.04)?,
                )?,
                mlp_norm: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &constant_bf16_values(hidden as usize, 1.0)?,
                )?,
                gate_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[intermediate, hidden])?,
                        0.035,
                    )?,
                )?,
                up_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[intermediate, hidden])?,
                        0.035,
                    )?,
                )?,
                down_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[hidden, intermediate])?,
                        0.035,
                    )?,
                )?,
            });
        }

        Ok(Self {
            config,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    fn validate_for(&self, config: &QwenConfig) -> Result<(), Status> {
        if !self.config.same_model_shape(config) {
            return Err(Status::InvalidArgument);
        }
        if self.config.device_ordinal != config.device_ordinal
            || self.config.stream != config.stream
        {
            return Err(Status::InvalidArgument);
        }
        let expected_layers =
            usize::try_from(config.num_layers).map_err(|_| Status::InvalidArgument)?;
        if self.layers.len() != expected_layers {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

struct QwenLayerWeights {
    attn_norm: DeviceBuffer<u16>,
    q_proj: DeviceBuffer<u16>,
    k_proj: DeviceBuffer<u16>,
    v_proj: DeviceBuffer<u16>,
    o_proj: DeviceBuffer<u16>,
    mlp_norm: DeviceBuffer<u16>,
    gate_proj: DeviceBuffer<u16>,
    up_proj: DeviceBuffer<u16>,
    down_proj: DeviceBuffer<u16>,
}

#[derive(Clone, Copy)]
struct QwenLayerPtrs {
    attn_norm: ffi::DevicePtr,
    q_proj: ffi::DevicePtr,
    k_proj: ffi::DevicePtr,
    v_proj: ffi::DevicePtr,
    o_proj: ffi::DevicePtr,
    mlp_norm: ffi::DevicePtr,
    gate_proj: ffi::DevicePtr,
    up_proj: ffi::DevicePtr,
    down_proj: ffi::DevicePtr,
}

impl QwenLayerWeights {
    fn ptrs(&self) -> QwenLayerPtrs {
        QwenLayerPtrs {
            attn_norm: self.attn_norm.as_device_ptr(),
            q_proj: self.q_proj.as_device_ptr(),
            k_proj: self.k_proj.as_device_ptr(),
            v_proj: self.v_proj.as_device_ptr(),
            o_proj: self.o_proj.as_device_ptr(),
            mlp_norm: self.mlp_norm.as_device_ptr(),
            gate_proj: self.gate_proj.as_device_ptr(),
            up_proj: self.up_proj.as_device_ptr(),
            down_proj: self.down_proj.as_device_ptr(),
        }
    }
}

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
    config: QwenConfig,
    weights: QwenWeights,
    engine: Engine,
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
        let engine = Engine::new(config.engine_config())?;
        let mut qscb_workspace = DeviceBuffer::empty(config.device_ordinal);
        qscb_workspace.ensure(config.qscb_workspace_bytes)?;
        Ok(Self {
            scratch: RunnerScratch::new(config.device_ordinal),
            qscb_workspace,
            config,
            weights,
            engine,
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
        self.live_request_id = None;
        self.live_tokens.clear();
        self.last_next_tokens.clear();
        self.last_logits_rows = 0;
        self.last_logits_vocab_size = 0;
        Ok(())
    }

    pub fn live_tokens(&self) -> &[i32] {
        &self.live_tokens
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

        let mut generated_tokens = try_vec_with_capacity(
            usize::try_from(request.max_new_tokens).map_err(|_| Status::InvalidArgument)?,
        )?;
        self.live_tokens
            .try_reserve(
                usize::try_from(total_tokens)
                    .map_err(|_| Status::InvalidArgument)?
                    .saturating_sub(self.live_tokens.len()),
            )
            .map_err(|_| Status::OutOfMemory)?;

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
        rebuilt_live_tokens
            .try_reserve(usize::try_from(total_tokens).map_err(|_| Status::InvalidArgument)?)
            .map_err(|_| Status::OutOfMemory)?;

        let fresh_engine = Engine::new(self.config.engine_config())?;
        let old_engine = mem::replace(&mut self.engine, fresh_engine);
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
        self.live_tokens
            .try_reserve(tokens.len())
            .map_err(|_| Status::OutOfMemory)?;

        self.engine.begin_append(AppendBatch {
            request_ids: &[request_id],
            token_indptr: &[
                0,
                i32::try_from(tokens.len()).map_err(|_| Status::InvalidArgument)?,
            ],
            tokens,
        })?;

        let result = self
            .execute_active_batch(BatchRun {
                tokens,
                start_pos,
                kind: ActiveRunKind::Append,
            })
            .and_then(|sampled| {
                self.engine
                    .commit_batch(Commit {
                        accepted_token_counts: None,
                    })
                    .map(|_| sampled)
            });
        match result {
            Ok(sampled) => {
                self.live_tokens.extend_from_slice(tokens);
                self.live_request_id = Some(request_id);
                self.last_next_tokens = sampled;
                Ok(())
            }
            Err(status) => {
                let _ = self.engine.abort_batch();
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
        self.live_tokens
            .try_reserve(1)
            .map_err(|_| Status::OutOfMemory)?;
        self.engine.begin_decode(DecodeBatch {
            request_ids: &[request_id],
            tokens: &[token],
        })?;

        let result = self
            .execute_active_batch(BatchRun {
                tokens: &[token],
                start_pos,
                kind: ActiveRunKind::Decode,
            })
            .and_then(|sampled| {
                self.engine
                    .commit_batch(Commit {
                        accepted_token_counts: None,
                    })
                    .map(|_| sampled)
            });
        match result {
            Ok(sampled) => {
                self.live_tokens.push(token);
                self.live_request_id = Some(request_id);
                self.last_next_tokens = sampled;
                Ok(())
            }
            Err(status) => {
                let _ = self.engine.abort_batch();
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

        self.embedding_gather(rows)?;

        let hidden = self.config.hidden_size;
        let intermediate = self.config.intermediate_size;
        let kv_hidden = self.config.kv_hidden_size()?;
        let mut layer_input = self.scratch.norm.as_device_ptr();
        let layer0 = self
            .weights
            .layers
            .first()
            .ok_or(Status::InternalError)?
            .ptrs();
        self.rmsnorm(
            self.scratch.residual.as_device_ptr(),
            layer0.attn_norm,
            self.scratch.norm.as_device_ptr(),
            rows,
        )?;

        for layer_idx in 0..self.config.num_layers {
            let layer = self.weights.layers[layer_idx as usize].ptrs();
            self.gemm_bf16(
                layer_input,
                rows,
                hidden,
                layer.q_proj,
                self.scratch.q.as_device_ptr(),
                hidden,
                GemmOut::Bf16,
            )?;
            self.gemm_bf16(
                layer_input,
                rows,
                hidden,
                layer.k_proj,
                self.scratch.k.as_device_ptr(),
                kv_hidden,
                GemmOut::Bf16,
            )?;
            self.gemm_bf16(
                layer_input,
                rows,
                hidden,
                layer.v_proj,
                self.scratch.v.as_device_ptr(),
                kv_hidden,
                GemmOut::Bf16,
            )?;

            let engine_layer = EngineLayer::bf16_attention(
                layer_idx,
                self.attention_heads(
                    self.scratch.q.as_device_ptr(),
                    rows,
                    self.config.num_q_heads,
                )?,
                self.attention_heads(
                    self.scratch.k.as_device_ptr(),
                    rows,
                    self.config.num_kv_heads,
                )?,
                self.attention_heads(
                    self.scratch.v.as_device_ptr(),
                    rows,
                    self.config.num_kv_heads,
                )?,
                self.attention_heads(
                    self.scratch.attn_out.as_device_ptr(),
                    rows,
                    self.config.num_q_heads,
                )?,
                self.scratch.positions.as_device_ptr(),
            );
            unsafe {
                match run.kind {
                    ActiveRunKind::Append => self.engine.append_layer(&engine_layer)?,
                    ActiveRunKind::Decode => self.engine.decode_layer(&engine_layer)?,
                }
            }

            self.gemm_bf16(
                self.scratch.attn_out.as_device_ptr(),
                rows,
                hidden,
                layer.o_proj,
                self.scratch.attn_proj.as_device_ptr(),
                hidden,
                GemmOut::Bf16,
            )?;
            self.fused_add_rmsnorm(
                self.scratch.attn_proj.as_device_ptr(),
                self.scratch.residual.as_device_ptr(),
                layer.mlp_norm,
                rows,
            )?;
            self.gemm_bf16(
                self.scratch.attn_proj.as_device_ptr(),
                rows,
                hidden,
                layer.gate_proj,
                self.scratch.gate.as_device_ptr(),
                intermediate,
                GemmOut::Bf16,
            )?;
            self.gemm_bf16(
                self.scratch.attn_proj.as_device_ptr(),
                rows,
                hidden,
                layer.up_proj,
                self.scratch.up.as_device_ptr(),
                intermediate,
                GemmOut::Bf16,
            )?;
            self.silu_and_mul(rows)?;
            self.gemm_bf16(
                self.scratch.mlp.as_device_ptr(),
                rows,
                intermediate,
                layer.down_proj,
                self.scratch.mlp_out.as_device_ptr(),
                hidden,
                GemmOut::Bf16,
            )?;

            let next_weight = if layer_idx + 1 == self.config.num_layers {
                self.weights.final_norm.as_device_ptr()
            } else {
                self.weights.layers[(layer_idx + 1) as usize]
                    .ptrs()
                    .attn_norm
            };
            self.fused_add_rmsnorm(
                self.scratch.mlp_out.as_device_ptr(),
                self.scratch.residual.as_device_ptr(),
                next_weight,
                rows,
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

    fn upload_batch_inputs(&mut self, tokens: &[i32], start_pos: u32) -> Result<(), Status> {
        self.scratch.token_ids.upload(self.config.stream, tokens)?;
        let mut positions = try_vec_with_capacity(tokens.len())?;
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

    fn attention_heads(
        &self,
        data: ffi::DevicePtr,
        rows: u32,
        heads: u32,
    ) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, heads, self.config.head_dim)
    }

    fn embedding_gather(&mut self, rows: u32) -> Result<(), Status> {
        let desc = EmbeddingGatherBf16::with_options(
            DVec::<{ ffi::DTYPE_I32 }>::contiguous(self.scratch.token_ids.as_device_ptr(), rows)?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.weights.token_embedding.as_device_ptr(),
                self.config.vocab_size,
                self.config.hidden_size,
            )?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.residual.as_device_ptr(),
                rows,
                self.config.hidden_size,
            )?,
            None,
            true,
        )?;
        let mut ops = self.engine.kernel_ops();
        unsafe { ops.embedding_gather_bf16(&desc) }
    }

    fn rmsnorm(
        &mut self,
        x: ffi::DevicePtr,
        weight: ffi::DevicePtr,
        out: ffi::DevicePtr,
        rows: u32,
    ) -> Result<(), Status> {
        let desc = RmsNormBf16::new(
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(x, rows, self.config.hidden_size)?,
            DVec::<{ ffi::DTYPE_BF16 }>::contiguous(weight, self.config.hidden_size)?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(out, rows, self.config.hidden_size)?,
            self.config.rms_norm_eps,
        )?;
        let mut ops = self.engine.kernel_ops();
        unsafe { ops.rmsnorm_bf16(&desc) }
    }

    fn fused_add_rmsnorm(
        &mut self,
        x: ffi::DevicePtr,
        residual: ffi::DevicePtr,
        weight: ffi::DevicePtr,
        rows: u32,
    ) -> Result<(), Status> {
        let desc = FusedAddRmsNormBf16::new(
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(x, rows, self.config.hidden_size)?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(residual, rows, self.config.hidden_size)?,
            DVec::<{ ffi::DTYPE_BF16 }>::contiguous(weight, self.config.hidden_size)?,
            self.config.rms_norm_eps,
        )?;
        let mut ops = self.engine.kernel_ops();
        unsafe { ops.fused_add_rmsnorm_bf16(&desc) }
    }

    fn gemm_bf16(
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
            GemmOut::Bf16 => Bf16OrF32Mat::Bf16(DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                out,
                rows,
                out_features,
            )?),
            GemmOut::F32 => Bf16OrF32Mat::F32(DMat::<{ ffi::DTYPE_F32 }>::contiguous(
                out,
                rows,
                out_features,
            )?),
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
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(x, rows, in_features)?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(weight, out_features, in_features)?,
            out,
            workspace,
        )?;
        let mut ops = self.engine.kernel_ops();
        unsafe { ops.gemm_bf16(&desc) }
    }

    fn silu_and_mul(&mut self, rows: u32) -> Result<(), Status> {
        let desc = SiluAndMulBf16::new(
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.gate.as_device_ptr(),
                rows,
                self.config.intermediate_size,
            )?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.up.as_device_ptr(),
                rows,
                self.config.intermediate_size,
            )?,
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.mlp.as_device_ptr(),
                rows,
                self.config.intermediate_size,
            )?,
        )?;
        let mut ops = self.engine.kernel_ops();
        unsafe { ops.silu_and_mul_bf16(&desc) }
    }

    fn sample_logits(&mut self, rows: u32) -> Result<Vec<i32>, Status> {
        let logits = DMat::<{ ffi::DTYPE_F32 }>::contiguous(
            self.scratch.logits.as_device_ptr(),
            rows,
            self.config.vocab_size,
        )?;
        if self.config.logits_soft_cap > 0.0 {
            let desc = LogitsSoftCapF32::new(logits, self.config.logits_soft_cap)?;
            let mut ops = self.engine.kernel_ops();
            unsafe { ops.logits_soft_cap_f32(&desc)? };
        }
        let desc = GreedyArgmaxF32::new(
            logits,
            DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                self.scratch.next_token_ids.as_device_ptr(),
                rows,
            )?,
        )?;
        {
            let mut ops = self.engine.kernel_ops();
            unsafe { ops.greedy_argmax_f32(&desc)? };
        }

        let row_count = usize::try_from(rows).map_err(|_| Status::InvalidArgument)?;
        let mut sampled = try_vec_with_capacity(row_count)?;
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

#[derive(Clone, Copy)]
struct BatchRun<'a> {
    tokens: &'a [i32],
    start_pos: u32,
    kind: ActiveRunKind,
}

#[derive(Clone, Copy)]
enum ActiveRunKind {
    Append,
    Decode,
}

#[derive(Clone, Copy)]
enum GemmOut {
    Bf16,
    F32,
}

struct RunnerScratch {
    token_ids: DeviceBuffer<i32>,
    positions: DeviceBuffer<i32>,
    residual: DeviceBuffer<u16>,
    norm: DeviceBuffer<u16>,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    attn_out: DeviceBuffer<u16>,
    attn_proj: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    up: DeviceBuffer<u16>,
    mlp: DeviceBuffer<u16>,
    mlp_out: DeviceBuffer<u16>,
    logits: DeviceBuffer<f32>,
    next_token_ids: DeviceBuffer<i32>,
}

impl RunnerScratch {
    fn new(device_ordinal: i32) -> Self {
        Self {
            token_ids: DeviceBuffer::empty(device_ordinal),
            positions: DeviceBuffer::empty(device_ordinal),
            residual: DeviceBuffer::empty(device_ordinal),
            norm: DeviceBuffer::empty(device_ordinal),
            q: DeviceBuffer::empty(device_ordinal),
            k: DeviceBuffer::empty(device_ordinal),
            v: DeviceBuffer::empty(device_ordinal),
            attn_out: DeviceBuffer::empty(device_ordinal),
            attn_proj: DeviceBuffer::empty(device_ordinal),
            gate: DeviceBuffer::empty(device_ordinal),
            up: DeviceBuffer::empty(device_ordinal),
            mlp: DeviceBuffer::empty(device_ordinal),
            mlp_out: DeviceBuffer::empty(device_ordinal),
            logits: DeviceBuffer::empty(device_ordinal),
            next_token_ids: DeviceBuffer::empty(device_ordinal),
        }
    }

    fn ensure(&mut self, config: &QwenConfig, rows: u32) -> Result<(), Status> {
        let hidden = checked_usize_product(&[rows, config.hidden_size])?;
        let kv_hidden = checked_usize_product(&[rows, config.kv_hidden_size()?])?;
        let intermediate = checked_usize_product(&[rows, config.intermediate_size])?;
        let logits = checked_usize_product(&[rows, config.vocab_size])?;
        let rows = usize::try_from(rows).map_err(|_| Status::InvalidArgument)?;

        self.token_ids.ensure(rows)?;
        self.positions.ensure(rows)?;
        self.residual.ensure(hidden)?;
        self.norm.ensure(hidden)?;
        self.q.ensure(hidden)?;
        self.k.ensure(kv_hidden)?;
        self.v.ensure(kv_hidden)?;
        self.attn_out.ensure(hidden)?;
        self.attn_proj.ensure(hidden)?;
        self.gate.ensure(intermediate)?;
        self.up.ensure(intermediate)?;
        self.mlp.ensure(intermediate)?;
        self.mlp_out.ensure(hidden)?;
        self.logits.ensure(logits)?;
        self.next_token_ids.ensure(rows)?;
        Ok(())
    }
}

struct DeviceBuffer<T> {
    ptr: *mut T,
    cap: usize,
    device_ordinal: i32,
}

impl<T> DeviceBuffer<T> {
    fn empty(device_ordinal: i32) -> Self {
        Self {
            ptr: ptr::null_mut(),
            cap: 0,
            device_ordinal,
        }
    }

    fn from_slice(device_ordinal: i32, stream: *mut c_void, values: &[T]) -> Result<Self, Status>
    where
        T: Copy,
    {
        let mut buffer = Self::empty(device_ordinal);
        buffer.upload(stream, values)?;
        synchronize_stream(stream)?;
        Ok(buffer)
    }

    fn ensure(&mut self, len: usize) -> Result<(), Status> {
        activate_device(self.device_ordinal)?;
        if len == 0 || self.cap >= len {
            return Ok(());
        }
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or(Status::InvalidArgument)?;
        let mut next = ptr::null_mut();
        result_from_cuda(unsafe { cuda::cudaMalloc(&mut next, bytes) })?;
        if !self.ptr.is_null() {
            unsafe {
                cuda::cudaFree(self.ptr.cast());
            }
        }
        self.ptr = next.cast();
        self.cap = len;
        Ok(())
    }

    fn upload(&mut self, stream: *mut c_void, values: &[T]) -> Result<(), Status>
    where
        T: Copy,
    {
        if values.is_empty() {
            return Ok(());
        }
        self.ensure(values.len())?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                self.ptr.cast(),
                values.as_ptr().cast(),
                mem::size_of_val(values),
                cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        })
    }

    fn download(&self, stream: *mut c_void, out: &mut [T]) -> Result<(), Status>
    where
        T: Copy,
    {
        if out.is_empty() {
            return Ok(());
        }
        if self.ptr.is_null() || out.len() > self.cap {
            return Err(Status::InvalidArgument);
        }
        activate_device(self.device_ordinal)?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                out.as_mut_ptr().cast(),
                self.ptr.cast(),
                mem::size_of_val(out),
                cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        })?;
        synchronize_stream(stream)
    }

    fn as_device_ptr(&self) -> ffi::DevicePtr {
        self.ptr.cast()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = activate_device(self.device_ordinal);
            unsafe {
                cuda::cudaFree(self.ptr.cast());
            }
        }
    }
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_f491_4f6c_dd1d)) >> 32) as u32
    }

    fn next_unit_f32(&mut self) -> f32 {
        let bits = 0x3f80_0000 | (self.next_u32() >> 9);
        f32::from_bits(bits) - 1.0
    }
}

fn random_bf16_values(
    rng: &mut DeterministicRng,
    count: usize,
    scale: f32,
) -> Result<Vec<u16>, Status> {
    let mut out = try_vec_with_capacity(count)?;
    for _ in 0..count {
        let value = (rng.next_unit_f32() * 2.0 - 1.0) * scale;
        out.push(f32_to_bf16_bits(value));
    }
    Ok(out)
}

fn constant_bf16_values(count: usize, value: f32) -> Result<Vec<u16>, Status> {
    let mut out = try_vec_with_capacity(count)?;
    out.resize(count, f32_to_bf16_bits(value));
    Ok(out)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn validate_token_ids(tokens: &[i32], vocab_size: u32) -> Result<(), Status> {
    let vocab_size = i32::try_from(vocab_size).map_err(|_| Status::Unsupported)?;
    for token in tokens {
        if *token < 0 || *token >= vocab_size {
            return Err(Status::InvalidArgument);
        }
    }
    Ok(())
}

fn checked_u32_product(values: &[u32]) -> Result<u32, Status> {
    let mut product = 1u32;
    for value in values {
        product = product.checked_mul(*value).ok_or(Status::InvalidArgument)?;
    }
    Ok(product)
}

fn checked_usize_product(values: &[u32]) -> Result<usize, Status> {
    let mut product = 1usize;
    for value in values {
        product = product
            .checked_mul(usize::try_from(*value).map_err(|_| Status::InvalidArgument)?)
            .ok_or(Status::InvalidArgument)?;
    }
    Ok(product)
}

fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, Status> {
    let mut vec = Vec::new();
    vec.try_reserve(capacity).map_err(|_| Status::OutOfMemory)?;
    Ok(vec)
}

fn try_clone_slice<T: Copy>(slice: &[T]) -> Result<Vec<T>, Status> {
    let mut out = try_vec_with_capacity(slice.len())?;
    out.extend_from_slice(slice);
    Ok(out)
}

fn activate_device(device_ordinal: i32) -> Result<(), Status> {
    if device_ordinal < 0 {
        return Ok(());
    }
    result_from_cuda(unsafe { cuda::cudaSetDevice(device_ordinal) })
}

fn resolve_device_ordinal(device_ordinal: i32) -> Result<i32, Status> {
    if device_ordinal >= 0 {
        return Ok(device_ordinal);
    }
    let mut current = 0;
    result_from_cuda(unsafe { cuda::cudaGetDevice(&mut current) })?;
    Ok(current)
}

fn synchronize_stream(stream: *mut c_void) -> Result<(), Status> {
    result_from_cuda(unsafe { cuda::cudaStreamSynchronize(stream) })
}

fn result_from_cuda(err: i32) -> Result<(), Status> {
    if err == cuda::CUDA_SUCCESS {
        Ok(())
    } else if err == cuda::CUDA_ERROR_MEMORY_ALLOCATION {
        Err(Status::OutOfMemory)
    } else {
        Err(Status::CudaError)
    }
}
