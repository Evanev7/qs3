use crate::{
    QWEN36_FULL_ATTN_GROUP_SIZE, QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS,
    QWEN36_FULL_ATTN_KV_HIDDEN, QWEN36_FULL_ATTN_Q_HEADS, QWEN36_FULL_ATTN_Q_HIDDEN,
    QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_NUM_V_HEADS,
    QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM, QWEN36_HIDDEN_SIZE,
    QWEN36_MOE_INTERMEDIATE_SIZE, QWEN36_MOE_MAX_EXPERTS, QWEN36_MOE_MAX_TOP_K,
    QWEN36_MOE_NUM_EXPERTS, QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, QWEN36_MOE_TOP_K,
    engine::{
        DynDType, EngineConfig, KvLayout, Status, validate_supported_attention_grouping,
        validate_supported_attention_head_dim,
    },
    model::resolve_device_ordinal,
};

use std::{ffi::c_void, ptr};

/// Inference-relevant Qwen3.6 MoE fields from HF config.json.
/// `output_router_logits` and `router_aux_loss_coef` are omitted because they
/// do not change token inference execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenMoeConfig {
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
}

impl QwenMoeConfig {
    const fn randomized_tiny_fixture() -> Self {
        Self {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        }
    }

    pub const fn qwen36_35b_a3b() -> Self {
        Self {
            num_experts: QWEN36_MOE_NUM_EXPERTS,
            num_experts_per_tok: QWEN36_MOE_TOP_K,
            moe_intermediate_size: QWEN36_MOE_INTERMEDIATE_SIZE,
            shared_expert_intermediate_size: QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE,
        }
    }

    pub(super) fn validate(self, hidden_size: u32) -> Result<(), Status> {
        if self.num_experts == 0 || self.num_experts_per_tok == 0 || self.moe_intermediate_size == 0
        {
            return Err(Status::InvalidArgument);
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(Status::InvalidArgument);
        }
        if self.num_experts_per_tok > QWEN36_MOE_MAX_TOP_K
            || self.num_experts > QWEN36_MOE_MAX_EXPERTS
        {
            return Err(Status::Unsupported);
        }
        if !hidden_size.is_multiple_of(8)
            || !self.moe_intermediate_size.is_multiple_of(8)
            || !self.shared_expert_intermediate_size.is_multiple_of(8)
        {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct QwenGdnShape {
    num_key_heads: u32,
    num_value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    conv_kernel_dim: u32,
}

impl QwenGdnShape {
    const fn qwen36_moe() -> Self {
        Self {
            num_key_heads: QWEN36_GDN_NUM_K_HEADS,
            num_value_heads: QWEN36_GDN_NUM_V_HEADS,
            key_head_dim: QWEN36_GDN_KEY_DIM,
            value_head_dim: QWEN36_GDN_VALUE_DIM,
            conv_kernel_dim: QWEN36_GDN_CONV_WIDTH,
        }
    }

    #[cfg(test)]
    pub(super) const fn qwen36_dense_27b() -> Self {
        Self {
            num_key_heads: QWEN36_GDN_NUM_K_HEADS,
            num_value_heads: 48,
            key_head_dim: QWEN36_GDN_KEY_DIM,
            value_head_dim: QWEN36_GDN_VALUE_DIM,
            conv_kernel_dim: QWEN36_GDN_CONV_WIDTH,
        }
    }

    pub(super) fn packed_dim(self) -> Result<u32, Status> {
        let qk = self
            .num_key_heads
            .checked_mul(self.key_head_dim)
            .and_then(|dim| dim.checked_mul(2))
            .ok_or(Status::InvalidArgument)?;
        let v = self.output_dim()?;
        qk.checked_add(v).ok_or(Status::InvalidArgument)
    }

    pub(super) fn output_dim(self) -> Result<u32, Status> {
        self.num_value_heads
            .checked_mul(self.value_head_dim)
            .ok_or(Status::InvalidArgument)
    }

    pub(super) fn validate_supported_runner_shape(self) -> Result<(), Status> {
        if self != Self::qwen36_moe() {
            return Err(Status::Unsupported);
        }
        Ok(())
    }

    pub(super) fn validate_config(self, config: &QwenConfig) -> Result<(), Status> {
        self.validate_supported_runner_shape()?;
        if self.packed_dim()? != QWEN36_GDN_PACKED_DIM
            || self.output_dim()? != QWEN36_GDN_OUTPUT_DIM
            || self.conv_kernel_dim != QWEN36_GDN_CONV_WIDTH
        {
            return Err(Status::InternalError);
        }
        if config.hidden_size != QWEN36_HIDDEN_SIZE {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QwenLayerPattern {
    FullAttentionOnly,
    Qwen36HybridGdn { gdn: QwenGdnShape },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QwenBlockKind {
    FullAttention,
    LinearAttention,
}

impl QwenLayerPattern {
    pub(super) fn validate_schedule(self, num_layers: u32) -> Result<(), Status> {
        match self {
            Self::FullAttentionOnly => Ok(()),
            Self::Qwen36HybridGdn { .. } => {
                if num_layers.is_multiple_of(4) {
                    Ok(())
                } else {
                    Err(Status::InvalidArgument)
                }
            }
        }
    }

    pub(super) fn block_kind(self, layer_idx: u32) -> QwenBlockKind {
        match self {
            Self::FullAttentionOnly => QwenBlockKind::FullAttention,
            // Qwen3.5/Qwen3.6 hybrid GDN layers repeat three linear-attention
            // blocks followed by one full-attention block.
            Self::Qwen36HybridGdn { .. } => {
                if layer_idx % 4 == 3 {
                    QwenBlockKind::FullAttention
                } else {
                    QwenBlockKind::LinearAttention
                }
            }
        }
    }

    pub(super) fn full_attention_layer_count(self, num_layers: u32) -> u32 {
        match self {
            Self::FullAttentionOnly => num_layers,
            Self::Qwen36HybridGdn { .. } => num_layers / 4,
        }
    }

    pub(super) fn gdn_layer_count(self, num_layers: u32) -> u32 {
        num_layers - self.full_attention_layer_count(num_layers)
    }

    pub(super) fn full_attention_layer_index(self, model_layer_idx: u32) -> Option<u32> {
        match self {
            Self::FullAttentionOnly => Some(model_layer_idx),
            Self::Qwen36HybridGdn { .. } => {
                if model_layer_idx % 4 == 3 {
                    Some(model_layer_idx / 4)
                } else {
                    None
                }
            }
        }
    }

    pub(super) fn gdn_layer_index(self, model_layer_idx: u32) -> Option<u32> {
        match self {
            Self::FullAttentionOnly => None,
            Self::Qwen36HybridGdn { .. } => {
                if model_layer_idx % 4 == 3 {
                    None
                } else {
                    Some(model_layer_idx - model_layer_idx / 4)
                }
            }
        }
    }

    pub(super) fn gdn_shape(self) -> Option<QwenGdnShape> {
        match self {
            Self::FullAttentionOnly => None,
            Self::Qwen36HybridGdn { gdn } => Some(gdn),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct QwenModelShape {
    pub(super) layer_pattern: QwenLayerPattern,
}

impl QwenModelShape {
    const fn full_attention_only() -> Self {
        Self {
            layer_pattern: QwenLayerPattern::FullAttentionOnly,
        }
    }

    const fn qwen36_moe_gdn() -> Self {
        Self {
            layer_pattern: QwenLayerPattern::Qwen36HybridGdn {
                gdn: QwenGdnShape::qwen36_moe(),
            },
        }
    }

    pub(super) fn validate(self, config: &QwenConfig) -> Result<(), Status> {
        self.layer_pattern.validate_schedule(config.num_layers)?;
        match config.moe {
            Some(moe) => {
                moe.validate(config.hidden_size)?;
                if config.intermediate_size != moe.moe_intermediate_size {
                    return Err(Status::InvalidArgument);
                }
            }
            None => {
                if !config.hidden_size.is_multiple_of(8)
                    || !config.intermediate_size.is_multiple_of(8)
                {
                    return Err(Status::InvalidArgument);
                }
            }
        }
        if let Some(gdn) = self.layer_pattern.gdn_shape() {
            if config.moe != Some(QwenMoeConfig::qwen36_35b_a3b()) {
                return Err(Status::Unsupported);
            }
            gdn.validate_config(config)?;
        }
        config.validate_full_attention_shape()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QwenConfig {
    pub device_ordinal: i32,
    pub stream: *mut c_void,
    pub num_layers: u32,
    pub max_live_requests: u32,
    pub max_batch_rows: u32,
    pub max_batch_tokens: u32,
    pub max_seq_len: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub moe: Option<QwenMoeConfig>,
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
    pub(super) model_shape: QwenModelShape,
}

impl QwenConfig {
    pub(crate) fn qwen36_bf16_runtime(
        device_ordinal: i32,
        stream: *mut c_void,
        num_layers: u32,
        vocab_size: u32,
        rms_norm_eps: f32,
        rope_theta: f32,
        logits_soft_cap: f32,
        max_seq_len: u32,
    ) -> Result<Self, Status> {
        let page_size = 4;
        let config = Self {
            device_ordinal,
            stream,
            num_layers,
            max_live_requests: 1,
            max_batch_rows: 1,
            max_batch_tokens: max_seq_len,
            max_seq_len,
            max_pages: max_seq_len.div_ceil(page_size),
            page_size,
            hidden_size: QWEN36_HIDDEN_SIZE,
            intermediate_size: QWEN36_MOE_INTERMEDIATE_SIZE,
            moe: Some(QwenMoeConfig::qwen36_35b_a3b()),
            vocab_size,
            num_q_heads: QWEN36_FULL_ATTN_Q_HEADS,
            num_kv_heads: QWEN36_FULL_ATTN_KV_HEADS,
            head_dim: QWEN36_FULL_ATTN_HEAD_DIM,
            rms_norm_eps,
            rope_theta,
            rope_scale: 1.0,
            logits_soft_cap,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
            qscb_workspace_bytes: 64 << 20,
            model_shape: QwenModelShape::qwen36_moe_gdn(),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn randomized_dense_tiny_fixture(device_ordinal: i32) -> Self {
        Self {
            device_ordinal,
            stream: ptr::null_mut(),
            num_layers: 2,
            max_live_requests: 1,
            max_batch_rows: 1,
            max_batch_tokens: 16,
            max_seq_len: 16,
            max_pages: 8,
            page_size: 4,
            hidden_size: QWEN36_HIDDEN_SIZE,
            intermediate_size: 256,
            moe: None,
            vocab_size: 64,
            num_q_heads: QWEN36_FULL_ATTN_Q_HEADS,
            num_kv_heads: QWEN36_FULL_ATTN_KV_HEADS,
            head_dim: QWEN36_FULL_ATTN_HEAD_DIM,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            logits_soft_cap: 0.0,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
            qscb_workspace_bytes: 16 << 20,
            model_shape: QwenModelShape::full_attention_only(),
        }
    }

    pub fn randomized_moe_tiny_fixture(device_ordinal: i32) -> Self {
        let moe = QwenMoeConfig::randomized_tiny_fixture();
        let mut config = Self::randomized_dense_tiny_fixture(device_ordinal);
        config.intermediate_size = moe.moe_intermediate_size;
        config.moe = Some(moe);
        config
    }

    pub fn randomized_shared_moe_tiny_fixture(device_ordinal: i32) -> Self {
        let mut config = Self::randomized_moe_tiny_fixture(device_ordinal);
        config.moe = Some(QwenMoeConfig {
            shared_expert_intermediate_size: 32,
            ..QwenMoeConfig::randomized_tiny_fixture()
        });
        config
    }

    pub fn randomized_qwen36_moe_gdn_one_block_fixture(device_ordinal: i32) -> Self {
        let moe = QwenMoeConfig::qwen36_35b_a3b();
        Self {
            device_ordinal,
            stream: ptr::null_mut(),
            num_layers: 4,
            max_live_requests: 1,
            max_batch_rows: 1,
            max_batch_tokens: 8,
            max_seq_len: 8,
            max_pages: 2,
            page_size: 4,
            hidden_size: QWEN36_HIDDEN_SIZE,
            intermediate_size: moe.moe_intermediate_size,
            moe: Some(moe),
            vocab_size: 32,
            num_q_heads: QWEN36_FULL_ATTN_Q_HEADS,
            num_kv_heads: QWEN36_FULL_ATTN_KV_HEADS,
            head_dim: QWEN36_FULL_ATTN_HEAD_DIM,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            logits_soft_cap: 0.0,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
            qscb_workspace_bytes: 64 << 20,
            model_shape: QwenModelShape::qwen36_moe_gdn(),
        }
    }

    pub fn validate(&self) -> Result<(), Status> {
        if self.num_layers == 0
            || self.max_live_requests == 0
            || self.max_batch_rows == 0
            || self.max_batch_tokens == 0
            || self.max_seq_len == 0
            || self.max_pages == 0
            || self.page_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
        {
            return Err(Status::InvalidArgument);
        }
        if self.attention_layer_count() == 0
            || self.num_q_heads == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(Status::InvalidArgument);
        }
        if self.max_live_requests != 1 || self.max_batch_rows != 1 {
            return Err(Status::Unsupported);
        }
        if self.max_batch_tokens < self.max_batch_rows {
            return Err(Status::InvalidArgument);
        }
        if self.vocab_size > i32::MAX as u32 {
            return Err(Status::Unsupported);
        }
        self.model_shape.validate(self)?;
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

    pub(super) fn resolved_device_config(&self) -> Result<Self, Status> {
        let mut config = *self;
        config.device_ordinal = resolve_device_ordinal(config.device_ordinal)?;
        Ok(config)
    }

    pub(super) fn kv_hidden_size(&self) -> Result<u32, Status> {
        self.num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or(Status::InvalidArgument)
    }

    pub(super) fn q_hidden_size(&self) -> Result<u32, Status> {
        self.num_q_heads
            .checked_mul(self.head_dim)
            .ok_or(Status::InvalidArgument)
    }

    pub(super) fn validate_full_attention_shape(&self) -> Result<(), Status> {
        if self.hidden_size != QWEN36_HIDDEN_SIZE {
            return Err(Status::InvalidArgument);
        }
        validate_supported_attention_grouping(self.num_q_heads, self.num_kv_heads)?;
        if self.num_q_heads / self.num_kv_heads != QWEN36_FULL_ATTN_GROUP_SIZE {
            return Err(Status::Unsupported);
        }
        validate_supported_attention_head_dim(self.head_dim)?;
        if self.head_dim != QWEN36_FULL_ATTN_HEAD_DIM {
            return Err(Status::Unsupported);
        }
        let q_hidden = self.q_hidden_size()?;
        let kv_hidden = self.kv_hidden_size()?;
        if q_hidden != QWEN36_FULL_ATTN_Q_HIDDEN || kv_hidden != QWEN36_FULL_ATTN_KV_HIDDEN {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }

    pub(super) fn layer_kind(&self, layer_idx: u32) -> QwenBlockKind {
        self.model_shape.layer_pattern.block_kind(layer_idx)
    }

    pub(super) fn has_gdn_layers(&self) -> bool {
        self.gdn_layer_count() != 0
    }

    pub(super) fn attention_layer_count(&self) -> u32 {
        self.model_shape
            .layer_pattern
            .full_attention_layer_count(self.num_layers)
    }

    pub(super) fn gdn_layer_count(&self) -> u32 {
        self.model_shape
            .layer_pattern
            .gdn_layer_count(self.num_layers)
    }

    pub(super) fn attention_layer_index(&self, model_layer_idx: u32) -> Result<u32, Status> {
        if model_layer_idx >= self.num_layers {
            return Err(Status::InvalidArgument);
        }
        self.model_shape
            .layer_pattern
            .full_attention_layer_index(model_layer_idx)
            .ok_or(Status::InternalError)
    }

    pub(super) fn gdn_layer_index(&self, model_layer_idx: u32) -> Result<u32, Status> {
        if model_layer_idx >= self.num_layers {
            return Err(Status::InvalidArgument);
        }
        self.model_shape
            .layer_pattern
            .gdn_layer_index(model_layer_idx)
            .ok_or(Status::InternalError)
    }

    pub(super) fn moe_config(&self) -> Option<QwenMoeConfig> {
        self.moe
    }

    pub(super) fn engine_config(&self) -> EngineConfig {
        let attention_layers = self.attention_layer_count();
        EngineConfig {
            device_ordinal: self.device_ordinal,
            stream: self.stream,
            num_layers: attention_layers,
            max_live_requests: self.max_live_requests,
            max_batch_rows: self.max_batch_rows,
            max_batch_tokens: self.max_batch_tokens,
            max_seq_len: self.max_seq_len,
            max_pages: self.max_pages,
            page_size: self.page_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            activation_dtype: DynDType::BF16,
            kv_dtype: DynDType::BF16,
            kv_layout: KvLayout::NHD,
            rope_theta: self.rope_theta,
            rope_scale: self.rope_scale,
            logits_soft_cap: self.logits_soft_cap,
            qsfi_float_workspace_bytes: self.qsfi_float_workspace_bytes,
            qsfi_int_workspace_bytes: self.qsfi_int_workspace_bytes,
            qsfi_host_int_workspace_bytes: self.qsfi_host_int_workspace_bytes,
        }
    }

    pub(super) fn same_model_shape(&self, other: &Self) -> bool {
        self.num_layers == other.num_layers
            && self.hidden_size == other.hidden_size
            && self.intermediate_size == other.intermediate_size
            && self.moe == other.moe
            && self.model_shape == other.model_shape
            && self.vocab_size == other.vocab_size
            && self.num_q_heads == other.num_q_heads
            && self.num_kv_heads == other.num_kv_heads
            && self.head_dim == other.head_dim
    }
}
