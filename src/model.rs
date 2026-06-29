use crate::engine::{
    AppendBatch, Commit, DType, DecodeBatch, Engine, EngineConfig, EngineLayer, EngineTrait,
    KvLayout, RequestId, Status, validate_supported_attention_grouping,
    validate_supported_attention_head_dim,
};
use crate::ffi::{self, cuda};
use crate::runtime::device_tensor::{DMat, DTensor3, DVec};
use crate::runtime::kernels::{
    Activation, Bf16Gemm, Bf16Heads, Bf16OrF32Mat, Bf16OrF32Vec, EmbeddingGatherBf16, FloatStorage,
    FusedAddRmsNormBf16, GdnCausalConv1dBf16, GdnCausalConv1dBf16Args, GdnConvState, GdnDecodeBf16,
    GdnDecodeBf16Args, GdnForgetGateOutput, GdnPostConvPrepareBf16, GdnPostConvPrepareBf16Args,
    GdnPrefillBf16, GdnPrefillBf16Args, GdnRecurrentState, GdnRmsNormGatedBf16,
    GdnRmsNormGatedBf16Args, GreedyArgmaxF32, LogitsSoftCapF32, MoeBf16Execute, MoeBf16ExecuteArgs,
    MoeBf16PlanConfig, MoePlan, RmsNormBf16, RouterScore, RouterTopK, SiluAndMulBf16, Workspace,
};

use std::ffi::c_void;
use std::{mem, ptr};

const QWEN_MOE_MAX_TOP_K: u32 = 16;
const QWEN_MOE_MAX_EXPERTS: u32 = 4096;
const QWEN36_GDN_NUM_Q_HEADS: u32 = 16;
const QWEN36_GDN_NUM_K_HEADS: u32 = 16;
const QWEN36_GDN_NUM_V_HEADS: u32 = 32;
const QWEN36_GDN_KEY_DIM: u32 = 128;
const QWEN36_GDN_VALUE_DIM: u32 = 128;
const QWEN36_GDN_CONV_WIDTH: u32 = 4;
const QWEN36_GDN_CONV_STATE: u32 = QWEN36_GDN_CONV_WIDTH - 1;
const QWEN36_GDN_PACKED_DIM: u32 =
    2 * QWEN36_GDN_NUM_K_HEADS * QWEN36_GDN_KEY_DIM + QWEN36_GDN_NUM_V_HEADS * QWEN36_GDN_VALUE_DIM;
const QWEN36_GDN_OUTPUT_DIM: u32 = QWEN36_GDN_NUM_V_HEADS * QWEN36_GDN_VALUE_DIM;
const QWEN36_GDN_STATE_SLOTS_PER_LAYER: u32 = 2;
const GDN_ENGINE_NUM_Q_HEADS: u32 = 2;
const GDN_ENGINE_NUM_KV_HEADS: u32 = 2;
const GDN_ENGINE_HEAD_DIM: u32 = 64;
const QWEN36_MOE_ROUTER_SCORE: RouterScore = RouterScore::Softmax;
const QWEN36_MOE_ROUTER_RENORMALIZE: bool = true;
const QWEN36_MOE_ROUTER_SCALING_FACTOR: f32 = 1.0;
const _: () = assert!(QWEN36_GDN_PACKED_DIM == 8192);
const _: () = assert!(QWEN36_GDN_OUTPUT_DIM == 4096);
const _: () = assert!(QWEN36_GDN_NUM_Q_HEADS == 16);
const _: () = assert!(QWEN36_GDN_NUM_K_HEADS == 16);
const _: () = assert!(QWEN36_GDN_NUM_V_HEADS == 32);
const _: () = assert!(QWEN36_GDN_KEY_DIM == 128);
const _: () = assert!(QWEN36_GDN_VALUE_DIM == 128);
const _: () = assert!(QWEN36_GDN_CONV_STATE == 3);

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
            num_experts: 256,
            num_experts_per_tok: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
        }
    }

    fn validate(self, hidden_size: u32) -> Result<(), Status> {
        if self.num_experts == 0 || self.num_experts_per_tok == 0 || self.moe_intermediate_size == 0
        {
            return Err(Status::InvalidArgument);
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(Status::InvalidArgument);
        }
        if self.num_experts_per_tok > QWEN_MOE_MAX_TOP_K || self.num_experts > QWEN_MOE_MAX_EXPERTS
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
struct QwenGdnShape {
    num_key_heads: u32,
    num_value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    conv_kernel_dim: u32,
}

impl QwenGdnShape {
    const fn qwen36_moe() -> Self {
        Self {
            num_key_heads: 16,
            num_value_heads: 32,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
        }
    }

    #[cfg(test)]
    const fn qwen36_dense_27b() -> Self {
        Self {
            num_key_heads: 16,
            num_value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
        }
    }

    fn packed_dim(self) -> Result<u32, Status> {
        let qk = self
            .num_key_heads
            .checked_mul(self.key_head_dim)
            .and_then(|dim| dim.checked_mul(2))
            .ok_or(Status::InvalidArgument)?;
        let v = self.output_dim()?;
        qk.checked_add(v).ok_or(Status::InvalidArgument)
    }

    fn output_dim(self) -> Result<u32, Status> {
        self.num_value_heads
            .checked_mul(self.value_head_dim)
            .ok_or(Status::InvalidArgument)
    }

    fn validate_supported_runner_shape(self) -> Result<(), Status> {
        if self != Self::qwen36_moe() {
            return Err(Status::Unsupported);
        }
        Ok(())
    }

    fn validate_config(self, config: &QwenConfig) -> Result<(), Status> {
        self.validate_supported_runner_shape()?;
        if self.packed_dim()? != QWEN36_GDN_PACKED_DIM
            || self.output_dim()? != QWEN36_GDN_OUTPUT_DIM
            || self.conv_kernel_dim != QWEN36_GDN_CONV_WIDTH
        {
            return Err(Status::InternalError);
        }
        if config.hidden_size != self.packed_dim()?
            || config.num_q_heads != self.num_key_heads
            || config.num_kv_heads != self.num_key_heads
            || config.head_dim != self.key_head_dim
        {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenLayerPattern {
    FullAttentionOnly,
    Qwen36HybridGdn { gdn: QwenGdnShape },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenBlockKind {
    FullAttention,
    LinearAttention,
}

impl QwenLayerPattern {
    fn block_kind(self, layer_idx: u32) -> QwenBlockKind {
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

    fn gdn_shape(self) -> Option<QwenGdnShape> {
        match self {
            Self::FullAttentionOnly => None,
            Self::Qwen36HybridGdn { gdn } => Some(gdn),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QwenModelShape {
    layer_pattern: QwenLayerPattern,
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

    fn validate(self, config: &QwenConfig) -> Result<(), Status> {
        match config.moe {
            Some(moe) => {
                moe.validate(config.hidden_size)?;
                if config.intermediate_size != moe.moe_intermediate_size {
                    return Err(Status::InvalidArgument);
                }
                if config.has_full_attention_layers() && moe.shared_expert_intermediate_size != 0 {
                    return Err(Status::Unsupported);
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
        } else {
            config.validate_full_attention_shape()?;
        }
        Ok(())
    }
}

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
    model_shape: QwenModelShape,
}

impl QwenConfig {
    pub fn randomized_dense_tiny_fixture(device_ordinal: i32) -> Self {
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
            moe: None,
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

    pub fn randomized_qwen36_moe_gdn_one_block_fixture(device_ordinal: i32) -> Self {
        let moe = QwenMoeConfig::qwen36_35b_a3b();
        Self {
            device_ordinal,
            stream: ptr::null_mut(),
            num_layers: 1,
            max_live_requests: 1,
            max_batch_size: 1,
            max_seq_len: 8,
            max_pages: 2,
            page_size: 4,
            hidden_size: QWEN36_GDN_PACKED_DIM,
            intermediate_size: moe.moe_intermediate_size,
            moe: Some(moe),
            vocab_size: 32,
            num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
            num_kv_heads: QWEN36_GDN_NUM_K_HEADS,
            head_dim: QWEN36_GDN_KEY_DIM,
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
        if self.has_gdn_layers() && self.has_full_attention_layers() {
            return Err(Status::Unsupported);
        }
        Ok(())
    }

    fn resolved_device_config(&self) -> Result<Self, Status> {
        let mut config = *self;
        config.device_ordinal = resolve_device_ordinal(config.device_ordinal)?;
        Ok(config)
    }

    fn kv_hidden_size(&self) -> Result<u32, Status> {
        self.num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or(Status::InvalidArgument)
    }

    fn validate_full_attention_shape(&self) -> Result<(), Status> {
        validate_supported_attention_grouping(self.num_q_heads, self.num_kv_heads)?;
        let q_hidden = self
            .num_q_heads
            .checked_mul(self.head_dim)
            .ok_or(Status::InvalidArgument)?;
        if q_hidden != self.hidden_size {
            return Err(Status::InvalidArgument);
        }
        let _ = self.kv_hidden_size()?;
        validate_supported_attention_head_dim(self.head_dim)
    }

    fn layer_kind(&self, layer_idx: u32) -> QwenBlockKind {
        self.model_shape.layer_pattern.block_kind(layer_idx)
    }

    fn has_gdn_layers(&self) -> bool {
        (0..self.num_layers)
            .any(|layer_idx| self.layer_kind(layer_idx) == QwenBlockKind::LinearAttention)
    }

    fn has_full_attention_layers(&self) -> bool {
        (0..self.num_layers)
            .any(|layer_idx| self.layer_kind(layer_idx) == QwenBlockKind::FullAttention)
    }

    fn moe_config(&self) -> Option<QwenMoeConfig> {
        self.moe
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
            num_q_heads: if self.has_gdn_layers() {
                GDN_ENGINE_NUM_Q_HEADS
            } else {
                self.num_q_heads
            },
            num_kv_heads: if self.has_gdn_layers() {
                GDN_ENGINE_NUM_KV_HEADS
            } else {
                self.num_kv_heads
            },
            head_dim: if self.has_gdn_layers() {
                GDN_ENGINE_HEAD_DIM
            } else {
                self.head_dim
            },
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
            && self.moe == other.moe
            && self.model_shape == other.model_shape
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
        let intermediate = config.intermediate_size;
        let moe = config.moe_config();
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
        for layer_idx in 0..config.num_layers {
            if config.layer_kind(layer_idx) == QwenBlockKind::LinearAttention {
                layers.push(QwenLayerWeights::Gdn(QwenGdnWeights {
                    norm: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(hidden as usize, 1.0)?,
                    )?,
                    in_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_PACKED_DIM, hidden])?,
                            0.01,
                        )?,
                    )?,
                    gate_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_OUTPUT_DIM, hidden])?,
                            0.01,
                        )?,
                    )?,
                    a_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_NUM_V_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    b_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_NUM_V_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    conv_weight: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH])?,
                            0.25,
                        )?,
                    )?,
                    conv_bias: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(QWEN36_GDN_PACKED_DIM as usize, 0.0)?,
                    )?,
                    a_log: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_f32_values(QWEN36_GDN_NUM_V_HEADS as usize, -2.0)?,
                    )?,
                    dt_bias: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_f32_values(QWEN36_GDN_NUM_V_HEADS as usize, -1.0)?,
                    )?,
                    rms_weight: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(QWEN36_GDN_VALUE_DIM as usize, 1.0)?,
                    )?,
                    out_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[hidden, QWEN36_GDN_OUTPUT_DIM])?,
                            0.01,
                        )?,
                    )?,
                }));
                continue;
            }

            let kv_hidden = config.kv_hidden_size()?;
            let mlp = if let Some(moe) = moe {
                QwenMlpWeights::Moe {
                    router_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[moe.num_experts, hidden])?,
                            0.03,
                        )?,
                    )?,
                    gate_up_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[
                                moe.num_experts,
                                2,
                                moe.moe_intermediate_size,
                                hidden,
                            ])?,
                            0.03,
                        )?,
                    )?,
                    down_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[
                                moe.num_experts,
                                hidden,
                                moe.moe_intermediate_size,
                            ])?,
                            0.03,
                        )?,
                    )?,
                }
            } else {
                QwenMlpWeights::Dense {
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
                }
            };

            layers.push(QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
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
                mlp,
            }));
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
        let expected_layers = config.num_layers as usize;
        if self.layers.len() != expected_layers {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

enum QwenLayerWeights {
    AttentionMlp(QwenAttentionMlpWeights),
    Gdn(QwenGdnWeights),
}

struct QwenAttentionMlpWeights {
    attn_norm: DeviceBuffer<u16>,
    q_proj: DeviceBuffer<u16>,
    k_proj: DeviceBuffer<u16>,
    v_proj: DeviceBuffer<u16>,
    o_proj: DeviceBuffer<u16>,
    mlp_norm: DeviceBuffer<u16>,
    mlp: QwenMlpWeights,
}

struct QwenGdnWeights {
    norm: DeviceBuffer<u16>,
    in_proj: DeviceBuffer<u16>,
    gate_proj: DeviceBuffer<u16>,
    a_proj: DeviceBuffer<u16>,
    b_proj: DeviceBuffer<u16>,
    conv_weight: DeviceBuffer<u16>,
    conv_bias: DeviceBuffer<u16>,
    a_log: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    rms_weight: DeviceBuffer<u16>,
    out_proj: DeviceBuffer<u16>,
}

enum QwenMlpWeights {
    Dense {
        gate_proj: DeviceBuffer<u16>,
        up_proj: DeviceBuffer<u16>,
        down_proj: DeviceBuffer<u16>,
    },
    Moe {
        router_proj: DeviceBuffer<u16>,
        gate_up_proj: DeviceBuffer<u16>,
        down_proj: DeviceBuffer<u16>,
    },
}

#[derive(Clone, Copy)]
enum QwenMlpPtrs {
    Dense {
        gate_proj: ffi::DevicePtr,
        up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    },
    Moe {
        router_proj: ffi::DevicePtr,
        gate_up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    },
}

impl QwenMlpWeights {
    fn ptrs(&self) -> QwenMlpPtrs {
        match self {
            Self::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => QwenMlpPtrs::Dense {
                gate_proj: gate_proj.as_device_ptr(),
                up_proj: up_proj.as_device_ptr(),
                down_proj: down_proj.as_device_ptr(),
            },
            Self::Moe {
                router_proj,
                gate_up_proj,
                down_proj,
            } => QwenMlpPtrs::Moe {
                router_proj: router_proj.as_device_ptr(),
                gate_up_proj: gate_up_proj.as_device_ptr(),
                down_proj: down_proj.as_device_ptr(),
            },
        }
    }
}

#[derive(Clone, Copy)]
enum QwenLayerPtrs {
    AttentionMlp(QwenAttentionMlpPtrs),
    Gdn(QwenGdnPtrs),
}

#[derive(Clone, Copy)]
struct QwenAttentionMlpPtrs {
    attn_norm: ffi::DevicePtr,
    q_proj: ffi::DevicePtr,
    k_proj: ffi::DevicePtr,
    v_proj: ffi::DevicePtr,
    o_proj: ffi::DevicePtr,
    mlp_norm: ffi::DevicePtr,
    mlp: QwenMlpPtrs,
}

impl QwenLayerWeights {
    fn ptrs(&self) -> QwenLayerPtrs {
        match self {
            Self::AttentionMlp(layer) => QwenLayerPtrs::AttentionMlp(QwenAttentionMlpPtrs {
                attn_norm: layer.attn_norm.as_device_ptr(),
                q_proj: layer.q_proj.as_device_ptr(),
                k_proj: layer.k_proj.as_device_ptr(),
                v_proj: layer.v_proj.as_device_ptr(),
                o_proj: layer.o_proj.as_device_ptr(),
                mlp_norm: layer.mlp_norm.as_device_ptr(),
                mlp: layer.mlp.ptrs(),
            }),
            Self::Gdn(layer) => QwenLayerPtrs::Gdn(QwenGdnPtrs {
                norm: layer.norm.as_device_ptr(),
                in_proj: layer.in_proj.as_device_ptr(),
                gate_proj: layer.gate_proj.as_device_ptr(),
                a_proj: layer.a_proj.as_device_ptr(),
                b_proj: layer.b_proj.as_device_ptr(),
                conv_weight: layer.conv_weight.as_device_ptr(),
                conv_bias: layer.conv_bias.as_device_ptr(),
                a_log: layer.a_log.as_device_ptr(),
                dt_bias: layer.dt_bias.as_device_ptr(),
                rms_weight: layer.rms_weight.as_device_ptr(),
                out_proj: layer.out_proj.as_device_ptr(),
            }),
        }
    }
}

impl QwenLayerPtrs {
    fn input_norm(self) -> ffi::DevicePtr {
        match self {
            Self::AttentionMlp(layer) => layer.attn_norm,
            Self::Gdn(layer) => layer.norm,
        }
    }
}

#[derive(Clone, Copy)]
struct QwenGdnPtrs {
    norm: ffi::DevicePtr,
    in_proj: ffi::DevicePtr,
    gate_proj: ffi::DevicePtr,
    a_proj: ffi::DevicePtr,
    b_proj: ffi::DevicePtr,
    conv_weight: ffi::DevicePtr,
    conv_bias: ffi::DevicePtr,
    a_log: ffi::DevicePtr,
    dt_bias: ffi::DevicePtr,
    rms_weight: ffi::DevicePtr,
    out_proj: ffi::DevicePtr,
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
        let moe_plan = if let Some(moe) = config
            .moe_config()
            .filter(|_| config.has_full_attention_layers())
        {
            let (plan, workspace_bytes) = {
                let mut ops = engine.kernel_ops();
                let plan = unsafe {
                    ops.create_moe_bf16_plan(MoeBf16PlanConfig {
                        max_num_tokens: config.max_seq_len,
                        hidden_size: config.hidden_size,
                        intermediate_size: moe.moe_intermediate_size,
                        num_experts: moe.num_experts,
                        top_k: moe.num_experts_per_tok,
                    })?
                };
                let workspace_bytes = unsafe { ops.moe_workspace_size(&plan, config.max_seq_len)? };
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

        let mut generated_tokens = try_vec_with_capacity(request.max_new_tokens as usize)?;
        self.live_tokens
            .try_reserve((total_tokens as usize).saturating_sub(self.live_tokens.len()))
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
            .try_reserve(total_tokens as usize)
            .map_err(|_| Status::OutOfMemory)?;

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
                self.commit_gdn_state();
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
                self.commit_gdn_state();
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
                    let kv_hidden = self.config.kv_hidden_size()?;
                    self.execute_attention_layer(
                        layer_idx,
                        rows,
                        hidden,
                        kv_hidden,
                        layer_input,
                        layer,
                        run.kind,
                    )?;
                    self.fused_add_rmsnorm(
                        self.scratch.attn_proj.as_device_ptr(),
                        self.scratch.residual.as_device_ptr(),
                        layer.mlp_norm,
                        rows,
                    )?;
                    match layer.mlp {
                        QwenMlpPtrs::Dense {
                            gate_proj,
                            up_proj,
                            down_proj,
                        } => self.execute_dense_mlp(
                            rows,
                            hidden,
                            intermediate,
                            gate_proj,
                            up_proj,
                            down_proj,
                        )?,
                        QwenMlpPtrs::Moe {
                            router_proj,
                            gate_up_proj,
                            down_proj,
                        } => self.execute_moe_mlp(
                            rows,
                            hidden,
                            router_proj,
                            gate_up_proj,
                            down_proj,
                        )?,
                    }
                    self.fused_add_rmsnorm(
                        self.scratch.mlp_out.as_device_ptr(),
                        self.scratch.residual.as_device_ptr(),
                        next_weight,
                        rows,
                    )?;
                    layer_input = self.scratch.mlp_out.as_device_ptr();
                }
                QwenLayerPtrs::Gdn(layer) => {
                    self.execute_gdn_layer(layer_idx, rows, hidden, layer_input, layer, run.kind)?;
                    self.fused_add_rmsnorm(
                        self.scratch.attn_proj.as_device_ptr(),
                        self.scratch.residual.as_device_ptr(),
                        next_weight,
                        rows,
                    )?;
                    layer_input = self.scratch.attn_proj.as_device_ptr();
                }
            }
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

    fn execute_attention_layer(
        &mut self,
        layer_idx: u32,
        rows: u32,
        hidden: u32,
        kv_hidden: u32,
        layer_input: ffi::DevicePtr,
        layer: QwenAttentionMlpPtrs,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
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
            match kind {
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
        )
    }

    fn execute_gdn_layer(
        &mut self,
        layer_idx: u32,
        rows: u32,
        hidden: u32,
        layer_input: ffi::DevicePtr,
        layer: QwenGdnPtrs,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        if hidden != QWEN36_GDN_PACKED_DIM {
            return Err(Status::InvalidArgument);
        }
        if matches!(kind, ActiveRunKind::Decode) && rows != 1 {
            return Err(Status::InvalidArgument);
        }
        let (_state_pool, live_slot, staged_slot, conv_state, recurrent_state) =
            self.gdn_state_views(layer_idx)?;
        let live_slot_i32 = i32::try_from(live_slot).map_err(|_| Status::InvalidArgument)?;
        let staged_slot_i32 = i32::try_from(staged_slot).map_err(|_| Status::InvalidArgument)?;
        self.scratch
            .gdn_state_indices
            .upload(self.config.stream, &[live_slot_i32])?;
        self.scratch
            .gdn_state_out_indices
            .upload(self.config.stream, &[staged_slot_i32])?;

        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.in_proj,
            self.scratch.gdn_packed.as_device_ptr(),
            QWEN36_GDN_PACKED_DIM,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.a_proj,
            self.scratch.gdn_a.as_device_ptr(),
            QWEN36_GDN_NUM_V_HEADS,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.b_proj,
            self.scratch.gdn_b.as_device_ptr(),
            QWEN36_GDN_NUM_V_HEADS,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.gate_proj,
            self.scratch.gdn_gate.as_device_ptr(),
            QWEN36_GDN_OUTPUT_DIM,
            GemmOut::Bf16,
        )?;

        let seq_indptr = if matches!(kind, ActiveRunKind::Append) {
            let rows_i32 = i32::try_from(rows).map_err(|_| Status::InvalidArgument)?;
            self.scratch
                .gdn_seq_indptr
                .upload(self.config.stream, &[0, rows_i32])?;
            Some(DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                self.scratch.gdn_seq_indptr.as_device_ptr(),
                2,
            )?)
        } else {
            None
        };

        let conv = GdnCausalConv1dBf16::new(GdnCausalConv1dBf16Args {
            x: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.gdn_packed.as_device_ptr(),
                rows,
                QWEN36_GDN_PACKED_DIM,
            )?,
            weight: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                layer.conv_weight,
                QWEN36_GDN_PACKED_DIM,
                QWEN36_GDN_CONV_WIDTH,
            )?,
            bias: Some(Bf16OrF32Vec::Bf16(DVec::<{ ffi::DTYPE_BF16 }>::contiguous(
                layer.conv_bias,
                QWEN36_GDN_PACKED_DIM,
            )?)),
            state: conv_state,
            state_read_indices: Some(DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                self.scratch.gdn_state_indices.as_device_ptr(),
                1,
            )?),
            state_write_indices: Some(DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                self.scratch.gdn_state_out_indices.as_device_ptr(),
                1,
            )?),
            seq_indptr,
            out: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.gdn_conv_out.as_device_ptr(),
                rows,
                QWEN36_GDN_PACKED_DIM,
            )?,
            batch_size: 1,
            activation: Activation::Silu,
            update_state: true,
        })?;
        {
            let mut ops = self.engine.kernel_ops();
            unsafe { ops.qwen36_gdn_causal_conv1d_bf16(&conv)? };
        }

        let post = GdnPostConvPrepareBf16::new(GdnPostConvPrepareBf16Args {
            conv_out: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.gdn_conv_out.as_device_ptr(),
                rows,
                QWEN36_GDN_PACKED_DIM,
            )?,
            a: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.gdn_a.as_device_ptr(),
                rows,
                QWEN36_GDN_NUM_V_HEADS,
            )?,
            b: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.gdn_b.as_device_ptr(),
                rows,
                QWEN36_GDN_NUM_V_HEADS,
            )?,
            a_log: DVec::<{ ffi::DTYPE_F32 }>::contiguous(layer.a_log, QWEN36_GDN_NUM_V_HEADS)?,
            dt_bias: DVec::<{ ffi::DTYPE_F32 }>::contiguous(layer.dt_bias, QWEN36_GDN_NUM_V_HEADS)?,
            q: self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?,
            k: self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?,
            v: self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?,
            g_out: None,
            beta_out: None,
            apply_qk_l2norm: false,
            l2norm_eps: self.config.rms_norm_eps,
            forget_gate_output: GdnForgetGateOutput::LogDecay,
        })?;
        {
            let mut ops = self.engine.kernel_ops();
            unsafe { ops.qwen36_gdn_post_conv_prepare_bf16(&post)? };
        }

        match kind {
            ActiveRunKind::Append => {
                let rows_i32 = i32::try_from(rows).map_err(|_| Status::InvalidArgument)?;
                self.scratch
                    .gdn_seq_indptr
                    .upload(self.config.stream, &[0, rows_i32])?;
                let prefill = GdnPrefillBf16::new(GdnPrefillBf16Args {
                    q: self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?,
                    k: self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?,
                    v: self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?,
                    a: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                        self.scratch.gdn_a.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    b: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                        self.scratch.gdn_b.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    a_log: DVec::<{ ffi::DTYPE_F32 }>::contiguous(
                        layer.a_log,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    dt_bias: DVec::<{ ffi::DTYPE_F32 }>::contiguous(
                        layer.dt_bias,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    state: recurrent_state,
                    seq_indptr: DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                        self.scratch.gdn_seq_indptr.as_device_ptr(),
                        2,
                    )?,
                    state_indices: DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                        self.scratch.gdn_state_indices.as_device_ptr(),
                        1,
                    )?,
                    state_out_indices: Some(DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                        self.scratch.gdn_state_out_indices.as_device_ptr(),
                        1,
                    )?),
                    out: self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?,
                    batch_size: 1,
                    scale: qwen36_gdn_scale(),
                    use_qk_l2norm: true,
                    disable_state_update: false,
                })?;
                let mut ops = self.engine.kernel_ops();
                unsafe { ops.gdn_prefill_bf16(&prefill)? };
                self.engine
                    .complete_append_layer_without_attention(layer_idx)?;
            }
            ActiveRunKind::Decode => {
                let decode = GdnDecodeBf16::new(GdnDecodeBf16Args {
                    q: self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?,
                    k: self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?,
                    v: self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?,
                    a: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                        self.scratch.gdn_a.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    b: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                        self.scratch.gdn_b.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    a_log: DVec::<{ ffi::DTYPE_F32 }>::contiguous(
                        layer.a_log,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    dt_bias: DVec::<{ ffi::DTYPE_F32 }>::contiguous(
                        layer.dt_bias,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    state: recurrent_state,
                    state_indices: DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                        self.scratch.gdn_state_indices.as_device_ptr(),
                        rows,
                    )?,
                    state_out_indices: Some(DVec::<{ ffi::DTYPE_I32 }>::contiguous(
                        self.scratch.gdn_state_out_indices.as_device_ptr(),
                        rows,
                    )?),
                    out: self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?,
                    scale: qwen36_gdn_scale(),
                    use_qk_l2norm: true,
                    disable_state_update: false,
                })?;
                let mut ops = self.engine.kernel_ops();
                unsafe { ops.gdn_decode_bf16(&decode)? };
                self.engine
                    .complete_decode_layer_without_attention(layer_idx)?;
            }
        }

        let gated = GdnRmsNormGatedBf16::new(GdnRmsNormGatedBf16Args {
            x: self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?,
            gate: self.gdn_v_heads(self.scratch.gdn_gate.as_device_ptr(), rows)?,
            weight: Bf16OrF32Vec::Bf16(DVec::<{ ffi::DTYPE_BF16 }>::contiguous(
                layer.rms_weight,
                QWEN36_GDN_VALUE_DIM,
            )?),
            out: self.gdn_v_heads(self.scratch.gdn_norm_out.as_device_ptr(), rows)?,
            eps: self.config.rms_norm_eps,
            gate_activation: Activation::Silu,
        })?;
        {
            let mut ops = self.engine.kernel_ops();
            unsafe { ops.qwen36_gdn_rmsnorm_gated_bf16(&gated)? };
        }

        self.gemm_bf16(
            self.scratch.gdn_norm_out.as_device_ptr(),
            rows,
            QWEN36_GDN_OUTPUT_DIM,
            layer.out_proj,
            self.scratch.attn_proj.as_device_ptr(),
            hidden,
            GemmOut::Bf16,
        )?;
        Ok(())
    }

    fn execute_dense_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        gate_proj: ffi::DevicePtr,
        up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    ) -> Result<(), Status> {
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            gate_proj,
            self.scratch.gate.as_device_ptr(),
            intermediate,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            up_proj,
            self.scratch.up.as_device_ptr(),
            intermediate,
            GemmOut::Bf16,
        )?;
        self.silu_and_mul(rows)?;
        self.gemm_bf16(
            self.scratch.mlp.as_device_ptr(),
            rows,
            intermediate,
            down_proj,
            self.scratch.mlp_out.as_device_ptr(),
            hidden,
            GemmOut::Bf16,
        )
    }

    fn execute_moe_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        router_proj: ffi::DevicePtr,
        gate_up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    ) -> Result<(), Status> {
        let moe = self.config.moe_config().ok_or(Status::InternalError)?;
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            router_proj,
            self.scratch.router_logits.as_device_ptr(),
            moe.num_experts,
            GemmOut::F32,
        )?;

        let router = RouterTopK::new(
            Bf16OrF32Mat::F32(DMat::<{ ffi::DTYPE_F32 }>::contiguous(
                self.scratch.router_logits.as_device_ptr(),
                rows,
                moe.num_experts,
            )?),
            DMat::<{ ffi::DTYPE_I32 }>::contiguous(
                self.scratch.topk_ids.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            DMat::<{ ffi::DTYPE_F32 }>::contiguous(
                self.scratch.topk_weights.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            QWEN36_MOE_ROUTER_SCORE,
            QWEN36_MOE_ROUTER_RENORMALIZE,
            QWEN36_MOE_ROUTER_SCALING_FACTOR,
        )?;
        {
            let mut ops = self.engine.kernel_ops();
            unsafe { ops.router_topk(&router)? };
        }

        let plan = self.moe_plan.as_ref().ok_or(Status::InternalError)?;
        let execute = MoeBf16Execute::new(MoeBf16ExecuteArgs {
            hidden: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.attn_proj.as_device_ptr(),
                rows,
                hidden,
            )?,
            topk_ids: DMat::<{ ffi::DTYPE_I32 }>::contiguous(
                self.scratch.topk_ids.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            topk_weights: DMat::<{ ffi::DTYPE_F32 }>::contiguous(
                self.scratch.topk_weights.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            gate_up_weight: DTensor3::<{ ffi::DTYPE_BF16 }>::contiguous(
                gate_up_proj,
                moe.num_experts,
                moe.moe_intermediate_size
                    .checked_mul(2)
                    .ok_or(Status::InvalidArgument)?,
                hidden,
            )?,
            down_weight: DTensor3::<{ ffi::DTYPE_BF16 }>::contiguous(
                down_proj,
                moe.num_experts,
                hidden,
                moe.moe_intermediate_size,
            )?,
            out: DMat::<{ ffi::DTYPE_BF16 }>::contiguous(
                self.scratch.mlp_out.as_device_ptr(),
                rows,
                hidden,
            )?,
            workspace: Workspace::new(
                self.scratch.moe_workspace.as_device_ptr(),
                self.scratch.moe_workspace.cap,
            )?,
        })?;
        let mut ops = self.engine.kernel_ops();
        unsafe { ops.moe_execute_bf16(plan, &execute) }
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

    fn gdn_q_heads(&self, data: ffi::DevicePtr, rows: u32) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_KEY_DIM)
    }

    fn gdn_k_heads(&self, data: ffi::DevicePtr, rows: u32) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_KEY_DIM)
    }

    fn gdn_v_heads(&self, data: ffi::DevicePtr, rows: u32) -> Result<Bf16Heads, Status> {
        Bf16Heads::contiguous(data, rows, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)
    }

    fn gdn_state_views(
        &self,
        layer_idx: u32,
    ) -> Result<(u32, u32, u32, GdnConvState, GdnRecurrentState), Status> {
        let state = self.gdn_state.as_ref().ok_or(Status::InternalError)?;
        let slots = state.layer_slots(layer_idx)?;
        let conv_state = GdnConvState::contiguous(
            state.conv.as_device_ptr(),
            FloatStorage::Bf16,
            state.state_pool,
        )?;
        let recurrent_state = GdnRecurrentState::contiguous(
            state.recurrent.as_device_ptr(),
            FloatStorage::Bf16,
            state.state_pool,
        )?;
        Ok((
            state.state_pool,
            slots.live_slot,
            slots.staged_slot,
            conv_state,
            recurrent_state,
        ))
    }

    fn commit_gdn_state(&mut self) {
        if let Some(state) = self.gdn_state.as_mut() {
            state.commit();
        }
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

        let row_count = rows as usize;
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

struct GdnLayerSlots {
    live_slot: u32,
    staged_slot: u32,
}

struct GdnState {
    conv: DeviceBuffer<u16>,
    recurrent: DeviceBuffer<u16>,
    live_slots: Vec<u32>,
    staged_slots: Vec<u32>,
    state_pool: u32,
}

impl GdnState {
    fn new(config: &QwenConfig) -> Result<Self, Status> {
        let state_pool = config
            .num_layers
            .checked_mul(QWEN36_GDN_STATE_SLOTS_PER_LAYER)
            .ok_or(Status::InvalidArgument)?;
        let conv_len =
            checked_usize_product(&[state_pool, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_STATE])?;
        let recurrent_len = checked_usize_product(&[
            state_pool,
            QWEN36_GDN_NUM_V_HEADS,
            QWEN36_GDN_VALUE_DIM,
            QWEN36_GDN_KEY_DIM,
        ])?;
        let mut live_slots = try_vec_with_capacity(config.num_layers as usize)?;
        let mut staged_slots = try_vec_with_capacity(config.num_layers as usize)?;
        for layer_idx in 0..config.num_layers {
            let base = layer_idx * QWEN36_GDN_STATE_SLOTS_PER_LAYER;
            live_slots.push(base);
            staged_slots.push(base.checked_add(1).ok_or(Status::InvalidArgument)?);
        }

        let mut state = Self {
            conv: DeviceBuffer::empty(config.device_ordinal),
            recurrent: DeviceBuffer::empty(config.device_ordinal),
            live_slots,
            staged_slots,
            state_pool,
        };
        state.conv.ensure(conv_len)?;
        state.recurrent.ensure(recurrent_len)?;
        state.zero(config)?;
        Ok(state)
    }

    fn reset(&mut self, config: &QwenConfig) -> Result<(), Status> {
        for layer_idx in 0..config.num_layers {
            let idx = layer_idx as usize;
            let base = layer_idx * QWEN36_GDN_STATE_SLOTS_PER_LAYER;
            self.live_slots[idx] = base;
            self.staged_slots[idx] = base.checked_add(1).ok_or(Status::InvalidArgument)?;
        }
        self.zero(config)
    }

    fn zero(&mut self, config: &QwenConfig) -> Result<(), Status> {
        self.conv.zero(self.conv.cap, config.stream)?;
        self.recurrent.zero(self.recurrent.cap, config.stream)
    }

    fn layer_slots(&self, layer_idx: u32) -> Result<GdnLayerSlots, Status> {
        let idx = layer_idx as usize;
        let live_slot = *self.live_slots.get(idx).ok_or(Status::InvalidArgument)?;
        let staged_slot = *self.staged_slots.get(idx).ok_or(Status::InvalidArgument)?;
        if live_slot >= self.state_pool || staged_slot >= self.state_pool {
            return Err(Status::InternalError);
        }
        Ok(GdnLayerSlots {
            live_slot,
            staged_slot,
        })
    }

    fn commit(&mut self) {
        for idx in 0..self.live_slots.len() {
            mem::swap(&mut self.live_slots[idx], &mut self.staged_slots[idx]);
        }
    }
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
    router_logits: DeviceBuffer<f32>,
    topk_ids: DeviceBuffer<i32>,
    topk_weights: DeviceBuffer<f32>,
    moe_workspace: DeviceBuffer<u8>,
    gdn_packed: DeviceBuffer<u16>,
    gdn_conv_out: DeviceBuffer<u16>,
    gdn_a: DeviceBuffer<u16>,
    gdn_b: DeviceBuffer<u16>,
    gdn_q: DeviceBuffer<u16>,
    gdn_k: DeviceBuffer<u16>,
    gdn_v: DeviceBuffer<u16>,
    gdn_recurrent_out: DeviceBuffer<u16>,
    gdn_gate: DeviceBuffer<u16>,
    gdn_norm_out: DeviceBuffer<u16>,
    gdn_seq_indptr: DeviceBuffer<i32>,
    gdn_state_indices: DeviceBuffer<i32>,
    gdn_state_out_indices: DeviceBuffer<i32>,
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
            router_logits: DeviceBuffer::empty(device_ordinal),
            topk_ids: DeviceBuffer::empty(device_ordinal),
            topk_weights: DeviceBuffer::empty(device_ordinal),
            moe_workspace: DeviceBuffer::empty(device_ordinal),
            gdn_packed: DeviceBuffer::empty(device_ordinal),
            gdn_conv_out: DeviceBuffer::empty(device_ordinal),
            gdn_a: DeviceBuffer::empty(device_ordinal),
            gdn_b: DeviceBuffer::empty(device_ordinal),
            gdn_q: DeviceBuffer::empty(device_ordinal),
            gdn_k: DeviceBuffer::empty(device_ordinal),
            gdn_v: DeviceBuffer::empty(device_ordinal),
            gdn_recurrent_out: DeviceBuffer::empty(device_ordinal),
            gdn_gate: DeviceBuffer::empty(device_ordinal),
            gdn_norm_out: DeviceBuffer::empty(device_ordinal),
            gdn_seq_indptr: DeviceBuffer::empty(device_ordinal),
            gdn_state_indices: DeviceBuffer::empty(device_ordinal),
            gdn_state_out_indices: DeviceBuffer::empty(device_ordinal),
            logits: DeviceBuffer::empty(device_ordinal),
            next_token_ids: DeviceBuffer::empty(device_ordinal),
        }
    }

    fn ensure(&mut self, config: &QwenConfig, rows: u32) -> Result<(), Status> {
        let hidden = checked_usize_product(&[rows, config.hidden_size])?;
        let logits = checked_usize_product(&[rows, config.vocab_size])?;
        let row_count = rows as usize;

        self.token_ids.ensure(row_count)?;
        self.positions.ensure(row_count)?;
        self.residual.ensure(hidden)?;
        self.norm.ensure(hidden)?;
        if config.has_gdn_layers() {
            self.gdn_packed
                .ensure(checked_usize_product(&[rows, QWEN36_GDN_PACKED_DIM])?)?;
            self.gdn_conv_out
                .ensure(checked_usize_product(&[rows, QWEN36_GDN_PACKED_DIM])?)?;
            self.gdn_a
                .ensure(checked_usize_product(&[rows, QWEN36_GDN_NUM_V_HEADS])?)?;
            self.gdn_b
                .ensure(checked_usize_product(&[rows, QWEN36_GDN_NUM_V_HEADS])?)?;
            self.gdn_q.ensure(checked_usize_product(&[
                rows,
                QWEN36_GDN_NUM_Q_HEADS,
                QWEN36_GDN_KEY_DIM,
            ])?)?;
            self.gdn_k.ensure(checked_usize_product(&[
                rows,
                QWEN36_GDN_NUM_K_HEADS,
                QWEN36_GDN_KEY_DIM,
            ])?)?;
            let gdn_out = checked_usize_product(&[rows, QWEN36_GDN_OUTPUT_DIM])?;
            self.gdn_v.ensure(gdn_out)?;
            self.gdn_recurrent_out.ensure(gdn_out)?;
            self.gdn_gate.ensure(gdn_out)?;
            self.gdn_norm_out.ensure(gdn_out)?;
            self.gdn_seq_indptr.ensure(2)?;
            self.gdn_state_indices.ensure(row_count.max(1))?;
            self.gdn_state_out_indices.ensure(row_count.max(1))?;
            self.attn_proj.ensure(hidden)?;
        } else {
            let kv_hidden = checked_usize_product(&[rows, config.kv_hidden_size()?])?;
            self.q.ensure(hidden)?;
            self.k.ensure(kv_hidden)?;
            self.v.ensure(kv_hidden)?;
            self.attn_out.ensure(hidden)?;
            self.attn_proj.ensure(hidden)?;
            if let Some(moe) = config.moe_config() {
                self.router_logits
                    .ensure(checked_usize_product(&[rows, moe.num_experts])?)?;
                let topk = checked_usize_product(&[rows, moe.num_experts_per_tok])?;
                self.topk_ids.ensure(topk)?;
                self.topk_weights.ensure(topk)?;
            } else {
                let intermediate = checked_usize_product(&[rows, config.intermediate_size])?;
                self.gate.ensure(intermediate)?;
                self.up.ensure(intermediate)?;
                self.mlp.ensure(intermediate)?;
            }
            self.mlp_out.ensure(hidden)?;
        }
        self.logits.ensure(logits)?;
        self.next_token_ids.ensure(row_count)?;
        Ok(())
    }

    fn ensure_moe_workspace(&mut self, bytes: usize) -> Result<(), Status> {
        self.moe_workspace.ensure(bytes)
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

    fn zero(&mut self, len: usize, stream: *mut c_void) -> Result<(), Status> {
        if len == 0 {
            return Ok(());
        }
        self.ensure(len)?;
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or(Status::InvalidArgument)?;
        result_from_cuda(unsafe { cuda::cudaMemsetAsync(self.ptr.cast(), 0, bytes, stream) })
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

fn constant_f32_values(count: usize, value: f32) -> Result<Vec<f32>, Status> {
    let mut out = try_vec_with_capacity(count)?;
    out.resize(count, value);
    Ok(out)
}

fn qwen36_gdn_scale() -> f32 {
    1.0 / (QWEN36_GDN_KEY_DIM as f32).sqrt()
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

fn checked_usize_product(values: &[u32]) -> Result<usize, Status> {
    let mut product = 1usize;
    for value in values {
        product = product
            .checked_mul(*value as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_moe_config_validation_rejects_invalid_config_json_shapes() {
        assert_eq!(
            QwenMoeConfig::qwen36_35b_a3b(),
            QwenMoeConfig {
                num_experts: 256,
                num_experts_per_tok: 8,
                moe_intermediate_size: 512,
                shared_expert_intermediate_size: 512,
            }
        );
        assert_eq!(
            QwenMoeConfig {
                num_experts: 4,
                num_experts_per_tok: 0,
                moe_intermediate_size: 64,
                shared_expert_intermediate_size: 0,
            }
            .validate(128),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            QwenMoeConfig {
                num_experts: 4,
                num_experts_per_tok: 5,
                moe_intermediate_size: 64,
                shared_expert_intermediate_size: 0,
            }
            .validate(128),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            QwenMoeConfig {
                num_experts: 32,
                num_experts_per_tok: 17,
                moe_intermediate_size: 64,
                shared_expert_intermediate_size: 0,
            }
            .validate(128),
            Err(Status::Unsupported)
        );
        assert_eq!(
            QwenMoeConfig {
                num_experts: 4097,
                num_experts_per_tok: 2,
                moe_intermediate_size: 64,
                shared_expert_intermediate_size: 0,
            }
            .validate(128),
            Err(Status::Unsupported)
        );
        assert_eq!(
            QwenMoeConfig {
                num_experts: 4,
                num_experts_per_tok: 2,
                moe_intermediate_size: 66,
                shared_expert_intermediate_size: 0,
            }
            .validate(128),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            QwenMoeConfig {
                num_experts: 4,
                num_experts_per_tok: 2,
                moe_intermediate_size: 64,
                shared_expert_intermediate_size: 66,
            }
            .validate(128),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn private_qwen36_gdn_validation_is_fixed_to_supported_moe_shape() {
        let mut dense_gdn = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
        dense_gdn.moe = None;
        dense_gdn.hidden_size = 10240;
        dense_gdn.intermediate_size = 17408;
        dense_gdn.model_shape = QwenModelShape {
            layer_pattern: QwenLayerPattern::Qwen36HybridGdn {
                gdn: QwenGdnShape::qwen36_dense_27b(),
            },
        };
        assert_eq!(dense_gdn.validate(), Err(Status::Unsupported));

        let mut wrong_hidden = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
        wrong_hidden.hidden_size = 4096;
        assert_eq!(wrong_hidden.validate(), Err(Status::InvalidArgument));

        let mut introduces_full_attention =
            QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
        introduces_full_attention.num_layers = 4;
        assert_eq!(
            introduces_full_attention.validate(),
            Err(Status::Unsupported)
        );
    }
}
