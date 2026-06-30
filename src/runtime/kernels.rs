#![allow(dead_code)]

use crate::{
    Status,
    ffi::{self, qscb, qscu, qsfi},
};

use super::device_tensor::{DMat, DTensor3, DVec};

use std::ptr;

const QWEN36_GDN_NUM_Q_HEADS: u32 = 16;
const QWEN36_GDN_NUM_K_HEADS: u32 = 16;
const QWEN36_GDN_NUM_V_HEADS: u32 = 32;
const QWEN36_GDN_KEY_DIM: u32 = 128;
const QWEN36_GDN_VALUE_DIM: u32 = 128;
const QWEN36_GDN_CONV_WIDTH: u32 = 4;
const QWEN36_GDN_CONV_STATE: u32 = QWEN36_GDN_CONV_WIDTH - 1;
const QWEN36_GDN_PACKED_DIM: u32 =
    2 * QWEN36_GDN_NUM_K_HEADS * QWEN36_GDN_KEY_DIM + QWEN36_GDN_NUM_V_HEADS * QWEN36_GDN_VALUE_DIM;
const ROUTER_MAX_TOP_K: u32 = 16;
const ROUTER_MAX_EXPERTS: u32 = 4096;

pub(crate) type MoePlan = qsfi::MoePlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Workspace {
    data: ffi::DevicePtr,
    bytes: usize,
}

impl Workspace {
    pub(crate) fn none() -> Self {
        Self {
            data: ptr::null_mut(),
            bytes: 0,
        }
    }

    pub(crate) fn new(data: ffi::DevicePtr, bytes: usize) -> Result<Self, Status> {
        let workspace = Self { data, bytes };
        workspace.validate()?;
        Ok(workspace)
    }

    fn validate(self) -> Result<(), Status> {
        if self.data.is_null() && self.bytes != 0 {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }

    fn tensor(self, dtype: ffi::DTypeRaw) -> Result<ffi::Tensor1, Status> {
        self.validate()?;
        let len = i64::try_from(self.bytes).map_err(|_| Status::InvalidArgument)?;
        Ok(ffi::Tensor1 {
            data: self.data,
            dtype,
            shape: [len],
            stride: [1],
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Bf16Heads {
    data: ffi::DevicePtr,
    tokens: u32,
    heads: u32,
    head_dim: u32,
    token_stride: u32,
    head_stride: u32,
}

impl Bf16Heads {
    pub(crate) fn contiguous(
        data: ffi::DevicePtr,
        tokens: u32,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Status> {
        Self::new(
            data,
            tokens,
            heads,
            head_dim,
            heads_mul(heads, head_dim)?,
            head_dim,
        )
    }

    pub(crate) fn new(
        data: ffi::DevicePtr,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        token_stride: u32,
        head_stride: u32,
    ) -> Result<Self, Status> {
        validate_ptr(data)?;
        validate_nonzero(&[tokens, heads, head_dim, token_stride, head_stride])?;
        if head_stride < head_dim || token_stride < heads_mul(heads, head_stride)? {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            data,
            tokens,
            heads,
            head_dim,
            token_stride,
            head_stride,
        })
    }

    pub(crate) fn tensor(self) -> ffi::Tensor3 {
        ffi::Tensor3 {
            data: self.data,
            dtype: ffi::DTYPE_BF16,
            shape: [self.tokens.into(), self.heads.into(), self.head_dim.into()],
            stride: [self.token_stride.into(), self.head_stride.into(), 1],
        }
    }

    fn same_shape(self, other: Self) -> bool {
        self.tokens == other.tokens && self.heads == other.heads && self.head_dim == other.head_dim
    }

    fn same_strides(self, other: Self) -> bool {
        self.token_stride == other.token_stride && self.head_stride == other.head_stride
    }

    fn is_contiguous(self) -> bool {
        self.head_stride == self.head_dim
            && self.heads.checked_mul(self.head_dim) == Some(self.token_stride)
    }

    fn require_contiguous(&self) -> Result<(), Status> {
        if !self.is_contiguous() {
            Err(Status::InvalidArgument)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Bf16OrF32Mat {
    Bf16(DMat<{ ffi::DTYPE_BF16 }>),
    F32(DMat<{ ffi::DTYPE_F32 }>),
}

impl Bf16OrF32Mat {
    fn rows(self) -> u32 {
        match self {
            Self::Bf16(mat) => mat.rows,
            Self::F32(mat) => mat.rows,
        }
    }

    fn cols(self) -> u32 {
        match self {
            Self::Bf16(mat) => mat.cols,
            Self::F32(mat) => mat.cols,
        }
    }

    fn tensor(self) -> ffi::Tensor2 {
        match self {
            Self::Bf16(mat) => mat.tensor(),
            Self::F32(mat) => mat.tensor(),
        }
    }

    fn is_contiguous(self) -> bool {
        match self {
            Self::Bf16(mat) => mat.is_contiguous(),
            Self::F32(mat) => mat.is_contiguous(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Bf16OrF32Vec {
    Bf16(DVec<{ ffi::DTYPE_BF16 }>),
    F32(DVec<{ ffi::DTYPE_F32 }>),
}

impl Bf16OrF32Vec {
    fn len(self) -> u32 {
        match self {
            Self::Bf16(vec) => vec.len,
            Self::F32(vec) => vec.len,
        }
    }

    fn tensor(self) -> ffi::Tensor1 {
        match self {
            Self::Bf16(vec) => vec.tensor(),
            Self::F32(vec) => vec.tensor(),
        }
    }

    fn is_contiguous(self) -> bool {
        match self {
            Self::Bf16(vec) => vec.is_contiguous(),
            Self::F32(vec) => vec.is_contiguous(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FloatStorage {
    Bf16,
    F32,
}

impl FloatStorage {
    fn dtype(self) -> ffi::DTypeRaw {
        match self {
            Self::Bf16 => ffi::DTYPE_BF16,
            Self::F32 => ffi::DTYPE_F32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Activation {
    None,
    Silu,
    Sigmoid,
}

impl Activation {
    fn raw(self) -> qscu::ActivationRaw {
        match self {
            Self::None => qscu::ACTIVATION_NONE,
            Self::Silu => qscu::ACTIVATION_SILU,
            Self::Sigmoid => qscu::ACTIVATION_SIGMOID,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GdnForgetGateOutput {
    LogDecay,
    LinearAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouterScore {
    Softmax,
    Sigmoid,
}

impl RouterScore {
    fn raw(self) -> qscu::RouterScoreRaw {
        match self {
            Self::Softmax => qscu::ROUTER_SCORE_SOFTMAX,
            Self::Sigmoid => qscu::ROUTER_SCORE_SIGMOID,
        }
    }
}

impl GdnForgetGateOutput {
    fn raw(self) -> qscu::GdnForgetGateOutputRaw {
        match self {
            Self::LogDecay => qscu::GDN_FORGET_LOG_DECAY,
            Self::LinearAlpha => qscu::GDN_FORGET_LINEAR_ALPHA,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GdnConvState {
    data: ffi::DevicePtr,
    dtype: FloatStorage,
    state_pool: u32,
}

impl GdnConvState {
    pub(crate) fn contiguous(
        data: ffi::DevicePtr,
        dtype: FloatStorage,
        state_pool: u32,
    ) -> Result<Self, Status> {
        validate_ptr(data)?;
        validate_nonzero(&[state_pool])?;
        Ok(Self {
            data,
            dtype,
            state_pool,
        })
    }

    fn tensor(self) -> ffi::Tensor3 {
        ffi::Tensor3 {
            data: self.data,
            dtype: self.dtype.dtype(),
            shape: [
                self.state_pool.into(),
                QWEN36_GDN_PACKED_DIM.into(),
                QWEN36_GDN_CONV_STATE.into(),
            ],
            stride: [
                (QWEN36_GDN_PACKED_DIM * QWEN36_GDN_CONV_STATE).into(),
                QWEN36_GDN_CONV_STATE.into(),
                1,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GdnRecurrentState {
    data: ffi::DevicePtr,
    dtype: FloatStorage,
    state_pool: u32,
}

impl GdnRecurrentState {
    pub(crate) fn contiguous(
        data: ffi::DevicePtr,
        dtype: FloatStorage,
        state_pool: u32,
    ) -> Result<Self, Status> {
        validate_ptr(data)?;
        validate_nonzero(&[state_pool])?;
        Ok(Self {
            data,
            dtype,
            state_pool,
        })
    }

    fn tensor(self) -> ffi::Tensor4 {
        let value_key = QWEN36_GDN_VALUE_DIM * QWEN36_GDN_KEY_DIM;
        ffi::Tensor4 {
            data: self.data,
            dtype: self.dtype.dtype(),
            shape: [
                self.state_pool.into(),
                QWEN36_GDN_NUM_V_HEADS.into(),
                QWEN36_GDN_VALUE_DIM.into(),
                QWEN36_GDN_KEY_DIM.into(),
            ],
            stride: [
                (QWEN36_GDN_NUM_V_HEADS * value_key).into(),
                value_key.into(),
                QWEN36_GDN_KEY_DIM.into(),
                1.into(),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GdnStateIndexPolicy {
    NegativeSkips,
    NonNegative,
}

impl GdnStateIndexPolicy {
    fn validate_host_indices(self, indices: &[i32], state_pool: u32) -> Result<(), Status> {
        validate_nonzero(&[state_pool])?;
        for &index in indices {
            if index < 0 {
                if self == Self::NegativeSkips {
                    continue;
                }
                return Err(Status::InvalidArgument);
            }
            if u32::try_from(index).map_err(|_| Status::InvalidArgument)? >= state_pool {
                return Err(Status::InvalidArgument);
            }
        }
        Ok(())
    }
}

pub(crate) struct KernelOps<'a> {
    stream: ffi::CudaStream,
    qsfi: &'a mut qsfi::Context,
    qscb: &'a mut qscb::Context,
}

impl<'a> KernelOps<'a> {
    pub(crate) fn new(
        stream: ffi::CudaStream,
        qsfi: &'a mut qsfi::Context,
        qscb: &'a mut qscb::Context,
    ) -> Self {
        Self { stream, qsfi, qscb }
    }

    pub(crate) unsafe fn embedding_gather_bf16(
        &mut self,
        desc: &EmbeddingGatherBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::embedding_gather_bf16(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn silu_and_mul_bf16(&mut self, desc: &SiluAndMulBf16) -> Result<(), Status> {
        unsafe { qscu::silu_and_mul_bf16(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn qwen36_shared_expert_gate_add_bf16(
        &mut self,
        desc: &Qwen36SharedExpertGateAddBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_shared_expert_gate_add_bf16(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn logits_soft_cap_f32(
        &mut self,
        desc: &LogitsSoftCapF32,
    ) -> Result<(), Status> {
        unsafe {
            qscu::logits_soft_cap_f32(
                &desc.logits,
                desc.rows,
                desc.vocab_size,
                desc.soft_cap,
                self.stream,
            )
        }
    }

    pub(crate) unsafe fn greedy_argmax_f32(
        &mut self,
        desc: &GreedyArgmaxF32,
    ) -> Result<(), Status> {
        unsafe { qscu::greedy_argmax_f32(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn router_topk(&mut self, desc: &RouterTopK) -> Result<(), Status> {
        unsafe { qscu::router_topk(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn gemm_bf16(&mut self, desc: &Bf16Gemm) -> Result<(), Status> {
        unsafe { self.qscb.gemm_bf16(&desc.raw) }
    }

    pub(crate) unsafe fn rmsnorm_bf16(&mut self, desc: &RmsNormBf16) -> Result<(), Status> {
        unsafe { self.qsfi.rmsnorm(&desc.raw) }
    }

    pub(crate) unsafe fn fused_add_rmsnorm_bf16(
        &mut self,
        desc: &FusedAddRmsNormBf16,
    ) -> Result<(), Status> {
        unsafe { self.qsfi.fused_add_rmsnorm(&desc.raw) }
    }

    pub(crate) unsafe fn rope_apply_bf16(&mut self, desc: &RopeApplyBf16) -> Result<(), Status> {
        unsafe { self.qsfi.rope_apply(&desc.raw) }
    }

    pub(crate) unsafe fn qwen36_gdn_causal_conv1d_bf16(
        &mut self,
        desc: &GdnCausalConv1dBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_gdn_causal_conv1d_bf16(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn qwen36_gdn_post_conv_prepare_bf16(
        &mut self,
        desc: &GdnPostConvPrepareBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_gdn_post_conv_prepare_bf16(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn qwen36_gdn_rmsnorm_gated_bf16(
        &mut self,
        desc: &GdnRmsNormGatedBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_gdn_rmsnorm_gated_bf16(&desc.raw, self.stream) }
    }

    pub(crate) unsafe fn gdn_decode_bf16(&mut self, desc: &GdnDecodeBf16) -> Result<(), Status> {
        unsafe { qscu::gdn_decode(self.qsfi, &desc.raw) }
    }

    pub(crate) unsafe fn gdn_prefill_bf16(&mut self, desc: &GdnPrefillBf16) -> Result<(), Status> {
        unsafe { qscu::gdn_prefill(self.qsfi, &desc.raw) }
    }

    pub(crate) unsafe fn create_moe_bf16_plan(
        &mut self,
        config: MoeBf16PlanConfig,
    ) -> Result<MoePlan, Status> {
        let desc = config.desc()?;
        unsafe { self.qsfi.create_moe_plan(&desc) }
    }

    pub(crate) unsafe fn moe_workspace_size(
        &mut self,
        plan: &MoePlan,
        num_tokens: u32,
    ) -> Result<usize, Status> {
        unsafe { self.qsfi.moe_workspace_size(plan, num_tokens) }
    }

    pub(crate) unsafe fn moe_execute_bf16(
        &mut self,
        plan: &MoePlan,
        desc: &MoeBf16Execute,
    ) -> Result<(), Status> {
        unsafe { self.qsfi.moe_execute_bf16(plan, &desc.raw) }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddingGatherBf16 {
    raw: qscu::EmbeddingGatherDesc,
}

impl EmbeddingGatherBf16 {
    pub(crate) fn new(
        token_ids: DVec<{ ffi::DTYPE_I32 }>,
        embedding: DMat<{ ffi::DTYPE_BF16 }>,
        out: DMat<{ ffi::DTYPE_BF16 }>,
    ) -> Result<Self, Status> {
        Self::with_options(token_ids, embedding, out, None, false)
    }

    pub(crate) fn with_options(
        token_ids: DVec<{ ffi::DTYPE_I32 }>,
        embedding: DMat<{ ffi::DTYPE_BF16 }>,
        out: DMat<{ ffi::DTYPE_BF16 }>,
        padding_token_id: Option<i32>,
        validate_token_ids: bool,
    ) -> Result<Self, Status> {
        token_ids.require_contiguous()?;
        embedding.require_contiguous()?;
        out.require_contiguous()?;
        if out.rows != token_ids.len || out.cols != embedding.cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::EmbeddingGatherDesc {
                token_ids: token_ids.tensor(),
                embedding: embedding.tensor(),
                out: out.tensor(),
                padding_token_id: padding_token_id.unwrap_or(-1),
                validate_token_ids: u32::from(validate_token_ids),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SiluAndMulBf16 {
    raw: qscu::SiluAndMulDesc,
}

impl SiluAndMulBf16 {
    pub(crate) fn new(
        gate: DMat<{ ffi::DTYPE_BF16 }>,
        up: DMat<{ ffi::DTYPE_BF16 }>,
        out: DMat<{ ffi::DTYPE_BF16 }>,
    ) -> Result<Self, Status> {
        gate.require_contiguous()?;
        up.require_contiguous()?;
        out.require_contiguous()?;
        if !gate.same_shape(up) || !gate.same_shape(out) {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::SiluAndMulDesc {
                gate: gate.tensor(),
                up: up.tensor(),
                out: out.tensor(),
                num_tokens: gate.rows,
                intermediate_size: gate.cols,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36SharedExpertGateAddBf16 {
    raw: qscu::Qwen36SharedExpertGateAddDesc,
}

impl Qwen36SharedExpertGateAddBf16 {
    pub(crate) fn new(
        gate_logits: Bf16OrF32Mat,
        shared: DMat<{ ffi::DTYPE_BF16 }>,
        out: DMat<{ ffi::DTYPE_BF16 }>,
    ) -> Result<Self, Status> {
        if !gate_logits.is_contiguous() || !shared.is_contiguous() || !out.is_contiguous() {
            return Err(Status::InvalidArgument);
        }
        if gate_logits.rows() != shared.rows || gate_logits.cols() != 1 || !shared.same_shape(out) {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::Qwen36SharedExpertGateAddDesc {
                gate_logits: gate_logits.tensor(),
                shared: shared.tensor(),
                out: out.tensor(),
                num_tokens: shared.rows,
                hidden_size: shared.cols,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogitsSoftCapF32 {
    logits: ffi::Tensor2,
    rows: u32,
    vocab_size: u32,
    soft_cap: f32,
}

impl LogitsSoftCapF32 {
    pub(crate) fn new(logits: DMat<{ ffi::DTYPE_F32 }>, soft_cap: f32) -> Result<Self, Status> {
        logits.require_contiguous()?;
        validate_soft_cap(soft_cap)?;
        Ok(Self {
            logits: logits.tensor(),
            rows: logits.rows,
            vocab_size: logits.cols,
            soft_cap,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GreedyArgmaxF32 {
    raw: qscu::SamplingDesc,
}

impl GreedyArgmaxF32 {
    pub(crate) fn new(
        logits: DMat<{ ffi::DTYPE_F32 }>,
        next_token_ids: DVec<{ ffi::DTYPE_I32 }>,
    ) -> Result<Self, Status> {
        logits.require_contiguous()?;
        next_token_ids.require_contiguous()?;
        if next_token_ids.len != logits.rows {
            return Err(Status::InvalidArgument);
        }
        if logits.cols > i32::MAX as u32 {
            return Err(Status::Unsupported);
        }
        Ok(Self {
            raw: qscu::SamplingDesc {
                logits: logits.tensor(),
                uniform_samples: zero_tensor1(ffi::DTYPE_F32),
                next_token_ids: next_token_ids.tensor(),
                selected_logprobs: zero_tensor1(ffi::DTYPE_F32),
                selected_probs: zero_tensor1(ffi::DTYPE_F32),
                batch_size: logits.rows,
                vocab_size: logits.cols,
                top_k: 0,
                top_p: 0.0,
                min_p: 0.0,
                temperature: 0.0,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RouterTopK {
    raw: qscu::RouterTopkDesc,
}

impl RouterTopK {
    pub(crate) fn new(
        logits: Bf16OrF32Mat,
        topk_ids: DMat<{ ffi::DTYPE_I32 }>,
        topk_weights: DMat<{ ffi::DTYPE_F32 }>,
        score: RouterScore,
        renormalize: bool,
        routed_scaling_factor: f32,
    ) -> Result<Self, Status> {
        if !logits.is_contiguous() || !topk_ids.is_contiguous() || !topk_weights.is_contiguous() {
            return Err(Status::InvalidArgument);
        }
        if logits.rows() == 0
            || logits.cols() == 0
            || topk_ids.rows != logits.rows()
            || topk_weights.rows != logits.rows()
            || topk_ids.cols != topk_weights.cols
            || topk_ids.cols == 0
        {
            return Err(Status::InvalidArgument);
        }
        if topk_ids.cols > ROUTER_MAX_TOP_K
            || logits.cols() > ROUTER_MAX_EXPERTS
            || topk_ids.cols > logits.cols()
        {
            return Err(Status::Unsupported);
        }
        if !routed_scaling_factor.is_finite() || routed_scaling_factor <= 0.0 {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::RouterTopkDesc {
                logits: logits.tensor(),
                topk_ids: topk_ids.tensor(),
                topk_weights: topk_weights.tensor(),
                num_tokens: logits.rows(),
                num_experts: logits.cols(),
                top_k: topk_ids.cols,
                score: score.raw(),
                renormalize: u32::from(renormalize),
                routed_scaling_factor,
            },
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MoeBf16PlanConfig {
    pub(crate) max_num_tokens: u32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_experts: u32,
    pub(crate) top_k: u32,
}

impl MoeBf16PlanConfig {
    fn desc(self) -> Result<ffi::MoePlanDesc, Status> {
        validate_nonzero(&[
            self.max_num_tokens,
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k,
        ])?;
        if self.top_k > self.num_experts {
            return Err(Status::InvalidArgument);
        }
        Ok(ffi::MoePlanDesc {
            backend: ffi::MOE_BACKEND_FLASHINFER_STAGED_BF16,
            route_mode: ffi::MOE_ROUTE_PRECOMPUTED_TOPK,
            max_num_tokens: self.max_num_tokens,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_experts: self.num_experts,
            top_k: self.top_k,
            local_expert_offset: 0,
            local_num_experts: self.num_experts,
            activation_dtype: ffi::DTYPE_BF16,
            weight_dtype: ffi::DTYPE_BF16,
            output_dtype: ffi::DTYPE_BF16,
            reserved0: 0,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeBf16ExecuteArgs {
    pub(crate) hidden: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) topk_ids: DMat<{ ffi::DTYPE_I32 }>,
    pub(crate) topk_weights: DMat<{ ffi::DTYPE_F32 }>,
    pub(crate) gate_up_weight: DTensor3<{ ffi::DTYPE_BF16 }>,
    pub(crate) down_weight: DTensor3<{ ffi::DTYPE_BF16 }>,
    pub(crate) out: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) workspace: Workspace,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeBf16Execute {
    raw: ffi::MoeBf16ExecuteDesc,
}

impl MoeBf16Execute {
    pub(crate) fn new(args: MoeBf16ExecuteArgs) -> Result<Self, Status> {
        args.hidden.require_contiguous()?;
        args.topk_ids.require_contiguous()?;
        args.topk_weights.require_contiguous()?;
        args.gate_up_weight.require_contiguous()?;
        args.down_weight.require_contiguous()?;
        args.out.require_contiguous()?;
        if args.hidden.rows == 0
            || args.topk_ids.rows != args.hidden.rows
            || args.topk_weights.rows != args.hidden.rows
            || args.topk_ids.cols != args.topk_weights.cols
            || args.topk_ids.cols == 0
            || !args.hidden.same_shape(args.out)
        {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::MoeBf16ExecuteDesc {
                hidden: args.hidden.tensor(),
                topk_ids: args.topk_ids.tensor(),
                topk_weights: args.topk_weights.tensor(),
                gate_up_weight: args.gate_up_weight.tensor(),
                down_weight: args.down_weight.tensor(),
                out: args.out.tensor(),
                workspace: args.workspace.tensor(ffi::DTYPE_U8)?,
                num_tokens: args.hidden.rows,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Bf16Gemm {
    raw: qscb::Bf16GemmDesc,
}

impl Bf16Gemm {
    pub(crate) fn new(
        x: DMat<{ ffi::DTYPE_BF16 }>,
        weight: DMat<{ ffi::DTYPE_BF16 }>,
        out: Bf16OrF32Mat,
        workspace: Workspace,
    ) -> Result<Self, Status> {
        Self::with_alpha_beta(x, weight, out, workspace, 0.0, 0.0)
    }

    pub(crate) fn with_alpha_beta(
        x: DMat<{ ffi::DTYPE_BF16 }>,
        weight: DMat<{ ffi::DTYPE_BF16 }>,
        out: Bf16OrF32Mat,
        workspace: Workspace,
        alpha: f32,
        beta: f32,
    ) -> Result<Self, Status> {
        workspace.validate()?;
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(Status::InvalidArgument);
        }
        if weight.cols != x.cols || out.rows() != x.rows || out.cols() != weight.rows {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscb::Bf16GemmDesc {
                x: x.tensor(),
                weight: weight.tensor(),
                out: out.tensor(),
                rows: x.rows,
                in_features: x.cols,
                out_features: weight.rows,
                alpha,
                beta,
                workspace: workspace.data,
                workspace_bytes: workspace.bytes,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsNormBf16 {
    raw: ffi::RmsnormDesc,
}

impl RmsNormBf16 {
    pub(crate) fn new(
        x: DMat<{ ffi::DTYPE_BF16 }>,
        weight: DVec<{ ffi::DTYPE_BF16 }>,
        out: DMat<{ ffi::DTYPE_BF16 }>,
        eps: f32,
    ) -> Result<Self, Status> {
        validate_eps(eps)?;
        weight.require_contiguous()?;
        if !x.same_shape(out) || weight.len != x.cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::RmsnormDesc {
                x: x.tensor(),
                weight: weight.tensor(),
                out: out.tensor(),
                hidden_size: x.cols,
                eps,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedAddRmsNormBf16 {
    raw: ffi::FusedAddRmsnormDesc,
}

impl FusedAddRmsNormBf16 {
    pub(crate) fn new(
        x: DMat<{ ffi::DTYPE_BF16 }>,
        residual_inout: DMat<{ ffi::DTYPE_BF16 }>,
        weight: DVec<{ ffi::DTYPE_BF16 }>,
        eps: f32,
    ) -> Result<Self, Status> {
        validate_eps(eps)?;
        weight.require_contiguous()?;
        if !x.same_shape(residual_inout) || weight.len != x.cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::FusedAddRmsnormDesc {
                x: x.tensor(),
                residual_inout: residual_inout.tensor(),
                weight: weight.tensor(),
                out: x.tensor(),
                hidden_size: x.cols,
                eps,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeApplyBf16 {
    raw: ffi::RopeApplyDesc,
}

impl RopeApplyBf16 {
    pub(crate) fn new(
        q: Bf16Heads,
        k: Bf16Heads,
        q_out: Bf16Heads,
        k_out: Bf16Heads,
        positions: DVec<{ ffi::DTYPE_I32 }>,
    ) -> Result<Self, Status> {
        Self::with_params(q, k, q_out, k_out, positions, 0.0, 0.0)
    }

    pub(crate) fn with_params(
        q: Bf16Heads,
        k: Bf16Heads,
        q_out: Bf16Heads,
        k_out: Bf16Heads,
        positions: DVec<{ ffi::DTYPE_I32 }>,
        rope_scale: f32,
        rope_theta: f32,
    ) -> Result<Self, Status> {
        require_supported_rope_head_dim(q.head_dim)?;
        if q.head_dim != k.head_dim
            || !q.same_shape(q_out)
            || !k.same_shape(k_out)
            || k.tokens != q.tokens
            || positions.len != q.tokens
        {
            return Err(Status::InvalidArgument);
        }
        if q_out.data == q.data && !q.same_strides(q_out) {
            return Err(Status::InvalidArgument);
        }
        if k_out.data == k.data && !k.same_strides(k_out) {
            return Err(Status::InvalidArgument);
        }
        positions.require_contiguous()?;
        if !rope_scale.is_finite()
            || rope_scale < 0.0
            || !rope_theta.is_finite()
            || rope_theta < 0.0
        {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::RopeApplyDesc {
                q: q.tensor(),
                k: k.tensor(),
                q_out: q_out.tensor(),
                k_out: k_out.tensor(),
                positions: positions.tensor(),
                num_qo_heads: q.heads,
                num_kv_heads: k.heads,
                head_dim: q.head_dim,
                rope_scale,
                rope_theta,
                interleave: 0,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnCausalConv1dBf16Args {
    pub(crate) x: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) weight: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) bias: Option<Bf16OrF32Vec>,
    pub(crate) state: GdnConvState,
    pub(crate) state_read_indices: Option<DVec<{ ffi::DTYPE_I32 }>>,
    pub(crate) state_write_indices: Option<DVec<{ ffi::DTYPE_I32 }>>,
    pub(crate) seq_indptr: Option<DVec<{ ffi::DTYPE_I32 }>>,
    pub(crate) out: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) batch_size: u32,
    pub(crate) activation: Activation,
    pub(crate) update_state: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnCausalConv1dBf16 {
    raw: qscu::GdnCausalConv1dDesc,
}

impl GdnCausalConv1dBf16 {
    pub(crate) fn new(args: GdnCausalConv1dBf16Args) -> Result<Self, Status> {
        validate_nonzero(&[args.batch_size])?;
        args.x.require_contiguous()?;
        args.weight.require_contiguous()?;
        args.out.require_contiguous()?;
        if args.x.rows == 0
            || args.x.cols != QWEN36_GDN_PACKED_DIM
            || args.weight.rows != QWEN36_GDN_PACKED_DIM
            || args.weight.cols != QWEN36_GDN_CONV_WIDTH
            || args.out.rows != args.x.rows
            || args.out.cols != QWEN36_GDN_PACKED_DIM
        {
            return Err(Status::InvalidArgument);
        }
        if let Some(bias) = args.bias {
            require_float_vec(bias, QWEN36_GDN_PACKED_DIM)?;
        }
        if let Some(indices) = args.state_read_indices {
            require_i32_vec(indices, args.batch_size)?;
        }
        if let Some(indices) = args.state_write_indices {
            require_i32_vec(indices, args.batch_size)?;
        }
        if args.state_read_indices.is_none() && args.state_write_indices.is_none() {
            return Err(Status::InvalidArgument);
        }
        let seq_indptr = if let Some(seq_indptr) = args.seq_indptr {
            let expected = args
                .batch_size
                .checked_add(1)
                .ok_or(Status::InvalidArgument)?;
            require_i32_vec(seq_indptr, expected)?;
            seq_indptr.data
        } else {
            if args.batch_size != args.x.rows {
                return Err(Status::InvalidArgument);
            }
            ptr::null_mut()
        };
        if !matches!(args.activation, Activation::None | Activation::Silu) {
            return Err(Status::Unsupported);
        }

        Ok(Self {
            raw: qscu::GdnCausalConv1dDesc {
                x: args.x.tensor(),
                weight: args.weight.tensor(),
                bias: args
                    .bias
                    .map_or(zero_tensor1(ffi::DTYPE_BF16), |bias| bias.tensor()),
                state: args.state.tensor(),
                state_read_indices: args
                    .state_read_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                state_write_indices: args
                    .state_write_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                seq_indptr,
                out: args.out.tensor(),
                num_tokens: args.x.rows,
                batch_size: args.batch_size,
                activation: args.activation.raw(),
                update_state: u32::from(args.update_state),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPostConvPrepareBf16Args {
    pub(crate) conv_out: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) a: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) b: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) a_log: DVec<{ ffi::DTYPE_F32 }>,
    pub(crate) dt_bias: DVec<{ ffi::DTYPE_F32 }>,
    pub(crate) q: Bf16Heads,
    pub(crate) k: Bf16Heads,
    pub(crate) v: Bf16Heads,
    pub(crate) g_out: Option<DMat<{ ffi::DTYPE_F32 }>>,
    pub(crate) beta_out: Option<DMat<{ ffi::DTYPE_F32 }>>,
    pub(crate) apply_qk_l2norm: bool,
    pub(crate) l2norm_eps: f32,
    pub(crate) forget_gate_output: GdnForgetGateOutput,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPostConvPrepareBf16 {
    raw: qscu::GdnPostConvPrepareDesc,
}

impl GdnPostConvPrepareBf16 {
    pub(crate) fn new(args: GdnPostConvPrepareBf16Args) -> Result<Self, Status> {
        validate_eps(args.l2norm_eps)?;
        args.conv_out.require_contiguous()?;
        args.a.require_contiguous()?;
        args.b.require_contiguous()?;
        args.a_log.require_contiguous()?;
        args.dt_bias.require_contiguous()?;
        args.q.require_contiguous()?;
        args.k.require_contiguous()?;
        args.v.require_contiguous()?;
        let tokens = args.conv_out.rows;
        if args.conv_out.cols != QWEN36_GDN_PACKED_DIM
            || args.a.rows != tokens
            || args.a.cols != QWEN36_GDN_NUM_V_HEADS
            || args.b.rows != tokens
            || args.b.cols != QWEN36_GDN_NUM_V_HEADS
            || args.a_log.len != QWEN36_GDN_NUM_V_HEADS
            || args.dt_bias.len != QWEN36_GDN_NUM_V_HEADS
        {
            return Err(Status::InvalidArgument);
        }
        require_qwen36_gdn_heads(args.q, QWEN36_GDN_NUM_Q_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.k, QWEN36_GDN_NUM_K_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.v, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        if let Some(g_out) = args.g_out {
            g_out.require_contiguous()?;
            if g_out.rows != tokens || g_out.cols != QWEN36_GDN_NUM_V_HEADS {
                return Err(Status::InvalidArgument);
            }
        }
        if let Some(beta_out) = args.beta_out {
            beta_out.require_contiguous()?;
            if beta_out.rows != tokens || beta_out.cols != QWEN36_GDN_NUM_V_HEADS {
                return Err(Status::InvalidArgument);
            }
        }

        Ok(Self {
            raw: qscu::GdnPostConvPrepareDesc {
                conv_out: args.conv_out.tensor(),
                a: args.a.tensor(),
                b: args.b.tensor(),
                a_log: args.a_log.tensor(),
                dt_bias: args.dt_bias.tensor(),
                q: args.q.tensor(),
                k: args.k.tensor(),
                v: args.v.tensor(),
                g_out: args
                    .g_out
                    .map_or(zero_tensor2(ffi::DTYPE_F32), DMat::tensor),
                beta_out: args
                    .beta_out
                    .map_or(zero_tensor2(ffi::DTYPE_F32), DMat::tensor),
                num_tokens: tokens,
                apply_qk_l2norm: u32::from(args.apply_qk_l2norm),
                l2norm_eps: args.l2norm_eps,
                forget_gate_output: args.forget_gate_output.raw(),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnRmsNormGatedBf16Args {
    pub(crate) x: Bf16Heads,
    pub(crate) gate: Bf16Heads,
    pub(crate) weight: Bf16OrF32Vec,
    pub(crate) out: Bf16Heads,
    pub(crate) eps: f32,
    pub(crate) gate_activation: Activation,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnRmsNormGatedBf16 {
    raw: qscu::GdnRmsnormGatedDesc,
}

impl GdnRmsNormGatedBf16 {
    pub(crate) fn new(args: GdnRmsNormGatedBf16Args) -> Result<Self, Status> {
        validate_eps(args.eps)?;
        if !matches!(args.gate_activation, Activation::Silu | Activation::Sigmoid) {
            return Err(Status::Unsupported);
        }
        args.x.require_contiguous()?;
        args.gate.require_contiguous()?;
        args.out.require_contiguous()?;
        require_float_vec(args.weight, QWEN36_GDN_VALUE_DIM)?;
        let tokens = args.x.tokens;
        require_qwen36_gdn_heads(args.x, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.gate, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.out, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        Ok(Self {
            raw: qscu::GdnRmsnormGatedDesc {
                x: args.x.tensor(),
                gate: args.gate.tensor(),
                weight: args.weight.tensor(),
                out: args.out.tensor(),
                num_tokens: tokens,
                eps: args.eps,
                gate_activation: args.gate_activation.raw(),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnDecodeBf16Args {
    pub(crate) q: Bf16Heads,
    pub(crate) k: Bf16Heads,
    pub(crate) v: Bf16Heads,
    pub(crate) a: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) b: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) a_log: DVec<{ ffi::DTYPE_F32 }>,
    pub(crate) dt_bias: DVec<{ ffi::DTYPE_F32 }>,
    pub(crate) state: GdnRecurrentState,
    pub(crate) state_indices: DVec<{ ffi::DTYPE_I32 }>,
    pub(crate) state_out_indices: Option<DVec<{ ffi::DTYPE_I32 }>>,
    pub(crate) out: Bf16Heads,
    pub(crate) scale: f32,
    pub(crate) use_qk_l2norm: bool,
    pub(crate) disable_state_update: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnDecodeBf16 {
    raw: qscu::GdnDecodeDesc,
}

impl GdnDecodeBf16 {
    pub(crate) fn new(args: GdnDecodeBf16Args) -> Result<Self, Status> {
        validate_gdn_scale(args.scale)?;
        let tokens = args.q.tokens;
        validate_gdn_recurrent_tensors(
            args.q,
            args.k,
            args.v,
            args.a,
            args.b,
            args.a_log,
            args.dt_bias,
            args.out,
            tokens,
        )?;
        require_gdn_state_index_vec(
            args.state_indices,
            tokens,
            GdnStateIndexPolicy::NegativeSkips,
        )?;
        if let Some(indices) = args.state_out_indices {
            require_gdn_state_index_vec(indices, tokens, GdnStateIndexPolicy::NegativeSkips)?;
        }
        Ok(Self {
            raw: qscu::GdnDecodeDesc {
                q: args.q.tensor(),
                k: args.k.tensor(),
                v: args.v.tensor(),
                a: args.a.tensor(),
                b: args.b.tensor(),
                a_log: args.a_log.tensor(),
                dt_bias: args.dt_bias.tensor(),
                state: args.state.tensor(),
                state_indices: args.state_indices.tensor(),
                state_out_indices: args
                    .state_out_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                out: args.out.tensor(),
                num_tokens: tokens,
                num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
                num_k_heads: QWEN36_GDN_NUM_K_HEADS,
                num_v_heads: QWEN36_GDN_NUM_V_HEADS,
                key_dim: QWEN36_GDN_KEY_DIM,
                value_dim: QWEN36_GDN_VALUE_DIM,
                state_layout: qscu::GDN_STATE_LAYOUT_VK,
                scale: args.scale,
                use_qk_l2norm: u32::from(args.use_qk_l2norm),
                disable_state_update: u32::from(args.disable_state_update),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPrefillBf16Args {
    pub(crate) q: Bf16Heads,
    pub(crate) k: Bf16Heads,
    pub(crate) v: Bf16Heads,
    pub(crate) a: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) b: DMat<{ ffi::DTYPE_BF16 }>,
    pub(crate) a_log: DVec<{ ffi::DTYPE_F32 }>,
    pub(crate) dt_bias: DVec<{ ffi::DTYPE_F32 }>,
    pub(crate) state: GdnRecurrentState,
    pub(crate) seq_indptr: DVec<{ ffi::DTYPE_I32 }>,
    pub(crate) state_indices: DVec<{ ffi::DTYPE_I32 }>,
    pub(crate) state_out_indices: Option<DVec<{ ffi::DTYPE_I32 }>>,
    pub(crate) out: Bf16Heads,
    pub(crate) batch_size: u32,
    pub(crate) scale: f32,
    pub(crate) use_qk_l2norm: bool,
    pub(crate) disable_state_update: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPrefillBf16 {
    raw: qscu::GdnPrefillDesc,
}

impl GdnPrefillBf16 {
    pub(crate) fn new(args: GdnPrefillBf16Args) -> Result<Self, Status> {
        validate_gdn_scale(args.scale)?;
        validate_nonzero(&[args.batch_size])?;
        let total_tokens = args.q.tokens;
        validate_gdn_recurrent_tensors(
            args.q,
            args.k,
            args.v,
            args.a,
            args.b,
            args.a_log,
            args.dt_bias,
            args.out,
            total_tokens,
        )?;
        let seq_indptr_len = args
            .batch_size
            .checked_add(1)
            .ok_or(Status::InvalidArgument)?;
        require_i32_vec(args.seq_indptr, seq_indptr_len)?;
        require_gdn_state_index_vec(
            args.state_indices,
            args.batch_size,
            GdnStateIndexPolicy::NegativeSkips,
        )?;
        if let Some(indices) = args.state_out_indices {
            require_gdn_state_index_vec(
                indices,
                args.batch_size,
                GdnStateIndexPolicy::NegativeSkips,
            )?;
        }
        Ok(Self {
            raw: qscu::GdnPrefillDesc {
                q: args.q.tensor(),
                k: args.k.tensor(),
                v: args.v.tensor(),
                a: args.a.tensor(),
                b: args.b.tensor(),
                a_log: args.a_log.tensor(),
                dt_bias: args.dt_bias.tensor(),
                state: args.state.tensor(),
                seq_indptr: args.seq_indptr.data,
                state_indices: args.state_indices.tensor(),
                state_out_indices: args
                    .state_out_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                out: args.out.tensor(),
                batch_size: args.batch_size,
                total_tokens,
                num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
                num_k_heads: QWEN36_GDN_NUM_K_HEADS,
                num_v_heads: QWEN36_GDN_NUM_V_HEADS,
                key_dim: QWEN36_GDN_KEY_DIM,
                value_dim: QWEN36_GDN_VALUE_DIM,
                state_layout: qscu::GDN_STATE_LAYOUT_VK,
                scale: args.scale,
                use_qk_l2norm: u32::from(args.use_qk_l2norm),
                disable_state_update: u32::from(args.disable_state_update),
            },
        })
    }
}

fn validate_ptr(data: ffi::DevicePtr) -> Result<(), Status> {
    if data.is_null() {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_nonzero(values: &[u32]) -> Result<(), Status> {
    if values.contains(&0) {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn heads_mul(lhs: u32, rhs: u32) -> Result<u32, Status> {
    lhs.checked_mul(rhs).ok_or(Status::InvalidArgument)
}

fn validate_eps(eps: f32) -> Result<(), Status> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_soft_cap(soft_cap: f32) -> Result<(), Status> {
    if soft_cap.is_nan() || (soft_cap > 0.0 && !soft_cap.is_finite()) {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn require_supported_rope_head_dim(head_dim: u32) -> Result<(), Status> {
    if head_dim % 2 != 0 {
        return Err(Status::InvalidArgument);
    }
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        return Err(Status::Unsupported);
    }
    Ok(())
}

fn validate_gdn_scale(scale: f32) -> Result<(), Status> {
    if !scale.is_finite() || scale == 0.0 {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn require_float_vec(vec: Bf16OrF32Vec, expected_len: u32) -> Result<(), Status> {
    if vec.len() != expected_len || !vec.is_contiguous() {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn require_i32_vec(vec: DVec<{ ffi::DTYPE_I32 }>, expected_len: u32) -> Result<(), Status> {
    if vec.len != expected_len || !vec.is_contiguous() {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn require_gdn_state_index_vec(
    vec: DVec<{ ffi::DTYPE_I32 }>,
    expected_len: u32,
    _policy: GdnStateIndexPolicy,
) -> Result<(), Status> {
    require_i32_vec(vec, expected_len)
}

fn require_qwen36_gdn_heads(
    heads: Bf16Heads,
    expected_heads: u32,
    expected_tokens: u32,
) -> Result<(), Status> {
    if heads.tokens != expected_tokens
        || heads.heads != expected_heads
        || heads.head_dim != QWEN36_GDN_KEY_DIM
    {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_gdn_recurrent_tensors(
    q: Bf16Heads,
    k: Bf16Heads,
    v: Bf16Heads,
    a: DMat<{ ffi::DTYPE_BF16 }>,
    b: DMat<{ ffi::DTYPE_BF16 }>,
    a_log: DVec<{ ffi::DTYPE_F32 }>,
    dt_bias: DVec<{ ffi::DTYPE_F32 }>,
    out: Bf16Heads,
    total_tokens: u32,
) -> Result<(), Status> {
    require_qwen36_gdn_heads(q, QWEN36_GDN_NUM_Q_HEADS, total_tokens)?;
    require_qwen36_gdn_heads(k, QWEN36_GDN_NUM_K_HEADS, total_tokens)?;
    require_qwen36_gdn_heads(v, QWEN36_GDN_NUM_V_HEADS, total_tokens)?;
    require_qwen36_gdn_heads(out, QWEN36_GDN_NUM_V_HEADS, total_tokens)?;
    if a.rows != total_tokens
        || a.cols != QWEN36_GDN_NUM_V_HEADS
        || b.rows != total_tokens
        || b.cols != QWEN36_GDN_NUM_V_HEADS
        || a_log.len != QWEN36_GDN_NUM_V_HEADS
        || dt_bias.len != QWEN36_GDN_NUM_V_HEADS
    {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn zero_tensor1(dtype: ffi::DTypeRaw) -> ffi::Tensor1 {
    ffi::Tensor1 {
        data: ptr::null_mut(),
        dtype,
        shape: [0],
        stride: [0],
    }
}

fn zero_tensor2(dtype: ffi::DTypeRaw) -> ffi::Tensor2 {
    ffi::Tensor2 {
        data: ptr::null_mut(),
        dtype,
        shape: [0, 0],
        stride: [0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::c_void;

    fn device_ptr(offset: usize) -> ffi::DevicePtr {
        (0x1000usize + offset) as *mut c_void
    }

    fn bf16_mat(offset: usize, rows: u32, cols: u32) -> DMat<{ ffi::DTYPE_BF16 }> {
        DMat::contiguous(device_ptr(offset), rows, cols).unwrap()
    }

    fn f32_mat(offset: usize, rows: u32, cols: u32) -> DMat<{ ffi::DTYPE_F32 }> {
        DMat::contiguous(device_ptr(offset), rows, cols).unwrap()
    }

    fn bf16_vec(offset: usize, len: u32) -> DVec<{ ffi::DTYPE_BF16 }> {
        DVec::contiguous(device_ptr(offset), len).unwrap()
    }

    fn f32_vec(offset: usize, len: u32) -> DVec<{ ffi::DTYPE_F32 }> {
        DVec::contiguous(device_ptr(offset), len).unwrap()
    }

    fn i32_vec(offset: usize, len: u32) -> DVec<{ ffi::DTYPE_I32 }> {
        DVec::contiguous(device_ptr(offset), len).unwrap()
    }

    fn heads(offset: usize, tokens: u32, heads: u32, head_dim: u32) -> Bf16Heads {
        Bf16Heads::contiguous(device_ptr(offset), tokens, heads, head_dim).unwrap()
    }

    fn q_heads(offset: usize, tokens: u32) -> Bf16Heads {
        heads(offset, tokens, QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_KEY_DIM)
    }

    fn k_heads(offset: usize, tokens: u32) -> Bf16Heads {
        heads(offset, tokens, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_KEY_DIM)
    }

    fn v_heads(offset: usize, tokens: u32) -> Bf16Heads {
        heads(offset, tokens, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)
    }

    fn recurrent_state(offset: usize) -> GdnRecurrentState {
        GdnRecurrentState::contiguous(device_ptr(offset), FloatStorage::Bf16, 4).unwrap()
    }

    #[test]
    fn generic_vec_and_mat_aliases_preserve_tensor_metadata() {
        let bf16_vec = bf16_vec(1, 4).tensor();
        assert_eq!(bf16_vec.dtype, ffi::DTYPE_BF16);
        assert_eq!(bf16_vec.shape, [4]);
        assert_eq!(bf16_vec.stride, [1]);

        let f32_vec = f32_vec(2, 5).tensor();
        assert_eq!(f32_vec.dtype, ffi::DTYPE_F32);

        let i32_vec = i32_vec(3, 6).tensor();
        assert_eq!(i32_vec.dtype, ffi::DTYPE_I32);

        let bf16_mat = bf16_mat(4, 2, 3);
        let f32_mat = DMat::<{ ffi::DTYPE_F32 }>::new(device_ptr(5), 2, 3, 8).unwrap();
        assert!(bf16_mat.same_shape(f32_mat));

        let f32_tensor = f32_mat.tensor();
        assert_eq!(f32_tensor.dtype, ffi::DTYPE_F32);
        assert_eq!(f32_tensor.shape, [2, 3]);
        assert_eq!(f32_tensor.stride, [8, 1]);

        let i32_tensor = DMat::<{ ffi::DTYPE_I32 }>::contiguous(device_ptr(6), 3, 2)
            .unwrap()
            .tensor();
        assert_eq!(i32_tensor.dtype, ffi::DTYPE_I32);
    }

    #[test]
    fn handles_reject_null_zero_and_bad_strides() {
        assert!(matches!(
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(ptr::null_mut(), 1, 1),
            Err(Status::InvalidArgument)
        ));
        assert!(matches!(
            DMat::<{ ffi::DTYPE_BF16 }>::contiguous(device_ptr(1), 0, 1),
            Err(Status::InvalidArgument)
        ));
        assert!(matches!(
            DMat::<{ ffi::DTYPE_BF16 }>::new(device_ptr(2), 2, 4, 3),
            Err(Status::InvalidArgument)
        ));
        assert!(matches!(
            Bf16Heads::new(device_ptr(3), 2, 4, 64, 255, 64),
            Err(Status::InvalidArgument)
        ));
        assert!(matches!(
            Workspace::new(ptr::null_mut(), 1),
            Err(Status::InvalidArgument)
        ));
    }

    #[test]
    fn qscu_descriptor_builders_validate_shapes_and_modes() {
        let gate = bf16_mat(10, 2, 8);
        let up = bf16_mat(11, 2, 8);
        let out = bf16_mat(12, 2, 8);
        assert!(SiluAndMulBf16::new(gate, up, out).is_ok());
        let padded_gate = DMat::new(device_ptr(13), 2, 8, 16).unwrap();
        assert!(matches!(
            SiluAndMulBf16::new(padded_gate, up, out),
            Err(Status::InvalidArgument)
        ));

        assert!(
            Qwen36SharedExpertGateAddBf16::new(
                Bf16OrF32Mat::F32(f32_mat(110, 2, 1)),
                bf16_mat(111, 2, 8),
                bf16_mat(112, 2, 8),
            )
            .is_ok()
        );
        assert!(
            Qwen36SharedExpertGateAddBf16::new(
                Bf16OrF32Mat::Bf16(bf16_mat(113, 2, 1)),
                bf16_mat(114, 2, 8),
                bf16_mat(115, 2, 8),
            )
            .is_ok()
        );
        assert!(matches!(
            Qwen36SharedExpertGateAddBf16::new(
                Bf16OrF32Mat::F32(f32_mat(116, 2, 2)),
                bf16_mat(117, 2, 8),
                bf16_mat(118, 2, 8),
            ),
            Err(Status::InvalidArgument)
        ));

        let token_ids = i32_vec(14, 2);
        let embedding = bf16_mat(15, 128, 8);
        assert!(EmbeddingGatherBf16::new(token_ids, embedding, out).is_ok());
        assert!(matches!(
            EmbeddingGatherBf16::new(token_ids, embedding, bf16_mat(16, 2, 7)),
            Err(Status::InvalidArgument)
        ));

        let logits = f32_mat(17, 2, 128);
        assert!(LogitsSoftCapF32::new(logits, 30.0).is_ok());
        assert!(LogitsSoftCapF32::new(logits, f32::NEG_INFINITY).is_ok());
        assert!(matches!(
            LogitsSoftCapF32::new(logits, f32::NAN),
            Err(Status::InvalidArgument)
        ));

        assert!(GreedyArgmaxF32::new(logits, token_ids).is_ok());
        assert!(matches!(
            GreedyArgmaxF32::new(logits, i32_vec(18, 1)),
            Err(Status::InvalidArgument)
        ));
        let huge_vocab = DMat::contiguous(device_ptr(19), 1, i32::MAX as u32 + 1).unwrap();
        assert!(matches!(
            GreedyArgmaxF32::new(huge_vocab, i32_vec(20, 1)),
            Err(Status::Unsupported)
        ));

        assert!(
            RouterTopK::new(
                Bf16OrF32Mat::F32(logits),
                DMat::<{ ffi::DTYPE_I32 }>::contiguous(device_ptr(21), 2, 4).unwrap(),
                f32_mat(22, 2, 4),
                RouterScore::Softmax,
                true,
                1.0,
            )
            .is_ok()
        );
        assert!(
            RouterTopK::new(
                Bf16OrF32Mat::Bf16(bf16_mat(23, 2, 128)),
                DMat::<{ ffi::DTYPE_I32 }>::contiguous(device_ptr(24), 2, 4).unwrap(),
                f32_mat(25, 2, 4),
                RouterScore::Sigmoid,
                false,
                0.5,
            )
            .is_ok()
        );
        assert!(matches!(
            RouterTopK::new(
                Bf16OrF32Mat::F32(logits),
                DMat::<{ ffi::DTYPE_I32 }>::contiguous(device_ptr(26), 2, ROUTER_MAX_TOP_K + 1)
                    .unwrap(),
                f32_mat(27, 2, ROUTER_MAX_TOP_K + 1),
                RouterScore::Softmax,
                true,
                1.0,
            ),
            Err(Status::Unsupported)
        ));
        assert!(matches!(
            RouterTopK::new(
                Bf16OrF32Mat::F32(logits),
                DMat::<{ ffi::DTYPE_I32 }>::contiguous(device_ptr(28), 2, 4).unwrap(),
                f32_mat(29, 2, 4),
                RouterScore::Softmax,
                true,
                0.0,
            ),
            Err(Status::InvalidArgument)
        ));
    }

    #[test]
    fn qscb_and_qsfi_descriptor_builders_accept_padded_rows() {
        let x = DMat::new(device_ptr(30), 4, 128, 160).unwrap();
        let weight = DMat::new(device_ptr(31), 256, 128, 128).unwrap();
        let out = DMat::new(device_ptr(32), 4, 256, 320).unwrap();
        assert!(Bf16Gemm::new(x, weight, Bf16OrF32Mat::F32(out), Workspace::none()).is_ok());
        assert!(matches!(
            Bf16Gemm::new(
                x,
                weight,
                Bf16OrF32Mat::F32(DMat::new(device_ptr(33), 4, 128, 128).unwrap()),
                Workspace::none(),
            ),
            Err(Status::InvalidArgument)
        ));

        let norm_out = DMat::new(device_ptr(34), 4, 128, 160).unwrap();
        assert!(RmsNormBf16::new(x, bf16_vec(35, 128), norm_out, 1.0e-6).is_ok());
        assert!(matches!(
            RmsNormBf16::new(x, bf16_vec(36, 64), norm_out, 1.0e-6),
            Err(Status::InvalidArgument)
        ));
        assert!(matches!(
            RmsNormBf16::new(x, bf16_vec(37, 128), norm_out, 0.0),
            Err(Status::InvalidArgument)
        ));

        let residual = DMat::new(device_ptr(38), 4, 128, 192).unwrap();
        assert!(FusedAddRmsNormBf16::new(x, residual, bf16_vec(39, 128), 1.0e-6).is_ok());

        let q = Bf16Heads::new(device_ptr(40), 2, 4, 128, 1024, 128).unwrap();
        let k = Bf16Heads::new(device_ptr(41), 2, 2, 128, 512, 128).unwrap();
        let q_out = Bf16Heads::new(device_ptr(42), 2, 4, 128, 1024, 128).unwrap();
        let k_out = Bf16Heads::new(device_ptr(43), 2, 2, 128, 512, 128).unwrap();
        assert!(RopeApplyBf16::new(q, k, q_out, k_out, i32_vec(44, 2)).is_ok());
        assert!(matches!(
            RopeApplyBf16::new(
                heads(45, 2, 4, 96),
                heads(46, 2, 2, 96),
                heads(47, 2, 4, 96),
                heads(48, 2, 2, 96),
                i32_vec(49, 2),
            ),
            Err(Status::Unsupported)
        ));
        let alias_bad_stride = Bf16Heads::contiguous(device_ptr(40), 2, 4, 128).unwrap();
        assert!(matches!(
            RopeApplyBf16::new(q, k, alias_bad_stride, k_out, i32_vec(50, 2)),
            Err(Status::InvalidArgument)
        ));
    }

    #[test]
    fn gdn_prep_descriptors_enforce_qwen36_shapes() {
        let tokens = 2;
        let post = GdnPostConvPrepareBf16Args {
            conv_out: bf16_mat(60, tokens, QWEN36_GDN_PACKED_DIM),
            a: bf16_mat(61, tokens, QWEN36_GDN_NUM_V_HEADS),
            b: bf16_mat(62, tokens, QWEN36_GDN_NUM_V_HEADS),
            a_log: f32_vec(63, QWEN36_GDN_NUM_V_HEADS),
            dt_bias: f32_vec(64, QWEN36_GDN_NUM_V_HEADS),
            q: q_heads(65, tokens),
            k: k_heads(66, tokens),
            v: v_heads(67, tokens),
            g_out: Some(f32_mat(68, tokens, QWEN36_GDN_NUM_V_HEADS)),
            beta_out: None,
            apply_qk_l2norm: true,
            l2norm_eps: 1.0e-6,
            forget_gate_output: GdnForgetGateOutput::LinearAlpha,
        };
        assert!(GdnPostConvPrepareBf16::new(post).is_ok());

        let mut bad_post = post;
        bad_post.q = heads(69, tokens, 8, QWEN36_GDN_KEY_DIM);
        assert!(matches!(
            GdnPostConvPrepareBf16::new(bad_post),
            Err(Status::InvalidArgument)
        ));

        let gated = GdnRmsNormGatedBf16Args {
            x: v_heads(70, tokens),
            gate: v_heads(71, tokens),
            weight: Bf16OrF32Vec::F32(f32_vec(72, QWEN36_GDN_VALUE_DIM)),
            out: v_heads(73, tokens),
            eps: 1.0e-6,
            gate_activation: Activation::Silu,
        };
        assert!(GdnRmsNormGatedBf16::new(gated).is_ok());
        let mut bad_gated = gated;
        bad_gated.gate_activation = Activation::None;
        assert!(matches!(
            GdnRmsNormGatedBf16::new(bad_gated),
            Err(Status::Unsupported)
        ));

        let conv = GdnCausalConv1dBf16Args {
            x: bf16_mat(74, tokens, QWEN36_GDN_PACKED_DIM),
            weight: bf16_mat(75, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH),
            bias: Some(Bf16OrF32Vec::Bf16(bf16_vec(76, QWEN36_GDN_PACKED_DIM))),
            state: GdnConvState::contiguous(device_ptr(77), FloatStorage::F32, 3).unwrap(),
            state_read_indices: Some(i32_vec(78, tokens)),
            state_write_indices: None,
            seq_indptr: None,
            out: bf16_mat(79, tokens, QWEN36_GDN_PACKED_DIM),
            batch_size: tokens,
            activation: Activation::Silu,
            update_state: true,
        };
        assert!(GdnCausalConv1dBf16::new(conv).is_ok());
        let mut bad_conv = conv;
        bad_conv.activation = Activation::Sigmoid;
        assert!(matches!(
            GdnCausalConv1dBf16::new(bad_conv),
            Err(Status::Unsupported)
        ));
    }

    #[test]
    fn gdn_recurrent_descriptors_validate_decode_and_prefill_shapes() {
        let tokens = 3;
        let decode = GdnDecodeBf16Args {
            q: q_heads(90, tokens),
            k: k_heads(91, tokens),
            v: v_heads(92, tokens),
            a: bf16_mat(93, tokens, QWEN36_GDN_NUM_V_HEADS),
            b: bf16_mat(94, tokens, QWEN36_GDN_NUM_V_HEADS),
            a_log: f32_vec(95, QWEN36_GDN_NUM_V_HEADS),
            dt_bias: f32_vec(96, QWEN36_GDN_NUM_V_HEADS),
            state: recurrent_state(97),
            state_indices: i32_vec(98, tokens),
            state_out_indices: None,
            out: v_heads(99, tokens),
            scale: 0.08838835,
            use_qk_l2norm: true,
            disable_state_update: false,
        };
        assert!(GdnDecodeBf16::new(decode).is_ok());

        let mut bad_decode = decode;
        bad_decode.scale = 0.0;
        assert!(matches!(
            GdnDecodeBf16::new(bad_decode),
            Err(Status::InvalidArgument)
        ));
        let mut bad_indices = decode;
        bad_indices.state_indices = i32_vec(100, 2);
        assert!(matches!(
            GdnDecodeBf16::new(bad_indices),
            Err(Status::InvalidArgument)
        ));

        let prefill = GdnPrefillBf16Args {
            q: decode.q,
            k: decode.k,
            v: decode.v,
            a: decode.a,
            b: decode.b,
            a_log: decode.a_log,
            dt_bias: decode.dt_bias,
            state: decode.state,
            seq_indptr: i32_vec(101, 3),
            state_indices: i32_vec(102, 2),
            state_out_indices: Some(i32_vec(103, 2)),
            out: decode.out,
            batch_size: 2,
            scale: decode.scale,
            use_qk_l2norm: decode.use_qk_l2norm,
            disable_state_update: decode.disable_state_update,
        };
        assert!(GdnPrefillBf16::new(prefill).is_ok());
        let mut bad_prefill = prefill;
        bad_prefill.seq_indptr = i32_vec(104, 2);
        assert!(matches!(
            GdnPrefillBf16::new(bad_prefill),
            Err(Status::InvalidArgument)
        ));
    }

    #[test]
    fn gdn_state_index_policy_validates_host_indices() {
        assert_eq!(
            GdnStateIndexPolicy::NegativeSkips.validate_host_indices(&[-1, 0, 3], 4),
            Ok(())
        );
        assert_eq!(
            GdnStateIndexPolicy::NonNegative.validate_host_indices(&[-1, 0], 4),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            GdnStateIndexPolicy::NegativeSkips.validate_host_indices(&[4], 4),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            GdnStateIndexPolicy::NegativeSkips.validate_host_indices(&[0], 0),
            Err(Status::InvalidArgument)
        );
    }
}
