#![allow(dead_code)]

use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_Q_HIDDEN, QWEN36_FULL_ATTN_ROTARY_DIM,
    QWEN36_GDN_CONV_STATE, QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS,
    QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM,
    QWEN36_MOE_MAX_EXPERTS, QWEN36_MOE_MAX_TOP_K, Status,
    ffi::{self, qscb, qscu, qsfi},
};

pub(crate) mod cublas;
pub(crate) mod cuda;
mod dtype;
pub(crate) mod flashinfer;
mod tensor;

#[cfg(test)]
pub(crate) use cublas::Bf16Gemm;
pub(crate) use cublas::Cublas;
pub(crate) use cuda::Cuda;
#[cfg(test)]
pub(crate) use cuda::{
    EmbeddingGatherBf16, GdnCausalConv1dBf16, GdnCausalConv1dBf16Args, GdnDecodeBf16,
    GdnDecodeBf16Args, GdnPostConvPrepareBf16, GdnPostConvPrepareBf16Args, GdnPrefillBf16,
    GdnPrefillBf16Args, GdnRmsNormGatedBf16, GdnRmsNormGatedBf16Args, GreedyArgmaxF32,
    LogitsSoftCapF32, Qwen36FullAttentionOutputGateBf16, Qwen36SharedExpertGateAddBf16, RouterTopK,
    SiluAndMulBf16,
};
pub(crate) use dtype::{BF16, F32, I32};
pub(crate) use flashinfer::FlashInfer;
#[cfg(test)]
pub(crate) use flashinfer::{FusedAddRmsNormBf16, RmsNormBf16, RopeApplyBf16};
pub(crate) use tensor::{DMat, DTensor3, DVec};

use std::ptr;

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
    Bf16(DMat<BF16>),
    F32(DMat<F32>),
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
    Bf16(DVec<BF16>),
    F32(DVec<F32>),
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

pub(crate) struct Operators<'a> {
    stream: &'a ffi::CudaStream,
    flashinfer: &'a mut qsfi::Context,
    cublas: &'a mut qscb::Context,
}

impl<'a> Operators<'a> {
    pub(crate) fn new(
        stream: &'a ffi::CudaStream,
        flashinfer: &'a mut qsfi::Context,
        cublas: &'a mut qscb::Context,
    ) -> Self {
        Self {
            stream,
            flashinfer,
            cublas,
        }
    }

    pub(crate) fn cuda(&mut self) -> Cuda<'_> {
        Cuda::new(self.stream, self.flashinfer)
    }

    pub(crate) fn cublas(&mut self) -> Cublas<'_> {
        Cublas::new(self.cublas)
    }

    pub(crate) fn flashinfer(&mut self) -> FlashInfer<'_> {
        FlashInfer::new(self.flashinfer)
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

fn require_supported_rope_dims(head_dim: u32, rotary_dim: u32) -> Result<(), Status> {
    if head_dim == 0 || rotary_dim == 0 {
        return Err(Status::InvalidArgument);
    }
    if head_dim % 2 != 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err(Status::InvalidArgument);
    }
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        return Err(Status::Unsupported);
    }
    if head_dim == QWEN36_FULL_ATTN_HEAD_DIM && rotary_dim != QWEN36_FULL_ATTN_ROTARY_DIM {
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

fn require_i32_vec(vec: DVec<I32>, expected_len: u32) -> Result<(), Status> {
    if vec.len != expected_len || !vec.is_contiguous() {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn require_gdn_state_index_vec(
    vec: DVec<I32>,
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
    a: DMat<BF16>,
    b: DMat<BF16>,
    a_log: DVec<BF16>,
    dt_bias: DVec<BF16>,
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
mod tests;
