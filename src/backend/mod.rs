#![allow(dead_code)]

use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_ROTARY_DIM, QWEN36_GDN_CONV_STATE,
    QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_NUM_V_HEADS,
    QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM, Status,
    ffi::{self, sys},
};

mod dtype;
pub(crate) mod qscb;
pub(crate) mod qscu;
pub(crate) mod qsfi;
mod tensor;

pub(crate) use dtype::{BF16, F32, I32, DeviceElement};
pub(crate) use qscb::Qscb;
pub(crate) use qscu::Qscu;
pub(crate) use qsfi::Qsfi;
pub(crate) use tensor::{DMat, DTensor3, DVec};

use std::ptr;

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
pub(crate) enum RouterScore {
    Softmax,
    Sigmoid,
}

impl RouterScore {
    fn raw(self) -> sys::qscu_router_score {
        match self {
            Self::Softmax => sys::QSCU_ROUTER_SCORE_SOFTMAX,
            Self::Sigmoid => sys::QSCU_ROUTER_SCORE_SIGMOID,
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
    qsfi: &'a mut Qsfi,
    qscb: &'a mut Qscb,
}

impl<'a> Operators<'a> {
    pub(crate) fn new(stream: &'a ffi::CudaStream, qsfi: &'a mut Qsfi, qscb: &'a mut Qscb) -> Self {
        Self { stream, qsfi, qscb }
    }

    pub(crate) fn qscu(&mut self) -> Qscu<'_> {
        Qscu::new(self.stream, self.qsfi)
    }

    pub(crate) fn qscb(&mut self) -> &mut Qscb {
        self.qscb
    }

    pub(crate) fn qsfi(&mut self) -> &mut Qsfi {
        self.qsfi
    }
}

fn validate_ptr(data: ffi::DevicePtr) -> Result<(), Status> {
    if data.is_null() {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn result_from_raw(status: ffi::StatusRaw) -> Result<(), Status> {
    match status {
        ffi::sys::QSFI_STATUS_OK => Ok(()),
        ffi::sys::QSFI_STATUS_INVALID_ARGUMENT => Err(Status::InvalidArgument),
        ffi::sys::QSFI_STATUS_UNSUPPORTED => Err(Status::Unsupported),
        ffi::sys::QSFI_STATUS_OUT_OF_MEMORY => Err(Status::OutOfMemory),
        ffi::sys::QSFI_STATUS_CUDA_ERROR => Err(Status::CudaError),
        ffi::sys::QSFI_STATUS_BACKEND_ERROR => Err(Status::BackendError),
        ffi::sys::QSFI_STATUS_INTERNAL_ERROR => Err(Status::InternalError),
        _ => Err(Status::InternalError),
    }
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
