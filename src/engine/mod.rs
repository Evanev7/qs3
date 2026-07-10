use crate::{QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_Q_HEADS, ffi};

mod attention;
mod core;

pub use core::CoreState;
pub(crate) use core::EngineCore;
pub struct Engine {
    inner: Box<attention::AttentionSession>,
}

impl Engine {
    pub(crate) fn operators(&mut self) -> crate::backend::Operators<'_> {
        self.inner.operators()
    }
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self, Status> {
        attention::AttentionSession::new(config).map(|inner| Self { inner })
    }

    pub fn reset(&mut self) -> Result<(), Status> {
        self.inner.core.reset()
    }

    pub fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        self.inner.core.release_requests(request_ids)
    }

    pub fn state(&self) -> Result<CoreState<'_>, Status> {
        self.inner.core.state()
    }

    pub fn begin_append(&mut self, batch: AppendBatch<'_>) -> Result<(), Status> {
        self.inner.prepare_append(batch)
    }

    pub unsafe fn append_attention(&mut self, layer: &AttentionLayer) -> Result<(), Status> {
        unsafe { self.inner.execute_append_attention(layer) }
    }

    pub fn begin_decode(&mut self, batch: DecodeBatch<'_>) -> Result<(), Status> {
        self.inner.prepare_decode(batch)
    }

    pub unsafe fn decode_attention(&mut self, layer: &AttentionLayer) -> Result<(), Status> {
        unsafe { self.inner.execute_decode_attention(layer) }
    }

    pub fn commit_batch(&mut self, commit: Commit<'_>) -> Result<(), Status> {
        self.inner.commit_batch(commit)
    }

    pub fn abort_batch(&mut self) -> Result<(), Status> {
        self.inner.core.abort_batch()
    }
}

pub type RequestId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    InvalidArgument,
    Unsupported,
    OutOfMemory,
    CudaError,
    BackendError,
    InternalError,
    Unreachable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynDType {
    F32,
    F16,
    BF16,
    FP8E4M3,
    FP8E5M2,
    NVFP4E2M1,
    MXFP4E2M1,
    MXFP8E4M3,
    I32,
    U32,
    I8,
    U8,
}

impl DynDType {
    pub fn bits(self) -> usize {
        match self {
            DynDType::F32 | DynDType::I32 | DynDType::U32 => 32,
            DynDType::F16 | DynDType::BF16 => 16,
            DynDType::FP8E4M3
            | DynDType::FP8E5M2
            | DynDType::MXFP8E4M3
            | DynDType::I8
            | DynDType::U8 => 8,
            DynDType::NVFP4E2M1 | DynDType::MXFP4E2M1 => 4,
        }
    }

    pub fn storage_bytes_for(self, elements: usize) -> Result<usize, Status> {
        let bits = elements
            .checked_mul(self.bits())
            .ok_or(Status::InvalidArgument)?;
        bits.checked_add(7)
            .ok_or(Status::InvalidArgument)
            .map(|bits| bits / 8)
    }

    fn is_runtime_supported(self) -> bool {
        matches!(self, DynDType::F16 | DynDType::BF16)
    }

    pub(crate) fn to_raw(self) -> ffi::DTypeRaw {
        match self {
            DynDType::F32 => ffi::DTYPE_F32,
            DynDType::F16 => ffi::DTYPE_F16,
            DynDType::BF16 => ffi::DTYPE_BF16,
            DynDType::FP8E4M3 => ffi::DTYPE_FP8_E4M3,
            DynDType::FP8E5M2 => ffi::DTYPE_FP8_E5M2,
            DynDType::NVFP4E2M1 => ffi::DTYPE_NVFP4_E2M1,
            DynDType::MXFP4E2M1 => ffi::DTYPE_MXFP4_E2M1,
            DynDType::MXFP8E4M3 => ffi::DTYPE_MXFP8_E4M3,
            DynDType::I32 => ffi::DTYPE_I32,
            DynDType::U32 => ffi::DTYPE_U32,
            DynDType::I8 => ffi::DTYPE_I8,
            DynDType::U8 => ffi::DTYPE_U8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvLayout {
    NHD,
    HND,
}

impl KvLayout {
    pub(crate) fn to_raw(self) -> ffi::KvLayoutRaw {
        match self {
            KvLayout::NHD => ffi::KV_LAYOUT_NHD,
            KvLayout::HND => ffi::KV_LAYOUT_HND,
        }
    }
}

pub(crate) fn validate_supported_attention_grouping(
    num_q_heads: u32,
    num_kv_heads: u32,
) -> Result<(), Status> {
    if num_q_heads == 0 || num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(Status::InvalidArgument);
    }
    if num_q_heads != QWEN36_FULL_ATTN_Q_HEADS || num_kv_heads != QWEN36_FULL_ATTN_KV_HEADS {
        return Err(Status::Unsupported);
    }
    Ok(())
}

pub(crate) fn validate_supported_attention_head_dim(head_dim: u32) -> Result<(), Status> {
    if head_dim == 0 {
        return Err(Status::InvalidArgument);
    }
    if head_dim != QWEN36_FULL_ATTN_HEAD_DIM {
        return Err(Status::Unsupported);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchKind {
    Append,
    Decode,
}

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub device_ordinal: i32,
    pub stream: *mut std::ffi::c_void,
    pub num_layers: u32,
    pub max_live_requests: u32,
    pub max_batch_rows: u32,
    pub max_batch_tokens: u32,
    pub max_seq_len: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub activation_dtype: DynDType,
    pub kv_dtype: DynDType,
    pub kv_layout: KvLayout,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub logits_soft_cap: f32,
    pub qsfi_float_workspace_bytes: usize,
    pub qsfi_int_workspace_bytes: usize,
    pub qsfi_host_int_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AppendBatch<'a> {
    pub request_ids: &'a [RequestId],
    pub token_indptr: &'a [i32],
    pub tokens: &'a [i32],
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeBatch<'a> {
    pub request_ids: &'a [RequestId],
    pub tokens: &'a [i32],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Commit<'a> {
    pub accepted_token_counts: Option<&'a [u32]>,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionLayer {
    pub layer_idx: u32,
    pub q: ffi::Tensor3,
    pub k: ffi::Tensor3,
    pub v: ffi::Tensor3,
    pub o: ffi::Tensor3,
    pub q_rope_offset: ffi::DevicePtr,
    pub lse: ffi::DevicePtr,
    pub q_scale: f32,
    pub k_scale: f32,
    pub v_scale: f32,
}

impl AttentionLayer {
    pub(crate) fn bf16_attention(
        layer_idx: u32,
        q: crate::backend::Bf16Heads,
        k: crate::backend::Bf16Heads,
        v: crate::backend::Bf16Heads,
        o: crate::backend::Bf16Heads,
        q_rope_offset: ffi::DevicePtr,
    ) -> Self {
        Self {
            layer_idx,
            q: q.tensor(),
            k: k.tensor(),
            v: v.tensor(),
            o: o.tensor(),
            q_rope_offset,
            lse: std::ptr::null_mut(),
            q_scale: 0.0,
            k_scale: 0.0,
            v_scale: 0.0,
        }
    }
}
