use crate::dtype::I32;
use crate::{
    backend,
    constants::attention::{HEAD_DIM, NUM_KV_HEADS, NUM_Q_HEADS},
    ffi,
    memory::CudaCtx,
};
use std::rc::Rc;

mod attention;
mod core;

pub use core::CoreState;
pub(crate) use core::EngineCore;
pub struct Engine {
    inner: Box<attention::AttentionSession>,
}

impl Engine {
    pub(crate) fn operators(&mut self) -> backend::Operators<'_> {
        self.inner.operators()
    }
}

impl Engine {
    pub fn new(ctx: Rc<CudaCtx>, config: EngineConfig) -> Result<Self, Status> {
        attention::AttentionSession::new(ctx, config).map(|inner| Self { inner })
    }

    pub(crate) fn fresh_prefix_state(&self) -> Result<attention::PrefixState, Status> {
        attention::PrefixState::new(
            self.inner.ctx.clone(),
            EngineCore::new(self.inner.prefix.core.config())?,
        )
    }

    pub(crate) fn replace_prefix_state(
        &mut self,
        prefix: attention::PrefixState,
    ) -> Result<attention::PrefixState, Status> {
        // The existing plans/descriptors must still describe this cache geometry,
        // device and stream. Validate before installing any candidate state.
        if prefix.core.config() != self.inner.prefix.core.config() {
            return Err(Status::InvalidArgument);
        }
        Ok(std::mem::replace(&mut self.inner.prefix, prefix))
    }

    pub fn reset(&mut self) -> Result<(), Status> {
        self.inner.prefix.core.reset()
    }

    pub fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        self.inner.prefix.core.release_requests(request_ids)
    }

    pub fn state(&self) -> Result<CoreState<'_>, Status> {
        self.inner.prefix.core.state()
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
        self.inner.prefix.core.abort_batch()
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

use crate::dtype::DynDType;

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
    if num_q_heads != NUM_Q_HEADS || num_kv_heads != NUM_KV_HEADS {
        return Err(Status::Unsupported);
    }
    Ok(())
}

pub(crate) fn validate_supported_attention_head_dim(head_dim: u32) -> Result<(), Status> {
    if head_dim == 0 {
        return Err(Status::InvalidArgument);
    }
    if head_dim != HEAD_DIM {
        return Err(Status::Unsupported);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchKind {
    Append,
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineConfig {
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
    pub q_rope_offset: ffi::ErasedDevicePtr,
    pub lse: ffi::ErasedDevicePtr,
    pub q_scale: f32,
    pub k_scale: f32,
    pub v_scale: f32,
}

impl AttentionLayer {
    pub(crate) fn bf16_attention(
        layer_idx: u32,
        q: backend::Bf16Heads,
        k: backend::Bf16Heads,
        v: backend::Bf16Heads,
        o: backend::Bf16Heads,
        q_rope_offset: backend::DVec<I32>,
    ) -> Self {
        Self {
            layer_idx,
            q: q.tensor(),
            k: k.tensor(),
            v: v.tensor(),
            o: o.tensor(),
            q_rope_offset: q_rope_offset.tensor().data,
            lse: std::ptr::null_mut(),
            q_scale: 0.0,
            k_scale: 0.0,
            v_scale: 0.0,
        }
    }
}
