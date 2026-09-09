use super::{
    QwenConfig, activate_device, checked_usize_product, result_from_cuda, synchronize_stream,
};
use crate::{
    QWEN36_FULL_ATTN_Q_PROJ_OUT, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS,
    QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM,
    engine::Status,
    ffi::{self, cuda},
};

use std::{ffi::c_void, mem, ptr};

pub(super) struct RunnerScratch {
    pub(super) token_ids: DeviceBuffer<i32>,
    pub(super) positions: DeviceBuffer<i32>,
    pub(super) residual: DeviceBuffer<u16>,
    pub(super) norm: DeviceBuffer<u16>,
    pub(super) q_proj_out: DeviceBuffer<u16>,
    pub(super) q: DeviceBuffer<u16>,
    pub(super) k: DeviceBuffer<u16>,
    pub(super) v: DeviceBuffer<u16>,
    pub(super) attn_out: DeviceBuffer<u16>,
    pub(super) attn_proj: DeviceBuffer<u16>,
    pub(super) attn_gate: DeviceBuffer<u16>,
    pub(super) gate: DeviceBuffer<u16>,
    pub(super) up: DeviceBuffer<u16>,
    pub(super) mlp: DeviceBuffer<u16>,
    pub(super) mlp_out: DeviceBuffer<u16>,
    pub(super) shared_gate: DeviceBuffer<u16>,
    pub(super) shared_up: DeviceBuffer<u16>,
    pub(super) shared_mlp: DeviceBuffer<u16>,
    pub(super) shared_out: DeviceBuffer<u16>,
    pub(super) shared_gate_logits: DeviceBuffer<f32>,
    pub(super) router_logits: DeviceBuffer<u16>,
    pub(super) topk_ids: DeviceBuffer<i32>,
    pub(super) topk_weights: DeviceBuffer<f32>,
    pub(super) moe_workspace: DeviceBuffer<u8>,
    pub(super) gdn_packed: DeviceBuffer<u16>,
    pub(super) gdn_conv_out: DeviceBuffer<u16>,
    pub(super) gdn_a: DeviceBuffer<u16>,
    pub(super) gdn_b: DeviceBuffer<u16>,
    pub(super) gdn_q: DeviceBuffer<u16>,
    pub(super) gdn_k: DeviceBuffer<u16>,
    pub(super) gdn_v: DeviceBuffer<u16>,
    pub(super) gdn_recurrent_out: DeviceBuffer<u16>,
    pub(super) gdn_gate: DeviceBuffer<u16>,
    pub(super) gdn_norm_out: DeviceBuffer<u16>,
    pub(super) gdn_seq_indptr: DeviceBuffer<i32>,
    pub(super) gdn_state_indices: DeviceBuffer<i32>,
    pub(super) gdn_state_out_indices: DeviceBuffer<i32>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) next_token_ids: DeviceBuffer<i32>,
}

impl RunnerScratch {
    pub(super) fn new(device_ordinal: i32) -> Self {
        Self {
            token_ids: DeviceBuffer::empty(device_ordinal),
            positions: DeviceBuffer::empty(device_ordinal),
            residual: DeviceBuffer::empty(device_ordinal),
            norm: DeviceBuffer::empty(device_ordinal),
            q_proj_out: DeviceBuffer::empty(device_ordinal),
            q: DeviceBuffer::empty(device_ordinal),
            k: DeviceBuffer::empty(device_ordinal),
            v: DeviceBuffer::empty(device_ordinal),
            attn_out: DeviceBuffer::empty(device_ordinal),
            attn_proj: DeviceBuffer::empty(device_ordinal),
            attn_gate: DeviceBuffer::empty(device_ordinal),
            gate: DeviceBuffer::empty(device_ordinal),
            up: DeviceBuffer::empty(device_ordinal),
            mlp: DeviceBuffer::empty(device_ordinal),
            mlp_out: DeviceBuffer::empty(device_ordinal),
            shared_gate: DeviceBuffer::empty(device_ordinal),
            shared_up: DeviceBuffer::empty(device_ordinal),
            shared_mlp: DeviceBuffer::empty(device_ordinal),
            shared_out: DeviceBuffer::empty(device_ordinal),
            shared_gate_logits: DeviceBuffer::empty(device_ordinal),
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

    pub(super) fn ensure(&mut self, config: &QwenConfig, rows: u32) -> Result<(), Status> {
        let hidden = checked_usize_product(&[rows, config.hidden_size])?;
        let logits = config.vocab_size as usize;
        let row_count = rows as usize;

        self.token_ids.ensure(row_count)?;
        self.positions.ensure(row_count)?;
        self.residual.ensure(hidden)?;
        self.norm.ensure(hidden)?;
        let q_hidden = checked_usize_product(&[rows, config.q_hidden_size()?])?;
        let q_proj_out = checked_usize_product(&[rows, QWEN36_FULL_ATTN_Q_PROJ_OUT])?;
        let kv_hidden = checked_usize_product(&[rows, config.kv_hidden_size()?])?;
        self.q_proj_out.ensure(q_proj_out)?;
        self.q.ensure(q_hidden)?;
        self.k.ensure(kv_hidden)?;
        self.v.ensure(kv_hidden)?;
        self.attn_out.ensure(q_hidden)?;
        self.attn_proj.ensure(hidden)?;
        self.attn_gate.ensure(q_hidden)?;
        if let Some(moe) = config.moe_config() {
            self.router_logits
                .ensure(checked_usize_product(&[rows, moe.num_experts])?)?;
            let topk = checked_usize_product(&[rows, moe.num_experts_per_tok])?;
            self.topk_ids.ensure(topk)?;
            self.topk_weights.ensure(topk)?;
            if moe.shared_expert_intermediate_size != 0 {
                let shared_intermediate =
                    checked_usize_product(&[rows, moe.shared_expert_intermediate_size])?;
                self.shared_gate.ensure(shared_intermediate)?;
                self.shared_up.ensure(shared_intermediate)?;
                self.shared_mlp.ensure(shared_intermediate)?;
                self.shared_out.ensure(hidden)?;
                self.shared_gate_logits.ensure(row_count)?;
            }
        } else {
            let intermediate = checked_usize_product(&[rows, config.intermediate_size])?;
            self.gate.ensure(intermediate)?;
            self.up.ensure(intermediate)?;
            self.mlp.ensure(intermediate)?;
        }
        self.mlp_out.ensure(hidden)?;
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
        }
        self.logits.ensure(logits)?;
        self.next_token_ids.ensure(1)?;
        Ok(())
    }

    pub(super) fn ensure_moe_workspace(&mut self, bytes: usize) -> Result<(), Status> {
        self.moe_workspace.ensure(bytes)
    }
}

pub(crate) struct DeviceBuffer<T> {
    pub(super) ptr: *mut T,
    pub(super) cap: usize,
    pub(super) device_ordinal: i32,
}

impl<T> DeviceBuffer<T> {
    /// # Safety
    ///
    /// `ptr` must be non-null, properly aligned, uniquely owned, cover at
    /// least `cap` elements, and be releasable with `cudaFree` after activating
    /// `device_ordinal`.
    pub(crate) unsafe fn from_raw_parts(device_ordinal: i32, ptr: *mut T, cap: usize) -> Self {
        debug_assert!(!ptr.is_null());
        debug_assert!(cap != 0);
        Self {
            ptr,
            cap,
            device_ordinal,
        }
    }

    pub(super) fn empty(device_ordinal: i32) -> Self {
        Self {
            ptr: ptr::null_mut(),
            cap: 0,
            device_ordinal,
        }
    }

    pub(super) fn from_slice(
        device_ordinal: i32,
        stream: *mut c_void,
        values: &[T],
    ) -> Result<Self, Status>
    where
        T: Copy,
    {
        let mut buffer = Self::empty(device_ordinal);
        buffer.upload(stream, values)?;
        synchronize_stream(stream)?;
        Ok(buffer)
    }

    pub(super) fn ensure(&mut self, len: usize) -> Result<(), Status> {
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

    pub(super) fn upload(&mut self, stream: *mut c_void, values: &[T]) -> Result<(), Status>
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

    pub(super) fn download(&self, stream: *mut c_void, out: &mut [T]) -> Result<(), Status>
    where
        T: Copy,
    {
        self.download_range(stream, 0, out)
    }

    pub(super) fn download_range(
        &self,
        stream: *mut c_void,
        offset: usize,
        out: &mut [T],
    ) -> Result<(), Status>
    where
        T: Copy,
    {
        if out.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(out.len())
            .ok_or(Status::InvalidArgument)?;
        if self.ptr.is_null() || end > self.cap {
            return Err(Status::InvalidArgument);
        }
        activate_device(self.device_ordinal)?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                out.as_mut_ptr().cast(),
                self.ptr.add(offset).cast(),
                mem::size_of_val(out),
                cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        })?;
        synchronize_stream(stream)
    }

    pub(super) fn zero(&mut self, len: usize, stream: *mut c_void) -> Result<(), Status> {
        if len == 0 {
            return Ok(());
        }
        self.ensure(len)?;
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or(Status::InvalidArgument)?;
        result_from_cuda(unsafe { cuda::cudaMemsetAsync(self.ptr.cast(), 0, bytes, stream) })
    }

    pub(super) fn as_device_ptr(&self) -> ffi::DevicePtr {
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

// These views retain the allocation's element type and check its extent. Like
// the backend views, they are non-owning; unsafe launches require the owner to
// remain alive until stream completion. Batch execution borrows those owners.
impl<T: crate::backend::DeviceElement> DeviceBuffer<T> {
    pub(super) fn matrix(
        &self,
        rows: u32,
        cols: u32,
    ) -> Result<crate::backend::DMat<T::DType>, Status> {
        self.matrix_at(0, rows, cols)
    }

    pub(super) fn matrix_at(
        &self,
        offset: usize,
        rows: u32,
        cols: u32,
    ) -> Result<crate::backend::DMat<T::DType>, Status> {
        let len = checked_usize_product(&[rows, cols])?;
        self.check_view_len(offset.checked_add(len).ok_or(Status::InvalidArgument)?)?;
        // SAFETY: DeviceBuffer owns `cap` elements; the checked sum above places
        // offset within that allocation (or one past it for a rejected empty
        // shape). Borrowing self keeps it live during this pointer calculation.
        // Constructing the descriptor neither dereferences nor launches work.
        crate::backend::DMat::contiguous(unsafe { self.ptr.add(offset).cast() }, rows, cols)
    }

    pub(super) fn vector(&self, len: u32) -> Result<crate::backend::DVec<T::DType>, Status> {
        self.check_view_len(len as usize)?;
        crate::backend::DVec::contiguous(self.as_device_ptr(), len)
    }

    pub(super) fn tensor3(
        &self,
        a: u32,
        b: u32,
        c: u32,
    ) -> Result<crate::backend::DTensor3<T::DType>, Status> {
        self.check_view_len(checked_usize_product(&[a, b, c])?)?;
        crate::backend::DTensor3::contiguous(self.as_device_ptr(), a, b, c)
    }
}

impl<T> DeviceBuffer<T> {
    fn check_view_len(&self, len: usize) -> Result<(), Status> {
        if self.ptr.is_null() || len == 0 || len > self.cap {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

impl DeviceBuffer<u16> {
    pub(super) fn heads(
        &self,
        rows: u32,
        heads: u32,
        dim: u32,
    ) -> Result<crate::backend::Bf16Heads, Status> {
        self.check_view_len(checked_usize_product(&[rows, heads, dim])?)?;
        crate::backend::Bf16Heads::contiguous(self.as_device_ptr(), rows, heads, dim)
    }
}

impl DeviceBuffer<u8> {
    pub(super) fn workspace(&self, bytes: usize) -> Result<crate::backend::Workspace, Status> {
        if bytes == 0 {
            return Ok(crate::backend::Workspace::none());
        }
        self.check_view_len(bytes)?;
        crate::backend::Workspace::new(self.as_device_ptr(), bytes)
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;
    use crate::backend::{BF16, DMat, F32};
    use std::mem::ManuallyDrop;

    // Host backing is sufficient for descriptor checks; no kernel is launched.
    // ManuallyDrop prevents DeviceBuffer from passing host memory to cudaFree.
    #[test]
    fn typed_views_check_capacity_offsets_and_shape_overflow() {
        let mut storage = [0u16; 8];
        let buffer = ManuallyDrop::new(DeviceBuffer {
            ptr: storage.as_mut_ptr(),
            cap: storage.len(),
            device_ordinal: -1,
        });
        let matrix: DMat<BF16> = buffer.matrix(2, 4).unwrap();
        // SAFETY: storage backs the entire matrix and is live for both calls.
        unsafe {
            assert_eq!(matrix.row(1).unwrap(), buffer.matrix_at(4, 1, 4).unwrap());
            assert!(matrix.row(2).is_err());
        }
        assert!(buffer.matrix(3, 3).is_err());
        assert!(buffer.matrix_at(5, 1, 4).is_err());
        assert!(buffer.matrix_at(usize::MAX, 1, 1).is_err());
        assert!(buffer.matrix(u32::MAX, u32::MAX).is_err());
        assert!(buffer.matrix(0, 4).is_err());
        assert!(buffer.vector(8).is_ok());
        assert!(buffer.vector(9).is_err());
        assert!(buffer.tensor3(2, 2, 2).is_ok());
        assert!(buffer.tensor3(2, 2, 3).is_err());
        assert!(buffer.heads(1, 2, 4).is_ok());
        assert!(buffer.heads(2, 2, 4).is_err());

        let mut floats = [0f32; 8];
        let buffer = ManuallyDrop::new(DeviceBuffer {
            ptr: floats.as_mut_ptr(),
            cap: floats.len(),
            device_ordinal: -1,
        });
        let matrix: DMat<F32> = buffer.matrix(2, 4).unwrap();
        // SAFETY: floats backs the entire matrix and remains live here.
        assert_eq!(
            unsafe { matrix.row(1) }.unwrap(),
            buffer.matrix_at(4, 1, 4).unwrap()
        );
    }

    #[test]
    fn workspace_view_checks_requested_bytes_and_empty_storage() {
        let empty = DeviceBuffer::<u8>::empty(-1);
        assert!(empty.workspace(0).is_ok());
        assert!(empty.workspace(1).is_err());
        let mut bytes = [0u8; 32];
        let buffer = ManuallyDrop::new(DeviceBuffer {
            ptr: bytes.as_mut_ptr(),
            cap: bytes.len(),
            device_ordinal: -1,
        });
        assert!(buffer.workspace(16).is_ok());
        assert!(buffer.workspace(32).is_ok());
        assert!(buffer.workspace(33).is_err());
    }
}
