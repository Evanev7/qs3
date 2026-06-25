use super::{
    AppendBatch, BatchKind, Commit, CoreState, DType, DecodeBatch, KvLayout, RequestId, Session,
    SessionConfig, SessionCore, SessionLayer, Status, qsfi, qsfi_sys,
};

use std::ffi::c_void;
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_MEMORY_ALLOCATION: i32 = 2;
const CUDA_MEMORY_TYPE_DEVICE: i32 = 2;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

type QsfiTensorDesc = qsfi::TensorDesc;
type QsfiAttentionDesc = qsfi::AttentionDesc;
type QsfiPagedKvCache = qsfi::PagedKvCache;
type QsfiPagedKvPlan = qsfi::PagedKvPlan;
type QsfiQoPlan = qsfi::QoPlan;
type QsfiPagedKvTable = qsfi::PagedKvTable;
type QsfiBatchDecodeExecuteDesc = qsfi::BatchDecodeExecuteDesc;
type QsfiBatchPrefillExecuteDesc = qsfi::BatchPrefillExecuteDesc;
type QsfiAppendDecode = qsfi::AppendDecode;
type QsfiAppendPrefill = qsfi::AppendPrefill;

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaPointerAttributes {
    memory_type: i32,
    device: i32,
    device_pointer: *mut c_void,
    host_pointer: *mut c_void,
}

unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaPointerGetAttributes(attributes: *mut CudaPointerAttributes, ptr: *const c_void) -> i32;
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QsSessionConfig {
    device_ordinal: i32,
    stream: qsfi::CudaStream,
    num_layers: u32,
    max_live_requests: u32,
    max_batch_size: u32,
    max_seq_len: u32,
    max_pages: u32,
    page_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    vocab_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    activation_dtype: qsfi::DTypeRaw,
    kv_dtype: qsfi::DTypeRaw,
    kv_layout: qsfi::KvLayoutRaw,
    rope_theta: f32,
    rope_scale: f32,
    logits_soft_cap: f32,
    qsfi_float_workspace_bytes: usize,
    qsfi_int_workspace_bytes: usize,
    qsfi_host_int_workspace_bytes: usize,
}

impl QsSessionConfig {
    fn from_session_config(config: SessionConfig) -> Self {
        Self {
            device_ordinal: config.device_ordinal,
            stream: config.stream,
            num_layers: config.num_layers,
            max_live_requests: config.max_live_requests,
            max_batch_size: config.max_batch_size,
            max_seq_len: config.max_seq_len,
            max_pages: config.max_pages,
            page_size: config.page_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            num_q_heads: config.num_q_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            activation_dtype: dtype_to_raw(config.activation_dtype),
            kv_dtype: dtype_to_raw(config.kv_dtype),
            kv_layout: layout_to_raw(config.kv_layout),
            rope_theta: config.rope_theta,
            rope_scale: config.rope_scale,
            logits_soft_cap: config.logits_soft_cap,
            qsfi_float_workspace_bytes: config.qsfi_float_workspace_bytes,
            qsfi_int_workspace_bytes: config.qsfi_int_workspace_bytes,
            qsfi_host_int_workspace_bytes: config.qsfi_host_int_workspace_bytes,
        }
    }
}

#[repr(C)]
pub struct QsSessionState {
    batch_kind: BatchKind,
    live_request_count: u32,
    batch_size: u32,
    batch_token_count: u32,
    live_num_indices: u32,
    allocated_pages: u32,
    free_page_count: u32,
    max_pages: u32,
    page_size: u32,
    live_request_ids: *const RequestId,
    live_seq_lens: *const i32,
    live_kv_indptr: *const i32,
    live_kv_indices: *const i32,
    live_last_page_len: *const i32,
    free_pages: *const i32,
    batch_request_ids: *const RequestId,
    batch_tokens: *const i32,
    batch_qo_indptr: *const i32,
    batch_kv_indptr: *const i32,
    batch_kv_indices: *const i32,
    batch_last_page_len: *const i32,
    batch_rope_pos_offset: *const i32,
    batch_append_batch_indices: *const i32,
    batch_append_positions: *const i32,
    d_batch_tokens: qsfi::DevicePtr,
    d_batch_qo_indptr: qsfi::DevicePtr,
    d_batch_kv_indptr: qsfi::DevicePtr,
    d_batch_kv_indices: qsfi::DevicePtr,
    d_batch_last_page_len: qsfi::DevicePtr,
    d_batch_rope_pos_offset: qsfi::DevicePtr,
    d_batch_append_batch_indices: qsfi::DevicePtr,
    d_batch_append_positions: qsfi::DevicePtr,
}

#[repr(C)]
pub struct QsSessionAppendBatch {
    request_ids: *const RequestId,
    token_indptr: *const i32,
    tokens: *const i32,
    batch_size: u32,
    token_count: u32,
}

#[repr(C)]
pub struct QsSessionDecodeBatch {
    request_ids: *const RequestId,
    tokens: *const i32,
    batch_size: u32,
}

#[repr(C)]
pub struct QsSessionCommit {
    accepted_token_counts: *const u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QsSessionLayer {
    layer_idx: u32,
    q: QsfiTensorDesc,
    k: QsfiTensorDesc,
    v: QsfiTensorDesc,
    o: QsfiTensorDesc,
    q_rope_offset: qsfi::DevicePtr,
    lse: qsfi::DevicePtr,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
}

struct DeviceI32Buffer {
    data: *mut i32,
    cap: usize,
}

impl DeviceI32Buffer {
    fn new() -> Self {
        Self {
            data: ptr::null_mut(),
            cap: 0,
        }
    }

    fn ensure(&mut self, device_ordinal: i32, count: usize) -> Status {
        if count == 0 || self.cap >= count {
            return Status::Ok;
        }
        let bytes = match count.checked_mul(mem::size_of::<i32>()) {
            Some(bytes) => bytes,
            None => return Status::InvalidArgument,
        };
        let status = activate_device(device_ordinal);
        if status != Status::Ok {
            return status;
        }
        let mut next = ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut next, bytes) };
        if err != CUDA_SUCCESS {
            return status_from_cuda(err);
        }
        if !self.data.is_null() {
            unsafe {
                cudaFree(self.data.cast());
            }
        }
        self.data = next.cast();
        self.cap = count;
        Status::Ok
    }

    fn upload(&mut self, device_ordinal: i32, stream: *mut c_void, values: &[i32]) -> Status {
        if values.is_empty() {
            return Status::Ok;
        }
        let status = self.ensure(device_ordinal, values.len());
        if status != Status::Ok {
            return status;
        }
        let bytes = match values.len().checked_mul(mem::size_of::<i32>()) {
            Some(bytes) => bytes,
            None => return Status::InvalidArgument,
        };
        let err = unsafe {
            cudaMemcpyAsync(
                self.data.cast(),
                values.as_ptr().cast(),
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        status_from_cuda(err)
    }

    fn free(&mut self) {
        if !self.data.is_null() {
            unsafe {
                cudaFree(self.data.cast());
            }
        }
        self.data = ptr::null_mut();
        self.cap = 0;
    }

    fn device_ptr_if(&self, present: bool) -> *mut c_void {
        if present {
            self.data.cast()
        } else {
            ptr::null_mut()
        }
    }
}

struct LayerCache {
    k: *mut c_void,
    v: *mut c_void,
}

struct PlanCache {
    plan: Option<qsfi::Plan>,
    batch_size: u32,
    num_indices: u32,
    total_tokens: u32,
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    valid: bool,
}

impl PlanCache {
    fn new() -> Self {
        Self {
            plan: None,
            batch_size: 0,
            num_indices: 0,
            total_tokens: 0,
            qo_indptr: Vec::new(),
            kv_indptr: Vec::new(),
            valid: false,
        }
    }

    fn matches(
        &self,
        batch_size: u32,
        num_indices: u32,
        total_tokens: u32,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
    ) -> bool {
        self.valid
            && self.plan.is_some()
            && self.batch_size == batch_size
            && self.num_indices == num_indices
            && self.total_tokens == total_tokens
            && self.qo_indptr == qo_indptr
            && self.kv_indptr == kv_indptr
    }

    fn destroy(&mut self) {
        self.plan = None;
        self.qo_indptr.clear();
        self.kv_indptr.clear();
        self.valid = false;
    }
}

pub struct QsSession {
    core: SessionCore,
    stream: *mut c_void,
    ctx: qsfi::Context,
    append_attention: QsfiAttentionDesc,
    decode_attention: QsfiAttentionDesc,
    layer_caches: Vec<LayerCache>,
    d_batch_tokens: DeviceI32Buffer,
    d_batch_qo_indptr: DeviceI32Buffer,
    d_batch_kv_indptr: DeviceI32Buffer,
    d_batch_kv_indices: DeviceI32Buffer,
    d_batch_last_page_len: DeviceI32Buffer,
    d_batch_rope_pos_offset: DeviceI32Buffer,
    d_batch_append_batch_indices: DeviceI32Buffer,
    d_batch_append_positions: DeviceI32Buffer,
    append_plan: PlanCache,
    decode_plan: PlanCache,
}

impl QsSession {
    fn new(config: &QsSessionConfig) -> Result<Box<Self>, Status> {
        let core_config = SessionConfig {
            device_ordinal: config.device_ordinal,
            stream: config.stream,
            num_layers: config.num_layers,
            max_live_requests: config.max_live_requests,
            max_batch_size: config.max_batch_size,
            max_seq_len: config.max_seq_len,
            max_pages: config.max_pages,
            page_size: config.page_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            num_q_heads: config.num_q_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            activation_dtype: dtype_from_raw(config.activation_dtype)?,
            kv_dtype: dtype_from_raw(config.kv_dtype)?,
            kv_layout: layout_from_raw(config.kv_layout)?,
            rope_theta: config.rope_theta,
            rope_scale: config.rope_scale,
            logits_soft_cap: config.logits_soft_cap,
            qsfi_float_workspace_bytes: config.qsfi_float_workspace_bytes,
            qsfi_int_workspace_bytes: config.qsfi_int_workspace_bytes,
            qsfi_host_int_workspace_bytes: config.qsfi_host_int_workspace_bytes,
        };
        let core = SessionCore::new(core_config)?;
        let mut ctx = qsfi::Context::new(config.device_ordinal, config.stream)?;
        ctx.reserve_scratch(
            config.qsfi_float_workspace_bytes,
            config.qsfi_int_workspace_bytes,
            config.qsfi_host_int_workspace_bytes,
        )?;
        let mut session = Box::new(Self {
            append_attention: make_attention(config, qsfi::MASK_MODE_CAUSAL),
            decode_attention: make_attention(config, qsfi::MASK_MODE_NONE),
            core,
            stream: config.stream,
            ctx,
            layer_caches: Vec::new(),
            d_batch_tokens: DeviceI32Buffer::new(),
            d_batch_qo_indptr: DeviceI32Buffer::new(),
            d_batch_kv_indptr: DeviceI32Buffer::new(),
            d_batch_kv_indices: DeviceI32Buffer::new(),
            d_batch_last_page_len: DeviceI32Buffer::new(),
            d_batch_rope_pos_offset: DeviceI32Buffer::new(),
            d_batch_append_batch_indices: DeviceI32Buffer::new(),
            d_batch_append_positions: DeviceI32Buffer::new(),
            append_plan: PlanCache::new(),
            decode_plan: PlanCache::new(),
        });
        session.allocate_layer_caches()?;
        Ok(session)
    }

    fn allocate_layer_caches(&mut self) -> Result<(), Status> {
        let config = self.core.config();
        let elems = (config.max_pages as usize)
            .checked_mul(config.page_size as usize)
            .and_then(|v| v.checked_mul(config.num_kv_heads as usize))
            .and_then(|v| v.checked_mul(config.head_dim as usize))
            .ok_or(Status::InvalidArgument)?;
        let bytes = elems
            .checked_mul(dtype_size(config.kv_dtype)?)
            .ok_or(Status::InvalidArgument)?;
        self.layer_caches
            .try_reserve_exact(config.num_layers as usize)
            .map_err(|_| Status::OutOfMemory)?;
        let status = activate_device(config.device_ordinal);
        if status != Status::Ok {
            return Err(status);
        }
        for _ in 0..config.num_layers {
            let mut k = ptr::null_mut();
            let mut v = ptr::null_mut();
            let err = unsafe { cudaMalloc(&mut k, bytes) };
            if err != CUDA_SUCCESS {
                return Err(status_from_cuda(err));
            }
            let err = unsafe { cudaMalloc(&mut v, bytes) };
            if err != CUDA_SUCCESS {
                unsafe {
                    cudaFree(k);
                }
                return Err(status_from_cuda(err));
            }
            self.layer_caches.push(LayerCache { k, v });
        }
        Ok(())
    }

    fn upload_active_batch(&mut self) -> Status {
        let config = self.core.config();
        let uploads = [
            self.d_batch_tokens.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_tokens(),
            ),
            self.d_batch_qo_indptr.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_qo_indptr(),
            ),
            self.d_batch_kv_indptr.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_kv_indptr(),
            ),
            self.d_batch_kv_indices.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_kv_indices(),
            ),
            self.d_batch_last_page_len.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_last_page_len(),
            ),
            self.d_batch_rope_pos_offset.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_rope_pos_offset(),
            ),
            self.d_batch_append_batch_indices.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_append_batch_indices(),
            ),
            self.d_batch_append_positions.upload(
                config.device_ordinal,
                self.stream,
                self.core.batch_append_positions(),
            ),
        ];
        uploads
            .into_iter()
            .find(|status| *status != Status::Ok)
            .unwrap_or(Status::Ok)
    }

    fn ensure_append_plan(&mut self) -> Status {
        let num_indices = match u32::try_from(self.core.batch_kv_indices().len()) {
            Ok(v) => v,
            Err(_) => return Status::InvalidArgument,
        };
        if self.append_plan.matches(
            self.core.batch_size(),
            num_indices,
            self.core.batch_token_count(),
            self.core.batch_qo_indptr(),
            self.core.batch_kv_indptr(),
        ) {
            return Status::Ok;
        }

        self.append_plan.destroy();
        let qo = QsfiQoPlan {
            indptr: ptr_or_null(self.core.batch_qo_indptr()),
            batch_size: self.core.batch_size(),
            total_tokens: self.core.batch_token_count(),
        };
        let page_table = QsfiPagedKvPlan {
            indptr: ptr_or_null(self.core.batch_kv_indptr()),
            indices: ptr_or_null(self.core.batch_kv_indices()),
            last_page_len: ptr_or_null(self.core.batch_last_page_len()),
            batch_size: self.core.batch_size(),
            num_indices,
        };
        let plan = match unsafe {
            self.ctx
                .create_prefill_plan(&self.append_attention, &qo, &page_table)
        } {
            Ok(plan) => plan,
            Err(status) => return status,
        };
        self.append_plan.plan = Some(plan);
        self.append_plan.batch_size = self.core.batch_size();
        self.append_plan.num_indices = num_indices;
        self.append_plan.total_tokens = self.core.batch_token_count();
        self.append_plan.qo_indptr = self.core.batch_qo_indptr().to_vec();
        self.append_plan.kv_indptr = self.core.batch_kv_indptr().to_vec();
        self.append_plan.valid = true;
        Status::Ok
    }

    fn ensure_decode_plan(&mut self) -> Status {
        let num_indices = match u32::try_from(self.core.batch_kv_indices().len()) {
            Ok(v) => v,
            Err(_) => return Status::InvalidArgument,
        };
        if self.decode_plan.matches(
            self.core.batch_size(),
            num_indices,
            self.core.batch_size(),
            &[],
            self.core.batch_kv_indptr(),
        ) {
            return Status::Ok;
        }

        self.decode_plan.destroy();
        let page_table = QsfiPagedKvPlan {
            indptr: ptr_or_null(self.core.batch_kv_indptr()),
            indices: ptr_or_null(self.core.batch_kv_indices()),
            last_page_len: ptr_or_null(self.core.batch_last_page_len()),
            batch_size: self.core.batch_size(),
            num_indices,
        };
        let plan = match unsafe {
            self.ctx
                .create_decode_plan(&self.decode_attention, &page_table)
        } {
            Ok(plan) => plan,
            Err(status) => return status,
        };
        self.decode_plan.plan = Some(plan);
        self.decode_plan.batch_size = self.core.batch_size();
        self.decode_plan.num_indices = num_indices;
        self.decode_plan.total_tokens = self.core.batch_size();
        self.decode_plan.qo_indptr.clear();
        self.decode_plan.kv_indptr = self.core.batch_kv_indptr().to_vec();
        self.decode_plan.valid = true;
        Status::Ok
    }

    fn make_kv_cache(&self, layer_idx: u32) -> Result<QsfiPagedKvCache, Status> {
        let idx = usize::try_from(layer_idx).map_err(|_| Status::InvalidArgument)?;
        let layer = self.layer_caches.get(idx).ok_or(Status::InvalidArgument)?;
        Ok(QsfiPagedKvCache {
            k: self.make_cache_tensor(layer.k)?,
            v: self.make_cache_tensor(layer.v)?,
            k_scale: zero_tensor(),
            v_scale: zero_tensor(),
        })
    }

    fn make_cache_tensor(&self, data: *mut c_void) -> Result<QsfiTensorDesc, Status> {
        let config = self.core.config();
        let mut tensor = zero_tensor();
        tensor.data = data;
        tensor.dtype = dtype_to_raw(config.kv_dtype);
        tensor.ndim = 4;
        tensor.shape[0] = config.max_pages as i64;
        tensor.shape[3] = config.head_dim as i64;
        tensor.stride[3] = 1;
        if config.kv_layout == KvLayout::NHD {
            tensor.shape[1] = config.page_size as i64;
            tensor.shape[2] = config.num_kv_heads as i64;
            tensor.stride[0] =
                checked_i64_product(&[config.page_size, config.num_kv_heads, config.head_dim])?;
            tensor.stride[1] = checked_i64_product(&[config.num_kv_heads, config.head_dim])?;
            tensor.stride[2] = config.head_dim as i64;
        } else {
            tensor.shape[1] = config.num_kv_heads as i64;
            tensor.shape[2] = config.page_size as i64;
            tensor.stride[0] =
                checked_i64_product(&[config.num_kv_heads, config.page_size, config.head_dim])?;
            tensor.stride[1] = checked_i64_product(&[config.page_size, config.head_dim])?;
            tensor.stride[2] = config.head_dim as i64;
        }
        Ok(tensor)
    }

    fn make_active_page_table(&self) -> QsfiPagedKvTable {
        QsfiPagedKvTable {
            indptr: self
                .d_batch_kv_indptr
                .device_ptr_if(!self.core.batch_kv_indptr().is_empty()),
            indices: self
                .d_batch_kv_indices
                .device_ptr_if(!self.core.batch_kv_indices().is_empty()),
            last_page_len: self
                .d_batch_last_page_len
                .device_ptr_if(!self.core.batch_last_page_len().is_empty()),
            rope_pos_offset: self
                .d_batch_rope_pos_offset
                .device_ptr_if(!self.core.batch_rope_pos_offset().is_empty()),
            batch_size: self.core.batch_size(),
            num_indices: u32::try_from(self.core.batch_kv_indices().len()).unwrap_or(u32::MAX),
        }
    }

    fn begin_append_batch(&mut self, batch: AppendBatch<'_>) -> Status {
        if let Err(status) =
            self.core
                .begin_append(batch.request_ids, batch.token_indptr, batch.tokens)
        {
            return status;
        }
        let mut status = self.upload_active_batch();
        if status == Status::Ok {
            status = self.ensure_append_plan();
        }
        if status != Status::Ok {
            let _ = self.core.abort_batch();
        }
        status
    }

    unsafe fn append_layer_desc(&mut self, layer: &QsSessionLayer) -> Status {
        if self.core.batch_kind() != BatchKind::Append || self.append_plan.plan.is_none() {
            return Status::InvalidArgument;
        }
        let kv_cache = match self.make_kv_cache(layer.layer_idx) {
            Ok(kv_cache) => kv_cache,
            Err(status) => return status,
        };
        let page_table = self.make_active_page_table();
        let append = QsfiAppendPrefill {
            k: layer.k,
            v: layer.v,
            batch_indices: self
                .d_batch_append_batch_indices
                .device_ptr_if(!self.core.batch_append_batch_indices().is_empty()),
            positions: self
                .d_batch_append_positions
                .device_ptr_if(!self.core.batch_append_positions().is_empty()),
            kv_cache,
            page_table,
            num_tokens: self.core.batch_token_count(),
        };
        if let Err(status) = unsafe {
            self.ctx
                .append_paged_kv_prefill(&self.append_attention, &append)
        } {
            return status;
        }
        let execute = QsfiBatchPrefillExecuteDesc {
            q: layer.q,
            q_rope_offset: layer.q_rope_offset,
            o: layer.o,
            lse: layer.lse,
            qo_indptr: self
                .d_batch_qo_indptr
                .device_ptr_if(!self.core.batch_qo_indptr().is_empty()),
            kv_cache,
            page_table,
            q_scale: layer.q_scale,
            k_scale: layer.k_scale,
            v_scale: layer.v_scale,
        };
        status_from_result(unsafe {
            self.ctx
                .execute_prefill(self.append_plan.plan.as_ref().unwrap(), &execute)
        })
    }

    fn begin_decode_batch(&mut self, batch: DecodeBatch<'_>) -> Status {
        if let Err(status) = self.core.begin_decode(batch.request_ids, batch.tokens) {
            return status;
        }
        let mut status = self.upload_active_batch();
        if status == Status::Ok {
            status = self.ensure_decode_plan();
        }
        if status != Status::Ok {
            let _ = self.core.abort_batch();
        }
        status
    }

    unsafe fn decode_layer_desc(&mut self, layer: &QsSessionLayer) -> Status {
        if self.core.batch_kind() != BatchKind::Decode || self.decode_plan.plan.is_none() {
            return Status::InvalidArgument;
        }
        let kv_cache = match self.make_kv_cache(layer.layer_idx) {
            Ok(kv_cache) => kv_cache,
            Err(status) => return status,
        };
        let page_table = self.make_active_page_table();
        let append = QsfiAppendDecode {
            k: layer.k,
            v: layer.v,
            kv_cache,
            page_table,
        };
        if let Err(status) = unsafe {
            self.ctx
                .append_paged_kv_decode(&self.decode_attention, &append)
        } {
            return status;
        }
        let execute = QsfiBatchDecodeExecuteDesc {
            q: layer.q,
            q_rope_offset: layer.q_rope_offset,
            o: layer.o,
            lse: layer.lse,
            kv_cache,
            page_table,
            q_scale: layer.q_scale,
            k_scale: layer.k_scale,
            v_scale: layer.v_scale,
        };
        status_from_result(unsafe {
            self.ctx
                .execute_decode(self.decode_plan.plan.as_ref().unwrap(), &execute)
        })
    }
}

impl Drop for QsSession {
    fn drop(&mut self) {
        self.append_plan.destroy();
        self.decode_plan.destroy();
        let config = self.core.config();
        let _ = activate_device(config.device_ordinal);
        for layer in &mut self.layer_caches {
            if !layer.k.is_null() {
                unsafe {
                    cudaFree(layer.k);
                }
                layer.k = ptr::null_mut();
            }
            if !layer.v.is_null() {
                unsafe {
                    cudaFree(layer.v);
                }
                layer.v = ptr::null_mut();
            }
        }
        self.d_batch_tokens.free();
        self.d_batch_qo_indptr.free();
        self.d_batch_kv_indptr.free();
        self.d_batch_kv_indices.free();
        self.d_batch_last_page_len.free();
        self.d_batch_rope_pos_offset.free();
        self.d_batch_append_batch_indices.free();
        self.d_batch_append_positions.free();
    }
}

impl QsSessionLayer {
    fn from_session_layer(layer: &SessionLayer) -> Self {
        Self {
            layer_idx: layer.layer_idx,
            q: layer.q,
            k: layer.k,
            v: layer.v,
            o: layer.o,
            q_rope_offset: layer.q_rope_offset,
            lse: layer.lse,
            q_scale: layer.q_scale,
            k_scale: layer.k_scale,
            v_scale: layer.v_scale,
        }
    }
}

pub struct RuntimeSession {
    inner: Box<QsSession>,
}

impl RuntimeSession {
    pub fn new(config: SessionConfig) -> Result<Self, Status> {
        let config = QsSessionConfig::from_session_config(config);
        QsSession::new(&config).map(|inner| Self { inner })
    }
}

impl Session for RuntimeSession {
    fn reset(&mut self) -> Result<(), Status> {
        self.inner.core.reset()
    }

    fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        self.inner.core.release_requests(request_ids)
    }

    fn state(&self) -> Result<CoreState<'_>, Status> {
        self.inner.core.state()
    }

    fn begin_append(&mut self, batch: AppendBatch<'_>) -> Result<(), Status> {
        result_from_status(self.inner.begin_append_batch(batch))
    }

    unsafe fn append_layer(&mut self, layer: &SessionLayer) -> Result<(), Status> {
        let layer = QsSessionLayer::from_session_layer(layer);
        result_from_status(unsafe { self.inner.append_layer_desc(&layer) })
    }

    fn begin_decode(&mut self, batch: DecodeBatch<'_>) -> Result<(), Status> {
        result_from_status(self.inner.begin_decode_batch(batch))
    }

    unsafe fn decode_layer(&mut self, layer: &SessionLayer) -> Result<(), Status> {
        let layer = QsSessionLayer::from_session_layer(layer);
        result_from_status(unsafe { self.inner.decode_layer_desc(&layer) })
    }

    fn commit_batch(&mut self, commit: Commit<'_>) -> Result<(), Status> {
        self.inner.core.commit_batch(commit.accepted_token_counts)
    }

    fn abort_batch(&mut self) -> Result<(), Status> {
        self.inner.core.abort_batch()
    }
}

fn ffi_guard(f: impl FnOnce() -> Status) -> Status {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(Status::InternalError)
}

fn ffi_guard_qsfi(f: impl FnOnce() -> Status) -> qsfi::StatusRaw {
    match ffi_guard(f) {
        Status::Ok => qsfi_sys::QSFI_STATUS_OK,
        Status::InvalidArgument => qsfi_sys::QSFI_STATUS_INVALID_ARGUMENT,
        Status::Unsupported => qsfi_sys::QSFI_STATUS_UNSUPPORTED,
        Status::OutOfMemory => qsfi_sys::QSFI_STATUS_OUT_OF_MEMORY,
        Status::CudaError => qsfi_sys::QSFI_STATUS_CUDA_ERROR,
        Status::BackendError => qsfi_sys::QSFI_STATUS_BACKEND_ERROR,
        Status::InternalError => qsfi_sys::QSFI_STATUS_INTERNAL_ERROR,
    }
}

fn status_from_result(result: Result<(), Status>) -> Status {
    match result {
        Ok(()) => Status::Ok,
        Err(status) => status,
    }
}

fn result_from_status(status: Status) -> Result<(), Status> {
    match status {
        Status::Ok => Ok(()),
        status => Err(status),
    }
}

fn status_from_cuda(err: i32) -> Status {
    if err == CUDA_SUCCESS {
        Status::Ok
    } else if err == CUDA_ERROR_MEMORY_ALLOCATION {
        Status::OutOfMemory
    } else {
        Status::CudaError
    }
}

fn activate_device(device_ordinal: i32) -> Status {
    if device_ordinal < 0 {
        return Status::Ok;
    }
    status_from_cuda(unsafe { cudaSetDevice(device_ordinal) })
}

fn pointer_host_readable<T>(ptr: *const T) -> bool {
    if ptr.is_null() {
        return false;
    }
    let mut attributes = CudaPointerAttributes {
        memory_type: 0,
        device: 0,
        device_pointer: ptr::null_mut(),
        host_pointer: ptr::null_mut(),
    };
    let err = unsafe { cudaPointerGetAttributes(&mut attributes, ptr.cast()) };
    if err != CUDA_SUCCESS {
        unsafe {
            cudaGetLastError();
        }
        return true;
    }
    attributes.memory_type != CUDA_MEMORY_TYPE_DEVICE
}

fn dtype_size(dtype: DType) -> Result<usize, Status> {
    match dtype {
        DType::F16 | DType::BF16 => Ok(2),
        _ => Err(Status::Unsupported),
    }
}

fn dtype_from_raw(raw: qsfi::DTypeRaw) -> Result<DType, Status> {
    match raw {
        qsfi::DTYPE_F16 => Ok(DType::F16),
        qsfi::DTYPE_BF16 => Ok(DType::BF16),
        _ => Err(Status::Unsupported),
    }
}

fn dtype_to_raw(dtype: DType) -> qsfi::DTypeRaw {
    match dtype {
        DType::F16 => qsfi::DTYPE_F16,
        DType::BF16 => qsfi::DTYPE_BF16,
        DType::FP8E4M3 => qsfi::DTYPE_FP8_E4M3,
        DType::FP8E5M2 => qsfi::DTYPE_FP8_E5M2,
        DType::NVFP4E2M1 => qsfi::DTYPE_NVFP4_E2M1,
    }
}

fn layout_from_raw(raw: qsfi::KvLayoutRaw) -> Result<KvLayout, Status> {
    match raw {
        qsfi::KV_LAYOUT_NHD => Ok(KvLayout::NHD),
        qsfi::KV_LAYOUT_HND => Ok(KvLayout::HND),
        _ => Err(Status::InvalidArgument),
    }
}

fn layout_to_raw(layout: KvLayout) -> qsfi::KvLayoutRaw {
    match layout {
        KvLayout::NHD => qsfi::KV_LAYOUT_NHD,
        KvLayout::HND => qsfi::KV_LAYOUT_HND,
    }
}

fn make_attention(config: &QsSessionConfig, mask_mode: qsfi::MaskModeRaw) -> QsfiAttentionDesc {
    QsfiAttentionDesc {
        num_qo_heads: config.num_q_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim_qk: config.head_dim,
        head_dim_vo: config.head_dim,
        page_size: config.page_size,
        q_dtype: config.activation_dtype,
        kv_dtype: config.kv_dtype,
        o_dtype: config.activation_dtype,
        kv_layout: config.kv_layout,
        pos_encoding: qsfi::POS_ENCODING_ROPE_LLAMA,
        mask_mode,
        window_left: -1,
        fixed_split_size: 0,
        sm_scale: 0.0,
        logits_soft_cap: config.logits_soft_cap,
        rope_scale: if config.rope_scale == 0.0 {
            1.0
        } else {
            config.rope_scale
        },
        rope_theta: if config.rope_theta == 0.0 {
            10000.0
        } else {
            config.rope_theta
        },
        disable_split_kv: 0,
        use_fp16_qk_reduction: 0,
    }
}

fn zero_tensor() -> QsfiTensorDesc {
    QsfiTensorDesc {
        data: ptr::null_mut(),
        dtype: qsfi::DTYPE_F16,
        ndim: 0,
        shape: [0i64; qsfi::MAX_TENSOR_DIMS],
        stride: [0i64; qsfi::MAX_TENSOR_DIMS],
    }
}

fn checked_i64_product(values: &[u32]) -> Result<i64, Status> {
    let mut product = 1u128;
    for value in values {
        product = product
            .checked_mul(*value as u128)
            .ok_or(Status::InvalidArgument)?;
    }
    if product > i64::MAX as u128 {
        return Err(Status::InvalidArgument);
    }
    Ok(product as i64)
}

fn ptr_or_null<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        ptr::null()
    } else {
        slice.as_ptr()
    }
}

unsafe fn slice_from_raw<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T], Status> {
    if len == 0 {
        return Ok(&[]);
    }
    if !pointer_host_readable(ptr) {
        return Err(Status::InvalidArgument);
    }
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

unsafe fn session_mut<'a>(session: *mut QsSession) -> Result<&'a mut QsSession, Status> {
    if session.is_null() {
        Err(Status::InvalidArgument)
    } else {
        Ok(unsafe { &mut *session })
    }
}

unsafe fn session_ref<'a>(session: *const QsSession) -> Result<&'a QsSession, Status> {
    if session.is_null() {
        Err(Status::InvalidArgument)
    } else {
        Ok(unsafe { &*session })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_create(
    config: *const QsSessionConfig,
    out: *mut *mut QsSession,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        if out.is_null() {
            return Status::InvalidArgument;
        }
        unsafe {
            *out = ptr::null_mut();
        }
        if config.is_null() {
            return Status::InvalidArgument;
        }
        match QsSession::new(unsafe { &*config }) {
            Ok(session) => {
                unsafe {
                    *out = Box::into_raw(session);
                }
                Status::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_destroy(session: *mut QsSession) {
    if session.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(session));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_reset(session: *mut QsSession) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        status_from_result(session.core.reset())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_release_requests(
    session: *mut QsSession,
    request_ids: *const RequestId,
    request_count: u32,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        let ids = match unsafe { slice_from_raw(request_ids, request_count as usize) } {
            Ok(ids) => ids,
            Err(status) => return status,
        };
        status_from_result(session.core.release_requests(ids))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_get_state(
    session: *const QsSession,
    out: *mut QsSessionState,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        if out.is_null() {
            return Status::InvalidArgument;
        }
        let session = match unsafe { session_ref(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        let state = match session.core.state() {
            Ok(state) => state,
            Err(status) => return status,
        };
        let ffi_state = QsSessionState {
            batch_kind: state.batch_kind,
            live_request_count: state.live_request_count,
            batch_size: state.batch_size,
            batch_token_count: state.batch_token_count,
            live_num_indices: state.live_num_indices,
            allocated_pages: state.allocated_pages,
            free_page_count: state.free_page_count,
            max_pages: state.max_pages,
            page_size: state.page_size,
            live_request_ids: ptr_or_null(state.live_request_ids),
            live_seq_lens: ptr_or_null(state.live_seq_lens),
            live_kv_indptr: ptr_or_null(state.live_kv_indptr),
            live_kv_indices: ptr_or_null(state.live_kv_indices),
            live_last_page_len: ptr_or_null(state.live_last_page_len),
            free_pages: ptr_or_null(state.free_pages),
            batch_request_ids: ptr_or_null(state.batch_request_ids),
            batch_tokens: ptr_or_null(state.batch_tokens),
            batch_qo_indptr: ptr_or_null(state.batch_qo_indptr),
            batch_kv_indptr: ptr_or_null(state.batch_kv_indptr),
            batch_kv_indices: ptr_or_null(state.batch_kv_indices),
            batch_last_page_len: ptr_or_null(state.batch_last_page_len),
            batch_rope_pos_offset: ptr_or_null(state.batch_rope_pos_offset),
            batch_append_batch_indices: ptr_or_null(state.batch_append_batch_indices),
            batch_append_positions: ptr_or_null(state.batch_append_positions),
            d_batch_tokens: session
                .d_batch_tokens
                .device_ptr_if(!state.batch_tokens.is_empty()),
            d_batch_qo_indptr: session
                .d_batch_qo_indptr
                .device_ptr_if(!state.batch_qo_indptr.is_empty()),
            d_batch_kv_indptr: session
                .d_batch_kv_indptr
                .device_ptr_if(!state.batch_kv_indptr.is_empty()),
            d_batch_kv_indices: session
                .d_batch_kv_indices
                .device_ptr_if(!state.batch_kv_indices.is_empty()),
            d_batch_last_page_len: session
                .d_batch_last_page_len
                .device_ptr_if(!state.batch_last_page_len.is_empty()),
            d_batch_rope_pos_offset: session
                .d_batch_rope_pos_offset
                .device_ptr_if(!state.batch_rope_pos_offset.is_empty()),
            d_batch_append_batch_indices: session
                .d_batch_append_batch_indices
                .device_ptr_if(!state.batch_append_batch_indices.is_empty()),
            d_batch_append_positions: session
                .d_batch_append_positions
                .device_ptr_if(!state.batch_append_positions.is_empty()),
        };
        unsafe {
            out.write(ffi_state);
        }
        Status::Ok
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_begin_append(
    session: *mut QsSession,
    batch: *const QsSessionAppendBatch,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if batch.is_null() {
            return Status::InvalidArgument;
        }
        let batch = unsafe { &*batch };
        let batch_size = batch.batch_size as usize;
        let token_count = batch.token_count as usize;
        let request_ids = match unsafe { slice_from_raw(batch.request_ids, batch_size) } {
            Ok(request_ids) => request_ids,
            Err(status) => return status,
        };
        let token_indptr = match batch_size
            .checked_add(1)
            .ok_or(Status::InvalidArgument)
            .and_then(|len| unsafe { slice_from_raw(batch.token_indptr, len) })
        {
            Ok(token_indptr) => token_indptr,
            Err(status) => return status,
        };
        let tokens = match unsafe { slice_from_raw(batch.tokens, token_count) } {
            Ok(tokens) => tokens,
            Err(status) => return status,
        };
        session.begin_append_batch(AppendBatch {
            request_ids,
            token_indptr,
            tokens,
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_append_layer(
    session: *mut QsSession,
    layer: *const QsSessionLayer,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if layer.is_null() {
            return Status::InvalidArgument;
        }
        let layer = unsafe { &*layer };
        unsafe { session.append_layer_desc(layer) }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_begin_decode(
    session: *mut QsSession,
    batch: *const QsSessionDecodeBatch,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if batch.is_null() {
            return Status::InvalidArgument;
        }
        let batch = unsafe { &*batch };
        let batch_size = batch.batch_size as usize;
        let request_ids = match unsafe { slice_from_raw(batch.request_ids, batch_size) } {
            Ok(request_ids) => request_ids,
            Err(status) => return status,
        };
        let tokens = match unsafe { slice_from_raw(batch.tokens, batch_size) } {
            Ok(tokens) => tokens,
            Err(status) => return status,
        };
        session.begin_decode_batch(DecodeBatch {
            request_ids,
            tokens,
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_decode_layer(
    session: *mut QsSession,
    layer: *const QsSessionLayer,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if layer.is_null() {
            return Status::InvalidArgument;
        }
        let layer = unsafe { &*layer };
        unsafe { session.decode_layer_desc(layer) }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_commit_batch(
    session: *mut QsSession,
    commit: *const QsSessionCommit,
) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if session.core.batch_kind() == BatchKind::None {
            return Status::InvalidArgument;
        }
        let accepted = if commit.is_null() {
            None
        } else {
            let commit = unsafe { &*commit };
            if commit.accepted_token_counts.is_null() {
                None
            } else {
                let len = session.core.batch_size() as usize;
                match unsafe { slice_from_raw(commit.accepted_token_counts, len) } {
                    Ok(counts) => Some(counts),
                    Err(status) => return status,
                }
            }
        };
        status_from_result(session.core.commit_batch(accepted))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qs_session_abort_batch(session: *mut QsSession) -> qsfi::StatusRaw {
    ffi_guard_qsfi(|| {
        let session = match unsafe { session_mut(session) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        status_from_result(session.core.abort_batch())
    })
}
