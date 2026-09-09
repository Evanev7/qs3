use crate::backend::{
    Operators,
    qscb::Qscb,
    qsfi::{Plan, Qsfi},
};
use crate::engine::{
    AppendBatch, AttentionLayer, Commit, DecodeBatch, DynDType, EngineConfig, EngineCore, KvLayout,
    Status, validate_supported_attention_grouping, validate_supported_attention_head_dim,
};
use crate::ext::{SafeVec, try_clone_slice};
use crate::ffi::{
    AppendDecode, AppendPrefill, AttentionDesc, BatchDecodeExecuteDesc, BatchPrefillExecuteDesc,
    DTYPE_F16, MASK_MODE_CAUSAL, MASK_MODE_NONE, MaskModeRaw, PagedKvCache, PagedKvPlan,
    PagedKvTable, QoPlan, Tensor3, Tensor4, cuda,
};
use std::ffi::c_void;
use std::mem;
use std::ptr;

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

    fn ensure(&mut self, device_ordinal: i32, count: usize) -> Result<(), Status> {
        if count == 0 || self.cap >= count {
            return Ok(());
        }
        let bytes = count
            .checked_mul(mem::size_of::<i32>())
            .ok_or(Status::InvalidArgument)?;
        activate_device(device_ordinal)?;
        let mut next = ptr::null_mut();
        let err = unsafe { cuda::cudaMalloc(&mut next, bytes) };
        result_from_cuda(err)?;
        if !self.data.is_null() {
            unsafe {
                cuda::cudaFree(self.data.cast());
            }
        }
        self.data = next.cast();
        self.cap = count;
        Ok(())
    }

    fn upload(
        &mut self,
        device_ordinal: i32,
        stream: *mut c_void,
        values: &[i32],
    ) -> Result<(), Status> {
        if values.is_empty() {
            return Ok(());
        }
        self.ensure(device_ordinal, values.len())?;
        let bytes = values
            .len()
            .checked_mul(mem::size_of::<i32>())
            .ok_or(Status::InvalidArgument)?;
        let err = unsafe {
            cuda::cudaMemcpyAsync(
                self.data.cast(),
                values.as_ptr().cast(),
                bytes,
                cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        result_from_cuda(err)
    }

    fn free(&mut self) {
        if !self.data.is_null() {
            unsafe {
                cuda::cudaFree(self.data.cast());
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
    plan: Option<Plan>,
    batch_size: u32,
    num_indices: u32,
    total_tokens: u32,
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    kv_indices: Vec<i32>,
    last_page_len: Vec<i32>,
    valid: bool,
}

#[derive(Clone, Copy)]
struct PlanKey<'a> {
    batch_size: u32,
    num_indices: u32,
    total_tokens: u32,
    qo_indptr: &'a [i32],
    kv_indptr: &'a [i32],
    kv_indices: &'a [i32],
    last_page_len: &'a [i32],
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
            kv_indices: Vec::new(),
            last_page_len: Vec::new(),
            valid: false,
        }
    }

    fn matches(&self, key: PlanKey<'_>) -> bool {
        self.valid && self.plan.is_some() && self.key_matches(key)
    }

    fn key_matches(&self, key: PlanKey<'_>) -> bool {
        self.batch_size == key.batch_size
            && self.num_indices == key.num_indices
            && self.total_tokens == key.total_tokens
            && self.qo_indptr == key.qo_indptr
            && self.kv_indptr == key.kv_indptr
            && self.kv_indices == key.kv_indices
            && self.last_page_len == key.last_page_len
    }

    fn destroy(&mut self) {
        self.plan = None;
        self.qo_indptr.clear();
        self.kv_indptr.clear();
        self.kv_indices.clear();
        self.last_page_len.clear();
        self.valid = false;
    }
}

// Request state is replaced transactionally; provider contexts, plans and
// device metadata remain owned by the execution session across replacements.
pub(crate) struct PrefixState {
    pub(crate) core: EngineCore,
    layer_caches: Vec<LayerCache>,
}

impl PrefixState {
    pub(super) fn new(core: EngineCore) -> Result<Self, Status> {
        let mut state = Self {
            core,
            layer_caches: Vec::new(),
        };
        state.allocate_layer_caches()?;
        Ok(state)
    }

    fn allocate_layer_caches(&mut self) -> Result<(), Status> {
        let config = self.core.config();
        let elems = (config.max_pages as usize)
            .checked_mul(config.page_size as usize)
            .and_then(|v| v.checked_mul(config.num_kv_heads as usize))
            .and_then(|v| v.checked_mul(config.head_dim as usize))
            .ok_or(Status::InvalidArgument)?;
        let bytes = config.kv_dtype.storage_bytes_for(elems)?;
        self.layer_caches.safe_reserve(config.num_layers as usize)?;
        activate_device(config.device_ordinal)?;
        for _ in 0..config.num_layers {
            let mut k = ptr::null_mut();
            let mut v = ptr::null_mut();
            let err = unsafe { cuda::cudaMalloc(&mut k, bytes) };
            result_from_cuda(err)?;
            let err = unsafe { cuda::cudaMalloc(&mut v, bytes) };
            if err != cuda::CUDA_SUCCESS {
                unsafe {
                    cuda::cudaFree(k);
                }
                return result_from_cuda(err);
            }
            self.layer_caches.push(LayerCache { k, v });
        }
        Ok(())
    }
}

impl Drop for PrefixState {
    fn drop(&mut self) {
        let _ = activate_device(self.core.config().device_ordinal);
        for layer in &mut self.layer_caches {
            if !layer.k.is_null() {
                unsafe {
                    cuda::cudaFree(layer.k);
                }
                layer.k = ptr::null_mut();
            }
            if !layer.v.is_null() {
                unsafe {
                    cuda::cudaFree(layer.v);
                }
                layer.v = ptr::null_mut();
            }
        }
    }
}

pub(crate) struct AttentionSession {
    pub(crate) prefix: PrefixState,
    stream: *mut c_void,
    qsfi: Qsfi,
    #[allow(dead_code)]
    qscb: Qscb,
    append_attention: AttentionDesc,
    decode_attention: AttentionDesc,
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

impl AttentionSession {
    pub(crate) fn new(config: EngineConfig) -> Result<Box<Self>, Status> {
        validate_runtime_config(&config)?;
        let core = EngineCore::new(config)?;
        let mut qsfi = Qsfi::new(config.device_ordinal, config.stream)?;
        qsfi.reserve_workspace(
            config.qsfi_float_workspace_bytes,
            config.qsfi_int_workspace_bytes,
            config.qsfi_host_int_workspace_bytes,
        )?;
        let qscb = Qscb::new(config.device_ordinal, config.stream)?;
        let session = Box::new(Self {
            append_attention: make_attention(&config, MASK_MODE_CAUSAL),
            decode_attention: make_attention(&config, MASK_MODE_NONE),
            prefix: PrefixState::new(core)?,
            stream: config.stream,
            qsfi,
            qscb,
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
        Ok(session)
    }

    #[allow(dead_code)]
    pub(crate) fn operators(&mut self) -> Operators<'_> {
        Operators::new(&self.stream, &mut self.qsfi, &mut self.qscb)
    }

    fn upload_active_batch(&mut self) -> Result<(), Status> {
        let config = self.prefix.core.config();
        let batch = self.prefix.core.active_batch()?;
        if batch.request_ids.len() != batch.size as usize {
            return Err(Status::InternalError);
        }
        // The copied metadata lives in AttentionSession-owned device buffers. All
        // launches that consume these pointers are enqueued on self.stream, and
        // the buffers are overwritten only by a later prepare on the same
        // stream, so stream order preserves their lifetime without events.
        self.d_batch_tokens
            .upload(config.device_ordinal, self.stream, batch.tokens)?;
        self.d_batch_qo_indptr
            .upload(config.device_ordinal, self.stream, batch.qo_indptr)?;
        self.d_batch_kv_indptr
            .upload(config.device_ordinal, self.stream, batch.kv_indptr)?;
        self.d_batch_kv_indices
            .upload(config.device_ordinal, self.stream, batch.kv_indices)?;
        self.d_batch_last_page_len.upload(
            config.device_ordinal,
            self.stream,
            batch.last_page_len,
        )?;
        self.d_batch_rope_pos_offset.upload(
            config.device_ordinal,
            self.stream,
            batch.rope_pos_offset,
        )?;
        self.d_batch_append_batch_indices.upload(
            config.device_ordinal,
            self.stream,
            batch.append_batch_indices,
        )?;
        self.d_batch_append_positions.upload(
            config.device_ordinal,
            self.stream,
            batch.append_positions,
        )
    }

    fn ensure_append_plan(&mut self) -> Result<(), Status> {
        let batch = self.prefix.core.active_batch()?;
        if batch.kind != crate::BatchKind::Append {
            return Err(Status::InternalError);
        }
        let num_indices =
            u32::try_from(batch.kv_indices.len()).map_err(|_| Status::InvalidArgument)?;
        let key = PlanKey {
            batch_size: batch.size,
            num_indices,
            total_tokens: batch.token_count,
            qo_indptr: batch.qo_indptr,
            kv_indptr: batch.kv_indptr,
            kv_indices: batch.kv_indices,
            last_page_len: batch.last_page_len,
        };
        if self.append_plan.matches(key) {
            return Ok(());
        }

        // Reserve candidate key storage before touching a reusable plan.
        let qo_indptr = try_clone_slice(key.qo_indptr)?;
        let kv_indptr = try_clone_slice(key.kv_indptr)?;
        let kv_indices = try_clone_slice(key.kv_indices)?;
        let last_page_len = try_clone_slice(key.last_page_len)?;
        self.append_plan.valid = false;
        let qo = QoPlan {
            indptr: ptr_or_null(batch.qo_indptr),
            batch_size: batch.size,
            total_tokens: batch.token_count,
        };
        let page_table = PagedKvPlan {
            indptr: ptr_or_null(batch.kv_indptr),
            indices: ptr_or_null(batch.kv_indices),
            last_page_len: ptr_or_null(batch.last_page_len),
            batch_size: batch.size,
            num_indices,
        };
        unsafe {
            self.qsfi.prepare_prefill_plan(
                &mut self.append_plan.plan,
                &self.append_attention,
                &qo,
                &page_table,
            )
        }?;
        self.append_plan.batch_size = batch.size;
        self.append_plan.num_indices = num_indices;
        self.append_plan.total_tokens = batch.token_count;
        self.append_plan.qo_indptr = qo_indptr;
        self.append_plan.kv_indptr = kv_indptr;
        self.append_plan.kv_indices = kv_indices;
        self.append_plan.last_page_len = last_page_len;
        self.append_plan.valid = true;
        Ok(())
    }

    fn ensure_decode_plan(&mut self) -> Result<(), Status> {
        let batch = self.prefix.core.active_batch()?;
        if batch.kind != crate::BatchKind::Decode {
            return Err(Status::InternalError);
        }
        let num_indices =
            u32::try_from(batch.kv_indices.len()).map_err(|_| Status::InvalidArgument)?;
        let key = PlanKey {
            batch_size: batch.size,
            num_indices,
            total_tokens: batch.size,
            qo_indptr: &[],
            kv_indptr: batch.kv_indptr,
            kv_indices: batch.kv_indices,
            last_page_len: batch.last_page_len,
        };
        if self.decode_plan.matches(key) {
            return Ok(());
        }

        // Reserve candidate key storage before touching a reusable plan.
        let qo_indptr = try_clone_slice(key.qo_indptr)?;
        let kv_indptr = try_clone_slice(key.kv_indptr)?;
        let kv_indices = try_clone_slice(key.kv_indices)?;
        let last_page_len = try_clone_slice(key.last_page_len)?;
        self.decode_plan.valid = false;
        let page_table = PagedKvPlan {
            indptr: ptr_or_null(batch.kv_indptr),
            indices: ptr_or_null(batch.kv_indices),
            last_page_len: ptr_or_null(batch.last_page_len),
            batch_size: batch.size,
            num_indices,
        };
        unsafe {
            self.qsfi.prepare_decode_plan(
                &mut self.decode_plan.plan,
                &self.decode_attention,
                &page_table,
            )
        }?;
        self.decode_plan.batch_size = batch.size;
        self.decode_plan.num_indices = num_indices;
        self.decode_plan.total_tokens = batch.size;
        self.decode_plan.qo_indptr = qo_indptr;
        self.decode_plan.kv_indptr = kv_indptr;
        self.decode_plan.kv_indices = kv_indices;
        self.decode_plan.last_page_len = last_page_len;
        self.decode_plan.valid = true;
        Ok(())
    }

    fn make_kv_cache(&self, layer_idx: u32) -> Result<PagedKvCache, Status> {
        let idx = usize::try_from(layer_idx).map_err(|_| Status::InvalidArgument)?;
        let layer = self
            .prefix
            .layer_caches
            .get(idx)
            .ok_or(Status::InvalidArgument)?;
        Ok(PagedKvCache {
            k: self.make_cache_tensor(layer.k)?,
            v: self.make_cache_tensor(layer.v)?,
            k_scale: zero_tensor4(),
            v_scale: zero_tensor4(),
        })
    }

    fn make_cache_tensor(&self, data: *mut c_void) -> Result<Tensor4, Status> {
        let config = self.prefix.core.config();
        let mut shape = [0i64; 4];
        let mut stride = [0i64; 4];
        shape[0] = config.max_pages as i64;
        shape[3] = config.head_dim as i64;
        stride[3] = 1;
        if config.kv_layout == KvLayout::NHD {
            shape[1] = config.page_size as i64;
            shape[2] = config.num_kv_heads as i64;
            stride[0] =
                checked_i64_product(&[config.page_size, config.num_kv_heads, config.head_dim])?;
            stride[1] = checked_i64_product(&[config.num_kv_heads, config.head_dim])?;
            stride[2] = config.head_dim as i64;
        } else {
            shape[1] = config.num_kv_heads as i64;
            shape[2] = config.page_size as i64;
            stride[0] =
                checked_i64_product(&[config.num_kv_heads, config.page_size, config.head_dim])?;
            stride[1] = checked_i64_product(&[config.page_size, config.head_dim])?;
            stride[2] = config.head_dim as i64;
        }
        Ok(Tensor4 {
            data,
            dtype: config.kv_dtype.to_raw(),
            shape,
            stride,
        })
    }

    fn make_active_page_table(&self) -> Result<PagedKvTable, Status> {
        let batch = self.prefix.core.active_batch()?;
        Ok(PagedKvTable {
            indptr: self
                .d_batch_kv_indptr
                .device_ptr_if(!batch.kv_indptr.is_empty()),
            indices: self
                .d_batch_kv_indices
                .device_ptr_if(!batch.kv_indices.is_empty()),
            last_page_len: self
                .d_batch_last_page_len
                .device_ptr_if(!batch.last_page_len.is_empty()),
            rope_pos_offset: self
                .d_batch_rope_pos_offset
                .device_ptr_if(!batch.rope_pos_offset.is_empty()),
            batch_size: batch.size,
            num_indices: u32::try_from(batch.kv_indices.len())
                .map_err(|_| Status::InvalidArgument)?,
        })
    }

    pub(crate) fn prepare_append(&mut self, batch: AppendBatch<'_>) -> Result<(), Status> {
        self.prefix
            .core
            .begin_append(batch.request_ids, batch.token_indptr, batch.tokens)?;
        if let Err(status) = self
            .upload_active_batch()
            .and_then(|_| self.ensure_append_plan())
        {
            let _ = self.prefix.core.abort_batch();
            return Err(status);
        }
        Ok(())
    }

    pub(crate) unsafe fn execute_append_attention(
        &mut self,
        layer: &AttentionLayer,
    ) -> Result<(), Status> {
        let pending_layer = self.prefix.core.pending_append_layer(layer.layer_idx)?;
        self.validate_append_layer(layer)?;
        let Some(plan) = self.append_plan.plan.as_ref() else {
            return Err(Status::InvalidArgument);
        };
        let kv_cache = self.make_kv_cache(pending_layer.layer_idx())?;
        let page_table = self.make_active_page_table()?;
        let batch = self.prefix.core.active_batch()?;
        let append = AppendPrefill {
            k: layer.k,
            v: layer.v,
            batch_indices: self
                .d_batch_append_batch_indices
                .device_ptr_if(!batch.append_batch_indices.is_empty()),
            positions: self
                .d_batch_append_positions
                .device_ptr_if(!batch.append_positions.is_empty()),
            kv_cache,
            page_table,
            num_tokens: batch.token_count,
        };
        // Deterministic descriptor failures are checked above, before K/V cache
        // mutation. Once this append launch succeeds, backend failures are not
        // rolled back: the active batch remains uncommitted, and callers must
        // abort it or overwrite/rebuild the same request prefix before reuse.
        unsafe {
            self.qsfi
                .append_paged_kv_prefill(&self.append_attention, &append)?;
        }
        let execute = BatchPrefillExecuteDesc {
            q: layer.q,
            q_rope_offset: layer.q_rope_offset,
            o: layer.o,
            lse: layer.lse,
            qo_indptr: self
                .d_batch_qo_indptr
                .device_ptr_if(!batch.qo_indptr.is_empty()),
            kv_cache,
            page_table,
            q_scale: layer.q_scale,
            k_scale: layer.k_scale,
            v_scale: layer.v_scale,
        };
        unsafe { self.qsfi.execute_prefill(plan, &execute) }?;
        self.prefix.core.complete_append_layer(pending_layer)
    }

    pub(crate) fn prepare_decode(&mut self, batch: DecodeBatch<'_>) -> Result<(), Status> {
        self.prefix.core.begin_decode(batch.request_ids, batch.tokens)?;
        if let Err(status) = self
            .upload_active_batch()
            .and_then(|_| self.ensure_decode_plan())
        {
            let _ = self.prefix.core.abort_batch();
            return Err(status);
        }
        Ok(())
    }

    pub(crate) unsafe fn execute_decode_attention(
        &mut self,
        layer: &AttentionLayer,
    ) -> Result<(), Status> {
        let pending_layer = self.prefix.core.pending_decode_layer(layer.layer_idx)?;
        self.validate_decode_layer(layer)?;
        let Some(plan) = self.decode_plan.plan.as_ref() else {
            return Err(Status::InvalidArgument);
        };
        let kv_cache = self.make_kv_cache(pending_layer.layer_idx())?;
        let page_table = self.make_active_page_table()?;
        let append = AppendDecode {
            k: layer.k,
            v: layer.v,
            kv_cache,
            page_table,
        };
        // See execute_append_attention: validation failures happen before this
        // launch; backend failures after it are non-rollbackable without
        // synchronizing and replaying cache contents.
        unsafe {
            self.qsfi
                .append_paged_kv_decode(&self.decode_attention, &append)?
        }
        let execute = BatchDecodeExecuteDesc {
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
        unsafe { self.qsfi.execute_decode(plan, &execute) }?;
        self.prefix.core.complete_decode_layer(pending_layer)
    }

    fn validate_append_layer(&self, layer: &AttentionLayer) -> Result<(), Status> {
        let config = self.prefix.core.config();
        let tokens = i64::from(self.prefix.core.active_batch()?.token_count);
        self.validate_attention_layer_common(layer, tokens)?;
        validate_tensor3_shape(
            &layer.k,
            config.kv_dtype.to_raw(),
            tokens,
            i64::from(config.num_kv_heads),
            i64::from(config.head_dim),
        )?;
        validate_tensor3_shape(
            &layer.v,
            config.kv_dtype.to_raw(),
            tokens,
            i64::from(config.num_kv_heads),
            i64::from(config.head_dim),
        )
    }

    fn validate_decode_layer(&self, layer: &AttentionLayer) -> Result<(), Status> {
        let config = self.prefix.core.config();
        let tokens = i64::from(self.prefix.core.active_batch()?.size);
        self.validate_attention_layer_common(layer, tokens)?;
        validate_tensor3_shape(
            &layer.k,
            config.kv_dtype.to_raw(),
            tokens,
            i64::from(config.num_kv_heads),
            i64::from(config.head_dim),
        )?;
        validate_tensor3_shape(
            &layer.v,
            config.kv_dtype.to_raw(),
            tokens,
            i64::from(config.num_kv_heads),
            i64::from(config.head_dim),
        )
    }

    fn validate_attention_layer_common(
        &self,
        layer: &AttentionLayer,
        tokens: i64,
    ) -> Result<(), Status> {
        let config = self.prefix.core.config();
        validate_tensor3_shape(
            &layer.q,
            config.activation_dtype.to_raw(),
            tokens,
            i64::from(config.num_q_heads),
            i64::from(config.head_dim),
        )?;
        validate_tensor3_shape(
            &layer.o,
            config.activation_dtype.to_raw(),
            tokens,
            i64::from(config.num_q_heads),
            i64::from(config.head_dim),
        )?;
        if layer.o.shape != layer.q.shape || default_one(layer.v_scale) != 1.0 {
            return Err(Status::Unsupported);
        }
        Ok(())
    }

    /// Commits only after a debug stream completion boundary.
    ///
    /// Release builds do not synchronize the stream here: a successful layer
    /// call means the relevant work has been enqueued, and the caller is
    /// responsible for ordering later consumers on the same stream or inserting
    /// its own event/synchronization boundary. Debug builds synchronize before
    /// committing so asynchronous launch/runtime failures are caught before core
    /// session state advances.
    pub(crate) fn commit_batch(&mut self, commit: Commit<'_>) -> Result<(), Status> {
        #[cfg(debug_assertions)]
        {
            let config = self.prefix.core.config();
            activate_device(config.device_ordinal)?;
            result_from_cuda(unsafe { cuda::cudaStreamSynchronize(self.stream) })?;
        }
        self.prefix.core.commit_batch(commit.accepted_token_counts)
    }
}

impl Drop for AttentionSession {
    fn drop(&mut self) {
        self.append_plan.destroy();
        self.decode_plan.destroy();
        let config = self.prefix.core.config();
        let _ = activate_device(config.device_ordinal);
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

fn result_from_cuda(err: i32) -> Result<(), Status> {
    if err == cuda::CUDA_SUCCESS {
        Ok(())
    } else if err == cuda::CUDA_ERROR_MEMORY_ALLOCATION {
        Err(Status::OutOfMemory)
    } else {
        Err(Status::CudaError)
    }
}

fn activate_device(device_ordinal: i32) -> Result<(), Status> {
    if device_ordinal < 0 {
        return Ok(());
    }
    result_from_cuda(unsafe { cuda::cudaSetDevice(device_ordinal) })
}

fn validate_runtime_config(config: &EngineConfig) -> Result<(), Status> {
    if !matches!(config.activation_dtype, DynDType::F16 | DynDType::BF16)
        || !matches!(config.kv_dtype, DynDType::F16 | DynDType::BF16)
    {
        return Err(Status::Unsupported);
    }
    if config.activation_dtype != config.kv_dtype {
        return Err(Status::Unsupported);
    }
    if !matches!(config.kv_layout, KvLayout::NHD | KvLayout::HND) {
        return Err(Status::InvalidArgument);
    }
    if config.num_layers == 0 {
        return Err(Status::InvalidArgument);
    }
    validate_supported_attention_grouping(config.num_q_heads, config.num_kv_heads)?;
    validate_supported_attention_head_dim(config.head_dim)?;
    Ok(())
}

fn make_attention(config: &EngineConfig, mask_mode: MaskModeRaw) -> AttentionDesc {
    AttentionDesc {
        num_qo_heads: config.num_q_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim_qk: config.head_dim,
        head_dim_vo: config.head_dim,
        page_size: config.page_size,
        q_dtype: config.activation_dtype.to_raw(),
        kv_dtype: config.kv_dtype.to_raw(),
        o_dtype: config.activation_dtype.to_raw(),
        kv_layout: config.kv_layout.to_raw(),
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

fn zero_tensor4() -> Tensor4 {
    Tensor4 {
        data: ptr::null_mut(),
        dtype: DTYPE_F16,
        shape: [0i64; 4],
        stride: [0i64; 4],
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

fn validate_tensor3_shape(
    tensor: &Tensor3,
    dtype: crate::ffi::DTypeRaw,
    dim0: i64,
    dim1: i64,
    dim2: i64,
) -> Result<(), Status> {
    if tensor.data.is_null() || tensor.dtype != dtype {
        return Err(Status::InvalidArgument);
    }
    if tensor.shape != [dim0, dim1, dim2] {
        return Err(Status::InvalidArgument);
    }
    for &stride in &tensor.stride {
        if stride <= 0 {
            return Err(Status::InvalidArgument);
        }
    }
    Ok(())
}

fn default_one(value: f32) -> f32 {
    if value == 0.0 { 1.0 } else { value }
}

fn ptr_or_null<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        ptr::null()
    } else {
        slice.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PlanCache, PlanKey, make_attention, validate_runtime_config, validate_tensor3_shape,
    };
    use crate::{
        QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_Q_HEADS,
        QWEN36_HIDDEN_SIZE,
        engine::{DynDType, EngineConfig, KvLayout, Status},
        ffi::{DTYPE_F16, MASK_MODE_CAUSAL, MASK_MODE_NONE, Tensor3},
    };

    use std::{ffi::c_void, ptr};

    fn tiny_config() -> EngineConfig {
        EngineConfig {
            device_ordinal: -1,
            stream: ptr::null_mut(),
            num_layers: 1,
            max_live_requests: 4,
            max_batch_rows: 3,
            max_batch_tokens: 8,
            max_seq_len: 8,
            max_pages: 8,
            page_size: 4,
            hidden_size: QWEN36_HIDDEN_SIZE,
            intermediate_size: 0,
            vocab_size: 0,
            num_q_heads: QWEN36_FULL_ATTN_Q_HEADS,
            num_kv_heads: QWEN36_FULL_ATTN_KV_HEADS,
            head_dim: QWEN36_FULL_ATTN_HEAD_DIM,
            activation_dtype: DynDType::F16,
            kv_dtype: DynDType::F16,
            kv_layout: KvLayout::NHD,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            logits_soft_cap: 0.0,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
        }
    }

    #[test]
    fn plan_cache_key_includes_full_page_table_metadata() {
        let mut cache = PlanCache::new();
        cache.batch_size = 2;
        cache.num_indices = 3;
        cache.total_tokens = 5;
        cache.qo_indptr = vec![0, 2, 5];
        cache.kv_indptr = vec![0, 1, 3];
        cache.kv_indices = vec![7, 8, 9];
        cache.last_page_len = vec![1, 4];
        cache.valid = true;

        let key = PlanKey {
            batch_size: 2,
            num_indices: 3,
            total_tokens: 5,
            qo_indptr: &[0, 2, 5],
            kv_indptr: &[0, 1, 3],
            kv_indices: &[7, 8, 9],
            last_page_len: &[1, 4],
        };
        assert!(!cache.matches(key));
        assert!(cache.key_matches(key));
        assert!(!cache.key_matches(PlanKey {
            kv_indices: &[7, 99, 9],
            ..key
        }));
        assert!(!cache.key_matches(PlanKey {
            last_page_len: &[1, 3],
            ..key
        }));
    }

    #[test]
    fn runtime_config_rejects_unsupported_attention_head_dim() {
        let mut config = tiny_config();
        config.head_dim = 80;
        assert_eq!(validate_runtime_config(&config), Err(Status::Unsupported));

        for head_dim in [64, 128, 512] {
            let mut config = tiny_config();
            config.head_dim = head_dim;
            assert_eq!(validate_runtime_config(&config), Err(Status::Unsupported));
        }
    }

    #[test]
    fn runtime_config_rejects_non_qwen36_attention_grouping() {
        let mut config = tiny_config();
        config.num_q_heads = config.num_kv_heads;
        assert_eq!(validate_runtime_config(&config), Err(Status::Unsupported));

        let mut config = tiny_config();
        config.num_q_heads = 8;
        config.num_kv_heads = 1;
        assert_eq!(validate_runtime_config(&config), Err(Status::Unsupported));

        let mut config = tiny_config();
        config.num_q_heads = 32;
        config.num_kv_heads = 4;
        assert_eq!(validate_runtime_config(&config), Err(Status::Unsupported));
    }

    #[test]
    fn runtime_config_accepts_supported_flashinfer_attention_shape() {
        let config = tiny_config();
        assert_eq!(validate_runtime_config(&config), Ok(()));
    }

    #[test]
    fn attention_descriptors_rely_on_pre_rotated_qk() {
        let config = tiny_config();
        assert_eq!(
            make_attention(&config, MASK_MODE_CAUSAL).mask_mode,
            MASK_MODE_CAUSAL
        );
        assert_eq!(
            make_attention(&config, MASK_MODE_NONE).mask_mode,
            MASK_MODE_NONE
        );
    }

    #[test]
    fn runtime_config_rejects_zero_attention_layers() {
        let mut config = tiny_config();
        config.num_layers = 0;
        config.num_q_heads = 0;
        config.num_kv_heads = 0;
        config.head_dim = 0;
        assert_eq!(
            validate_runtime_config(&config),
            Err(Status::InvalidArgument)
        );

        config.num_q_heads = 16;
        config.num_kv_heads = 2;
        config.head_dim = 256;
        assert_eq!(
            validate_runtime_config(&config),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn tensor3_shape_validation_catches_descriptor_errors_before_append() {
        let valid = Tensor3 {
            data: 0x10usize as *mut c_void,
            dtype: DTYPE_F16,
            shape: [2, 3, 64],
            stride: [192, 64, 1],
        };
        assert_eq!(validate_tensor3_shape(&valid, DTYPE_F16, 2, 3, 64), Ok(()));

        let mut bad_dtype = valid;
        bad_dtype.dtype = crate::ffi::DTYPE_BF16;
        assert_eq!(
            validate_tensor3_shape(&bad_dtype, DTYPE_F16, 2, 3, 64),
            Err(Status::InvalidArgument)
        );

        let mut bad_shape = valid;
        bad_shape.shape[0] = 1;
        assert_eq!(
            validate_tensor3_shape(&bad_shape, DTYPE_F16, 2, 3, 64),
            Err(Status::InvalidArgument)
        );

        let mut null_data = valid;
        null_data.data = ptr::null_mut();
        assert_eq!(
            validate_tensor3_shape(&null_data, DTYPE_F16, 2, 3, 64),
            Err(Status::InvalidArgument)
        );

        let mut zero_outer_stride = valid;
        zero_outer_stride.stride[0] = 0;
        assert_eq!(
            validate_tensor3_shape(&zero_outer_stride, DTYPE_F16, 2, 3, 64),
            Err(Status::InvalidArgument)
        );

        let mut negative_inner_stride = valid;
        negative_inner_stride.stride[2] = -1;
        assert_eq!(
            validate_tensor3_shape(&negative_inner_stride, DTYPE_F16, 2, 3, 64),
            Err(Status::InvalidArgument)
        );
    }
}
