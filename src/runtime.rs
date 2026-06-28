use crate::engine::{
    AppendBatch, BatchKind, DecodeBatch, EngineConfig, EngineCore, EngineLayer, KvLayout, Status,
    try_clone_slice,
};
use crate::ffi::qsfi::{Context, Plan};
use crate::ffi::{
    AppendDecode, AppendPrefill, AttentionDesc, BatchDecodeExecuteDesc, BatchPrefillExecuteDesc,
    DTYPE_F16, MASK_MODE_CAUSAL, MASK_MODE_NONE, MaskModeRaw, POS_ENCODING_ROPE_LLAMA,
    PagedKvCache, PagedKvPlan, PagedKvTable, QoPlan, Tensor4, cuda,
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

pub(crate) struct EngineInner {
    pub(crate) core: EngineCore,
    stream: *mut c_void,
    ctx: Context,
    append_attention: AttentionDesc,
    decode_attention: AttentionDesc,
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

impl EngineInner {
    pub(crate) fn new(config: EngineConfig) -> Result<Box<Self>, Status> {
        let core = EngineCore::new(config)?;
        let mut ctx = Context::new(config.device_ordinal, config.stream)?;
        ctx.reserve_workspace(
            config.qsfi_float_workspace_bytes,
            config.qsfi_int_workspace_bytes,
            config.qsfi_host_int_workspace_bytes,
        )?;
        let mut session = Box::new(Self {
            append_attention: make_attention(&config, MASK_MODE_CAUSAL),
            decode_attention: make_attention(&config, MASK_MODE_NONE),
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
        let bytes = config.kv_dtype.storage_bytes_for(elems)?;
        self.layer_caches
            .try_reserve(config.num_layers as usize)
            .map_err(|_| Status::OutOfMemory)?;
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

    fn upload_active_batch(&mut self) -> Result<(), Status> {
        let config = self.core.config();
        self.d_batch_tokens
            .upload(config.device_ordinal, self.stream, self.core.batch_tokens())?;
        self.d_batch_qo_indptr.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_qo_indptr(),
        )?;
        self.d_batch_kv_indptr.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_kv_indptr(),
        )?;
        self.d_batch_kv_indices.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_kv_indices(),
        )?;
        self.d_batch_last_page_len.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_last_page_len(),
        )?;
        self.d_batch_rope_pos_offset.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_rope_pos_offset(),
        )?;
        self.d_batch_append_batch_indices.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_append_batch_indices(),
        )?;
        self.d_batch_append_positions.upload(
            config.device_ordinal,
            self.stream,
            self.core.batch_append_positions(),
        )
    }

    fn ensure_append_plan(&mut self) -> Result<(), Status> {
        let num_indices = u32::try_from(self.core.batch_kv_indices().len())
            .map_err(|_| Status::InvalidArgument)?;
        if self.append_plan.matches(
            self.core.batch_size(),
            num_indices,
            self.core.batch_token_count(),
            self.core.batch_qo_indptr(),
            self.core.batch_kv_indptr(),
        ) {
            return Ok(());
        }

        self.append_plan.destroy();
        let qo = QoPlan {
            indptr: ptr_or_null(self.core.batch_qo_indptr()),
            batch_size: self.core.batch_size(),
            total_tokens: self.core.batch_token_count(),
        };
        let page_table = PagedKvPlan {
            indptr: ptr_or_null(self.core.batch_kv_indptr()),
            indices: ptr_or_null(self.core.batch_kv_indices()),
            last_page_len: ptr_or_null(self.core.batch_last_page_len()),
            batch_size: self.core.batch_size(),
            num_indices,
        };
        let plan = unsafe {
            self.ctx
                .create_prefill_plan(&self.append_attention, &qo, &page_table)
        }?;
        self.append_plan.plan = Some(plan);
        self.append_plan.batch_size = self.core.batch_size();
        self.append_plan.num_indices = num_indices;
        self.append_plan.total_tokens = self.core.batch_token_count();
        self.append_plan.qo_indptr = try_clone_slice(self.core.batch_qo_indptr())?;
        self.append_plan.kv_indptr = try_clone_slice(self.core.batch_kv_indptr())?;
        self.append_plan.valid = true;
        Ok(())
    }

    fn ensure_decode_plan(&mut self) -> Result<(), Status> {
        let num_indices = u32::try_from(self.core.batch_kv_indices().len())
            .map_err(|_| Status::InvalidArgument)?;
        if self.decode_plan.matches(
            self.core.batch_size(),
            num_indices,
            self.core.batch_size(),
            &[],
            self.core.batch_kv_indptr(),
        ) {
            return Ok(());
        }

        self.decode_plan.destroy();
        let page_table = PagedKvPlan {
            indptr: ptr_or_null(self.core.batch_kv_indptr()),
            indices: ptr_or_null(self.core.batch_kv_indices()),
            last_page_len: ptr_or_null(self.core.batch_last_page_len()),
            batch_size: self.core.batch_size(),
            num_indices,
        };
        let plan = unsafe {
            self.ctx
                .create_decode_plan(&self.decode_attention, &page_table)
        }?;
        self.decode_plan.plan = Some(plan);
        self.decode_plan.batch_size = self.core.batch_size();
        self.decode_plan.num_indices = num_indices;
        self.decode_plan.total_tokens = self.core.batch_size();
        self.decode_plan.qo_indptr.clear();
        self.decode_plan.kv_indptr = try_clone_slice(self.core.batch_kv_indptr())?;
        self.decode_plan.valid = true;
        Ok(())
    }

    fn make_kv_cache(&self, layer_idx: u32) -> Result<PagedKvCache, Status> {
        let idx = usize::try_from(layer_idx).map_err(|_| Status::InvalidArgument)?;
        let layer = self.layer_caches.get(idx).ok_or(Status::InvalidArgument)?;
        Ok(PagedKvCache {
            k: self.make_cache_tensor(layer.k)?,
            v: self.make_cache_tensor(layer.v)?,
            k_scale: zero_tensor4(),
            v_scale: zero_tensor4(),
        })
    }

    fn make_cache_tensor(&self, data: *mut c_void) -> Result<Tensor4, Status> {
        let config = self.core.config();
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

    fn make_active_page_table(&self) -> PagedKvTable {
        PagedKvTable {
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

    pub(crate) fn prepare_append(&mut self, batch: AppendBatch<'_>) -> Result<(), Status> {
        self.core
            .begin_append(batch.request_ids, batch.token_indptr, batch.tokens)?;
        if let Err(status) = self
            .upload_active_batch()
            .and_then(|_| self.ensure_append_plan())
        {
            let _ = self.core.abort_batch();
            return Err(status);
        }
        Ok(())
    }

    pub(crate) unsafe fn execute_append_layer(
        &mut self,
        layer: &EngineLayer,
    ) -> Result<(), Status> {
        if self.core.batch_kind() != BatchKind::Append || self.append_plan.plan.is_none() {
            return Err(Status::InvalidArgument);
        }
        let kv_cache = self.make_kv_cache(layer.layer_idx)?;
        let page_table = self.make_active_page_table();
        let append = AppendPrefill {
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
        unsafe {
            self.ctx
                .append_paged_kv_prefill(&self.append_attention, &append)?;
        }
        let execute = BatchPrefillExecuteDesc {
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
        unsafe {
            self.ctx
                .execute_prefill(self.append_plan.plan.as_ref().unwrap(), &execute)
        }
    }

    pub(crate) fn prepare_decode(&mut self, batch: DecodeBatch<'_>) -> Result<(), Status> {
        self.core.begin_decode(batch.request_ids, batch.tokens)?;
        if let Err(status) = self
            .upload_active_batch()
            .and_then(|_| self.ensure_decode_plan())
        {
            let _ = self.core.abort_batch();
            return Err(status);
        }
        Ok(())
    }

    pub(crate) unsafe fn execute_decode_layer(
        &mut self,
        layer: &EngineLayer,
    ) -> Result<(), Status> {
        if self.core.batch_kind() != BatchKind::Decode || self.decode_plan.plan.is_none() {
            return Err(Status::InvalidArgument);
        }
        let kv_cache = self.make_kv_cache(layer.layer_idx)?;
        let page_table = self.make_active_page_table();
        let append = AppendDecode {
            k: layer.k,
            v: layer.v,
            kv_cache,
            page_table,
        };
        unsafe {
            self.ctx
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
        unsafe {
            self.ctx
                .execute_decode(self.decode_plan.plan.as_ref().unwrap(), &execute)
        }
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.append_plan.destroy();
        self.decode_plan.destroy();
        let config = self.core.config();
        let _ = activate_device(config.device_ordinal);
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
        pos_encoding: POS_ENCODING_ROPE_LLAMA,
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

fn ptr_or_null<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        ptr::null()
    } else {
        slice.as_ptr()
    }
}
