pub mod sys {
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/qsfi_bindings.rs"));
}

use crate::engine::Status;

use std::ptr::{self, NonNull};

pub type StatusRaw = sys::qsfi_status;
pub type CudaStream = sys::qsfi_cuda_stream;
pub type DevicePtr = sys::qsfi_device_ptr;
pub type DTypeRaw = sys::qsfi_dtype;
pub type KvLayoutRaw = sys::qsfi_kv_layout;
pub type MaskModeRaw = sys::qsfi_mask_mode;
pub type PosEncodingRaw = sys::qsfi_pos_encoding;

pub type Tensor1 = sys::qsfi_tensor1;
pub type Tensor2 = sys::qsfi_tensor2;
pub type Tensor3 = sys::qsfi_tensor3;
pub type Tensor4 = sys::qsfi_tensor4;
pub type Tensor5 = sys::qsfi_tensor5;
pub type Tensor6 = sys::qsfi_tensor6;
pub type AttentionDesc = sys::qsfi_attention_desc;
pub type PagedKvCache = sys::qsfi_paged_kv_cache;
pub type PagedKvPlan = sys::qsfi_paged_kv_plan;
pub type QoPlan = sys::qsfi_qo_plan;
pub type PagedKvTable = sys::qsfi_paged_kv_table;
pub type BatchDecodeExecuteDesc = sys::qsfi_batch_decode_execute_desc;
pub type BatchPrefillExecuteDesc = sys::qsfi_batch_prefill_execute_desc;
pub type AppendDecode = sys::qsfi_append_decode_desc;
pub type AppendPrefill = sys::qsfi_append_prefill_desc;

pub const DTYPE_F32: DTypeRaw = sys::QSFI_DTYPE_F32;
pub const DTYPE_F16: DTypeRaw = sys::QSFI_DTYPE_F16;
pub const DTYPE_BF16: DTypeRaw = sys::QSFI_DTYPE_BF16;
pub const DTYPE_FP8_E4M3: DTypeRaw = sys::QSFI_DTYPE_FP8_E4M3;
pub const DTYPE_FP8_E5M2: DTypeRaw = sys::QSFI_DTYPE_FP8_E5M2;
pub const DTYPE_NVFP4_E2M1: DTypeRaw = sys::QSFI_DTYPE_NVFP4_E2M1;
pub const DTYPE_MXFP4_E2M1: DTypeRaw = sys::QSFI_DTYPE_MXFP4_E2M1;
pub const DTYPE_MXFP8_E4M3: DTypeRaw = sys::QSFI_DTYPE_MXFP8_E4M3;
pub const DTYPE_I32: DTypeRaw = sys::QSFI_DTYPE_I32;
pub const DTYPE_U32: DTypeRaw = sys::QSFI_DTYPE_U32;
pub const DTYPE_I8: DTypeRaw = sys::QSFI_DTYPE_I8;
pub const DTYPE_U8: DTypeRaw = sys::QSFI_DTYPE_U8;

pub const KV_LAYOUT_NHD: KvLayoutRaw = sys::QSFI_KV_LAYOUT_NHD;
pub const KV_LAYOUT_HND: KvLayoutRaw = sys::QSFI_KV_LAYOUT_HND;

pub const POS_ENCODING_ROPE_LLAMA: PosEncodingRaw = sys::QSFI_POS_ENCODING_ROPE_LLAMA;
pub const POS_ENCODING_NONE: PosEncodingRaw = sys::QSFI_POS_ENCODING_NONE;
pub const MASK_MODE_NONE: MaskModeRaw = sys::QSFI_MASK_MODE_NONE;
pub const MASK_MODE_CAUSAL: MaskModeRaw = sys::QSFI_MASK_MODE_CAUSAL;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlanKind {
    Decode,
    Prefill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlanShape {
    batch_size: u32,
    num_indices: u32,
    total_tokens: u32,
}

fn result_from_raw(status: StatusRaw) -> Result<(), Status> {
    match status {
        sys::QSFI_STATUS_OK => Ok(()),
        sys::QSFI_STATUS_INVALID_ARGUMENT => Err(Status::InvalidArgument),
        sys::QSFI_STATUS_UNSUPPORTED => Err(Status::Unsupported),
        sys::QSFI_STATUS_OUT_OF_MEMORY => Err(Status::OutOfMemory),
        sys::QSFI_STATUS_CUDA_ERROR => Err(Status::CudaError),
        sys::QSFI_STATUS_BACKEND_ERROR => Err(Status::BackendError),
        sys::QSFI_STATUS_INTERNAL_ERROR => Err(Status::InternalError),
        _ => unreachable!(),
    }
}

fn default_one(value: f32) -> f32 {
    if value == 0.0 { 1.0 } else { value }
}

fn valid_dtype(dtype: DTypeRaw) -> bool {
    matches!(
        dtype,
        DTYPE_F32
            | DTYPE_F16
            | DTYPE_BF16
            | DTYPE_FP8_E4M3
            | DTYPE_FP8_E5M2
            | DTYPE_NVFP4_E2M1
            | DTYPE_MXFP4_E2M1
            | DTYPE_MXFP8_E4M3
            | DTYPE_I32
            | DTYPE_U32
            | DTYPE_I8
            | DTYPE_U8
    )
}

fn supported_attention_dtype(dtype: DTypeRaw) -> bool {
    matches!(dtype, DTYPE_F16 | DTYPE_BF16)
}

trait TensorLike {
    fn data(&self) -> DevicePtr;
    fn dtype(&self) -> DTypeRaw;
    fn shape(&self) -> &[i64];
    fn stride(&self) -> &[i64];
}

macro_rules! impl_tensor_like {
    ($ty:ty) => {
        impl TensorLike for $ty {
            fn data(&self) -> DevicePtr {
                self.data
            }

            fn dtype(&self) -> DTypeRaw {
                self.dtype
            }

            fn shape(&self) -> &[i64] {
                &self.shape
            }

            fn stride(&self) -> &[i64] {
                &self.stride
            }
        }
    };
}

impl_tensor_like!(Tensor1);
impl_tensor_like!(Tensor2);
impl_tensor_like!(Tensor3);
impl_tensor_like!(Tensor4);
impl_tensor_like!(Tensor5);
impl_tensor_like!(Tensor6);

fn validate_attention_desc(attention: &AttentionDesc) -> Result<(), Status> {
    if attention.num_qo_heads == 0
        || attention.num_kv_heads == 0
        || attention.head_dim_qk == 0
        || attention.head_dim_vo == 0
        || attention.page_size == 0
    {
        return Err(Status::InvalidArgument);
    }
    if !attention
        .num_qo_heads
        .is_multiple_of(attention.num_kv_heads)
    {
        return Err(Status::InvalidArgument);
    }
    if attention.head_dim_qk != attention.head_dim_vo {
        return Err(Status::Unsupported);
    }
    if !matches!(attention.kv_layout, KV_LAYOUT_NHD | KV_LAYOUT_HND) {
        return Err(Status::InvalidArgument);
    }
    if !matches!(
        attention.pos_encoding,
        POS_ENCODING_NONE | POS_ENCODING_ROPE_LLAMA
    ) {
        return Err(Status::Unsupported);
    }
    if !valid_dtype(attention.q_dtype)
        || !valid_dtype(attention.kv_dtype)
        || !valid_dtype(attention.o_dtype)
    {
        return Err(Status::InvalidArgument);
    }
    if attention.q_dtype != attention.kv_dtype || attention.q_dtype != attention.o_dtype {
        return Err(Status::Unsupported);
    }
    if !supported_attention_dtype(attention.q_dtype) {
        return Err(Status::Unsupported);
    }
    if attention.use_fp16_qk_reduction != 0 {
        return Err(Status::Unsupported);
    }
    Ok(())
}

fn validate_tensor<T: TensorLike>(tensor: &T, expected_dtype: DTypeRaw) -> Result<(), Status> {
    if tensor.data().is_null() {
        return Err(Status::InvalidArgument);
    }
    if tensor.dtype() != expected_dtype {
        return Err(Status::InvalidArgument);
    }
    for (&shape, &stride) in tensor.shape().iter().zip(tensor.stride()) {
        if shape <= 0 || stride <= 0 {
            return Err(Status::InvalidArgument);
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_paged_kv_plan_slices(
    attention: &AttentionDesc,
    indptr: &[i32],
    indices: &[i32],
    last_page_len: &[i32],
) -> Result<PlanShape, Status> {
    let batch_size = last_page_len.len();
    if batch_size == 0 || indptr.len() != batch_size + 1 {
        return Err(Status::InvalidArgument);
    }
    if indptr[0] != 0 {
        return Err(Status::InvalidArgument);
    }
    for i in 0..batch_size {
        let begin = indptr[i];
        let end = indptr[i + 1];
        let pages = end.checked_sub(begin).ok_or(Status::InvalidArgument)?;
        let last_len = last_page_len[i];
        if begin < 0 || end < begin {
            return Err(Status::InvalidArgument);
        }
        if pages == 0 {
            if last_len != 0 {
                return Err(Status::InvalidArgument);
            }
        } else if last_len <= 0 || last_len > attention.page_size as i32 {
            return Err(Status::InvalidArgument);
        }
    }
    if indptr[batch_size] < 0 || indptr[batch_size] as usize != indices.len() {
        return Err(Status::InvalidArgument);
    }
    Ok(PlanShape {
        batch_size: u32::try_from(batch_size).map_err(|_| Status::InvalidArgument)?,
        num_indices: u32::try_from(indices.len()).map_err(|_| Status::InvalidArgument)?,
        total_tokens: u32::try_from(batch_size).map_err(|_| Status::InvalidArgument)?,
    })
}

#[cfg(test)]
fn validate_qo_plan_slices(indptr: &[i32], total_tokens: u32) -> Result<PlanShape, Status> {
    let total_tokens_i32 = i32::try_from(total_tokens).map_err(|_| Status::InvalidArgument)?;
    if indptr.len() < 2 || indptr[0] != 0 {
        return Err(Status::InvalidArgument);
    }
    let batch_size = indptr.len() - 1;
    for i in 0..batch_size {
        if indptr[i] < 0 || indptr[i + 1] < indptr[i] {
            return Err(Status::InvalidArgument);
        }
    }
    if indptr[batch_size] != total_tokens_i32 {
        return Err(Status::InvalidArgument);
    }
    Ok(PlanShape {
        batch_size: u32::try_from(batch_size).map_err(|_| Status::InvalidArgument)?,
        num_indices: 0,
        total_tokens,
    })
}

fn validate_paged_kv_plan_desc(page_table: &PagedKvPlan) -> Result<PlanShape, Status> {
    if page_table.batch_size == 0
        || page_table.indptr.is_null()
        || page_table.last_page_len.is_null()
    {
        return Err(Status::InvalidArgument);
    }
    if page_table.num_indices != 0 && page_table.indices.is_null() {
        return Err(Status::InvalidArgument);
    }
    Ok(PlanShape {
        batch_size: page_table.batch_size,
        num_indices: page_table.num_indices,
        total_tokens: page_table.batch_size,
    })
}

fn validate_qo_plan_desc(qo: &QoPlan) -> Result<PlanShape, Status> {
    if qo.batch_size == 0 || qo.indptr.is_null() {
        return Err(Status::InvalidArgument);
    }
    Ok(PlanShape {
        batch_size: qo.batch_size,
        num_indices: 0,
        total_tokens: qo.total_tokens,
    })
}

fn validate_kv_cache_desc(
    attention: &AttentionDesc,
    kv_cache: &PagedKvCache,
) -> Result<u32, Status> {
    validate_tensor(&kv_cache.k, attention.kv_dtype)?;
    validate_tensor(&kv_cache.v, attention.kv_dtype)?;
    for i in 0..4 {
        if kv_cache.k.shape[i] != kv_cache.v.shape[i]
            || kv_cache.k.stride[i] != kv_cache.v.stride[i]
        {
            return Err(Status::InvalidArgument);
        }
    }
    if !kv_cache.k_scale.data.is_null() || !kv_cache.v_scale.data.is_null() {
        return Err(Status::Unsupported);
    }
    if attention.kv_layout == KV_LAYOUT_NHD {
        if kv_cache.k.shape[1] != attention.page_size as i64
            || kv_cache.k.shape[2] != attention.num_kv_heads as i64
            || kv_cache.k.shape[3] != attention.head_dim_qk as i64
        {
            return Err(Status::InvalidArgument);
        }
    } else if kv_cache.k.shape[1] != attention.num_kv_heads as i64
        || kv_cache.k.shape[2] != attention.page_size as i64
        || kv_cache.k.shape[3] != attention.head_dim_qk as i64
    {
        return Err(Status::InvalidArgument);
    }
    u32::try_from(kv_cache.k.shape[0]).map_err(|_| Status::InvalidArgument)
}

fn validate_page_table_exec_desc(
    table: &PagedKvTable,
    plan_shape: PlanShape,
) -> Result<(), Status> {
    if table.indptr.is_null() || table.indices.is_null() || table.last_page_len.is_null() {
        return Err(Status::InvalidArgument);
    }
    if table.batch_size != plan_shape.batch_size || table.num_indices != plan_shape.num_indices {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_decode_execute_desc(
    attention: &AttentionDesc,
    plan_shape: PlanShape,
    desc: &BatchDecodeExecuteDesc,
) -> Result<(), Status> {
    validate_tensor(&desc.q, attention.q_dtype)?;
    validate_tensor(&desc.o, attention.o_dtype)?;
    if desc.q.shape[0] != plan_shape.batch_size as i64
        || desc.q.shape[1] != attention.num_qo_heads as i64
        || desc.q.shape[2] != attention.head_dim_qk as i64
    {
        return Err(Status::InvalidArgument);
    }
    for i in 0..3 {
        if desc.o.shape[i] != desc.q.shape[i] {
            return Err(Status::InvalidArgument);
        }
    }
    if default_one(desc.v_scale) != 1.0 {
        return Err(Status::Unsupported);
    }
    validate_kv_cache_desc(attention, &desc.kv_cache)?;
    validate_page_table_exec_desc(&desc.page_table, plan_shape)
}

fn validate_decode_plan_execute(
    kind: PlanKind,
    attention: &AttentionDesc,
    plan_shape: PlanShape,
    desc: &BatchDecodeExecuteDesc,
) -> Result<(), Status> {
    if kind != PlanKind::Decode {
        return Err(Status::InvalidArgument);
    }
    validate_decode_execute_desc(attention, plan_shape, desc)
}

fn validate_prefill_execute_desc(
    attention: &AttentionDesc,
    plan_shape: PlanShape,
    desc: &BatchPrefillExecuteDesc,
) -> Result<(), Status> {
    if desc.qo_indptr.is_null() {
        return Err(Status::InvalidArgument);
    }
    validate_tensor(&desc.q, attention.q_dtype)?;
    validate_tensor(&desc.o, attention.o_dtype)?;
    if desc.q.shape[0] != plan_shape.total_tokens as i64
        || desc.q.shape[1] != attention.num_qo_heads as i64
        || desc.q.shape[2] != attention.head_dim_qk as i64
    {
        return Err(Status::InvalidArgument);
    }
    for i in 0..3 {
        if desc.o.shape[i] != desc.q.shape[i] {
            return Err(Status::InvalidArgument);
        }
    }
    if default_one(desc.v_scale) != 1.0 {
        return Err(Status::Unsupported);
    }
    validate_kv_cache_desc(attention, &desc.kv_cache)?;
    validate_page_table_exec_desc(&desc.page_table, plan_shape)
}

fn validate_prefill_plan_execute(
    kind: PlanKind,
    attention: &AttentionDesc,
    plan_shape: PlanShape,
    desc: &BatchPrefillExecuteDesc,
) -> Result<(), Status> {
    if kind != PlanKind::Prefill {
        return Err(Status::InvalidArgument);
    }
    validate_prefill_execute_desc(attention, plan_shape, desc)
}

pub struct Context {
    raw: NonNull<sys::qsfi_context>,
}

impl Context {
    pub fn new(device_ordinal: i32, stream: CudaStream) -> Result<Self, Status> {
        let desc = sys::qsfi_context_desc {
            device_ordinal,
            stream,
        };
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe { sys::qsfi_context_create(&desc, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self { raw })
    }

    pub fn reserve_workspace(
        &mut self,
        float_workspace_bytes: usize,
        int_workspace_bytes: usize,
        host_int_workspace_bytes: usize,
    ) -> Result<(), Status> {
        result_from_raw(unsafe {
            sys::qsfi_context_reserve_workspace(
                self.raw.as_ptr(),
                float_workspace_bytes,
                int_workspace_bytes,
                host_int_workspace_bytes,
            )
        })
    }

    pub unsafe fn create_decode_plan(
        &mut self,
        attention: &AttentionDesc,
        page_table: &PagedKvPlan,
    ) -> Result<Plan, Status> {
        validate_attention_desc(attention)?;
        if attention.mask_mode != MASK_MODE_NONE {
            return Err(Status::Unsupported);
        }
        let shape = validate_paged_kv_plan_desc(page_table)?;
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe {
            sys::qsfi_batch_decode_plan_create(self.raw.as_ptr(), attention, page_table, &mut raw)
        })?;
        Plan::from_decode_raw(raw, *attention, shape)
    }

    pub unsafe fn execute_decode(
        &mut self,
        plan: &Plan,
        desc: &BatchDecodeExecuteDesc,
    ) -> Result<(), Status> {
        validate_decode_plan_execute(plan.kind, &plan.attention, plan.shape, desc)?;
        let PlanRaw::Decode(raw) = plan.raw else {
            return Err(Status::InvalidArgument);
        };
        result_from_raw(unsafe {
            sys::qsfi_batch_decode_execute(self.raw.as_ptr(), raw.as_ptr(), desc)
        })
    }

    pub unsafe fn create_prefill_plan(
        &mut self,
        attention: &AttentionDesc,
        qo: &QoPlan,
        page_table: &PagedKvPlan,
    ) -> Result<Plan, Status> {
        validate_attention_desc(attention)?;
        if !matches!(attention.mask_mode, MASK_MODE_NONE | MASK_MODE_CAUSAL) {
            return Err(Status::Unsupported);
        }
        let qo_shape = validate_qo_plan_desc(qo)?;
        let page_shape = validate_paged_kv_plan_desc(page_table)?;
        if qo_shape.batch_size != page_shape.batch_size {
            return Err(Status::InvalidArgument);
        }
        let shape = PlanShape {
            total_tokens: qo_shape.total_tokens,
            ..page_shape
        };
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe {
            sys::qsfi_batch_prefill_plan_create(
                self.raw.as_ptr(),
                attention,
                qo,
                page_table,
                &mut raw,
            )
        })?;
        Plan::from_prefill_raw(raw, *attention, shape)
    }

    pub unsafe fn execute_prefill(
        &mut self,
        plan: &Plan,
        desc: &BatchPrefillExecuteDesc,
    ) -> Result<(), Status> {
        validate_prefill_plan_execute(plan.kind, &plan.attention, plan.shape, desc)?;
        let PlanRaw::Prefill(raw) = plan.raw else {
            return Err(Status::InvalidArgument);
        };
        result_from_raw(unsafe {
            sys::qsfi_batch_prefill_execute(self.raw.as_ptr(), raw.as_ptr(), desc)
        })
    }

    pub unsafe fn append_paged_kv_decode(
        &mut self,
        attention: &AttentionDesc,
        append: &AppendDecode,
    ) -> Result<(), Status> {
        validate_attention_desc(attention)?;
        result_from_raw(unsafe {
            sys::qsfi_append_paged_kv_decode(self.raw.as_ptr(), attention, append)
        })
    }

    pub unsafe fn append_paged_kv_prefill(
        &mut self,
        attention: &AttentionDesc,
        append: &AppendPrefill,
    ) -> Result<(), Status> {
        validate_attention_desc(attention)?;
        result_from_raw(unsafe {
            sys::qsfi_append_paged_kv_prefill(self.raw.as_ptr(), attention, append)
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            sys::qsfi_context_destroy(self.raw.as_ptr());
        }
    }
}

#[derive(Clone, Copy)]
enum PlanRaw {
    Decode(NonNull<sys::qsfi_batch_decode_plan>),
    Prefill(NonNull<sys::qsfi_batch_prefill_plan>),
}

pub struct Plan {
    raw: PlanRaw,
    kind: PlanKind,
    attention: AttentionDesc,
    shape: PlanShape,
}

impl Plan {
    fn from_decode_raw(
        raw: *mut sys::qsfi_batch_decode_plan,
        attention: AttentionDesc,
        shape: PlanShape,
    ) -> Result<Self, Status> {
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self {
            raw: PlanRaw::Decode(raw),
            kind: PlanKind::Decode,
            attention,
            shape,
        })
    }

    fn from_prefill_raw(
        raw: *mut sys::qsfi_batch_prefill_plan,
        attention: AttentionDesc,
        shape: PlanShape,
    ) -> Result<Self, Status> {
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self {
            raw: PlanRaw::Prefill(raw),
            kind: PlanKind::Prefill,
            attention,
            shape,
        })
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            match self.raw {
                PlanRaw::Decode(raw) => sys::qsfi_batch_decode_plan_destroy(raw.as_ptr()),
                PlanRaw::Prefill(raw) => sys::qsfi_batch_prefill_plan_destroy(raw.as_ptr()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    fn device_ptr(offset: usize) -> DevicePtr {
        (0x1000usize + offset) as *mut c_void
    }

    fn zero_tensor4() -> Tensor4 {
        Tensor4 {
            data: ptr::null_mut(),
            dtype: DTYPE_F16,
            shape: [0; 4],
            stride: [0; 4],
        }
    }

    fn tensor3(data: DevicePtr, dtype: DTypeRaw, shape: [i64; 3], stride: [i64; 3]) -> Tensor3 {
        Tensor3 {
            data,
            dtype,
            shape,
            stride,
        }
    }

    fn tensor4(data: DevicePtr, dtype: DTypeRaw, shape: [i64; 4], stride: [i64; 4]) -> Tensor4 {
        Tensor4 {
            data,
            dtype,
            shape,
            stride,
        }
    }

    fn attention(layout: KvLayoutRaw) -> AttentionDesc {
        AttentionDesc {
            num_qo_heads: 4,
            num_kv_heads: 2,
            head_dim_qk: 64,
            head_dim_vo: 64,
            page_size: 4,
            q_dtype: DTYPE_F16,
            kv_dtype: DTYPE_F16,
            o_dtype: DTYPE_F16,
            kv_layout: layout,
            pos_encoding: POS_ENCODING_NONE,
            mask_mode: MASK_MODE_NONE,
            window_left: -1,
            fixed_split_size: 0,
            sm_scale: 0.0,
            logits_soft_cap: 0.0,
            rope_scale: 1.0,
            rope_theta: 10000.0,
            disable_split_kv: 0,
            use_fp16_qk_reduction: 0,
        }
    }

    fn kv_cache(layout: KvLayoutRaw) -> PagedKvCache {
        let shape = if layout == KV_LAYOUT_NHD {
            [8, 4, 2, 64]
        } else {
            [8, 2, 4, 64]
        };
        let stride = if layout == KV_LAYOUT_NHD {
            [512, 128, 64, 1]
        } else {
            [512, 256, 64, 1]
        };
        PagedKvCache {
            k: tensor4(device_ptr(1), DTYPE_F16, shape, stride),
            v: tensor4(device_ptr(2), DTYPE_F16, shape, stride),
            k_scale: zero_tensor4(),
            v_scale: zero_tensor4(),
        }
    }

    fn exec_table(shape: PlanShape) -> PagedKvTable {
        PagedKvTable {
            indptr: device_ptr(3),
            indices: device_ptr(4),
            last_page_len: device_ptr(5),
            rope_pos_offset: ptr::null_mut(),
            batch_size: shape.batch_size,
            num_indices: shape.num_indices,
        }
    }

    #[test]
    fn tensor_validation_rejects_bad_shape_dtype_and_pointer() {
        let valid = tensor3(device_ptr(1), DTYPE_F16, [2, 4, 64], [256, 64, 1]);
        assert_eq!(validate_tensor(&valid, DTYPE_F16), Ok(()));

        let mut null_data = valid;
        null_data.data = ptr::null_mut();
        assert_eq!(
            validate_tensor(&null_data, DTYPE_F16),
            Err(Status::InvalidArgument)
        );

        let mut bad_dtype = valid;
        bad_dtype.dtype = DTYPE_BF16;
        assert_eq!(
            validate_tensor(&bad_dtype, DTYPE_F16),
            Err(Status::InvalidArgument)
        );

        let mut bad_shape = valid;
        bad_shape.shape[1] = 0;
        assert_eq!(
            validate_tensor(&bad_shape, DTYPE_F16),
            Err(Status::InvalidArgument)
        );

        let mut bad_stride = valid;
        bad_stride.stride[2] = -1;
        assert_eq!(
            validate_tensor(&bad_stride, DTYPE_F16),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn attention_desc_validation_matches_supported_surface() {
        let valid = attention(KV_LAYOUT_NHD);
        assert_eq!(validate_attention_desc(&valid), Ok(()));

        let mut zero_heads = valid;
        zero_heads.num_qo_heads = 0;
        assert_eq!(
            validate_attention_desc(&zero_heads),
            Err(Status::InvalidArgument)
        );

        let mut bad_gqa = valid;
        bad_gqa.num_qo_heads = 3;
        assert_eq!(
            validate_attention_desc(&bad_gqa),
            Err(Status::InvalidArgument)
        );

        let mut split_head_dim = valid;
        split_head_dim.head_dim_vo = 32;
        assert_eq!(
            validate_attention_desc(&split_head_dim),
            Err(Status::Unsupported)
        );

        let mut bad_layout = valid;
        bad_layout.kv_layout = 99;
        assert_eq!(
            validate_attention_desc(&bad_layout),
            Err(Status::InvalidArgument)
        );

        let mut invalid_dtype = valid;
        invalid_dtype.q_dtype = 99;
        assert_eq!(
            validate_attention_desc(&invalid_dtype),
            Err(Status::InvalidArgument)
        );

        let mut mixed_dtype = valid;
        mixed_dtype.o_dtype = DTYPE_BF16;
        assert_eq!(
            validate_attention_desc(&mixed_dtype),
            Err(Status::Unsupported)
        );

        let mut fp8 = valid;
        fp8.q_dtype = DTYPE_FP8_E4M3;
        fp8.kv_dtype = DTYPE_FP8_E4M3;
        fp8.o_dtype = DTYPE_FP8_E4M3;
        assert_eq!(validate_attention_desc(&fp8), Err(Status::Unsupported));
    }

    #[test]
    fn paged_kv_plan_validation_checks_csr_shape() {
        let attention = attention(KV_LAYOUT_NHD);
        let shape =
            validate_paged_kv_plan_slices(&attention, &[0, 0, 2], &[3, 4], &[0, 4]).unwrap();
        assert_eq!(
            shape,
            PlanShape {
                batch_size: 2,
                num_indices: 2,
                total_tokens: 2
            }
        );

        assert_eq!(
            validate_paged_kv_plan_slices(&attention, &[1, 2], &[3], &[4]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_paged_kv_plan_slices(&attention, &[0, 2, 1], &[3], &[4, 1]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_paged_kv_plan_slices(&attention, &[0, 0], &[], &[1]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_paged_kv_plan_slices(&attention, &[0, 1], &[3], &[0]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_paged_kv_plan_slices(&attention, &[0, 1], &[3], &[5]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_paged_kv_plan_slices(&attention, &[0, 2], &[3], &[1]),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn plan_desc_validation_checks_pointer_shape_without_reading_contents() {
        let indptr = device_ptr(20).cast_const().cast();
        let indices = device_ptr(21).cast_const().cast();
        let last_page_len = device_ptr(22).cast_const().cast();
        let page_table = PagedKvPlan {
            indptr,
            indices,
            last_page_len,
            batch_size: 2,
            num_indices: 3,
        };
        assert_eq!(
            validate_paged_kv_plan_desc(&page_table),
            Ok(PlanShape {
                batch_size: 2,
                num_indices: 3,
                total_tokens: 2
            })
        );

        let empty_indices = PagedKvPlan {
            indices: ptr::null(),
            num_indices: 0,
            ..page_table
        };
        assert_eq!(
            validate_paged_kv_plan_desc(&empty_indices),
            Ok(PlanShape {
                batch_size: 2,
                num_indices: 0,
                total_tokens: 2
            })
        );

        let missing_indices = PagedKvPlan {
            indices: ptr::null(),
            ..page_table
        };
        assert_eq!(
            validate_paged_kv_plan_desc(&missing_indices),
            Err(Status::InvalidArgument)
        );

        let qo = QoPlan {
            indptr,
            batch_size: 2,
            total_tokens: 5,
        };
        assert_eq!(
            validate_qo_plan_desc(&qo),
            Ok(PlanShape {
                batch_size: 2,
                num_indices: 0,
                total_tokens: 5
            })
        );

        let missing_qo = QoPlan {
            indptr: ptr::null(),
            ..qo
        };
        assert_eq!(
            validate_qo_plan_desc(&missing_qo),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn qo_plan_validation_checks_token_indptr() {
        let shape = validate_qo_plan_slices(&[0, 3, 4], 4).unwrap();
        assert_eq!(
            shape,
            PlanShape {
                batch_size: 2,
                num_indices: 0,
                total_tokens: 4
            }
        );
        assert_eq!(
            validate_qo_plan_slices(&[], 0),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_qo_plan_slices(&[1, 2], 2),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_qo_plan_slices(&[0, 3, 2], 2),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_qo_plan_slices(&[0, 3], 2),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            validate_qo_plan_slices(&[0, i32::MAX], u32::MAX),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn kv_cache_validation_checks_layout_shape_and_scale_support() {
        let nhd = attention(KV_LAYOUT_NHD);
        assert_eq!(
            validate_kv_cache_desc(&nhd, &kv_cache(KV_LAYOUT_NHD)),
            Ok(8)
        );

        let hnd = attention(KV_LAYOUT_HND);
        assert_eq!(
            validate_kv_cache_desc(&hnd, &kv_cache(KV_LAYOUT_HND)),
            Ok(8)
        );

        assert_eq!(
            validate_kv_cache_desc(&nhd, &kv_cache(KV_LAYOUT_HND)),
            Err(Status::InvalidArgument)
        );

        let mut mismatched_v = kv_cache(KV_LAYOUT_NHD);
        mismatched_v.v.shape[0] = 7;
        assert_eq!(
            validate_kv_cache_desc(&nhd, &mismatched_v),
            Err(Status::InvalidArgument)
        );

        let mut quant_scale = kv_cache(KV_LAYOUT_NHD);
        quant_scale.k_scale.data = device_ptr(6);
        assert_eq!(
            validate_kv_cache_desc(&nhd, &quant_scale),
            Err(Status::Unsupported)
        );
    }

    #[test]
    fn execute_desc_validation_checks_shapes_and_page_table() {
        let attention = attention(KV_LAYOUT_NHD);
        let shape = PlanShape {
            batch_size: 2,
            num_indices: 3,
            total_tokens: 5,
        };
        let decode_q = tensor3(device_ptr(10), DTYPE_F16, [2, 4, 64], [256, 64, 1]);
        let prefill_q = tensor3(device_ptr(11), DTYPE_F16, [5, 4, 64], [256, 64, 1]);
        let mut decode = BatchDecodeExecuteDesc {
            q: decode_q,
            q_rope_offset: ptr::null_mut(),
            o: decode_q,
            lse: ptr::null_mut(),
            kv_cache: kv_cache(KV_LAYOUT_NHD),
            page_table: exec_table(shape),
            q_scale: 0.0,
            k_scale: 0.0,
            v_scale: 0.0,
        };
        assert_eq!(
            validate_decode_execute_desc(&attention, shape, &decode),
            Ok(())
        );
        assert_eq!(
            validate_decode_plan_execute(PlanKind::Decode, &attention, shape, &decode),
            Ok(())
        );
        assert_eq!(
            validate_decode_plan_execute(PlanKind::Prefill, &attention, shape, &decode),
            Err(Status::InvalidArgument)
        );

        decode.o.shape[0] = 1;
        assert_eq!(
            validate_decode_execute_desc(&attention, shape, &decode),
            Err(Status::InvalidArgument)
        );

        decode.o.shape[0] = 2;
        decode.page_table.num_indices = 2;
        assert_eq!(
            validate_decode_execute_desc(&attention, shape, &decode),
            Err(Status::InvalidArgument)
        );

        let mut prefill = BatchPrefillExecuteDesc {
            q: prefill_q,
            q_rope_offset: ptr::null_mut(),
            o: prefill_q,
            lse: ptr::null_mut(),
            qo_indptr: device_ptr(12),
            kv_cache: kv_cache(KV_LAYOUT_NHD),
            page_table: exec_table(shape),
            q_scale: 0.0,
            k_scale: 0.0,
            v_scale: 0.0,
        };
        assert_eq!(
            validate_prefill_execute_desc(&attention, shape, &prefill),
            Ok(())
        );
        assert_eq!(
            validate_prefill_plan_execute(PlanKind::Prefill, &attention, shape, &prefill),
            Ok(())
        );
        assert_eq!(
            validate_prefill_plan_execute(PlanKind::Decode, &attention, shape, &prefill),
            Err(Status::InvalidArgument)
        );

        prefill.qo_indptr = ptr::null_mut();
        assert_eq!(
            validate_prefill_execute_desc(&attention, shape, &prefill),
            Err(Status::InvalidArgument)
        );

        prefill.qo_indptr = device_ptr(12);
        prefill.v_scale = 2.0;
        assert_eq!(
            validate_prefill_execute_desc(&attention, shape, &prefill),
            Err(Status::Unsupported)
        );
    }
}
