pub mod sys {
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/qsfi_bindings.rs"));
}

use crate::engine::Status;

use std::ptr::{self, NonNull};

pub type StatusRaw = sys::qsfi_status_t;
pub type CudaStream = sys::qsfi_cuda_stream_t;
pub type DevicePtr = sys::qsfi_device_ptr_t;
pub type DTypeRaw = sys::qsfi_dtype_t;
pub type KvLayoutRaw = sys::qsfi_kv_layout_t;
pub type MaskModeRaw = sys::qsfi_mask_mode_t;
pub type PosEncodingRaw = sys::qsfi_pos_encoding_t;

pub type TensorDesc = sys::qsfi_tensor_desc_t;
pub type AttentionDesc = sys::qsfi_attention_desc_t;
pub type PagedKvCache = sys::qsfi_paged_kv_cache_t;
pub type PagedKvPlan = sys::qsfi_paged_kv_plan_t;
pub type QoPlan = sys::qsfi_qo_plan_t;
pub type PagedKvTable = sys::qsfi_paged_kv_table_t;
pub type BatchDecodeExecuteDesc = sys::qsfi_batch_decode_execute_desc_t;
pub type BatchPrefillExecuteDesc = sys::qsfi_batch_prefill_execute_desc_t;
pub type AppendDecode = sys::qsfi_append_decode_t;
pub type AppendPrefill = sys::qsfi_append_prefill_t;

pub const MAX_TENSOR_DIMS: usize = sys::QSFI_MAX_TENSOR_DIMS as usize;

pub const DTYPE_F16: DTypeRaw = sys::QSFI_DTYPE_F16;
pub const DTYPE_BF16: DTypeRaw = sys::QSFI_DTYPE_BF16;
pub const DTYPE_FP8_E4M3: DTypeRaw = sys::QSFI_DTYPE_FP8_E4M3;
pub const DTYPE_FP8_E5M2: DTypeRaw = sys::QSFI_DTYPE_FP8_E5M2;
pub const DTYPE_NVFP4_E2M1: DTypeRaw = sys::QSFI_DTYPE_NVFP4_E2M1;

pub const KV_LAYOUT_NHD: KvLayoutRaw = sys::QSFI_KV_LAYOUT_NHD;
pub const KV_LAYOUT_HND: KvLayoutRaw = sys::QSFI_KV_LAYOUT_HND;

pub const POS_ENCODING_ROPE_LLAMA: PosEncodingRaw = sys::QSFI_POS_ENCODING_ROPE_LLAMA;
pub const MASK_MODE_NONE: MaskModeRaw = sys::QSFI_MASK_MODE_NONE;
pub const MASK_MODE_CAUSAL: MaskModeRaw = sys::QSFI_MASK_MODE_CAUSAL;

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

pub struct Context {
    raw: NonNull<sys::qsfi_context_t>,
}

impl Context {
    pub fn new(device_ordinal: i32, stream: CudaStream) -> Result<Self, Status> {
        let desc = sys::qsfi_context_desc_t {
            device_ordinal,
            stream,
        };
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe { sys::qsfi_context_create(&desc, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self { raw })
    }

    pub fn reserve_scratch(
        &mut self,
        float_workspace_bytes: usize,
        int_workspace_bytes: usize,
        host_int_workspace_bytes: usize,
    ) -> Result<(), Status> {
        result_from_raw(unsafe {
            sys::qsfi_context_reserve_scratch(
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
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe {
            sys::qsfi_batch_decode_plan_create(self.raw.as_ptr(), attention, page_table, &mut raw)
        })?;
        Plan::from_raw(raw)
    }

    pub unsafe fn execute_decode(
        &mut self,
        plan: &Plan,
        desc: &BatchDecodeExecuteDesc,
    ) -> Result<(), Status> {
        result_from_raw(unsafe {
            sys::qsfi_batch_decode_execute(self.raw.as_ptr(), plan.raw.as_ptr(), desc)
        })
    }

    pub unsafe fn create_prefill_plan(
        &mut self,
        attention: &AttentionDesc,
        qo: &QoPlan,
        page_table: &PagedKvPlan,
    ) -> Result<Plan, Status> {
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
        Plan::from_raw(raw)
    }

    pub unsafe fn execute_prefill(
        &mut self,
        plan: &Plan,
        desc: &BatchPrefillExecuteDesc,
    ) -> Result<(), Status> {
        result_from_raw(unsafe {
            sys::qsfi_batch_prefill_execute(self.raw.as_ptr(), plan.raw.as_ptr(), desc)
        })
    }

    pub unsafe fn append_paged_kv_decode(
        &mut self,
        attention: &AttentionDesc,
        append: &AppendDecode,
    ) -> Result<(), Status> {
        result_from_raw(unsafe {
            sys::qsfi_append_paged_kv_decode(self.raw.as_ptr(), attention, append)
        })
    }

    pub unsafe fn append_paged_kv_prefill(
        &mut self,
        attention: &AttentionDesc,
        append: &AppendPrefill,
    ) -> Result<(), Status> {
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

pub struct Plan {
    raw: NonNull<sys::qsfi_plan_t>,
}

impl Plan {
    fn from_raw(raw: *mut sys::qsfi_plan_t) -> Result<Self, Status> {
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self { raw })
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            sys::qsfi_plan_destroy(self.raw.as_ptr());
        }
    }
}
