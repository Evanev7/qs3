pub(crate) mod cuda;
pub(crate) mod qscb;
pub(crate) mod qscu;
pub(crate) mod qsfi;
mod sys {
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/ffi_bindings.rs"));
}

use crate::engine::Status;

pub type StatusRaw = sys::qsfi_status;
pub type ErrorInfo = sys::qsfi_error_info;
pub type ErrorSourceRaw = sys::qsfi_error_source;
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
pub type RmsnormDesc = sys::qsfi_rmsnorm_desc;
pub type FusedAddRmsnormDesc = sys::qsfi_fused_add_rmsnorm_desc;
pub type RopeApplyDesc = sys::qsfi_rope_apply_desc;
pub type PagedKvCache = sys::qsfi_paged_kv_cache;
pub type PagedKvPlan = sys::qsfi_paged_kv_plan;
pub type QoPlan = sys::qsfi_qo_plan;
pub type PagedKvTable = sys::qsfi_paged_kv_table;
pub type BatchDecodeExecuteDesc = sys::qsfi_batch_decode_execute_desc;
pub type BatchPrefillExecuteDesc = sys::qsfi_batch_prefill_execute_desc;
pub type AppendDecode = sys::qsfi_append_decode_desc;
pub type AppendPrefill = sys::qsfi_append_prefill_desc;
pub type MoePlanDesc = sys::qsfi_moe_plan_desc;
pub type MoeBf16ExecuteDesc = sys::qsfi_moe_bf16_execute_desc;
pub type MoeNvfp4ExecuteDesc = sys::qsfi_moe_nvfp4_execute_desc;
pub type MoeBackendRaw = sys::qsfi_moe_backend;
pub type MoeRouteModeRaw = sys::qsfi_moe_route_mode;

pub(crate) type ContextRaw = sys::qsfi_context;
pub(crate) type BatchDecodePlanRaw = sys::qsfi_batch_decode_plan;
pub(crate) type BatchPrefillPlanRaw = sys::qsfi_batch_prefill_plan;
#[allow(dead_code)]
pub(crate) type MoePlanRaw = sys::qsfi_moe_plan;

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

pub const ERROR_SOURCE_NONE: ErrorSourceRaw = sys::QSFI_ERROR_SOURCE_NONE;
pub const ERROR_SOURCE_QSFI: ErrorSourceRaw = sys::QSFI_ERROR_SOURCE_QSFI;
pub const ERROR_SOURCE_CUDA: ErrorSourceRaw = sys::QSFI_ERROR_SOURCE_CUDA;
pub const ERROR_SOURCE_FLASHINFER: ErrorSourceRaw = sys::QSFI_ERROR_SOURCE_FLASHINFER;
pub const ERROR_SOURCE_CUBLASLT: ErrorSourceRaw = sys::QSFI_ERROR_SOURCE_CUBLASLT;

pub const MOE_BACKEND_FLASHINFER_STAGED_BF16: MoeBackendRaw =
    sys::QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16;
pub const MOE_BACKEND_FLASHINFER_FUSED_BF16: MoeBackendRaw =
    sys::QSFI_MOE_BACKEND_FLASHINFER_FUSED_BF16;
pub const MOE_BACKEND_FLASHINFER_NVFP4: MoeBackendRaw = sys::QSFI_MOE_BACKEND_FLASHINFER_NVFP4;
pub const MOE_ROUTE_PRECOMPUTED_TOPK: MoeRouteModeRaw = sys::QSFI_MOE_ROUTE_PRECOMPUTED_TOPK;
pub const MOE_ROUTE_ROUTER_LOGITS: MoeRouteModeRaw = sys::QSFI_MOE_ROUTE_ROUTER_LOGITS;

pub(crate) fn result_from_raw(status: StatusRaw) -> Result<(), Status> {
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

pub(crate) unsafe fn context_create(
    device_ordinal: i32,
    stream: CudaStream,
    out: *mut *mut ContextRaw,
) -> StatusRaw {
    let desc = sys::qsfi_context_desc {
        device_ordinal,
        stream,
    };
    unsafe { sys::qsfi_context_create(&desc, out) }
}

pub(crate) unsafe fn context_reserve_workspace(
    ctx: *mut ContextRaw,
    float_workspace_bytes: usize,
    int_workspace_bytes: usize,
    host_int_workspace_bytes: usize,
) -> StatusRaw {
    unsafe {
        sys::qsfi_context_reserve_workspace(
            ctx,
            float_workspace_bytes,
            int_workspace_bytes,
            host_int_workspace_bytes,
        )
    }
}

pub(crate) unsafe fn context_destroy(ctx: *mut ContextRaw) {
    unsafe { sys::qsfi_context_destroy(ctx) };
}

pub(crate) unsafe fn batch_decode_plan_create(
    ctx: *mut ContextRaw,
    attention: *const AttentionDesc,
    page_table: *const PagedKvPlan,
    out: *mut *mut BatchDecodePlanRaw,
) -> StatusRaw {
    unsafe { sys::qsfi_batch_decode_plan_create(ctx, attention, page_table, out) }
}

pub(crate) unsafe fn batch_decode_execute(
    ctx: *mut ContextRaw,
    plan: *mut BatchDecodePlanRaw,
    desc: *const BatchDecodeExecuteDesc,
) -> StatusRaw {
    unsafe { sys::qsfi_batch_decode_execute(ctx, plan, desc) }
}

pub(crate) unsafe fn batch_decode_plan_destroy(plan: *mut BatchDecodePlanRaw) {
    unsafe { sys::qsfi_batch_decode_plan_destroy(plan) };
}

pub(crate) unsafe fn batch_prefill_plan_create(
    ctx: *mut ContextRaw,
    attention: *const AttentionDesc,
    qo: *const QoPlan,
    page_table: *const PagedKvPlan,
    out: *mut *mut BatchPrefillPlanRaw,
) -> StatusRaw {
    unsafe { sys::qsfi_batch_prefill_plan_create(ctx, attention, qo, page_table, out) }
}

pub(crate) unsafe fn batch_prefill_execute(
    ctx: *mut ContextRaw,
    plan: *mut BatchPrefillPlanRaw,
    desc: *const BatchPrefillExecuteDesc,
) -> StatusRaw {
    unsafe { sys::qsfi_batch_prefill_execute(ctx, plan, desc) }
}

pub(crate) unsafe fn batch_prefill_plan_destroy(plan: *mut BatchPrefillPlanRaw) {
    unsafe { sys::qsfi_batch_prefill_plan_destroy(plan) };
}

pub(crate) unsafe fn append_paged_kv_decode(
    ctx: *mut ContextRaw,
    attention: *const AttentionDesc,
    append: *const AppendDecode,
) -> StatusRaw {
    unsafe { sys::qsfi_append_paged_kv_decode(ctx, attention, append) }
}

pub(crate) unsafe fn append_paged_kv_prefill(
    ctx: *mut ContextRaw,
    attention: *const AttentionDesc,
    append: *const AppendPrefill,
) -> StatusRaw {
    unsafe { sys::qsfi_append_paged_kv_prefill(ctx, attention, append) }
}

#[allow(dead_code)]
pub(crate) unsafe fn rmsnorm(ctx: *mut ContextRaw, desc: *const RmsnormDesc) -> StatusRaw {
    unsafe { sys::qsfi_rmsnorm(ctx, desc) }
}

#[allow(dead_code)]
pub(crate) unsafe fn fused_add_rmsnorm(
    ctx: *mut ContextRaw,
    desc: *const FusedAddRmsnormDesc,
) -> StatusRaw {
    unsafe { sys::qsfi_fused_add_rmsnorm(ctx, desc) }
}

#[allow(dead_code)]
pub(crate) unsafe fn rope_apply(ctx: *mut ContextRaw, desc: *const RopeApplyDesc) -> StatusRaw {
    unsafe { sys::qsfi_rope_apply(ctx, desc) }
}

#[allow(dead_code)]
pub(crate) unsafe fn moe_plan_create(
    ctx: *mut ContextRaw,
    desc: *const MoePlanDesc,
    out: *mut *mut MoePlanRaw,
) -> StatusRaw {
    unsafe { sys::qsfi_moe_plan_create(ctx, desc, out) }
}

#[allow(dead_code)]
pub(crate) unsafe fn moe_plan_destroy(plan: *mut MoePlanRaw) {
    unsafe { sys::qsfi_moe_plan_destroy(plan) };
}

#[allow(dead_code)]
pub(crate) unsafe fn moe_workspace_size(
    ctx: *mut ContextRaw,
    plan: *mut MoePlanRaw,
    num_tokens: u32,
    device_bytes: *mut usize,
) -> StatusRaw {
    unsafe { sys::qsfi_moe_workspace_size(ctx, plan, num_tokens, device_bytes) }
}

#[allow(dead_code)]
pub(crate) unsafe fn moe_execute_bf16(
    ctx: *mut ContextRaw,
    plan: *mut MoePlanRaw,
    desc: *const MoeBf16ExecuteDesc,
) -> StatusRaw {
    unsafe { sys::qsfi_moe_execute_bf16(ctx, plan, desc) }
}

#[allow(dead_code)]
pub(crate) unsafe fn moe_execute_nvfp4(
    ctx: *mut ContextRaw,
    plan: *mut MoePlanRaw,
    desc: *const MoeNvfp4ExecuteDesc,
) -> StatusRaw {
    unsafe { sys::qsfi_moe_execute_nvfp4(ctx, plan, desc) }
}
