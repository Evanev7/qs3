#![allow(dead_code)]

use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_Q_HEADS,
    QWEN36_FULL_ATTN_ROTARY_DIM, engine::Status, ffi,
};

use super::sys;

use std::{
    mem::MaybeUninit,
    mem::{align_of, size_of},
    ptr::{self, NonNull},
};

use crate::ffi::{
    AppendDecode, AppendPrefill, AttentionDesc, BatchDecodeExecuteDesc, BatchPrefillExecuteDesc,
    CudaStream, DTYPE_BF16, DTYPE_F16, DTYPE_F32, DTYPE_FP8_E4M3, DTYPE_FP8_E5M2, DTYPE_I8,
    DTYPE_I32, DTYPE_MXFP4_E2M1, DTYPE_MXFP8_E4M3, DTYPE_NVFP4_E2M1, DTYPE_U8, DTYPE_U32, DTypeRaw,
    DevicePtr, FusedAddRmsnormDesc, KV_LAYOUT_HND, KV_LAYOUT_NHD, MASK_MODE_CAUSAL, MASK_MODE_NONE,
    MOE_BACKEND_FLASHINFER_NVFP4, MOE_BACKEND_FLASHINFER_STAGED_BF16, MOE_ROUTE_PRECOMPUTED_TOPK,
    MoeBf16ExecuteDesc, MoeNvfp4ExecuteDesc, MoePlanDesc, PagedKvCache, PagedKvPlan, PagedKvTable,
    QoPlan, RmsnormDesc, RopeApplyDesc, StatusRaw, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5,
    Tensor6,
};

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
    ffi::result_from_raw(status)
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
    if attention.head_dim_qk != QWEN36_FULL_ATTN_HEAD_DIM
        || attention.num_qo_heads != QWEN36_FULL_ATTN_Q_HEADS
        || attention.num_kv_heads != QWEN36_FULL_ATTN_KV_HEADS
    {
        return Err(Status::Unsupported);
    }
    if !matches!(attention.kv_layout, KV_LAYOUT_NHD | KV_LAYOUT_HND) {
        return Err(Status::InvalidArgument);
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
    let mut extent = 1i64;
    for (&shape, &stride) in tensor.shape().iter().zip(tensor.stride()) {
        if shape <= 0 || stride <= 0 {
            return Err(Status::InvalidArgument);
        }
        let dim_extent = shape
            .checked_sub(1)
            .and_then(|shape| shape.checked_mul(stride))
            .ok_or(Status::InvalidArgument)?;
        extent = extent
            .checked_add(dim_extent)
            .ok_or(Status::InvalidArgument)?;
    }
    // TODO: pair tensor descriptors with allocation extents once device buffers
    // carry them. For now, reject impossible logical extents cheaply.
    usize::try_from(extent).map_err(|_| Status::InvalidArgument)?;
    Ok(())
}

fn validate_supported_float_dtype(dtype: DTypeRaw) -> Result<(), Status> {
    if !valid_dtype(dtype) {
        return Err(Status::InvalidArgument);
    }
    if !matches!(dtype, DTYPE_F32 | DTYPE_BF16) {
        return Err(Status::Unsupported);
    }
    Ok(())
}

fn validate_u32_i64(value: i64) -> Result<(), Status> {
    u32::try_from(value)
        .map(|_| ())
        .map_err(|_| Status::InvalidArgument)
}

fn validate_eps(eps: f32) -> Result<(), Status> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_rmsnorm_weight_bias(weight_bias: f32) -> Result<(), Status> {
    if !weight_bias.is_finite() {
        return Err(Status::InvalidArgument);
    }
    if weight_bias != 0.0 && weight_bias != 1.0 {
        return Err(Status::Unsupported);
    }
    Ok(())
}

fn tensor1_is_contiguous(tensor: &Tensor1) -> bool {
    tensor.stride[0] == 1
}

fn tensor2_has_row_major_inner(tensor: &Tensor2) -> bool {
    tensor.stride[1] == 1 && tensor.stride[0] >= tensor.shape[1]
}

fn tensor2_is_contiguous(tensor: &Tensor2) -> bool {
    tensor.stride[1] == 1 && tensor.stride[0] == tensor.shape[1]
}

fn tensor3_has_row_major_inner(tensor: &Tensor3) -> bool {
    let Some(min_row_stride) = tensor.shape[1].checked_mul(tensor.stride[1]) else {
        return false;
    };
    tensor.stride[2] == 1
        && tensor.stride[1] >= tensor.shape[2]
        && tensor.stride[0] >= min_row_stride
}

fn tensor3_is_contiguous(tensor: &Tensor3) -> bool {
    let Some(row_stride) = tensor.shape[1].checked_mul(tensor.shape[2]) else {
        return false;
    };
    tensor.stride[2] == 1 && tensor.stride[1] == tensor.shape[2] && tensor.stride[0] == row_stride
}

fn same_shape2(a: &Tensor2, b: &Tensor2) -> bool {
    a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]
}

fn same_shape_and_stride2(a: &Tensor2, b: &Tensor2) -> bool {
    same_shape2(a, b) && a.stride[0] == b.stride[0] && a.stride[1] == b.stride[1]
}

fn same_shape3(a: &Tensor3, b: &Tensor3) -> bool {
    a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] && a.shape[2] == b.shape[2]
}

fn same_shape_and_stride3(a: &Tensor3, b: &Tensor3) -> bool {
    same_shape3(a, b)
        && a.stride[0] == b.stride[0]
        && a.stride[1] == b.stride[1]
        && a.stride[2] == b.stride[2]
}

fn validate_rmsnorm_common(
    x: &Tensor2,
    weight: &Tensor1,
    out: &Tensor2,
    hidden_size: u32,
    eps: f32,
) -> Result<(), Status> {
    if hidden_size == 0 {
        return Err(Status::InvalidArgument);
    }
    validate_supported_float_dtype(x.dtype)?;
    validate_tensor(x, x.dtype)?;
    validate_tensor(weight, x.dtype)?;
    validate_tensor(out, x.dtype)?;
    if x.shape[1] != i64::from(hidden_size)
        || out.shape[0] != x.shape[0]
        || out.shape[1] != x.shape[1]
        || weight.shape[0] != x.shape[1]
    {
        return Err(Status::InvalidArgument);
    }
    if !tensor2_has_row_major_inner(x) || !tensor2_has_row_major_inner(out) || weight.stride[0] != 1
    {
        return Err(Status::InvalidArgument);
    }
    validate_u32_i64(x.shape[0])?;
    validate_u32_i64(x.stride[0])?;
    validate_u32_i64(out.stride[0])?;
    validate_eps(eps)
}

fn validate_rmsnorm_desc(desc: &RmsnormDesc) -> Result<(), Status> {
    validate_rmsnorm_common(&desc.x, &desc.weight, &desc.out, desc.hidden_size, desc.eps)?;
    validate_rmsnorm_weight_bias(desc.weight_bias)
}

fn validate_fused_add_rmsnorm_desc(desc: &FusedAddRmsnormDesc) -> Result<(), Status> {
    validate_rmsnorm_common(&desc.x, &desc.weight, &desc.out, desc.hidden_size, desc.eps)?;
    validate_tensor(&desc.residual_inout, desc.x.dtype)?;
    if !same_shape2(&desc.x, &desc.residual_inout)
        || !tensor2_has_row_major_inner(&desc.residual_inout)
        || desc.out.data != desc.x.data
        || !same_shape_and_stride2(&desc.out, &desc.x)
    {
        return Err(Status::InvalidArgument);
    }
    validate_u32_i64(desc.residual_inout.stride[0])
}

fn supported_rope_head_dim(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 128 | 256 | 512)
}

fn validate_rope_apply_desc(desc: &RopeApplyDesc) -> Result<(), Status> {
    if desc.num_qo_heads == 0
        || desc.num_kv_heads == 0
        || desc.head_dim == 0
        || desc.rotary_dim == 0
    {
        return Err(Status::InvalidArgument);
    }
    if desc.head_dim % 2 != 0 || desc.rotary_dim % 2 != 0 || desc.rotary_dim > desc.head_dim {
        return Err(Status::InvalidArgument);
    }
    if !supported_rope_head_dim(desc.head_dim)
        || (desc.head_dim == QWEN36_FULL_ATTN_HEAD_DIM
            && desc.rotary_dim != QWEN36_FULL_ATTN_ROTARY_DIM)
        || desc.interleave != 0
    {
        return Err(Status::Unsupported);
    }
    if !desc.rope_scale.is_finite()
        || desc.rope_scale < 0.0
        || !desc.rope_theta.is_finite()
        || desc.rope_theta < 0.0
    {
        return Err(Status::InvalidArgument);
    }

    validate_supported_float_dtype(desc.q.dtype)?;
    validate_tensor(&desc.q, desc.q.dtype)?;
    validate_tensor(&desc.k, desc.q.dtype)?;
    validate_tensor(&desc.q_out, desc.q.dtype)?;
    validate_tensor(&desc.k_out, desc.q.dtype)?;
    if !matches!(desc.positions.dtype, DTYPE_I32 | DTYPE_U32) {
        return Err(Status::InvalidArgument);
    }
    validate_tensor(&desc.positions, desc.positions.dtype)?;
    if desc.positions.stride[0] != 1 {
        return Err(Status::InvalidArgument);
    }

    let num_tokens = desc.q.shape[0];
    if desc.k.shape[0] != num_tokens
        || desc.q_out.shape[0] != num_tokens
        || desc.k_out.shape[0] != num_tokens
        || desc.positions.shape[0] != num_tokens
        || desc.q.shape[1] != i64::from(desc.num_qo_heads)
        || desc.q_out.shape[1] != desc.q.shape[1]
        || desc.k.shape[1] != i64::from(desc.num_kv_heads)
        || desc.k_out.shape[1] != desc.k.shape[1]
        || desc.q.shape[2] != i64::from(desc.head_dim)
        || desc.k.shape[2] != i64::from(desc.head_dim)
        || desc.q_out.shape[2] != i64::from(desc.head_dim)
        || desc.k_out.shape[2] != i64::from(desc.head_dim)
        || !same_shape3(&desc.q, &desc.q_out)
        || !same_shape3(&desc.k, &desc.k_out)
    {
        return Err(Status::InvalidArgument);
    }
    if (desc.q_out.data == desc.q.data && !same_shape_and_stride3(&desc.q, &desc.q_out))
        || (desc.k_out.data == desc.k.data && !same_shape_and_stride3(&desc.k, &desc.k_out))
    {
        return Err(Status::InvalidArgument);
    }
    if !tensor3_has_row_major_inner(&desc.q)
        || !tensor3_has_row_major_inner(&desc.k)
        || !tensor3_has_row_major_inner(&desc.q_out)
        || !tensor3_has_row_major_inner(&desc.k_out)
    {
        return Err(Status::InvalidArgument);
    }

    validate_u32_i64(num_tokens)?;
    validate_u32_i64(desc.q.stride[0])?;
    validate_u32_i64(desc.q.stride[1])?;
    validate_u32_i64(desc.k.stride[0])?;
    validate_u32_i64(desc.k.stride[1])?;
    validate_u32_i64(desc.q_out.stride[0])?;
    validate_u32_i64(desc.q_out.stride[1])?;
    validate_u32_i64(desc.k_out.stride[0])?;
    validate_u32_i64(desc.k_out.stride[1])
}

fn validate_moe_plan_desc(desc: &MoePlanDesc) -> Result<(), Status> {
    if desc.max_num_tokens == 0
        || desc.hidden_size == 0
        || desc.intermediate_size == 0
        || desc.num_experts == 0
        || desc.top_k == 0
        || desc.local_num_experts == 0
    {
        return Err(Status::InvalidArgument);
    }
    let local_end = desc
        .local_expert_offset
        .checked_add(desc.local_num_experts)
        .ok_or(Status::InvalidArgument)?;
    if local_end > desc.num_experts {
        return Err(Status::InvalidArgument);
    }
    if desc.route_mode != MOE_ROUTE_PRECOMPUTED_TOPK {
        return Err(Status::Unsupported);
    }

    if desc.backend == MOE_BACKEND_FLASHINFER_STAGED_BF16 {
        if desc.local_expert_offset != 0 || desc.local_num_experts != desc.num_experts {
            return Err(Status::Unsupported);
        }
        if desc.activation_dtype != DTYPE_BF16
            || desc.weight_dtype != DTYPE_BF16
            || desc.output_dtype != DTYPE_BF16
        {
            return Err(Status::InvalidArgument);
        }
        if !desc.hidden_size.is_multiple_of(8) || !desc.intermediate_size.is_multiple_of(8) {
            return Err(Status::InvalidArgument);
        }
        return Ok(());
    }

    if desc.backend == MOE_BACKEND_FLASHINFER_NVFP4 {
        if desc.activation_dtype != DTYPE_NVFP4_E2M1
            || desc.weight_dtype != DTYPE_NVFP4_E2M1
            || desc.output_dtype != DTYPE_BF16
        {
            return Err(Status::InvalidArgument);
        }
        if !desc.hidden_size.is_multiple_of(16) || !desc.intermediate_size.is_multiple_of(16) {
            return Err(Status::InvalidArgument);
        }
        return Ok(());
    }

    Err(Status::Unsupported)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, Status> {
    let mask = alignment.checked_sub(1).ok_or(Status::InvalidArgument)?;
    value
        .checked_add(mask)
        .map(|value| value / alignment * alignment)
        .ok_or(Status::InvalidArgument)
}

fn take_workspace_bytes<T>(offset: &mut usize, count: usize) -> Result<(), Status> {
    *offset = align_up(*offset, 256.max(align_of::<T>()))?;
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(Status::InvalidArgument)?;
    *offset = (*offset)
        .checked_add(bytes)
        .ok_or(Status::InvalidArgument)?;
    Ok(())
}

fn take_bf16_workspace_bytes(offset: &mut usize, count: usize) -> Result<(), Status> {
    *offset = align_up(*offset, 256)?;
    let bytes = count.checked_mul(2).ok_or(Status::InvalidArgument)?;
    *offset = (*offset)
        .checked_add(bytes)
        .ok_or(Status::InvalidArgument)?;
    Ok(())
}

fn moe_workspace_bytes(desc: &MoePlanDesc, num_tokens: u32) -> Result<usize, Status> {
    if desc.backend != MOE_BACKEND_FLASHINFER_STAGED_BF16 {
        return Err(Status::Unsupported);
    }
    if num_tokens > desc.max_num_tokens {
        return Err(Status::InvalidArgument);
    }
    let max_routes = (num_tokens as usize)
        .checked_mul(desc.top_k as usize)
        .ok_or(Status::InvalidArgument)?;
    let hidden_elems = max_routes
        .checked_mul(desc.hidden_size as usize)
        .ok_or(Status::InvalidArgument)?;
    let gate_up_width = (desc.intermediate_size as usize)
        .checked_mul(2)
        .ok_or(Status::InvalidArgument)?;
    let gate_up_elems = max_routes
        .checked_mul(gate_up_width)
        .ok_or(Status::InvalidArgument)?;
    let act_elems = max_routes
        .checked_mul(desc.intermediate_size as usize)
        .ok_or(Status::InvalidArgument)?;
    let down_elems = hidden_elems;
    let local_experts = desc.local_num_experts as usize;

    let mut offset = 0usize;
    take_workspace_bytes::<i32>(&mut offset, local_experts)?;
    take_workspace_bytes::<i32>(&mut offset, local_experts + 1)?;
    take_workspace_bytes::<i32>(&mut offset, local_experts)?;
    take_workspace_bytes::<i32>(&mut offset, max_routes)?;
    take_workspace_bytes::<i32>(&mut offset, max_routes)?;
    take_workspace_bytes::<f32>(&mut offset, max_routes)?;
    take_bf16_workspace_bytes(&mut offset, hidden_elems)?;
    take_bf16_workspace_bytes(&mut offset, gate_up_elems)?;
    take_bf16_workspace_bytes(&mut offset, act_elems)?;
    take_bf16_workspace_bytes(&mut offset, down_elems)?;
    take_workspace_bytes::<i32>(&mut offset, local_experts * 3)?;
    take_workspace_bytes::<DevicePtr>(&mut offset, local_experts)?;
    take_workspace_bytes::<DevicePtr>(&mut offset, local_experts)?;
    take_workspace_bytes::<DevicePtr>(&mut offset, local_experts)?;
    take_workspace_bytes::<i64>(&mut offset, local_experts)?;
    take_workspace_bytes::<i64>(&mut offset, local_experts)?;
    take_workspace_bytes::<i64>(&mut offset, local_experts)?;
    align_up(offset, 256)
}

fn validate_moe_workspace_tensor(
    workspace: &Tensor1,
    min_workspace_bytes: Option<usize>,
) -> Result<(), Status> {
    if !matches!(workspace.dtype, DTYPE_U8 | DTYPE_I8) {
        return Err(Status::InvalidArgument);
    }
    validate_tensor(workspace, workspace.dtype)?;
    if !tensor1_is_contiguous(workspace) {
        return Err(Status::InvalidArgument);
    }
    if let Some(min_workspace_bytes) = min_workspace_bytes {
        let workspace_bytes =
            usize::try_from(workspace.shape[0]).map_err(|_| Status::InvalidArgument)?;
        if workspace_bytes < min_workspace_bytes {
            return Err(Status::InvalidArgument);
        }
    }
    Ok(())
}

fn validate_moe_bf16_execute_desc(
    plan_desc: &MoePlanDesc,
    desc: &MoeBf16ExecuteDesc,
) -> Result<(), Status> {
    if plan_desc.backend != MOE_BACKEND_FLASHINFER_STAGED_BF16 {
        return Err(Status::InvalidArgument);
    }
    if desc.num_tokens == 0 || desc.num_tokens > plan_desc.max_num_tokens {
        return Err(Status::InvalidArgument);
    }
    validate_tensor(&desc.hidden, DTYPE_BF16)?;
    validate_tensor(&desc.topk_ids, DTYPE_I32)?;
    validate_tensor(&desc.topk_weights, DTYPE_F32)?;
    validate_tensor(&desc.gate_up_weight, DTYPE_BF16)?;
    validate_tensor(&desc.down_weight, DTYPE_BF16)?;
    validate_tensor(&desc.out, DTYPE_BF16)?;

    if !tensor2_is_contiguous(&desc.hidden)
        || !tensor2_is_contiguous(&desc.topk_ids)
        || !tensor2_is_contiguous(&desc.topk_weights)
        || !tensor3_is_contiguous(&desc.gate_up_weight)
        || !tensor3_is_contiguous(&desc.down_weight)
        || !tensor2_is_contiguous(&desc.out)
    {
        return Err(Status::InvalidArgument);
    }

    if desc.hidden.shape != [i64::from(desc.num_tokens), i64::from(plan_desc.hidden_size)]
        || desc.topk_ids.shape != [i64::from(desc.num_tokens), i64::from(plan_desc.top_k)]
        || desc.topk_weights.shape != [i64::from(desc.num_tokens), i64::from(plan_desc.top_k)]
        || desc.out.shape != [i64::from(desc.num_tokens), i64::from(plan_desc.hidden_size)]
        || desc.gate_up_weight.shape
            != [
                i64::from(plan_desc.local_num_experts),
                2 * i64::from(plan_desc.intermediate_size),
                i64::from(plan_desc.hidden_size),
            ]
        || desc.down_weight.shape
            != [
                i64::from(plan_desc.local_num_experts),
                i64::from(plan_desc.hidden_size),
                i64::from(plan_desc.intermediate_size),
            ]
    {
        return Err(Status::InvalidArgument);
    }
    let workspace_bytes = moe_workspace_bytes(plan_desc, desc.num_tokens)?;
    validate_moe_workspace_tensor(&desc.workspace, Some(workspace_bytes))
}

fn validate_optional_tensor1(
    tensor: &Tensor1,
    expected_dtype: DTypeRaw,
    expected_shape: i64,
) -> Result<(), Status> {
    if tensor.data.is_null() {
        return Ok(());
    }
    validate_tensor(tensor, expected_dtype)?;
    if tensor.shape[0] != expected_shape || !tensor1_is_contiguous(tensor) {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_moe_nvfp4_execute_desc(
    plan_desc: &MoePlanDesc,
    desc: &MoeNvfp4ExecuteDesc,
) -> Result<(), Status> {
    if plan_desc.backend != MOE_BACKEND_FLASHINFER_NVFP4 {
        return Err(Status::InvalidArgument);
    }
    if desc.num_tokens == 0 || desc.num_tokens > plan_desc.max_num_tokens {
        return Err(Status::InvalidArgument);
    }
    validate_tensor(&desc.hidden_packed, DTYPE_U8)?;
    validate_tensor(&desc.hidden_scale, DTYPE_FP8_E4M3)?;
    validate_tensor(&desc.topk_ids, DTYPE_I32)?;
    validate_tensor(&desc.topk_weights, DTYPE_F32)?;
    validate_tensor(&desc.gate_up_weight_packed, DTYPE_U8)?;
    validate_tensor(&desc.gate_up_weight_scale, DTYPE_FP8_E4M3)?;
    validate_tensor(&desc.down_weight_packed, DTYPE_U8)?;
    validate_tensor(&desc.down_weight_scale, DTYPE_FP8_E4M3)?;
    validate_optional_tensor1(
        &desc.expert_output_scale,
        DTYPE_F32,
        i64::from(plan_desc.local_num_experts),
    )?;
    validate_tensor(&desc.out, DTYPE_BF16)?;

    if !tensor2_is_contiguous(&desc.hidden_packed)
        || !tensor2_is_contiguous(&desc.hidden_scale)
        || !tensor2_is_contiguous(&desc.topk_ids)
        || !tensor2_is_contiguous(&desc.topk_weights)
        || !tensor3_is_contiguous(&desc.gate_up_weight_packed)
        || !tensor3_is_contiguous(&desc.gate_up_weight_scale)
        || !tensor3_is_contiguous(&desc.down_weight_packed)
        || !tensor3_is_contiguous(&desc.down_weight_scale)
        || !tensor2_is_contiguous(&desc.out)
    {
        return Err(Status::InvalidArgument);
    }

    let tokens = i64::from(desc.num_tokens);
    let hidden = i64::from(plan_desc.hidden_size);
    let hidden_packed = i64::from(plan_desc.hidden_size / 2);
    let hidden_scale = i64::from(plan_desc.hidden_size / 16);
    let intermediate = i64::from(plan_desc.intermediate_size);
    let intermediate_packed = i64::from(plan_desc.intermediate_size / 2);
    let intermediate_scale = i64::from(plan_desc.intermediate_size / 16);
    let gate_up = 2 * intermediate;
    let local_experts = i64::from(plan_desc.local_num_experts);
    if desc.hidden_packed.shape != [tokens, hidden_packed]
        || desc.hidden_scale.shape != [tokens, hidden_scale]
        || desc.topk_ids.shape != [tokens, i64::from(plan_desc.top_k)]
        || desc.topk_weights.shape != [tokens, i64::from(plan_desc.top_k)]
        || desc.gate_up_weight_packed.shape != [local_experts, gate_up, hidden_packed]
        || desc.gate_up_weight_scale.shape != [local_experts, gate_up, hidden_scale]
        || desc.down_weight_packed.shape != [local_experts, hidden, intermediate_packed]
        || desc.down_weight_scale.shape != [local_experts, hidden, intermediate_scale]
        || desc.out.shape != [tokens, hidden]
    {
        return Err(Status::InvalidArgument);
    }
    validate_moe_workspace_tensor(&desc.workspace, None)
}

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

pub(crate) struct PagedKvPlanHost<'a> {
    raw: PagedKvPlan,
    shape: PlanShape,
    _indptr: &'a [i32],
    _indices: &'a [i32],
    _last_page_len: &'a [i32],
}

impl<'a> PagedKvPlanHost<'a> {
    pub(crate) fn new(
        attention: &AttentionDesc,
        indptr: &'a [i32],
        indices: &'a [i32],
        last_page_len: &'a [i32],
    ) -> Result<Self, Status> {
        let shape = validate_paged_kv_plan_slices(attention, indptr, indices, last_page_len)?;
        Ok(Self {
            raw: PagedKvPlan {
                indptr: indptr.as_ptr(),
                indices: if indices.is_empty() {
                    ptr::null()
                } else {
                    indices.as_ptr()
                },
                last_page_len: last_page_len.as_ptr(),
                batch_size: shape.batch_size,
                num_indices: shape.num_indices,
            },
            shape,
            _indptr: indptr,
            _indices: indices,
            _last_page_len: last_page_len,
        })
    }

    pub(crate) fn as_raw(&self) -> &PagedKvPlan {
        &self.raw
    }

    fn shape(&self) -> PlanShape {
        self.shape
    }
}

pub(crate) struct QoPlanHost<'a> {
    raw: QoPlan,
    shape: PlanShape,
    _indptr: &'a [i32],
}

impl<'a> QoPlanHost<'a> {
    pub(crate) fn new(indptr: &'a [i32], total_tokens: u32) -> Result<Self, Status> {
        let shape = validate_qo_plan_slices(indptr, total_tokens)?;
        Ok(Self {
            raw: QoPlan {
                indptr: indptr.as_ptr(),
                batch_size: shape.batch_size,
                total_tokens: shape.total_tokens,
            },
            shape,
            _indptr: indptr,
        })
    }

    pub(crate) fn as_raw(&self) -> &QoPlan {
        &self.raw
    }

    fn shape(&self) -> PlanShape {
        self.shape
    }
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

fn validate_append_decode_desc(
    attention: &AttentionDesc,
    append: &AppendDecode,
) -> Result<(), Status> {
    validate_tensor(&append.k, attention.kv_dtype)?;
    validate_tensor(&append.v, attention.kv_dtype)?;
    let batch_size = u32::try_from(append.k.shape[0]).map_err(|_| Status::InvalidArgument)?;
    if append.k.shape[1] != i64::from(attention.num_kv_heads)
        || append.k.shape[2] != i64::from(attention.head_dim_qk)
    {
        return Err(Status::InvalidArgument);
    }
    if !same_shape_and_stride3(&append.k, &append.v) {
        return Err(Status::InvalidArgument);
    }
    if !tensor3_is_contiguous(&append.k) {
        return Err(Status::Unsupported);
    }
    validate_kv_cache_desc(attention, &append.kv_cache)?;
    validate_page_table_exec_desc(
        &append.page_table,
        PlanShape {
            batch_size,
            num_indices: append.page_table.num_indices,
            total_tokens: batch_size,
        },
    )
}

fn validate_append_prefill_desc(
    attention: &AttentionDesc,
    append: &AppendPrefill,
) -> Result<(), Status> {
    if append.num_tokens == 0 {
        return Ok(());
    }
    if append.batch_indices.is_null() || append.positions.is_null() {
        return Err(Status::InvalidArgument);
    }
    validate_tensor(&append.k, attention.kv_dtype)?;
    validate_tensor(&append.v, attention.kv_dtype)?;
    if append.k.shape[0] != i64::from(append.num_tokens)
        || append.k.shape[1] != i64::from(attention.num_kv_heads)
        || append.k.shape[2] != i64::from(attention.head_dim_qk)
        || !same_shape3(&append.k, &append.v)
    {
        return Err(Status::InvalidArgument);
    }
    if !tensor3_has_row_major_inner(&append.k) || !tensor3_has_row_major_inner(&append.v) {
        return Err(Status::InvalidArgument);
    }
    validate_kv_cache_desc(attention, &append.kv_cache)?;
    if append.page_table.batch_size == 0 {
        return Err(Status::InvalidArgument);
    }
    validate_page_table_exec_desc(
        &append.page_table,
        PlanShape {
            batch_size: append.page_table.batch_size,
            num_indices: append.page_table.num_indices,
            total_tokens: append.num_tokens,
        },
    )
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

pub(crate) struct Context {
    raw: NonNull<ffi::ContextRaw>,
}

impl Context {
    pub(crate) fn new(device_ordinal: i32, stream: CudaStream) -> Result<Self, Status> {
        let desc = sys::qsfi_context_desc {
            device_ordinal,
            stream,
        };
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe { sys::qsfi_context_create(&desc, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self { raw })
    }

    pub(crate) fn reserve_workspace(
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

    pub(crate) fn as_raw(&mut self) -> *mut ffi::ContextRaw {
        self.raw.as_ptr()
    }

    pub(crate) fn last_error(&self) -> Result<ffi::ErrorInfo, Status> {
        let mut out = MaybeUninit::uninit();
        result_from_raw(unsafe {
            sys::qsfi_context_get_last_error(self.raw.as_ptr(), out.as_mut_ptr())
        })?;
        Ok(unsafe { out.assume_init() })
    }

    pub(crate) fn clear_last_error(&mut self) {
        unsafe {
            sys::qsfi_context_clear_last_error(self.raw.as_ptr());
        }
    }

    fn result_with_last_error(&self, status: ffi::StatusRaw) -> Result<(), Status> {
        result_from_raw(status).inspect_err(|_| _ = self.last_error())
    }

    pub(crate) unsafe fn create_decode_plan(
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
        self.result_with_last_error(unsafe {
            sys::qsfi_batch_decode_plan_create(self.raw.as_ptr(), attention, page_table, &mut raw)
        })?;
        Plan::from_decode_raw(raw, *attention, shape)
    }

    pub(crate) unsafe fn execute_decode(
        &mut self,
        plan: &Plan,
        desc: &BatchDecodeExecuteDesc,
    ) -> Result<(), Status> {
        validate_decode_plan_execute(plan.kind, &plan.attention, plan.shape, desc)?;
        let PlanRaw::Decode(raw) = plan.raw else {
            return Err(Status::InvalidArgument);
        };
        self.result_with_last_error(unsafe {
            sys::qsfi_batch_decode_execute(self.raw.as_ptr(), raw.as_ptr(), desc)
        })
    }

    pub(crate) unsafe fn create_prefill_plan(
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
        self.result_with_last_error(unsafe {
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

    pub(crate) unsafe fn execute_prefill(
        &mut self,
        plan: &Plan,
        desc: &BatchPrefillExecuteDesc,
    ) -> Result<(), Status> {
        validate_prefill_plan_execute(plan.kind, &plan.attention, plan.shape, desc)?;
        let PlanRaw::Prefill(raw) = plan.raw else {
            return Err(Status::InvalidArgument);
        };
        self.result_with_last_error(unsafe {
            sys::qsfi_batch_prefill_execute(self.raw.as_ptr(), raw.as_ptr(), desc)
        })
    }

    pub(crate) unsafe fn append_paged_kv_decode(
        &mut self,
        attention: &AttentionDesc,
        append: &AppendDecode,
    ) -> Result<(), Status> {
        validate_attention_desc(attention)?;
        validate_append_decode_desc(attention, append)?;
        self.result_with_last_error(unsafe {
            sys::qsfi_append_paged_kv_decode(self.raw.as_ptr(), attention, append)
        })
    }

    pub(crate) unsafe fn append_paged_kv_prefill(
        &mut self,
        attention: &AttentionDesc,
        append: &AppendPrefill,
    ) -> Result<(), Status> {
        validate_attention_desc(attention)?;
        validate_append_prefill_desc(attention, append)?;
        self.result_with_last_error(unsafe {
            sys::qsfi_append_paged_kv_prefill(self.raw.as_ptr(), attention, append)
        })
    }

    pub(crate) unsafe fn rmsnorm(&mut self, desc: &RmsnormDesc) -> Result<(), Status> {
        validate_rmsnorm_desc(desc)?;
        self.result_with_last_error(unsafe { sys::qsfi_rmsnorm(self.raw.as_ptr(), desc) })
    }

    pub(crate) unsafe fn fused_add_rmsnorm(
        &mut self,
        desc: &FusedAddRmsnormDesc,
    ) -> Result<(), Status> {
        validate_fused_add_rmsnorm_desc(desc)?;
        self.result_with_last_error(unsafe { sys::qsfi_fused_add_rmsnorm(self.raw.as_ptr(), desc) })
    }

    pub(crate) unsafe fn rope_apply(&mut self, desc: &RopeApplyDesc) -> Result<(), Status> {
        validate_rope_apply_desc(desc)?;
        self.result_with_last_error(unsafe { sys::qsfi_rope_apply(self.raw.as_ptr(), desc) })
    }

    pub(crate) unsafe fn create_moe_plan(&mut self, desc: &MoePlanDesc) -> Result<MoePlan, Status> {
        validate_moe_plan_desc(desc)?;
        let mut raw = ptr::null_mut();
        self.result_with_last_error(unsafe {
            sys::qsfi_moe_plan_create(self.raw.as_ptr(), desc, &mut raw)
        })?;
        MoePlan::from_raw(raw, *desc)
    }

    pub(crate) unsafe fn moe_workspace_size(
        &mut self,
        plan: &MoePlan,
        num_tokens: u32,
    ) -> Result<usize, Status> {
        moe_workspace_bytes(&plan.desc, num_tokens)?;
        let mut device_bytes = 0usize;
        self.result_with_last_error(unsafe {
            sys::qsfi_moe_workspace_size(
                self.raw.as_ptr(),
                plan.raw.as_ptr(),
                num_tokens,
                &mut device_bytes,
            )
        })?;
        Ok(device_bytes)
    }

    pub(crate) unsafe fn moe_execute_bf16(
        &mut self,
        plan: &MoePlan,
        desc: &MoeBf16ExecuteDesc,
    ) -> Result<(), Status> {
        validate_moe_bf16_execute_desc(&plan.desc, desc)?;
        self.result_with_last_error(unsafe {
            sys::qsfi_moe_execute_bf16(self.raw.as_ptr(), plan.raw.as_ptr(), desc)
        })
    }

    pub(crate) unsafe fn moe_execute_nvfp4(
        &mut self,
        plan: &MoePlan,
        desc: &MoeNvfp4ExecuteDesc,
    ) -> Result<(), Status> {
        validate_moe_nvfp4_execute_desc(&plan.desc, desc)?;
        self.result_with_last_error(unsafe {
            sys::qsfi_moe_execute_nvfp4(self.raw.as_ptr(), plan.raw.as_ptr(), desc)
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
    Decode(NonNull<ffi::BatchDecodePlanRaw>),
    Prefill(NonNull<ffi::BatchPrefillPlanRaw>),
}

pub(crate) struct Plan {
    raw: PlanRaw,
    kind: PlanKind,
    attention: AttentionDesc,
    shape: PlanShape,
}

impl Plan {
    fn from_decode_raw(
        raw: *mut ffi::BatchDecodePlanRaw,
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
        raw: *mut ffi::BatchPrefillPlanRaw,
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

pub(crate) struct MoePlan {
    raw: NonNull<ffi::MoePlanRaw>,
    desc: MoePlanDesc,
}

impl MoePlan {
    fn from_raw(raw: *mut ffi::MoePlanRaw, desc: MoePlanDesc) -> Result<Self, Status> {
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self { raw, desc })
    }
}

impl Drop for MoePlan {
    fn drop(&mut self) {
        unsafe {
            sys::qsfi_moe_plan_destroy(self.raw.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ffi::{KvLayoutRaw, MOE_ROUTE_ROUTER_LOGITS};
    use crate::{QWEN36_FULL_ATTN_KV_HIDDEN, QWEN36_FULL_ATTN_Q_HIDDEN};

    use super::*;
    use std::ffi::c_void;

    fn device_ptr(offset: usize) -> DevicePtr {
        (0x1000usize + offset) as *mut c_void
    }

    fn tensor1(data: DevicePtr, dtype: DTypeRaw, shape: [i64; 1], stride: [i64; 1]) -> Tensor1 {
        Tensor1 {
            data,
            dtype,
            shape,
            stride,
        }
    }

    fn tensor2(data: DevicePtr, dtype: DTypeRaw, shape: [i64; 2], stride: [i64; 2]) -> Tensor2 {
        Tensor2 {
            data,
            dtype,
            shape,
            stride,
        }
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
            num_qo_heads: QWEN36_FULL_ATTN_Q_HEADS,
            num_kv_heads: QWEN36_FULL_ATTN_KV_HEADS,
            head_dim_qk: QWEN36_FULL_ATTN_HEAD_DIM,
            head_dim_vo: QWEN36_FULL_ATTN_HEAD_DIM,
            page_size: 4,
            q_dtype: DTYPE_F16,
            kv_dtype: DTYPE_F16,
            o_dtype: DTYPE_F16,
            kv_layout: layout,
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
        let page_size = 4_i64;
        let max_pages = 8_i64;
        let kv_heads = i64::from(QWEN36_FULL_ATTN_KV_HEADS);
        let head_dim = i64::from(QWEN36_FULL_ATTN_HEAD_DIM);
        let kv_hidden = i64::from(QWEN36_FULL_ATTN_KV_HIDDEN);
        let page_hidden = page_size * kv_hidden;
        let shape = if layout == KV_LAYOUT_NHD {
            [max_pages, page_size, kv_heads, head_dim]
        } else {
            [max_pages, kv_heads, page_size, head_dim]
        };
        let stride = if layout == KV_LAYOUT_NHD {
            [page_hidden, kv_hidden, head_dim, 1]
        } else {
            [page_hidden, page_size * head_dim, head_dim, 1]
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

        let huge_extent = tensor3(device_ptr(1), DTYPE_F16, [i64::MAX, 2, 2], [2, 1, 1]);
        assert_eq!(
            validate_tensor(&huge_extent, DTYPE_F16),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn unknown_raw_status_maps_to_internal_error() {
        assert_eq!(result_from_raw(StatusRaw::MAX), Err(Status::InternalError));
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

        let mut non_divisible = valid;
        non_divisible.num_qo_heads = 3;
        assert_eq!(
            validate_attention_desc(&non_divisible),
            Err(Status::InvalidArgument)
        );

        let mut old_no_gqa = valid;
        old_no_gqa.num_qo_heads = 2;
        old_no_gqa.num_kv_heads = 2;
        old_no_gqa.head_dim_qk = 64;
        old_no_gqa.head_dim_vo = 64;
        assert_eq!(
            validate_attention_desc(&old_no_gqa),
            Err(Status::Unsupported)
        );

        let mut too_few_grouped_heads = valid;
        too_few_grouped_heads.num_qo_heads = 8;
        too_few_grouped_heads.num_kv_heads = 1;
        assert_eq!(
            validate_attention_desc(&too_few_grouped_heads),
            Err(Status::Unsupported)
        );

        let mut too_many_grouped_heads = valid;
        too_many_grouped_heads.num_qo_heads = 32;
        too_many_grouped_heads.num_kv_heads = 4;
        assert_eq!(
            validate_attention_desc(&too_many_grouped_heads),
            Err(Status::Unsupported)
        );

        let mut unsupported_head_dim = valid;
        unsupported_head_dim.head_dim_qk = 128;
        unsupported_head_dim.head_dim_vo = 128;
        assert_eq!(
            validate_attention_desc(&unsupported_head_dim),
            Err(Status::Unsupported)
        );

        let mut split_head_dim = valid;
        split_head_dim.head_dim_vo = 128;
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
        let host_plan = PagedKvPlanHost::new(&attention, &[0, 0, 2], &[3, 4], &[0, 4]).unwrap();
        assert_eq!(host_plan.shape(), shape);
        assert_eq!(host_plan.as_raw().batch_size, 2);
        assert_eq!(host_plan.as_raw().num_indices, 2);

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
        let qo_host = QoPlanHost::new(&[0, 3, 4], 4).unwrap();
        assert_eq!(qo_host.shape(), shape);
        assert_eq!(qo_host.as_raw().batch_size, 2);
        assert_eq!(qo_host.as_raw().total_tokens, 4);
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

        let mut bad_layout_shape = kv_cache(KV_LAYOUT_HND);
        bad_layout_shape.k.shape[1] = 3;
        bad_layout_shape.v.shape[1] = 3;
        assert_eq!(
            validate_kv_cache_desc(&nhd, &bad_layout_shape),
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
        let q_heads = i64::from(QWEN36_FULL_ATTN_Q_HEADS);
        let head_dim = i64::from(QWEN36_FULL_ATTN_HEAD_DIM);
        let q_hidden = i64::from(QWEN36_FULL_ATTN_Q_HIDDEN);
        let decode_q = tensor3(
            device_ptr(10),
            DTYPE_F16,
            [2, q_heads, head_dim],
            [q_hidden, head_dim, 1],
        );
        let prefill_q = tensor3(
            device_ptr(11),
            DTYPE_F16,
            [5, q_heads, head_dim],
            [q_hidden, head_dim, 1],
        );
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

    #[test]
    fn append_desc_validation_checks_kv_and_index_presence() {
        let attention = attention(KV_LAYOUT_NHD);
        let shape = PlanShape {
            batch_size: 2,
            num_indices: 3,
            total_tokens: 5,
        };
        let kv_heads = i64::from(QWEN36_FULL_ATTN_KV_HEADS);
        let head_dim = i64::from(QWEN36_FULL_ATTN_HEAD_DIM);
        let kv_hidden = i64::from(QWEN36_FULL_ATTN_KV_HIDDEN);
        let decode_kv = tensor3(
            device_ptr(70),
            DTYPE_F16,
            [2, kv_heads, head_dim],
            [kv_hidden, head_dim, 1],
        );
        let prefill_kv = tensor3(
            device_ptr(71),
            DTYPE_F16,
            [5, kv_heads, head_dim],
            [kv_hidden, head_dim, 1],
        );
        let decode = AppendDecode {
            k: decode_kv,
            v: decode_kv,
            kv_cache: kv_cache(KV_LAYOUT_NHD),
            page_table: exec_table(shape),
        };
        assert_eq!(validate_append_decode_desc(&attention, &decode), Ok(()));

        let mut bad_decode_shape = decode;
        bad_decode_shape.k.shape[0] = 1;
        assert_eq!(
            validate_append_decode_desc(&attention, &bad_decode_shape),
            Err(Status::InvalidArgument)
        );

        let mut bad_decode_stride = decode;
        bad_decode_stride.k.stride[0] = 768;
        bad_decode_stride.v.stride[0] = 768;
        assert_eq!(
            validate_append_decode_desc(&attention, &bad_decode_stride),
            Err(Status::Unsupported)
        );

        let prefill = AppendPrefill {
            k: prefill_kv,
            v: prefill_kv,
            batch_indices: device_ptr(72),
            positions: device_ptr(73),
            kv_cache: kv_cache(KV_LAYOUT_NHD),
            page_table: exec_table(shape),
            num_tokens: 5,
        };
        assert_eq!(validate_append_prefill_desc(&attention, &prefill), Ok(()));

        let mut missing_positions = prefill;
        missing_positions.positions = ptr::null_mut();
        assert_eq!(
            validate_append_prefill_desc(&attention, &missing_positions),
            Err(Status::InvalidArgument)
        );

        let zero_tokens = AppendPrefill {
            num_tokens: 0,
            batch_indices: ptr::null_mut(),
            positions: ptr::null_mut(),
            ..prefill
        };
        assert_eq!(
            validate_append_prefill_desc(&attention, &zero_tokens),
            Ok(())
        );

        let mut bad_prefill_tokens = prefill;
        bad_prefill_tokens.num_tokens = 4;
        assert_eq!(
            validate_append_prefill_desc(&attention, &bad_prefill_tokens),
            Err(Status::InvalidArgument)
        );
    }

    fn rmsnorm_desc(dtype: DTypeRaw) -> RmsnormDesc {
        RmsnormDesc {
            x: tensor2(device_ptr(30), dtype, [3, 128], [128, 1]),
            weight: tensor1(device_ptr(31), dtype, [128], [1]),
            out: tensor2(device_ptr(32), dtype, [3, 128], [128, 1]),
            hidden_size: 128,
            weight_bias: 0.0,
            eps: 1.0e-6,
        }
    }

    #[test]
    fn rmsnorm_validation_checks_dtype_shape_and_eps() {
        let valid = rmsnorm_desc(DTYPE_BF16);
        assert_eq!(validate_rmsnorm_desc(&valid), Ok(()));
        assert_eq!(validate_rmsnorm_desc(&rmsnorm_desc(DTYPE_F32)), Ok(()));

        let mut qwen_qk_norm = valid;
        qwen_qk_norm.weight_bias = 1.0;
        assert_eq!(validate_rmsnorm_desc(&qwen_qk_norm), Ok(()));

        let unsupported = rmsnorm_desc(DTYPE_F16);
        assert_eq!(
            validate_rmsnorm_desc(&unsupported),
            Err(Status::Unsupported)
        );

        let mut bad_eps = valid;
        bad_eps.eps = 0.0;
        assert_eq!(
            validate_rmsnorm_desc(&bad_eps),
            Err(Status::InvalidArgument)
        );

        let mut bad_bias = valid;
        bad_bias.weight_bias = 0.5;
        assert_eq!(validate_rmsnorm_desc(&bad_bias), Err(Status::Unsupported));

        bad_bias.weight_bias = f32::NAN;
        assert_eq!(
            validate_rmsnorm_desc(&bad_bias),
            Err(Status::InvalidArgument)
        );

        let mut bad_hidden = valid;
        bad_hidden.out.shape[1] = 64;
        assert_eq!(
            validate_rmsnorm_desc(&bad_hidden),
            Err(Status::InvalidArgument)
        );

        let mut bad_weight_stride = valid;
        bad_weight_stride.weight.stride[0] = 2;
        assert_eq!(
            validate_rmsnorm_desc(&bad_weight_stride),
            Err(Status::InvalidArgument)
        );
    }

    fn fused_add_rmsnorm_desc() -> FusedAddRmsnormDesc {
        let x = tensor2(device_ptr(40), DTYPE_BF16, [3, 128], [128, 1]);
        FusedAddRmsnormDesc {
            x,
            residual_inout: tensor2(device_ptr(41), DTYPE_BF16, [3, 128], [128, 1]),
            weight: tensor1(device_ptr(42), DTYPE_BF16, [128], [1]),
            out: x,
            hidden_size: 128,
            eps: 1.0e-6,
        }
    }

    #[test]
    fn fused_add_rmsnorm_validation_requires_out_alias() {
        let valid = fused_add_rmsnorm_desc();
        assert_eq!(validate_fused_add_rmsnorm_desc(&valid), Ok(()));

        let mut non_alias = valid;
        non_alias.out.data = device_ptr(43);
        assert_eq!(
            validate_fused_add_rmsnorm_desc(&non_alias),
            Err(Status::InvalidArgument)
        );

        let mut bad_residual_dtype = valid;
        bad_residual_dtype.residual_inout.dtype = DTYPE_F32;
        assert_eq!(
            validate_fused_add_rmsnorm_desc(&bad_residual_dtype),
            Err(Status::InvalidArgument)
        );
    }

    fn rope_desc(dtype: DTypeRaw) -> RopeApplyDesc {
        RopeApplyDesc {
            q: tensor3(device_ptr(50), dtype, [5, 4, 128], [512, 128, 1]),
            k: tensor3(device_ptr(51), dtype, [5, 2, 128], [256, 128, 1]),
            q_out: tensor3(device_ptr(52), dtype, [5, 4, 128], [512, 128, 1]),
            k_out: tensor3(device_ptr(53), dtype, [5, 2, 128], [256, 128, 1]),
            positions: tensor1(device_ptr(54), DTYPE_U32, [5], [1]),
            num_qo_heads: 4,
            num_kv_heads: 2,
            head_dim: 128,
            rotary_dim: 128,
            rope_scale: 1.0,
            rope_theta: 10000.0,
            interleave: 0,
        }
    }

    #[test]
    fn rope_apply_validation_checks_shapes_positions_and_alias_stride() {
        let valid = rope_desc(DTYPE_BF16);
        assert_eq!(validate_rope_apply_desc(&valid), Ok(()));
        assert_eq!(validate_rope_apply_desc(&rope_desc(DTYPE_F32)), Ok(()));

        let mut bad_positions_dtype = valid;
        bad_positions_dtype.positions.dtype = DTYPE_F32;
        assert_eq!(
            validate_rope_apply_desc(&bad_positions_dtype),
            Err(Status::InvalidArgument)
        );

        let mut bad_interleave = valid;
        bad_interleave.interleave = 1;
        assert_eq!(
            validate_rope_apply_desc(&bad_interleave),
            Err(Status::Unsupported)
        );

        let mut unsupported_head_dim = valid;
        unsupported_head_dim.head_dim = 96;
        unsupported_head_dim.q.shape[2] = 96;
        unsupported_head_dim.k.shape[2] = 96;
        unsupported_head_dim.q_out.shape[2] = 96;
        unsupported_head_dim.k_out.shape[2] = 96;
        unsupported_head_dim.rotary_dim = 96;
        assert_eq!(
            validate_rope_apply_desc(&unsupported_head_dim),
            Err(Status::Unsupported)
        );

        let mut invalid_rotary_dim = valid;
        invalid_rotary_dim.rotary_dim = 0;
        assert_eq!(
            validate_rope_apply_desc(&invalid_rotary_dim),
            Err(Status::InvalidArgument)
        );

        invalid_rotary_dim = valid;
        invalid_rotary_dim.rotary_dim = 127;
        assert_eq!(
            validate_rope_apply_desc(&invalid_rotary_dim),
            Err(Status::InvalidArgument)
        );

        invalid_rotary_dim = valid;
        invalid_rotary_dim.rotary_dim = 256;
        assert_eq!(
            validate_rope_apply_desc(&invalid_rotary_dim),
            Err(Status::InvalidArgument)
        );

        let mut qwen36_rotary = rope_desc(DTYPE_BF16);
        let head_dim = i64::from(QWEN36_FULL_ATTN_HEAD_DIM);
        let q_stride = i64::from(4 * QWEN36_FULL_ATTN_HEAD_DIM);
        let kv_stride = i64::from(QWEN36_FULL_ATTN_KV_HIDDEN);
        qwen36_rotary.q.shape[2] = head_dim;
        qwen36_rotary.q.stride = [q_stride, head_dim, 1];
        qwen36_rotary.k.shape[2] = head_dim;
        qwen36_rotary.k.stride = [kv_stride, head_dim, 1];
        qwen36_rotary.q_out.shape[2] = head_dim;
        qwen36_rotary.q_out.stride = [q_stride, head_dim, 1];
        qwen36_rotary.k_out.shape[2] = head_dim;
        qwen36_rotary.k_out.stride = [kv_stride, head_dim, 1];
        qwen36_rotary.head_dim = QWEN36_FULL_ATTN_HEAD_DIM;
        qwen36_rotary.rotary_dim = QWEN36_FULL_ATTN_ROTARY_DIM;
        assert_eq!(validate_rope_apply_desc(&qwen36_rotary), Ok(()));

        qwen36_rotary.rotary_dim = 128;
        assert_eq!(
            validate_rope_apply_desc(&qwen36_rotary),
            Err(Status::Unsupported)
        );

        let mut bad_inplace_stride = valid;
        bad_inplace_stride.q_out.data = bad_inplace_stride.q.data;
        bad_inplace_stride.q_out.stride[0] = 1024;
        assert_eq!(
            validate_rope_apply_desc(&bad_inplace_stride),
            Err(Status::InvalidArgument)
        );
    }

    fn moe_bf16_plan_desc() -> MoePlanDesc {
        MoePlanDesc {
            backend: MOE_BACKEND_FLASHINFER_STAGED_BF16,
            route_mode: MOE_ROUTE_PRECOMPUTED_TOPK,
            max_num_tokens: 8,
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 4,
            top_k: 2,
            local_expert_offset: 0,
            local_num_experts: 4,
            activation_dtype: DTYPE_BF16,
            weight_dtype: DTYPE_BF16,
            output_dtype: DTYPE_BF16,
            reserved0: 0,
        }
    }

    fn moe_nvfp4_plan_desc() -> MoePlanDesc {
        MoePlanDesc {
            backend: MOE_BACKEND_FLASHINFER_NVFP4,
            route_mode: MOE_ROUTE_PRECOMPUTED_TOPK,
            max_num_tokens: 8,
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 4,
            top_k: 2,
            local_expert_offset: 0,
            local_num_experts: 4,
            activation_dtype: DTYPE_NVFP4_E2M1,
            weight_dtype: DTYPE_NVFP4_E2M1,
            output_dtype: DTYPE_BF16,
            reserved0: 0,
        }
    }

    #[test]
    fn moe_plan_validation_checks_supported_backend_and_shape() {
        let valid = moe_bf16_plan_desc();
        assert_eq!(validate_moe_plan_desc(&valid), Ok(()));
        assert_eq!(validate_moe_plan_desc(&moe_nvfp4_plan_desc()), Ok(()));

        let mut router_logits = valid;
        router_logits.route_mode = MOE_ROUTE_ROUTER_LOGITS;
        assert_eq!(
            validate_moe_plan_desc(&router_logits),
            Err(Status::Unsupported)
        );

        let mut local_subset = valid;
        local_subset.local_num_experts = 2;
        assert_eq!(
            validate_moe_plan_desc(&local_subset),
            Err(Status::Unsupported)
        );

        let mut bad_dtype = valid;
        bad_dtype.activation_dtype = DTYPE_F16;
        assert_eq!(
            validate_moe_plan_desc(&bad_dtype),
            Err(Status::InvalidArgument)
        );

        let mut bad_multiple = valid;
        bad_multiple.hidden_size = 10;
        assert_eq!(
            validate_moe_plan_desc(&bad_multiple),
            Err(Status::InvalidArgument)
        );
    }

    fn moe_bf16_execute_desc(workspace_bytes: usize) -> MoeBf16ExecuteDesc {
        MoeBf16ExecuteDesc {
            hidden: tensor2(device_ptr(60), DTYPE_BF16, [3, 16], [16, 1]),
            topk_ids: tensor2(device_ptr(61), DTYPE_I32, [3, 2], [2, 1]),
            topk_weights: tensor2(device_ptr(62), DTYPE_F32, [3, 2], [2, 1]),
            gate_up_weight: tensor3(device_ptr(63), DTYPE_BF16, [4, 64, 16], [1024, 16, 1]),
            down_weight: tensor3(device_ptr(64), DTYPE_BF16, [4, 16, 32], [512, 32, 1]),
            out: tensor2(device_ptr(65), DTYPE_BF16, [3, 16], [16, 1]),
            workspace: tensor1(
                device_ptr(66),
                DTYPE_U8,
                [i64::try_from(workspace_bytes).unwrap()],
                [1],
            ),
            num_tokens: 3,
        }
    }

    #[test]
    fn moe_bf16_execute_validation_checks_shapes_and_workspace() {
        let plan = moe_bf16_plan_desc();
        let workspace_bytes = moe_workspace_bytes(&plan, 3).unwrap();
        assert!(workspace_bytes > 0);
        assert_eq!(
            moe_workspace_bytes(&plan, plan.max_num_tokens + 1),
            Err(Status::InvalidArgument)
        );

        let valid = moe_bf16_execute_desc(workspace_bytes);
        assert_eq!(validate_moe_bf16_execute_desc(&plan, &valid), Ok(()));

        let mut small_workspace = valid;
        small_workspace.workspace.shape[0] = i64::try_from(workspace_bytes - 1).unwrap();
        assert_eq!(
            validate_moe_bf16_execute_desc(&plan, &small_workspace),
            Err(Status::InvalidArgument)
        );

        let mut bad_gate_shape = valid;
        bad_gate_shape.gate_up_weight.shape[1] = 32;
        assert_eq!(
            validate_moe_bf16_execute_desc(&plan, &bad_gate_shape),
            Err(Status::InvalidArgument)
        );
    }

    fn moe_nvfp4_execute_desc() -> MoeNvfp4ExecuteDesc {
        MoeNvfp4ExecuteDesc {
            hidden_packed: tensor2(device_ptr(70), DTYPE_U8, [3, 8], [8, 1]),
            hidden_scale: tensor2(device_ptr(71), DTYPE_FP8_E4M3, [3, 1], [1, 1]),
            topk_ids: tensor2(device_ptr(72), DTYPE_I32, [3, 2], [2, 1]),
            topk_weights: tensor2(device_ptr(73), DTYPE_F32, [3, 2], [2, 1]),
            gate_up_weight_packed: tensor3(device_ptr(74), DTYPE_U8, [4, 64, 8], [512, 8, 1]),
            gate_up_weight_scale: tensor3(device_ptr(75), DTYPE_FP8_E4M3, [4, 64, 1], [64, 1, 1]),
            down_weight_packed: tensor3(device_ptr(76), DTYPE_U8, [4, 16, 16], [256, 16, 1]),
            down_weight_scale: tensor3(device_ptr(77), DTYPE_FP8_E4M3, [4, 16, 2], [32, 2, 1]),
            expert_output_scale: tensor1(ptr::null_mut(), DTYPE_F32, [0], [0]),
            out: tensor2(device_ptr(78), DTYPE_BF16, [3, 16], [16, 1]),
            workspace: tensor1(device_ptr(79), DTYPE_U8, [256], [1]),
            num_tokens: 3,
        }
    }

    #[test]
    fn moe_nvfp4_execute_validation_checks_declared_shapes() {
        let plan = moe_nvfp4_plan_desc();
        let valid = moe_nvfp4_execute_desc();
        assert_eq!(validate_moe_nvfp4_execute_desc(&plan, &valid), Ok(()));

        let mut bad_scale = valid;
        bad_scale.hidden_scale.shape[1] = 2;
        assert_eq!(
            validate_moe_nvfp4_execute_desc(&plan, &bad_scale),
            Err(Status::InvalidArgument)
        );

        let mut optional_scale = valid;
        optional_scale.expert_output_scale = tensor1(device_ptr(80), DTYPE_F32, [4], [1]);
        assert_eq!(
            validate_moe_nvfp4_execute_desc(&plan, &optional_scale),
            Ok(())
        );
    }
}
