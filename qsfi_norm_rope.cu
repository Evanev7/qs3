#include "qsfi_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/norm.cuh>
#include <flashinfer/pos_enc.cuh>

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>

namespace {

bool supported_float_dtype(qsfi_dtype dtype)
{
    return dtype == QSFI_DTYPE_BF16 || dtype == QSFI_DTYPE_F32;
}

bool supported_rope_head_dim(uint32_t head_dim)
{
    return head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

const char* dtype_name(qsfi_dtype dtype)
{
    switch (dtype) {
    case QSFI_DTYPE_F32:
        return "f32";
    case QSFI_DTYPE_BF16:
        return "bf16";
    default:
        return "unsupported";
    }
}

bool tensor2_has_row_major_inner(const qsfi_tensor2& tensor)
{
    return tensor.stride[1] == 1 && tensor.stride[0] >= tensor.shape[1];
}

bool tensor3_has_row_major_inner(const qsfi_tensor3& tensor)
{
    return tensor.stride[2] == 1 && tensor.stride[1] >= tensor.shape[2]
        && tensor.stride[0] >= tensor.shape[1] * tensor.stride[1];
}

bool same_shape_and_stride(const qsfi_tensor2& a, const qsfi_tensor2& b)
{
    return a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] && a.stride[0] == b.stride[0]
        && a.stride[1] == b.stride[1];
}

bool same_shape(const qsfi_tensor2& a, const qsfi_tensor2& b)
{
    return a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1];
}

bool same_shape_and_stride(const qsfi_tensor3& a, const qsfi_tensor3& b)
{
    return a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] && a.shape[2] == b.shape[2]
        && a.stride[0] == b.stride[0] && a.stride[1] == b.stride[1] && a.stride[2] == b.stride[2];
}

bool same_shape(const qsfi_tensor3& a, const qsfi_tensor3& b)
{
    return a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] && a.shape[2] == b.shape[2];
}

qsfi_status require_u32(qsfi_context* ctx, int64_t value, const char* name)
{
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
        return set_invalid_arg(ctx, "%s must fit in uint32_t", name);
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_eps(qsfi_context* ctx, float eps)
{
    if (!std::isfinite(eps) || eps <= 0.0f) {
        return set_invalid_arg(ctx, "rmsnorm eps must be finite and > 0");
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_rmsnorm_common(
    qsfi_context* ctx,
    const qsfi_tensor2& x,
    const qsfi_tensor1& weight,
    const qsfi_tensor2& out,
    uint32_t hidden_size,
    float eps,
    const char* op
)
{
    if (hidden_size == 0) {
        return set_invalid_arg(ctx, "%s hidden_size must be non-zero", op);
    }
    if (!supported_float_dtype(x.dtype)) {
        return set_unsupported(
            ctx,
            "%s supports only bf16/f32 tensors, got %s",
            op,
            dtype_name(x.dtype)
        );
    }

    qsfi_status status = validate_tensor(ctx, x, "x", x.dtype, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, weight, "weight", x.dtype, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, out, "out", x.dtype, 2);
    if (status != QSFI_STATUS_OK)
        return status;

    if (x.shape[1] != static_cast<int64_t>(hidden_size) || out.shape[1] != x.shape[1]
        || weight.shape[0] != x.shape[1]) {
        return set_invalid_arg(
            ctx,
            "%s tensors must use hidden_size as their innermost extent",
            op
        );
    }
    if (out.shape[0] != x.shape[0]) {
        return set_invalid_arg(ctx, "%s out row count must match x", op);
    }
    if (!tensor2_has_row_major_inner(x) || !tensor2_has_row_major_inner(out)
        || weight.stride[0] != 1) {
        return set_invalid_arg(
            ctx,
            "%s requires contiguous hidden dimension and contiguous weight",
            op
        );
    }
    status = require_u32(ctx, x.shape[0], "x.shape[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, x.stride[0], "x.stride[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, out.stride[0], "out.stride[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_eps(ctx, eps);
}

qsfi_status validate_rmsnorm_weight_bias(qsfi_context* ctx, float weight_bias)
{
    if (!std::isfinite(weight_bias)) {
        return set_invalid_arg(ctx, "rmsnorm weight_bias must be finite");
    }
    if (weight_bias != 0.0f && weight_bias != 1.0f) {
        return set_unsupported(ctx, "rmsnorm weight_bias must be 0.0 or 1.0");
    }
    return QSFI_STATUS_OK;
}

template <typename T> cudaError_t launch_rmsnorm(qsfi_context* ctx, const qsfi_rmsnorm_desc* desc)
{
    if (desc->weight_bias == 1.0f) {
        return flashinfer::norm::GemmaRMSNorm<T>(
            static_cast<T*>(desc->x.data),
            static_cast<T*>(desc->weight.data),
            static_cast<T*>(desc->out.data),
            static_cast<uint32_t>(desc->x.shape[0]),
            desc->hidden_size,
            static_cast<uint32_t>(desc->x.stride[0]),
            static_cast<uint32_t>(desc->out.stride[0]),
            desc->eps,
            false,
            ctx->stream
        );
    }
    return flashinfer::norm::RMSNorm<T>(
        static_cast<T*>(desc->x.data),
        static_cast<T*>(desc->weight.data),
        static_cast<T*>(desc->out.data),
        static_cast<uint32_t>(desc->x.shape[0]),
        desc->hidden_size,
        static_cast<uint32_t>(desc->x.stride[0]),
        static_cast<uint32_t>(desc->out.stride[0]),
        desc->eps,
        false,
        ctx->stream
    );
}

template <typename T>
cudaError_t launch_fused_add_rmsnorm(qsfi_context* ctx, const qsfi_fused_add_rmsnorm_desc* desc)
{
    return flashinfer::norm::FusedAddRMSNorm<T>(
        static_cast<T*>(desc->x.data),
        static_cast<T*>(desc->residual_inout.data),
        static_cast<T*>(desc->weight.data),
        static_cast<uint32_t>(desc->x.shape[0]),
        desc->hidden_size,
        static_cast<uint32_t>(desc->x.stride[0]),
        static_cast<uint32_t>(desc->residual_inout.stride[0]),
        desc->eps,
        false,
        ctx->stream
    );
}

qsfi_status validate_fused_add_rmsnorm(qsfi_context* ctx, const qsfi_fused_add_rmsnorm_desc* desc)
{
    qsfi_status status = validate_rmsnorm_common(
        ctx,
        desc->x,
        desc->weight,
        desc->out,
        desc->hidden_size,
        desc->eps,
        "fused_add_rmsnorm"
    );
    if (status != QSFI_STATUS_OK)
        return status;

    status = validate_tensor(ctx, desc->residual_inout, "residual_inout", desc->x.dtype, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    if (!same_shape(desc->x, desc->residual_inout)) {
        return set_invalid_arg(ctx, "fused_add_rmsnorm residual_inout shape must match x");
    }
    if (!tensor2_has_row_major_inner(desc->residual_inout)) {
        return set_invalid_arg(
            ctx,
            "fused_add_rmsnorm residual_inout hidden dimension must be contiguous"
        );
    }
    if (desc->out.data != desc->x.data || !same_shape_and_stride(desc->out, desc->x)) {
        return set_invalid_arg(ctx, "fused_add_rmsnorm out must alias x exactly");
    }
    return require_u32(ctx, desc->residual_inout.stride[0], "residual_inout.stride[0]");
}

qsfi_status validate_rope_apply(qsfi_context* ctx, const qsfi_rope_apply_desc* desc)
{
    if (desc == nullptr) {
        return set_invalid_arg(ctx, "rope_apply desc must not be null");
    }
    if (desc->num_qo_heads == 0 || desc->num_kv_heads == 0 || desc->head_dim == 0
        || desc->rotary_dim == 0) {
        return set_invalid_arg(ctx, "rope_apply dimensions must be non-zero");
    }
    if (desc->head_dim % 2 != 0) {
        return set_invalid_arg(ctx, "rope_apply head_dim must be even");
    }
    if (desc->rotary_dim % 2 != 0) {
        return set_invalid_arg(ctx, "rope_apply rotary_dim must be even");
    }
    if (desc->rotary_dim > desc->head_dim) {
        return set_invalid_arg(ctx, "rope_apply rotary_dim must be <= head_dim");
    }
    if (!supported_rope_head_dim(desc->head_dim)) {
        return set_unsupported(ctx, "rope_apply supports only head_dim 64/128/256/512");
    }
    if (desc->head_dim == 256 && desc->rotary_dim != 64) {
        return set_unsupported(ctx, "rope_apply qwen3.6 head_dim 256 requires rotary_dim 64");
    }
    if (desc->interleave != 0) {
        return set_unsupported(ctx, "rope_apply supports only NeoX/Llama interleave=false layout");
    }
    if (!std::isfinite(desc->rope_scale) || desc->rope_scale < 0.0f) {
        return set_invalid_arg(ctx, "rope_apply rope_scale must be finite and >= 0");
    }
    if (!std::isfinite(desc->rope_theta) || desc->rope_theta < 0.0f) {
        return set_invalid_arg(ctx, "rope_apply rope_theta must be finite and >= 0");
    }

    if (!supported_float_dtype(desc->q.dtype)) {
        return set_unsupported(
            ctx,
            "rope_apply supports only bf16/f32 q/k tensors, got %s",
            dtype_name(desc->q.dtype)
        );
    }
    qsfi_status status = validate_tensor(ctx, desc->q, "q", desc->q.dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->k, "k", desc->q.dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->q_out, "q_out", desc->q.dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->k_out, "k_out", desc->q.dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;

    if (desc->positions.dtype != QSFI_DTYPE_I32 && desc->positions.dtype != QSFI_DTYPE_U32) {
        return set_invalid_arg(ctx, "rope_apply positions dtype must be i32 or u32");
    }
    status = validate_tensor(ctx, desc->positions, "positions", desc->positions.dtype, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    if (desc->positions.stride[0] != 1) {
        return set_invalid_arg(ctx, "rope_apply positions must be contiguous");
    }

    const int64_t num_tokens = desc->q.shape[0];
    if (desc->k.shape[0] != num_tokens || desc->q_out.shape[0] != num_tokens
        || desc->k_out.shape[0] != num_tokens || desc->positions.shape[0] != num_tokens) {
        return set_invalid_arg(ctx, "rope_apply token dimensions must match");
    }
    if (desc->q.shape[1] != static_cast<int64_t>(desc->num_qo_heads)
        || desc->q_out.shape[1] != desc->q.shape[1]
        || desc->k.shape[1] != static_cast<int64_t>(desc->num_kv_heads)
        || desc->k_out.shape[1] != desc->k.shape[1]) {
        return set_invalid_arg(ctx, "rope_apply head dimensions must match descriptor");
    }
    if (desc->q.shape[2] != static_cast<int64_t>(desc->head_dim)
        || desc->k.shape[2] != static_cast<int64_t>(desc->head_dim)
        || desc->q_out.shape[2] != static_cast<int64_t>(desc->head_dim)
        || desc->k_out.shape[2] != static_cast<int64_t>(desc->head_dim)) {
        return set_invalid_arg(ctx, "rope_apply innermost dimensions must match head_dim");
    }
    if (!same_shape(desc->q, desc->q_out) || !same_shape(desc->k, desc->k_out)) {
        return set_invalid_arg(ctx, "rope_apply outputs must match input shapes");
    }
    if ((desc->q_out.data == desc->q.data && !same_shape_and_stride(desc->q, desc->q_out))
        || (desc->k_out.data == desc->k.data && !same_shape_and_stride(desc->k, desc->k_out))) {
        return set_invalid_arg(ctx, "rope_apply in-place outputs must match input strides");
    }
    if (!tensor3_has_row_major_inner(desc->q) || !tensor3_has_row_major_inner(desc->k)
        || !tensor3_has_row_major_inner(desc->q_out) || !tensor3_has_row_major_inner(desc->k_out)) {
        return set_invalid_arg(ctx, "rope_apply requires contiguous head_dim storage");
    }

    status = require_u32(ctx, num_tokens, "num_tokens");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->q.stride[0], "q.stride[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->q.stride[1], "q.stride[1]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->k.stride[0], "k.stride[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->k.stride[1], "k.stride[1]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->q_out.stride[0], "q_out.stride[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->q_out.stride[1], "q_out.stride[1]");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_u32(ctx, desc->k_out.stride[0], "k_out.stride[0]");
    if (status != QSFI_STATUS_OK)
        return status;
    return require_u32(ctx, desc->k_out.stride[1], "k_out.stride[1]");
}

template <typename T, typename IdType>
cudaError_t launch_rope_apply(qsfi_context* ctx, const qsfi_rope_apply_desc* desc)
{
    const float rope_scale = qsfi_default_one(desc->rope_scale);
    const float rope_theta = desc->rope_theta == 0.0f ? 10000.0f : desc->rope_theta;
    return flashinfer::BatchQKApplyRotaryPosIds<T, IdType>(
        static_cast<T*>(desc->q.data),
        static_cast<T*>(desc->k.data),
        static_cast<T*>(desc->q_out.data),
        static_cast<T*>(desc->k_out.data),
        static_cast<IdType*>(desc->positions.data),
        static_cast<uint32_t>(desc->q.shape[0]),
        desc->num_qo_heads,
        desc->num_kv_heads,
        desc->rotary_dim,
        desc->head_dim,
        static_cast<size_t>(desc->q.stride[0]),
        static_cast<size_t>(desc->q.stride[1]),
        static_cast<size_t>(desc->k.stride[0]),
        static_cast<size_t>(desc->k.stride[1]),
        static_cast<size_t>(desc->q_out.stride[0]),
        static_cast<size_t>(desc->q_out.stride[1]),
        static_cast<size_t>(desc->k_out.stride[0]),
        static_cast<size_t>(desc->k_out.stride[1]),
        false,
        rope_scale,
        rope_theta,
        ctx->stream
    );
}

template <typename T>
cudaError_t launch_rope_apply(qsfi_context* ctx, const qsfi_rope_apply_desc* desc)
{
    if (desc->positions.dtype == QSFI_DTYPE_U32) {
        return launch_rope_apply<T, uint32_t>(ctx, desc);
    }
    return launch_rope_apply<T, int32_t>(ctx, desc);
}

} // namespace

extern "C" {

qsfi_status qsfi_rmsnorm(qsfi_context* ctx, const qsfi_rmsnorm_desc* desc)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (desc == nullptr)
        return set_invalid_arg(ctx, "rmsnorm desc must not be null");
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_rmsnorm_common(
        ctx,
        desc->x,
        desc->weight,
        desc->out,
        desc->hidden_size,
        desc->eps,
        "rmsnorm"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_rmsnorm_weight_bias(ctx, desc->weight_bias);
    if (status != QSFI_STATUS_OK)
        return status;

    try {
        cudaError_t err = desc->x.dtype == QSFI_DTYPE_BF16
            ? launch_rmsnorm<__nv_bfloat16>(ctx, desc)
            : launch_rmsnorm<float>(ctx, desc);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer rmsnorm");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer rmsnorm", ex);
    }
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_fused_add_rmsnorm(qsfi_context* ctx, const qsfi_fused_add_rmsnorm_desc* desc)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (desc == nullptr)
        return set_invalid_arg(ctx, "fused_add_rmsnorm desc must not be null");
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_fused_add_rmsnorm(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;

    try {
        cudaError_t err = desc->x.dtype == QSFI_DTYPE_BF16
            ? launch_fused_add_rmsnorm<__nv_bfloat16>(ctx, desc)
            : launch_fused_add_rmsnorm<float>(ctx, desc);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer fused_add_rmsnorm");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer fused_add_rmsnorm", ex);
    }
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_rope_apply(qsfi_context* ctx, const qsfi_rope_apply_desc* desc)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_rope_apply(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;

    try {
        cudaError_t err = desc->q.dtype == QSFI_DTYPE_BF16
            ? launch_rope_apply<__nv_bfloat16>(ctx, desc)
            : launch_rope_apply<float>(ctx, desc);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer rope_apply");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer rope_apply", ex);
    }
    return QSFI_STATUS_OK;
}

} // extern "C"
