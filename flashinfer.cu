#include "flashinfer.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>

struct qsfi_context {
    int32_t device_ordinal;
    cudaStream_t stream;
    void* float_workspace;
    size_t float_workspace_bytes;
    void* int_workspace;
    size_t int_workspace_bytes;
    void* host_int_workspace;
    size_t host_int_workspace_bytes;
    uint64_t scratch_generation;
    qsfi_error_info_t last_error;
};

struct qsfi_plan {
    qsfi_plan_kind_t kind;
    qsfi_attention_desc_t attention;
    uint32_t batch_size;
    uint32_t num_indices;
    uint32_t total_tokens;
    uint64_t scratch_generation;
    flashinfer::DecodePlanInfo decode;
    flashinfer::PrefillPlanInfo prefill;
};

namespace {

void clear_error(qsfi_error_info_t* err)
{
    if (err == nullptr)
        return;
    err->status = QSFI_STATUS_OK;
    err->source = QSFI_ERROR_SOURCE_NONE;
    err->native_code = 0;
    err->message[0] = '\0';
}

qsfi_status_t set_error(
    qsfi_context_t* ctx,
    qsfi_status_t status,
    qsfi_error_source_t source,
    int32_t native_code,
    const char* fmt,
    ...
)
{
    if (ctx == nullptr)
        return status;
    ctx->last_error.status = status;
    ctx->last_error.source = source;
    ctx->last_error.native_code = native_code;
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(ctx->last_error.message, QSFI_ERROR_MESSAGE_BYTES, fmt, args);
    va_end(args);
    ctx->last_error.message[QSFI_ERROR_MESSAGE_BYTES - 1] = '\0';
    return status;
}

qsfi_status_t set_cuda_error(qsfi_context_t* ctx, cudaError_t err, const char* op)
{
    if (err == cudaSuccess)
        return QSFI_STATUS_OK;
    const qsfi_status_t status
        = (err == cudaErrorMemoryAllocation) ? QSFI_STATUS_OUT_OF_MEMORY : QSFI_STATUS_CUDA_ERROR;
    return set_error(
        ctx,
        status,
        QSFI_ERROR_SOURCE_CUDA,
        static_cast<int32_t>(err),
        "%s: %s",
        op,
        cudaGetErrorString(err)
    );
}

qsfi_status_t set_flashinfer_error(qsfi_context_t* ctx, const char* op, const std::exception& ex)
{
    return set_error(
        ctx,
        QSFI_STATUS_BACKEND_ERROR,
        QSFI_ERROR_SOURCE_FLASHINFER,
        0,
        "%s: %s",
        op,
        ex.what()
    );
}

qsfi_status_t activate_context(qsfi_context_t* ctx)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (ctx->device_ordinal < 0)
        return QSFI_STATUS_OK;
    cudaError_t err = cudaSetDevice(ctx->device_ordinal);
    if (err != cudaSuccess)
        return set_cuda_error(ctx, err, "cudaSetDevice");
    return QSFI_STATUS_OK;
}

bool valid_dtype(qsfi_dtype_t dtype)
{
    return dtype == QSFI_DTYPE_F16 || dtype == QSFI_DTYPE_BF16 || dtype == QSFI_DTYPE_FP8_E4M3
        || dtype == QSFI_DTYPE_FP8_E5M2 || dtype == QSFI_DTYPE_NVFP4_E2M1;
}

bool supported_attention_dtype(qsfi_dtype_t dtype)
{
    return dtype == QSFI_DTYPE_F16 || dtype == QSFI_DTYPE_BF16;
}

flashinfer::QKVLayout to_flashinfer_layout(qsfi_kv_layout_t layout)
{
    switch (layout) {
    case QSFI_KV_LAYOUT_HND:
        return flashinfer::QKVLayout::kHND;
    case QSFI_KV_LAYOUT_NHD:
        return flashinfer::QKVLayout::kNHD;
    }
}

float default_sm_scale(const qsfi_attention_desc_t& attention)
{
    if (attention.sm_scale != 0.0f)
        return attention.sm_scale;
    return 1.0f / std::sqrt(static_cast<float>(attention.head_dim_qk));
}

float default_one(float value)
{
    return value == 0.0f ? 1.0f : value;
}

qsfi_status_t require_scratch(qsfi_context_t* ctx)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (ctx->float_workspace == nullptr || ctx->int_workspace == nullptr
        || ctx->host_int_workspace == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "scratch workspaces are not reserved"
        );
    }
    return QSFI_STATUS_OK;
}

bool pointer_is_host_readable(const void* ptr)
{
    if (ptr == nullptr)
        return false;
    cudaPointerAttributes attr;
    std::memset(&attr, 0, sizeof(attr));
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return true;
    }
#if CUDART_VERSION >= 10000
    return attr.type != cudaMemoryTypeDevice;
#else
    return attr.memoryType == cudaMemoryTypeHost;
#endif
}

qsfi_status_t require_host_readable_i32(qsfi_context_t* ctx, const int32_t* ptr, const char* name)
{
    if (ptr == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s must not be null",
            name
        );
    }
    if (!pointer_is_host_readable(ptr)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s must be host-readable at plan time; use host or managed memory",
            name
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t validate_attention(qsfi_context_t* ctx, const qsfi_attention_desc_t* attention)
{
    if (attention == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "attention must not be null"
        );
    }
    if (attention->num_qo_heads == 0 || attention->num_kv_heads == 0 || attention->head_dim_qk == 0
        || attention->head_dim_vo == 0 || attention->page_size == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "attention dimensions must be non-zero"
        );
    }
    if (attention->num_qo_heads % attention->num_kv_heads != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "num_qo_heads must be divisible by num_kv_heads"
        );
    }
    if (attention->head_dim_qk != attention->head_dim_vo) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "different qk/vo head dimensions are not wired yet"
        );
    }
    if (attention->kv_layout != QSFI_KV_LAYOUT_NHD && attention->kv_layout != QSFI_KV_LAYOUT_HND) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "invalid kv_layout"
        );
    }
    if (attention->pos_encoding != QSFI_POS_ENCODING_NONE
        && attention->pos_encoding != QSFI_POS_ENCODING_ROPE_LLAMA) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "unsupported positional encoding"
        );
    }
    if (!valid_dtype(attention->q_dtype) || !valid_dtype(attention->kv_dtype)
        || !valid_dtype(attention->o_dtype)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "invalid attention dtype"
        );
    }
    if (attention->q_dtype != attention->kv_dtype || attention->q_dtype != attention->o_dtype) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "mixed q/kv/o dtypes are not wired yet"
        );
    }
    if (!supported_attention_dtype(attention->q_dtype)) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "only f16 and bf16 attention are wired initially"
        );
    }
    if (attention->use_fp16_qk_reduction != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "fp16 qk reduction is not wired yet"
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t validate_paged_kv_plan(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table
)
{
    if (page_table == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "page_table plan must not be null"
        );
    }
    if (page_table->batch_size == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "page_table batch_size must be non-zero"
        );
    }
    qsfi_status_t status = require_host_readable_i32(ctx, page_table->indptr, "page_table.indptr");
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_host_readable_i32(ctx, page_table->last_page_len, "page_table.last_page_len");
    if (status != QSFI_STATUS_OK)
        return status;
    if (page_table->num_indices != 0) {
        status = require_host_readable_i32(ctx, page_table->indices, "page_table.indices");
        if (status != QSFI_STATUS_OK)
            return status;
    }
    if (page_table->indptr[0] != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "page_table.indptr[0] must be 0"
        );
    }
    for (uint32_t i = 0; i < page_table->batch_size; ++i) {
        const int32_t begin = page_table->indptr[i];
        const int32_t end = page_table->indptr[i + 1];
        const int32_t pages = end - begin;
        const int32_t last_len = page_table->last_page_len[i];
        if (begin < 0 || end < begin) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "page_table.indptr must be monotonic"
            );
        }
        if (pages == 0) {
            if (last_len != 0) {
                return set_error(
                    ctx,
                    QSFI_STATUS_INVALID_ARGUMENT,
                    QSFI_ERROR_SOURCE_QSFI,
                    0,
                    "empty requests must have last_page_len 0"
                );
            }
        } else if (last_len <= 0 || last_len > static_cast<int32_t>(attention->page_size)) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "last_page_len entries must be in [1, page_size] for non-empty requests"
            );
        }
    }
    if (page_table->indptr[page_table->batch_size]
        != static_cast<int32_t>(page_table->num_indices)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "page_table.num_indices must match indptr[batch_size]"
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t validate_qo_plan(qsfi_context_t* ctx, const qsfi_qo_plan_t* qo)
{
    if (qo == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qo plan must not be null"
        );
    }
    if (qo->batch_size == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qo batch_size must be non-zero"
        );
    }
    qsfi_status_t status = require_host_readable_i32(ctx, qo->indptr, "qo.indptr");
    if (status != QSFI_STATUS_OK)
        return status;
    if (qo->indptr[0] != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qo.indptr[0] must be 0"
        );
    }
    for (uint32_t i = 0; i < qo->batch_size; ++i) {
        if (qo->indptr[i] < 0 || qo->indptr[i + 1] < qo->indptr[i]) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "qo.indptr must be monotonic"
            );
        }
    }
    if (qo->indptr[qo->batch_size] != static_cast<int32_t>(qo->total_tokens)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qo.total_tokens must match qo.indptr[batch_size]"
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t validate_tensor(
    qsfi_context_t* ctx,
    const qsfi_tensor_desc_t& tensor,
    const char* name,
    qsfi_dtype_t dtype,
    uint32_t ndim
)
{
    if (tensor.data == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s.data must not be null",
            name
        );
    }
    if (tensor.dtype != dtype) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s dtype does not match plan",
            name
        );
    }
    if (tensor.ndim != ndim) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s ndim mismatch",
            name
        );
    }
    for (uint32_t i = 0; i < ndim; ++i) {
        if (tensor.shape[i] <= 0 || tensor.stride[i] <= 0) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "%s shape/stride entries must be positive",
                name
            );
        }
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t validate_kv_cache(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t& attention,
    const qsfi_paged_kv_cache_t& kv_cache,
    uint32_t* out_num_pages
)
{
    qsfi_status_t status = validate_tensor(ctx, kv_cache.k, "kv_cache.k", attention.kv_dtype, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, kv_cache.v, "kv_cache.v", attention.kv_dtype, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    for (uint32_t i = 0; i < 4; ++i) {
        if (kv_cache.k.shape[i] != kv_cache.v.shape[i]
            || kv_cache.k.stride[i] != kv_cache.v.stride[i]) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "kv_cache k/v shapes and strides must match"
            );
        }
    }
    if (kv_cache.k_scale.data != nullptr || kv_cache.v_scale.data != nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "kv scale tensors are only for quantized kv paths, not wired yet"
        );
    }
    const int64_t num_pages = kv_cache.k.shape[0];
    if (attention.kv_layout == QSFI_KV_LAYOUT_NHD) {
        if (kv_cache.k.shape[1] != static_cast<int64_t>(attention.page_size)
            || kv_cache.k.shape[2] != static_cast<int64_t>(attention.num_kv_heads)
            || kv_cache.k.shape[3] != static_cast<int64_t>(attention.head_dim_qk)) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "NHD kv_cache shape must be [pages, page_size, kv_heads, head_dim]"
            );
        }
    } else {
        if (kv_cache.k.shape[1] != static_cast<int64_t>(attention.num_kv_heads)
            || kv_cache.k.shape[2] != static_cast<int64_t>(attention.page_size)
            || kv_cache.k.shape[3] != static_cast<int64_t>(attention.head_dim_qk)) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "HND kv_cache shape must be [pages, kv_heads, page_size, head_dim]"
            );
        }
    }
    if (out_num_pages != nullptr)
        *out_num_pages = static_cast<uint32_t>(num_pages);
    return QSFI_STATUS_OK;
}

qsfi_status_t validate_page_table_exec(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_paged_kv_table_t& table
)
{
    if (table.indptr == nullptr || table.indices == nullptr || table.last_page_len == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "execution page table device pointers must not be null"
        );
    }
    if (table.batch_size != plan->batch_size || table.num_indices != plan->num_indices) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "execution page table shape does not match plan"
        );
    }
    return QSFI_STATUS_OK;
}

template <typename T, flashinfer::PosEncodingMode Pos, bool Sliding, bool Logits>
cudaError_t decode_plan_impl(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    flashinfer::DecodePlanInfo* out
)
{
    using Params = flashinfer::BatchDecodeParams<T, T, T, int32_t>;
    using AttentionVariant = flashinfer::DefaultAttention<false, Sliding, Logits, false>;
    cudaError_t status = cudaSuccess;
    DISPATCH_HEAD_DIM(attention->head_dim_qk, HEAD_DIM, {
        DISPATCH_GQA_GROUP_SIZE(attention->num_qo_heads / attention->num_kv_heads, GROUP_SIZE, {
            auto work_estimation = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                GROUP_SIZE,
                HEAD_DIM,
                Pos,
                AttentionVariant,
                Params>;
            status = flashinfer::DecodePlan<HEAD_DIM, Pos, AttentionVariant, Params>(
                ctx->float_workspace,
                ctx->float_workspace_bytes,
                ctx->int_workspace,
                ctx->host_int_workspace,
                ctx->int_workspace_bytes,
                *out,
                const_cast<int32_t*>(page_table->indptr),
                page_table->batch_size,
                attention->num_qo_heads,
                attention->page_size,
                false,
                ctx->stream,
                work_estimation
            );
        });
    });
    return status;
}

template <typename T, flashinfer::PosEncodingMode Pos, bool Sliding>
cudaError_t decode_plan_logits(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    flashinfer::DecodePlanInfo* out
)
{
    if (attention->logits_soft_cap > 0.0f) {
        return decode_plan_impl<T, Pos, Sliding, true>(ctx, attention, page_table, out);
    }
    return decode_plan_impl<T, Pos, Sliding, false>(ctx, attention, page_table, out);
}

template <typename T, flashinfer::PosEncodingMode Pos>
cudaError_t decode_plan_sliding(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    flashinfer::DecodePlanInfo* out
)
{
    if (attention->window_left >= 0) {
        return decode_plan_logits<T, Pos, true>(ctx, attention, page_table, out);
    }
    return decode_plan_logits<T, Pos, false>(ctx, attention, page_table, out);
}

template <typename T>
cudaError_t decode_plan_dtype(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    flashinfer::DecodePlanInfo* out
)
{
    if (attention->pos_encoding == QSFI_POS_ENCODING_ROPE_LLAMA) {
        return decode_plan_sliding<T, flashinfer::PosEncodingMode::kRoPELlama>(
            ctx,
            attention,
            page_table,
            out
        );
    }
    return decode_plan_sliding<T, flashinfer::PosEncodingMode::kNone>(
        ctx,
        attention,
        page_table,
        out
    );
}

cudaError_t decode_plan_dispatch(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    flashinfer::DecodePlanInfo* out
)
{
    if (attention->q_dtype == QSFI_DTYPE_BF16) {
        return decode_plan_dtype<__nv_bfloat16>(ctx, attention, page_table, out);
    }
    return decode_plan_dtype<half>(ctx, attention, page_table, out);
}

template <typename T, flashinfer::PosEncodingMode Pos, bool Sliding, bool Logits>
cudaError_t decode_execute_impl(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    using Params = flashinfer::BatchDecodeParams<T, T, T, int32_t>;
    using AttentionVariant = flashinfer::DefaultAttention<false, Sliding, Logits, false>;
    const qsfi_attention_desc_t& attention = plan->attention;
    const flashinfer::QKVLayout layout = to_flashinfer_layout(attention.kv_layout);
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        attention.num_kv_heads,
        attention.page_size,
        attention.head_dim_qk,
        desc->page_table.batch_size,
        layout,
        static_cast<T*>(desc->kv_cache.k.data),
        static_cast<T*>(desc->kv_cache.v.data),
        desc->kv_cache.k.stride,
        static_cast<int32_t*>(desc->page_table.indices),
        static_cast<int32_t*>(desc->page_table.indptr),
        static_cast<int32_t*>(desc->page_table.last_page_len),
        static_cast<int32_t*>(desc->page_table.rope_pos_offset)
    );

    Params params;
    params.q = static_cast<T*>(desc->q.data);
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = static_cast<T*>(desc->o.data);
    params.lse = static_cast<float*>(desc->lse);
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = static_cast<uint32_t>(plan->decode.padded_batch_size);
    params.num_qo_heads = attention.num_qo_heads;
    params.q_stride_n = static_cast<int32_t>(desc->q.stride[0]);
    params.q_stride_h = static_cast<int32_t>(desc->q.stride[1]);
    params.window_left = attention.window_left;
    params.logits_soft_cap = attention.logits_soft_cap;
    params.sm_scale
        = default_sm_scale(attention) * default_one(desc->q_scale) * default_one(desc->k_scale);
    params.rope_rcp_scale = 1.0f / default_one(attention.rope_scale);
    params.rope_rcp_theta = 1.0f / (attention.rope_theta == 0.0f ? 10000.0f : attention.rope_theta);
    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->decode.request_indices_offset
    );
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->decode.kv_tile_indices_offset
    );
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->decode.o_indptr_offset
    );
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->decode.kv_chunk_size_ptr_offset
    );
    params.block_valid_mask = nullptr;
    params.partition_kv = false;

    T* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan->decode.split_kv) {
        tmp_v = flashinfer::GetPtrFromBaseOffset<T>(ctx->float_workspace, plan->decode.v_offset);
        tmp_s
            = flashinfer::GetPtrFromBaseOffset<float>(ctx->float_workspace, plan->decode.s_offset);
    }
    cudaError_t status = cudaSuccess;
    DISPATCH_HEAD_DIM(attention.head_dim_qk, HEAD_DIM, {
        status = flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, Pos, AttentionVariant>(
            params,
            tmp_v,
            tmp_s,
            desc->enable_pdl != 0,
            ctx->stream
        );
    });
    return status;
}

template <typename T, flashinfer::PosEncodingMode Pos, bool Sliding>
cudaError_t decode_execute_logits(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    if (plan->attention.logits_soft_cap > 0.0f) {
        return decode_execute_impl<T, Pos, Sliding, true>(ctx, plan, desc);
    }
    return decode_execute_impl<T, Pos, Sliding, false>(ctx, plan, desc);
}

template <typename T, flashinfer::PosEncodingMode Pos>
cudaError_t decode_execute_sliding(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    if (plan->attention.window_left >= 0) {
        return decode_execute_logits<T, Pos, true>(ctx, plan, desc);
    }
    return decode_execute_logits<T, Pos, false>(ctx, plan, desc);
}

template <typename T>
cudaError_t decode_execute_dtype(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    if (plan->attention.pos_encoding == QSFI_POS_ENCODING_ROPE_LLAMA) {
        return decode_execute_sliding<T, flashinfer::PosEncodingMode::kRoPELlama>(ctx, plan, desc);
    }
    return decode_execute_sliding<T, flashinfer::PosEncodingMode::kNone>(ctx, plan, desc);
}

cudaError_t decode_execute_dispatch(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    if (plan->attention.q_dtype == QSFI_DTYPE_BF16) {
        return decode_execute_dtype<__nv_bfloat16>(ctx, plan, desc);
    }
    return decode_execute_dtype<half>(ctx, plan, desc);
}

cudaError_t prefill_plan_dispatch(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_qo_plan_t* qo,
    const qsfi_paged_kv_plan_t* page_table,
    flashinfer::PrefillPlanInfo* out
)
{
    return flashinfer::PrefillPlan<int32_t>(
        ctx->float_workspace,
        ctx->float_workspace_bytes,
        ctx->int_workspace,
        ctx->host_int_workspace,
        ctx->int_workspace_bytes,
        *out,
        const_cast<int32_t*>(qo->indptr),
        const_cast<int32_t*>(page_table->indptr),
        qo->total_tokens,
        qo->batch_size,
        attention->num_qo_heads,
        attention->num_kv_heads,
        attention->head_dim_qk,
        attention->head_dim_vo,
        attention->page_size,
        false,
        sizeof(uint16_t),
        attention->window_left,
        attention->fixed_split_size,
        attention->disable_split_kv != 0,
        0,
        ctx->stream
    );
}

template <
    typename T,
    flashinfer::PosEncodingMode Pos,
    bool Sliding,
    bool Logits,
    flashinfer::MaskMode Mask>
cudaError_t prefill_execute_impl(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    using Params = flashinfer::BatchPrefillPagedParams<T, T, T, int32_t>;
    using AttentionVariant = flashinfer::DefaultAttention<false, Sliding, Logits, false>;
    const qsfi_attention_desc_t& attention = plan->attention;
    const flashinfer::QKVLayout layout = to_flashinfer_layout(attention.kv_layout);
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        attention.num_kv_heads,
        attention.page_size,
        attention.head_dim_qk,
        desc->page_table.batch_size,
        layout,
        static_cast<T*>(desc->kv_cache.k.data),
        static_cast<T*>(desc->kv_cache.v.data),
        desc->kv_cache.k.stride,
        static_cast<int32_t*>(desc->page_table.indices),
        static_cast<int32_t*>(desc->page_table.indptr),
        static_cast<int32_t*>(desc->page_table.last_page_len),
        static_cast<int32_t*>(desc->page_table.rope_pos_offset)
    );

    Params params;
    params.q = static_cast<T*>(desc->q.data);
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = static_cast<int32_t*>(desc->qo_indptr);
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<T*>(desc->o.data);
    params.lse = static_cast<float*>(desc->lse);
    params.maybe_alibi_slopes = nullptr;
    params.group_size = flashinfer::uint_fastdiv(attention.num_qo_heads / attention.num_kv_heads);
    params.num_qo_heads = attention.num_qo_heads;
    params.q_stride_n = static_cast<int32_t>(desc->q.stride[0]);
    params.q_stride_h = static_cast<int32_t>(desc->q.stride[1]);
    params.window_left = attention.window_left;
    params.logits_soft_cap = attention.logits_soft_cap;
    params.sm_scale
        = default_sm_scale(attention) * default_one(desc->q_scale) * default_one(desc->k_scale);
    params.rope_rcp_scale = 1.0f / default_one(attention.rope_scale);
    params.rope_rcp_theta = 1.0f / (attention.rope_theta == 0.0f ? 10000.0f : attention.rope_theta);
    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->prefill.request_indices_offset
    );
    params.qo_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->prefill.qo_tile_indices_offset
    );
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->prefill.kv_tile_indices_offset
    );
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->prefill.o_indptr_offset
    );
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        ctx->int_workspace,
        plan->prefill.kv_chunk_size_ptr_offset
    );
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.max_total_num_rows = static_cast<uint32_t>(plan->prefill.total_num_rows);
    params.padded_batch_size = static_cast<uint32_t>(plan->prefill.padded_batch_size);
    params.partition_kv = false;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    T* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan->prefill.split_kv) {
        params.merge_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
            ctx->int_workspace,
            plan->prefill.merge_indptr_offset
        );
        tmp_v = flashinfer::GetPtrFromBaseOffset<T>(ctx->float_workspace, plan->prefill.v_offset);
        tmp_s
            = flashinfer::GetPtrFromBaseOffset<float>(ctx->float_workspace, plan->prefill.s_offset);
    }

    cudaError_t status = cudaSuccess;
    DISPATCH_HEAD_DIM(attention.head_dim_qk, HEAD_DIM, {
        DISPATCH_CTA_TILE_Q(plan->prefill.cta_tile_q, CTA_TILE_Q, {
            status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
                CTA_TILE_Q,
                HEAD_DIM,
                HEAD_DIM,
                Pos,
                false,
                Mask,
                AttentionVariant,
                Params>(params, tmp_v, tmp_s, desc->enable_pdl != 0, ctx->stream);
        });
    });
    return status;
}

template <typename T, flashinfer::PosEncodingMode Pos, bool Sliding, bool Logits>
cudaError_t prefill_execute_mask(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (plan->attention.mask_mode == QSFI_MASK_MODE_CAUSAL) {
        return prefill_execute_impl<T, Pos, Sliding, Logits, flashinfer::MaskMode::kCausal>(
            ctx,
            plan,
            desc
        );
    }
    return prefill_execute_impl<T, Pos, Sliding, Logits, flashinfer::MaskMode::kNone>(
        ctx,
        plan,
        desc
    );
}

template <typename T, flashinfer::PosEncodingMode Pos, bool Sliding>
cudaError_t prefill_execute_logits(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (plan->attention.logits_soft_cap > 0.0f) {
        return prefill_execute_mask<T, Pos, Sliding, true>(ctx, plan, desc);
    }
    return prefill_execute_mask<T, Pos, Sliding, false>(ctx, plan, desc);
}

template <typename T, flashinfer::PosEncodingMode Pos>
cudaError_t prefill_execute_sliding(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (plan->attention.window_left >= 0) {
        return prefill_execute_logits<T, Pos, true>(ctx, plan, desc);
    }
    return prefill_execute_logits<T, Pos, false>(ctx, plan, desc);
}

template <typename T>
cudaError_t prefill_execute_dtype(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (plan->attention.pos_encoding == QSFI_POS_ENCODING_ROPE_LLAMA) {
        return prefill_execute_sliding<T, flashinfer::PosEncodingMode::kRoPELlama>(ctx, plan, desc);
    }
    return prefill_execute_sliding<T, flashinfer::PosEncodingMode::kNone>(ctx, plan, desc);
}

cudaError_t prefill_execute_dispatch(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (plan->attention.q_dtype == QSFI_DTYPE_BF16) {
        return prefill_execute_dtype<__nv_bfloat16>(ctx, plan, desc);
    }
    return prefill_execute_dtype<half>(ctx, plan, desc);
}

template <typename T>
cudaError_t append_decode_impl(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_decode_t* append
)
{
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        attention->num_kv_heads,
        attention->page_size,
        attention->head_dim_qk,
        append->page_table.batch_size,
        to_flashinfer_layout(attention->kv_layout),
        static_cast<T*>(append->kv_cache.k.data),
        static_cast<T*>(append->kv_cache.v.data),
        append->kv_cache.k.stride,
        static_cast<int32_t*>(append->page_table.indices),
        static_cast<int32_t*>(append->page_table.indptr),
        static_cast<int32_t*>(append->page_table.last_page_len),
        static_cast<int32_t*>(append->page_table.rope_pos_offset)
    );
    return flashinfer::AppendPagedKVCacheDecode(
        paged_kv,
        static_cast<T*>(append->k.data),
        static_cast<T*>(append->v.data),
        ctx->stream
    );
}

template <typename T>
cudaError_t append_prefill_impl(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_prefill_t* append
)
{
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        attention->num_kv_heads,
        attention->page_size,
        attention->head_dim_qk,
        append->page_table.batch_size,
        to_flashinfer_layout(attention->kv_layout),
        static_cast<T*>(append->kv_cache.k.data),
        static_cast<T*>(append->kv_cache.v.data),
        append->kv_cache.k.stride,
        static_cast<int32_t*>(append->page_table.indices),
        static_cast<int32_t*>(append->page_table.indptr),
        static_cast<int32_t*>(append->page_table.last_page_len),
        static_cast<int32_t*>(append->page_table.rope_pos_offset)
    );
    return flashinfer::AppendPagedKVCache(
        paged_kv,
        static_cast<T*>(append->k.data),
        static_cast<T*>(append->v.data),
        static_cast<int32_t*>(append->batch_indices),
        static_cast<int32_t*>(append->positions),
        append->num_tokens,
        static_cast<size_t>(append->k.stride[0]),
        static_cast<size_t>(append->k.stride[1]),
        static_cast<size_t>(append->v.stride[0]),
        static_cast<size_t>(append->v.stride[1]),
        ctx->stream
    );
}

cudaError_t append_decode_dispatch(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_decode_t* append
)
{
    if (attention->kv_dtype == QSFI_DTYPE_BF16) {
        return append_decode_impl<__nv_bfloat16>(ctx, attention, append);
    }
    return append_decode_impl<half>(ctx, attention, append);
}

cudaError_t append_prefill_dispatch(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_prefill_t* append
)
{
    if (attention->kv_dtype == QSFI_DTYPE_BF16) {
        return append_prefill_impl<__nv_bfloat16>(ctx, attention, append);
    }
    return append_prefill_impl<half>(ctx, attention, append);
}

qsfi_status_t validate_decode_execute(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    if (desc == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "decode execute desc must not be null"
        );
    }
    const qsfi_attention_desc_t& attention = plan->attention;
    qsfi_status_t status = validate_tensor(ctx, desc->q, "q", attention.q_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->o, "o", attention.o_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    if (desc->q.shape[0] != static_cast<int64_t>(plan->batch_size)
        || desc->q.shape[1] != static_cast<int64_t>(attention.num_qo_heads)
        || desc->q.shape[2] != static_cast<int64_t>(attention.head_dim_qk)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "decode q shape must be [batch, qo_heads, head_dim]"
        );
    }
    for (uint32_t i = 0; i < 3; ++i) {
        if (desc->o.shape[i] != desc->q.shape[i]) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "decode o shape must match q shape"
            );
        }
    }
    if (default_one(desc->v_scale) != 1.0f) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "v_scale other than 1 is not wired yet"
        );
    }
    status = validate_kv_cache(ctx, attention, desc->kv_cache, nullptr);
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_page_table_exec(ctx, plan, desc->page_table);
}

qsfi_status_t validate_prefill_execute(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (desc == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "prefill execute desc must not be null"
        );
    }
    if (desc->qo_indptr == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "prefill qo_indptr device pointer must not be null"
        );
    }
    const qsfi_attention_desc_t& attention = plan->attention;
    qsfi_status_t status = validate_tensor(ctx, desc->q, "q", attention.q_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->o, "o", attention.o_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    if (desc->q.shape[0] != static_cast<int64_t>(plan->total_tokens)
        || desc->q.shape[1] != static_cast<int64_t>(attention.num_qo_heads)
        || desc->q.shape[2] != static_cast<int64_t>(attention.head_dim_qk)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "prefill q shape must be [total_tokens, qo_heads, head_dim]"
        );
    }
    for (uint32_t i = 0; i < 3; ++i) {
        if (desc->o.shape[i] != desc->q.shape[i]) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "prefill o shape must match q shape"
            );
        }
    }
    if (default_one(desc->v_scale) != 1.0f) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "v_scale other than 1 is not wired yet"
        );
    }
    status = validate_kv_cache(ctx, attention, desc->kv_cache, nullptr);
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_page_table_exec(ctx, plan, desc->page_table);
}

} // namespace

extern "C" {

const char* qsfi_status_string(qsfi_status_t status)
{
    switch (status) {
    case QSFI_STATUS_OK:
        return "ok";
    case QSFI_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case QSFI_STATUS_UNSUPPORTED:
        return "unsupported";
    case QSFI_STATUS_OUT_OF_MEMORY:
        return "out of memory";
    case QSFI_STATUS_CUDA_ERROR:
        return "cuda error";
    case QSFI_STATUS_BACKEND_ERROR:
        return "backend error";
    case QSFI_STATUS_INTERNAL_ERROR:
        return "internal error";
    default:
        return "unknown status";
    }
}

qsfi_status_t qsfi_context_create(const qsfi_context_desc_t* desc, qsfi_context_t** out)
{
    if (out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    qsfi_context_t* ctx = new (std::nothrow) qsfi_context_t;
    if (ctx == nullptr)
        return QSFI_STATUS_OUT_OF_MEMORY;
    ctx->device_ordinal = desc != nullptr ? desc->device_ordinal : -1;
    ctx->stream = desc != nullptr ? static_cast<cudaStream_t>(desc->stream) : nullptr;
    ctx->float_workspace = nullptr;
    ctx->float_workspace_bytes = 0;
    ctx->int_workspace = nullptr;
    ctx->int_workspace_bytes = 0;
    ctx->host_int_workspace = nullptr;
    ctx->host_int_workspace_bytes = 0;
    ctx->scratch_generation = 0;
    clear_error(&ctx->last_error);
    if (ctx->device_ordinal >= 0) {
        cudaError_t err = cudaSetDevice(ctx->device_ordinal);
        if (err != cudaSuccess) {
            set_cuda_error(ctx, err, "cudaSetDevice");
            delete ctx;
            return QSFI_STATUS_CUDA_ERROR;
        }
    }
    *out = ctx;
    return QSFI_STATUS_OK;
}

void qsfi_context_destroy(qsfi_context_t* ctx)
{
    if (ctx == nullptr)
        return;
    if (ctx->device_ordinal >= 0)
        cudaSetDevice(ctx->device_ordinal);
    if (ctx->float_workspace != nullptr)
        cudaFree(ctx->float_workspace);
    if (ctx->int_workspace != nullptr)
        cudaFree(ctx->int_workspace);
    if (ctx->host_int_workspace != nullptr)
        cudaFreeHost(ctx->host_int_workspace);
    delete ctx;
}

qsfi_status_t qsfi_context_set_stream(qsfi_context_t* ctx, qsfi_cuda_stream_t stream)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    ctx->stream = static_cast<cudaStream_t>(stream);
    clear_error(&ctx->last_error);
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_context_reserve_scratch(
    qsfi_context_t* ctx,
    size_t float_workspace_bytes,
    size_t int_workspace_bytes,
    size_t host_int_workspace_bytes
)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    void* new_float = nullptr;
    void* new_int = nullptr;
    void* new_host_int = nullptr;
    if (float_workspace_bytes != 0) {
        cudaError_t err = cudaMalloc(&new_float, float_workspace_bytes);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "cudaMalloc float workspace");
    }
    if (int_workspace_bytes != 0) {
        cudaError_t err = cudaMalloc(&new_int, int_workspace_bytes);
        if (err != cudaSuccess) {
            if (new_float != nullptr)
                cudaFree(new_float);
            return set_cuda_error(ctx, err, "cudaMalloc int workspace");
        }
    }
    if (host_int_workspace_bytes != 0) {
        cudaError_t err
            = cudaHostAlloc(&new_host_int, host_int_workspace_bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            if (new_float != nullptr)
                cudaFree(new_float);
            if (new_int != nullptr)
                cudaFree(new_int);
            return set_cuda_error(ctx, err, "cudaHostAlloc host int workspace");
        }
    }
    if (ctx->float_workspace != nullptr)
        cudaFree(ctx->float_workspace);
    if (ctx->int_workspace != nullptr)
        cudaFree(ctx->int_workspace);
    if (ctx->host_int_workspace != nullptr)
        cudaFreeHost(ctx->host_int_workspace);
    ctx->float_workspace = new_float;
    ctx->float_workspace_bytes = float_workspace_bytes;
    ctx->int_workspace = new_int;
    ctx->int_workspace_bytes = int_workspace_bytes;
    ctx->host_int_workspace = new_host_int;
    ctx->host_int_workspace_bytes = host_int_workspace_bytes;
    ctx->scratch_generation += 1;
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_context_get_last_error(const qsfi_context_t* ctx, qsfi_error_info_t* out)
{
    if (ctx == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = ctx->last_error;
    return QSFI_STATUS_OK;
}

void qsfi_context_clear_last_error(qsfi_context_t* ctx)
{
    if (ctx == nullptr)
        return;
    clear_error(&ctx->last_error);
}

qsfi_status_t qsfi_load_kernels(qsfi_context_t* ctx, qsfi_kernel_flags_t modules)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    if ((modules & ~QSFI_KERNEL_MODULE_ALL) != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "unknown kernel module flags"
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_batch_decode_plan_create(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    qsfi_plan_t** out
)
{
    if (ctx == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    *out = nullptr;
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_scratch(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_attention(ctx, attention);
    if (status != QSFI_STATUS_OK)
        return status;
    if (attention->mask_mode != QSFI_MASK_MODE_NONE) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "decode mask modes are not wired; use QSFI_MASK_MODE_NONE"
        );
    }
    status = validate_paged_kv_plan(ctx, attention, page_table);
    if (status != QSFI_STATUS_OK)
        return status;
    qsfi_plan_t* plan = new (std::nothrow) qsfi_plan_t {};
    if (plan == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_OUT_OF_MEMORY,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "failed to allocate decode plan"
        );
    }
    plan->kind = QSFI_PLAN_BATCH_DECODE;
    plan->attention = *attention;
    plan->batch_size = page_table->batch_size;
    plan->num_indices = page_table->num_indices;
    plan->total_tokens = page_table->batch_size;
    plan->scratch_generation = ctx->scratch_generation;
    try {
        cudaError_t err = decode_plan_dispatch(ctx, attention, page_table, &plan->decode);
        if (err != cudaSuccess) {
            delete plan;
            return set_cuda_error(ctx, err, "flashinfer decode plan");
        }
    } catch (const std::exception& ex) {
        delete plan;
        return set_flashinfer_error(ctx, "flashinfer decode plan", ex);
    }
    *out = plan;
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_batch_decode_execute(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
)
{
    if (ctx == nullptr || plan == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    if (plan->kind != QSFI_PLAN_BATCH_DECODE) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "plan is not a decode plan"
        );
    }
    if (plan->scratch_generation != ctx->scratch_generation) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "scratch was reallocated after plan creation"
        );
    }
    status = validate_decode_execute(ctx, plan, desc);
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        cudaError_t err = decode_execute_dispatch(ctx, plan, desc);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer decode execute");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer decode execute", ex);
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_batch_prefill_plan_create(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_qo_plan_t* qo,
    const qsfi_paged_kv_plan_t* page_table,
    qsfi_plan_t** out
)
{
    if (ctx == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    *out = nullptr;
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = require_scratch(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_attention(ctx, attention);
    if (status != QSFI_STATUS_OK)
        return status;
    if (attention->mask_mode != QSFI_MASK_MODE_NONE
        && attention->mask_mode != QSFI_MASK_MODE_CAUSAL) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "prefill supports only none/causal mask modes initially"
        );
    }
    status = validate_qo_plan(ctx, qo);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_paged_kv_plan(ctx, attention, page_table);
    if (status != QSFI_STATUS_OK)
        return status;
    if (qo->batch_size != page_table->batch_size) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qo and page_table batch sizes must match"
        );
    }
    qsfi_plan_t* plan = new (std::nothrow) qsfi_plan_t {};
    if (plan == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_OUT_OF_MEMORY,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "failed to allocate prefill plan"
        );
    }
    plan->kind = QSFI_PLAN_BATCH_PREFILL;
    plan->attention = *attention;
    plan->batch_size = page_table->batch_size;
    plan->num_indices = page_table->num_indices;
    plan->total_tokens = qo->total_tokens;
    plan->scratch_generation = ctx->scratch_generation;
    try {
        cudaError_t err = prefill_plan_dispatch(ctx, attention, qo, page_table, &plan->prefill);
        if (err != cudaSuccess) {
            delete plan;
            return set_cuda_error(ctx, err, "flashinfer prefill plan");
        }
    } catch (const std::exception& ex) {
        delete plan;
        return set_flashinfer_error(ctx, "flashinfer prefill plan", ex);
    }
    *out = plan;
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_batch_prefill_execute(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
)
{
    if (ctx == nullptr || plan == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    if (plan->kind != QSFI_PLAN_BATCH_PREFILL) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "plan is not a prefill plan"
        );
    }
    if (plan->scratch_generation != ctx->scratch_generation) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "scratch was reallocated after plan creation"
        );
    }
    status = validate_prefill_execute(ctx, plan, desc);
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        cudaError_t err = prefill_execute_dispatch(ctx, plan, desc);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer prefill execute");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer prefill execute", ex);
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_plan_kind(const qsfi_plan_t* plan, qsfi_plan_kind_t* out)
{
    if (plan == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = plan->kind;
    return QSFI_STATUS_OK;
}

void qsfi_plan_destroy(qsfi_plan_t* plan)
{
    delete plan;
}

qsfi_status_t qsfi_append_paged_kv_decode(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_decode_t* append
)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_attention(ctx, attention);
    if (status != QSFI_STATUS_OK)
        return status;
    if (append == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "append decode desc must not be null"
        );
    }
    status = validate_tensor(ctx, append->k, "append.k", attention->kv_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, append->v, "append.v", attention->kv_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    if (append->k.shape[0] != static_cast<int64_t>(append->page_table.batch_size)
        || append->k.shape[1] != static_cast<int64_t>(attention->num_kv_heads)
        || append->k.shape[2] != static_cast<int64_t>(attention->head_dim_qk)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "decode append k shape must be [batch, kv_heads, head_dim]"
        );
    }
    for (uint32_t i = 0; i < 3; ++i) {
        if (append->v.shape[i] != append->k.shape[i]
            || append->v.stride[i] != append->k.stride[i]) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "decode append k/v shapes and strides must match"
            );
        }
    }
    if (append->k.stride[2] != 1
        || append->k.stride[1] != static_cast<int64_t>(attention->head_dim_qk)
        || append->k.stride[0]
            != static_cast<int64_t>(attention->num_kv_heads * attention->head_dim_qk)) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "decode append input must be contiguous [batch, kv_heads, head_dim]"
        );
    }
    status = validate_kv_cache(ctx, *attention, append->kv_cache, nullptr);
    if (status != QSFI_STATUS_OK)
        return status;
    qsfi_plan_t shape_plan {};
    shape_plan.batch_size = append->page_table.batch_size;
    shape_plan.num_indices = append->page_table.num_indices;
    status = validate_page_table_exec(ctx, &shape_plan, append->page_table);
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        cudaError_t err = append_decode_dispatch(ctx, attention, append);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer append decode");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer append decode", ex);
    }
    return QSFI_STATUS_OK;
}

qsfi_status_t qsfi_append_paged_kv_prefill(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_prefill_t* append
)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status_t status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_attention(ctx, attention);
    if (status != QSFI_STATUS_OK)
        return status;
    if (append == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "append prefill desc must not be null"
        );
    }
    if (append->num_tokens == 0)
        return QSFI_STATUS_OK;
    if (append->batch_indices == nullptr || append->positions == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "append prefill batch_indices and positions must not be null"
        );
    }
    status = validate_tensor(ctx, append->k, "append.k", attention->kv_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, append->v, "append.v", attention->kv_dtype, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    if (append->k.shape[0] != static_cast<int64_t>(append->num_tokens)
        || append->k.shape[1] != static_cast<int64_t>(attention->num_kv_heads)
        || append->k.shape[2] != static_cast<int64_t>(attention->head_dim_qk)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "prefill append k shape must be [num_tokens, kv_heads, head_dim]"
        );
    }
    for (uint32_t i = 0; i < 3; ++i) {
        if (append->v.shape[i] != append->k.shape[i]) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "prefill append v shape must match k shape"
            );
        }
    }
    status = validate_kv_cache(ctx, *attention, append->kv_cache, nullptr);
    if (status != QSFI_STATUS_OK)
        return status;
    qsfi_plan_t shape_plan {};
    shape_plan.batch_size = append->page_table.batch_size;
    shape_plan.num_indices = append->page_table.num_indices;
    status = validate_page_table_exec(ctx, &shape_plan, append->page_table);
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        cudaError_t err = append_prefill_dispatch(ctx, attention, append);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "flashinfer append prefill");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "flashinfer append prefill", ex);
    }
    return QSFI_STATUS_OK;
}

} // extern "C"
