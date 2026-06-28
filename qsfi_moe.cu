#include "qsfi_internal.h"
#include "qsfi_macros.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>
#include <flashinfer/gemm/group_gemm.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>

struct qsfi_moe_plan {
    qsfi_moe_plan_desc desc;
};

namespace {

constexpr uint32_t kThreads = 256;
constexpr size_t kWorkspaceAlignment = 256;

struct moe_workspace {
    int32_t* counts;
    int32_t* indptr;
    int32_t* write_offsets;
    int32_t* route_rows;
    int32_t* row_tokens;
    float* row_scales;
    __nv_bfloat16* expanded_hidden;
    __nv_bfloat16* gemm1_out;
    __nv_bfloat16* act;
    __nv_bfloat16* gemm2_out;
    int32_t* problems;
    void** x_ptrs;
    void** w_ptrs;
    void** y_ptrs;
    int64_t* x_ld;
    int64_t* w_ld;
    int64_t* y_ld;
    size_t bytes;
};

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T> T* take_workspace(char* base, size_t& offset, size_t count)
{
    offset = align_up(offset, std::max(kWorkspaceAlignment, alignof(T)));
    T* ptr = base != nullptr ? reinterpret_cast<T*>(base + offset) : nullptr;
    offset += count * sizeof(T);
    return ptr;
}

bool mul_overflows_size(size_t a, size_t b, size_t* out)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        return true;
    *out = a * b;
    return false;
}

qsfi_status compute_workspace_layout(
    qsfi_context* ctx,
    const qsfi_moe_plan_desc& desc,
    uint32_t num_tokens,
    void* base,
    moe_workspace* out
)
{
    if (num_tokens > desc.max_num_tokens) {
        return set_invalid_arg(ctx, "num_tokens exceeds qsfi_moe_plan_desc.max_num_tokens");
    }

    size_t max_routes = 0;
    if (mul_overflows_size(num_tokens, desc.top_k, &max_routes)) {
        return set_invalid_arg(ctx, "MoE expanded route count overflows size_t");
    }

    size_t hidden_elems = 0;
    size_t gate_up_elems = 0;
    size_t act_elems = 0;
    size_t down_elems = 0;
    size_t twice_intermediate = 0;
    if (mul_overflows_size(max_routes, desc.hidden_size, &hidden_elems)
        || mul_overflows_size(desc.intermediate_size, 2, &twice_intermediate)
        || mul_overflows_size(max_routes, twice_intermediate, &gate_up_elems)
        || mul_overflows_size(max_routes, desc.intermediate_size, &act_elems)
        || mul_overflows_size(max_routes, desc.hidden_size, &down_elems)) {
        return set_invalid_arg(ctx, "MoE workspace element count overflows size_t");
    }

    char* bytes = static_cast<char*>(base);
    size_t offset = 0;
    moe_workspace layout {};
    const size_t local_experts = desc.local_num_experts;
    layout.counts = take_workspace<int32_t>(bytes, offset, local_experts);
    layout.indptr = take_workspace<int32_t>(bytes, offset, local_experts + 1);
    layout.write_offsets = take_workspace<int32_t>(bytes, offset, local_experts);
    layout.route_rows = take_workspace<int32_t>(bytes, offset, max_routes);
    layout.row_tokens = take_workspace<int32_t>(bytes, offset, max_routes);
    layout.row_scales = take_workspace<float>(bytes, offset, max_routes);
    layout.expanded_hidden = take_workspace<__nv_bfloat16>(bytes, offset, hidden_elems);
    layout.gemm1_out = take_workspace<__nv_bfloat16>(bytes, offset, gate_up_elems);
    layout.act = take_workspace<__nv_bfloat16>(bytes, offset, act_elems);
    layout.gemm2_out = take_workspace<__nv_bfloat16>(bytes, offset, down_elems);
    layout.problems = take_workspace<int32_t>(bytes, offset, local_experts * 3);
    layout.x_ptrs = take_workspace<void*>(bytes, offset, local_experts);
    layout.w_ptrs = take_workspace<void*>(bytes, offset, local_experts);
    layout.y_ptrs = take_workspace<void*>(bytes, offset, local_experts);
    layout.x_ld = take_workspace<int64_t>(bytes, offset, local_experts);
    layout.w_ld = take_workspace<int64_t>(bytes, offset, local_experts);
    layout.y_ld = take_workspace<int64_t>(bytes, offset, local_experts);
    layout.bytes = align_up(offset, kWorkspaceAlignment);
    *out = layout;
    return QSFI_STATUS_OK;
}

bool tensor2_is_contiguous(const qsfi_tensor2& tensor)
{
    return tensor.stride[1] == 1 && tensor.stride[0] == tensor.shape[1];
}

bool tensor3_is_contiguous(const qsfi_tensor3& tensor)
{
    return tensor.stride[2] == 1 && tensor.stride[1] == tensor.shape[2]
        && tensor.stride[0] == tensor.shape[1] * tensor.shape[2];
}

bool tensor1_is_contiguous(const qsfi_tensor1& tensor)
{
    return tensor.stride[0] == 1;
}

qsfi_status validate_contiguous(qsfi_context* ctx, const char* name)
{
    return set_invalid_arg(ctx, "%s must be contiguous in qsfi MoE staged BF16 backend", name);
}

qsfi_status validate_plan_desc(qsfi_context* ctx, const qsfi_moe_plan_desc* desc)
{
    if (desc == nullptr) {
        return set_invalid_arg(ctx, "desc is null");
    }
    if (desc->max_num_tokens == 0 || desc->hidden_size == 0 || desc->intermediate_size == 0
        || desc->num_experts == 0 || desc->top_k == 0 || desc->local_num_experts == 0) {
        return set_invalid_arg(ctx, "MoE plan dimensions must be non-zero");
    }
    if (static_cast<uint64_t>(desc->local_expert_offset) + desc->local_num_experts
        > desc->num_experts) {
        return set_invalid_arg(ctx, "MoE local expert range exceeds num_experts");
    }
    if (desc->route_mode != QSFI_MOE_ROUTE_PRECOMPUTED_TOPK) {
        return set_unsupported(ctx, "MoE router-logits mode is not implemented yet");
    }
    if (desc->backend == QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16) {
        if (desc->local_expert_offset != 0 || desc->local_num_experts != desc->num_experts) {
            return set_unsupported(
                ctx,
                "staged BF16 MoE currently requires all experts to be local"
            );
        }
        if (desc->activation_dtype != QSFI_DTYPE_BF16 || desc->weight_dtype != QSFI_DTYPE_BF16
            || desc->output_dtype != QSFI_DTYPE_BF16) {
            return set_invalid_arg(
                ctx,
                "staged BF16 MoE requires bf16 activation, weight, and output dtypes"
            );
        }
        if (desc->hidden_size % 8 != 0 || desc->intermediate_size % 8 != 0) {
            return set_invalid_arg(
                ctx,
                "staged BF16 MoE requires hidden_size and intermediate_size divisible by 8"
            );
        }
        return QSFI_STATUS_OK;
    }
    if (desc->backend == QSFI_MOE_BACKEND_FLASHINFER_NVFP4) {
        if (desc->activation_dtype != QSFI_DTYPE_NVFP4_E2M1
            || desc->weight_dtype != QSFI_DTYPE_NVFP4_E2M1
            || desc->output_dtype != QSFI_DTYPE_BF16) {
            return set_invalid_arg(
                ctx,
                "NVFP4 MoE plan requires nvfp4 activation/weight and bf16 output dtypes"
            );
        }
        if (desc->hidden_size % 16 != 0 || desc->intermediate_size % 16 != 0) {
            return set_invalid_arg(
                ctx,
                "NVFP4 MoE requires hidden_size and intermediate_size divisible by 16"
            );
        }
        return QSFI_STATUS_OK;
    }
    return set_unsupported(ctx, "unsupported MoE backend");
}

qsfi_status
validate_moe_launch_limits(qsfi_context* ctx, const qsfi_moe_plan_desc& p, uint32_t num_tokens)
{
    const uint64_t max_routes = static_cast<uint64_t>(num_tokens) * p.top_k;
    if (max_routes > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return set_unsupported(ctx, "MoE route count exceeds int32 grouped-GEMM limits");
    }
    if (p.local_num_experts > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
        || p.hidden_size > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
        || p.intermediate_size > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
        || 2ull * p.intermediate_size
            > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return set_unsupported(ctx, "MoE dimensions exceed int32 grouped-GEMM limits");
    }

    const auto grid_ok = [](uint64_t work) {
        return (work + kThreads - 1) / kThreads <= std::numeric_limits<uint32_t>::max();
    };
    if (!grid_ok(max_routes) || !grid_ok(max_routes * static_cast<uint64_t>(p.hidden_size))
        || !grid_ok(max_routes * static_cast<uint64_t>(p.intermediate_size))
        || !grid_ok(static_cast<uint64_t>(num_tokens) * p.hidden_size)) {
        return set_unsupported(ctx, "MoE launch grid is too large");
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_bf16_execute(
    qsfi_context* ctx,
    const qsfi_moe_plan* plan,
    const qsfi_moe_bf16_execute_desc* desc,
    moe_workspace* layout
)
{
    if (plan == nullptr || desc == nullptr) {
        return set_invalid_arg(ctx, "plan/desc is null");
    }
    const qsfi_moe_plan_desc& p = plan->desc;
    if (p.backend != QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16) {
        return set_invalid_arg(ctx, "qsfi_moe_execute_bf16 requires a staged BF16 MoE plan");
    }
    if (desc->num_tokens == 0 || desc->num_tokens > p.max_num_tokens) {
        return set_invalid_arg(ctx, "invalid MoE num_tokens");
    }
    qsfi_status status = validate_tensor(ctx, desc->hidden, "hidden", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->topk_ids, "topk_ids", QSFI_DTYPE_I32, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->topk_weights, "topk_weights", QSFI_DTYPE_F32, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->gate_up_weight, "gate_up_weight", QSFI_DTYPE_BF16, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->down_weight, "down_weight", QSFI_DTYPE_BF16, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->out, "out", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, desc->workspace, "workspace", desc->workspace.dtype, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    if (desc->workspace.dtype != QSFI_DTYPE_U8 && desc->workspace.dtype != QSFI_DTYPE_I8)
        return set_invalid_arg(ctx, "MoE workspace dtype must be u8 or i8");
    if (!tensor2_is_contiguous(desc->hidden))
        return validate_contiguous(ctx, "hidden");
    if (!tensor2_is_contiguous(desc->topk_ids))
        return validate_contiguous(ctx, "topk_ids");
    if (!tensor2_is_contiguous(desc->topk_weights))
        return validate_contiguous(ctx, "topk_weights");
    if (!tensor3_is_contiguous(desc->gate_up_weight))
        return validate_contiguous(ctx, "gate_up_weight");
    if (!tensor3_is_contiguous(desc->down_weight))
        return validate_contiguous(ctx, "down_weight");
    if (!tensor2_is_contiguous(desc->out))
        return validate_contiguous(ctx, "out");
    if (!tensor1_is_contiguous(desc->workspace))
        return validate_contiguous(ctx, "workspace");

    if (desc->hidden.shape[0] != desc->num_tokens || desc->hidden.shape[1] != p.hidden_size
        || desc->topk_ids.shape[0] != desc->num_tokens || desc->topk_ids.shape[1] != p.top_k
        || desc->topk_weights.shape[0] != desc->num_tokens || desc->topk_weights.shape[1] != p.top_k
        || desc->out.shape[0] != desc->num_tokens || desc->out.shape[1] != p.hidden_size) {
        return set_invalid_arg(ctx, "MoE token/top-k tensor shape mismatch");
    }
    if (desc->gate_up_weight.shape[0] != p.local_num_experts
        || desc->gate_up_weight.shape[1] != 2ll * p.intermediate_size
        || desc->gate_up_weight.shape[2] != p.hidden_size
        || desc->down_weight.shape[0] != p.local_num_experts
        || desc->down_weight.shape[1] != p.hidden_size
        || desc->down_weight.shape[2] != p.intermediate_size) {
        return set_invalid_arg(ctx, "MoE expert weight shape mismatch");
    }

    status = compute_workspace_layout(ctx, p, desc->num_tokens, desc->workspace.data, layout);
    if (status != QSFI_STATUS_OK)
        return status;
    if (static_cast<size_t>(desc->workspace.shape[0]) < layout->bytes) {
        return set_invalid_arg(ctx, "MoE workspace is too small");
    }
    return validate_moe_launch_limits(ctx, p, desc->num_tokens);
}

__global__ void validate_routes_kernel(
    const int32_t* topk_ids,
    const float* topk_weights,
    uint32_t total_routes,
    uint32_t local_expert_offset,
    uint32_t local_num_experts,
    int* error
)
{
    const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= total_routes)
        return;
    const int32_t expert = topk_ids[route];
    const int32_t local = expert - static_cast<int32_t>(local_expert_offset);
    const float weight = topk_weights[route];
    if (local < 0 || local >= static_cast<int32_t>(local_num_experts) || !isfinite(weight)) {
        atomicExch(error, 1);
    }
}

__global__ void count_routes_kernel(
    const int32_t* topk_ids,
    uint32_t num_tokens,
    uint32_t top_k,
    uint32_t local_expert_offset,
    uint32_t local_num_experts,
    int32_t* counts
)
{
    const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_routes = num_tokens * top_k;
    if (route >= total_routes)
        return;
    const int32_t expert = topk_ids[route];
    const int32_t local = expert - static_cast<int32_t>(local_expert_offset);
    if (local < 0 || local >= static_cast<int32_t>(local_num_experts))
        return;
    atomicAdd(counts + local, 1);
}

__global__ void
prefix_sum_kernel(const int32_t* counts, int32_t* indptr, uint32_t local_num_experts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int32_t running = 0;
    indptr[0] = 0;
    for (uint32_t expert = 0; expert < local_num_experts; ++expert) {
        running += counts[expert];
        indptr[expert + 1] = running;
    }
}

__global__ void fill_route_rows_kernel(
    const int32_t* topk_ids,
    const float* topk_weights,
    uint32_t num_tokens,
    uint32_t top_k,
    uint32_t local_expert_offset,
    uint32_t local_num_experts,
    const int32_t* indptr,
    int32_t* write_offsets,
    int32_t* route_rows,
    int32_t* row_tokens,
    float* row_scales
)
{
    const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_routes = num_tokens * top_k;
    if (route >= total_routes)
        return;
    const int32_t expert = topk_ids[route];
    const int32_t local = expert - static_cast<int32_t>(local_expert_offset);
    if (local < 0 || local >= static_cast<int32_t>(local_num_experts))
        return;
    const int32_t row = indptr[local] + atomicAdd(write_offsets + local, 1);
    route_rows[route] = row;
    row_tokens[row] = static_cast<int32_t>(route / top_k);
    row_scales[row] = topk_weights[route];
}

__global__ void scatter_hidden_kernel(
    const __nv_bfloat16* hidden,
    const int32_t* route_rows,
    uint32_t num_tokens,
    uint32_t top_k,
    uint32_t hidden_size,
    __nv_bfloat16* expanded_hidden
)
{
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(num_tokens) * top_k * hidden_size;
    if (idx >= total)
        return;
    const uint32_t h = idx % hidden_size;
    const uint32_t route = idx / hidden_size;
    const int32_t row = route_rows[route];
    if (row < 0)
        return;
    const uint32_t token = route / top_k;
    expanded_hidden[static_cast<uint64_t>(row) * hidden_size + h]
        = hidden[static_cast<uint64_t>(token) * hidden_size + h];
}

__global__ void prepare_group_gemm_args_kernel(
    const int32_t* indptr,
    uint32_t local_num_experts,
    uint32_t k,
    uint32_t n,
    __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* y,
    int32_t* problems,
    void** x_ptrs,
    void** w_ptrs,
    void** y_ptrs,
    int64_t* x_ld,
    int64_t* w_ld,
    int64_t* y_ld
)
{
    const uint32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= local_num_experts)
        return;
    const int32_t start = indptr[expert];
    const int32_t end = indptr[expert + 1];
    const int32_t m = end - start;
    problems[expert * 3 + 0] = m;
    problems[expert * 3 + 1] = static_cast<int32_t>(n);
    problems[expert * 3 + 2] = static_cast<int32_t>(k);
    x_ptrs[expert] = x + static_cast<uint64_t>(start) * k;
    w_ptrs[expert] = const_cast<__nv_bfloat16*>(w + static_cast<uint64_t>(expert) * n * k);
    y_ptrs[expert] = y + static_cast<uint64_t>(start) * n;
    x_ld[expert] = static_cast<int64_t>(k);
    w_ld[expert] = static_cast<int64_t>(k);
    y_ld[expert] = static_cast<int64_t>(n);
}

__global__ void swiglu_kernel(
    const __nv_bfloat16* gate_up,
    uint32_t total_routes,
    uint32_t intermediate_size,
    __nv_bfloat16* out
)
{
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(total_routes) * intermediate_size;
    if (idx >= total)
        return;
    const uint32_t i = idx % intermediate_size;
    const uint32_t row = idx / intermediate_size;
    const uint64_t base = static_cast<uint64_t>(row) * (2ull * intermediate_size);
    const float gate = __bfloat162float(gate_up[base + i]);
    const float up = __bfloat162float(gate_up[base + intermediate_size + i]);
    const float sigmoid = 1.0f / (1.0f + __expf(-gate));
    out[idx] = __float2bfloat16(gate * sigmoid * up);
}

__global__ void finalize_kernel(
    const int32_t* route_rows,
    const float* row_scales,
    const __nv_bfloat16* expert_out,
    uint32_t num_tokens,
    uint32_t top_k,
    uint32_t hidden_size,
    __nv_bfloat16* out
)
{
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
    if (idx >= total)
        return;
    const uint32_t h = idx % hidden_size;
    const uint32_t token = idx / hidden_size;
    float acc = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k) {
        const int32_t row = route_rows[token * top_k + k];
        if (row >= 0) {
            const float scale = row_scales[row];
            const float value
                = __bfloat162float(expert_out[static_cast<uint64_t>(row) * hidden_size + h]);
            acc += scale * value;
        }
    }
    out[idx] = __float2bfloat16(acc);
}

qsfi_status validate_moe_routes(
    qsfi_context* ctx,
    const qsfi_moe_plan_desc& p,
    const qsfi_moe_bf16_execute_desc* desc,
    uint32_t total_routes
)
{
#if !QSFI_ENABLE_CHECKED_VALIDATION
    (void)ctx;
    (void)p;
    (void)desc;
    (void)total_routes;
    return QSFI_STATUS_OK;
#else
    qsfi_context_error_reporter errors { ctx };
    qsfi_checked_validation_flag validation(ctx->stream);
    qsfi_status status = validation.reset(
        errors,
        "cudaMalloc MoE route validation flag",
        "cudaMemsetAsync MoE route validation flag"
    );
    if (status != QSFI_STATUS_OK)
        return status;

    const uint32_t blocks = (total_routes + kThreads - 1) / kThreads;
    validate_routes_kernel<<<blocks, kThreads, 0, ctx->stream>>>(
        static_cast<const int32_t*>(desc->topk_ids.data),
        static_cast<const float*>(desc->topk_weights.data),
        total_routes,
        p.local_expert_offset,
        p.local_num_experts,
        validation.device_ptr()
    );
    status = validation.check_launch(errors, "launch MoE route validation");
    if (status != QSFI_STATUS_OK)
        return status;
    return validation.finish(
        errors,
        "copy MoE route validation flag",
        "cudaFree MoE route validation flag",
        "MoE routes must target local experts and weights must be finite"
    );
#endif
}

cudaError_t launch_bf16_moe(
    qsfi_context* ctx,
    const qsfi_moe_plan_desc& p,
    const qsfi_moe_bf16_execute_desc* desc,
    const moe_workspace& ws
)
{
    const uint32_t num_tokens = desc->num_tokens;
    const uint32_t max_routes = static_cast<uint32_t>(static_cast<uint64_t>(num_tokens) * p.top_k);
    const uint32_t local_experts = p.local_num_experts;
    cudaStream_t stream = ctx->stream;
    const uint32_t route_blocks = (max_routes + kThreads - 1) / kThreads;

    cudaError_t err = cudaMemsetAsync(ws.counts, 0, local_experts * sizeof(int32_t), stream);
    if (err != cudaSuccess)
        return err;
    err = cudaMemsetAsync(ws.write_offsets, 0, local_experts * sizeof(int32_t), stream);
    if (err != cudaSuccess)
        return err;
    err = cudaMemsetAsync(ws.route_rows, 0xff, max_routes * sizeof(int32_t), stream);
    if (err != cudaSuccess)
        return err;

    count_routes_kernel<<<route_blocks, kThreads, 0, stream>>>(
        static_cast<const int32_t*>(desc->topk_ids.data),
        num_tokens,
        p.top_k,
        p.local_expert_offset,
        p.local_num_experts,
        ws.counts
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    prefix_sum_kernel<<<1, 1, 0, stream>>>(ws.counts, ws.indptr, local_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    fill_route_rows_kernel<<<route_blocks, kThreads, 0, stream>>>(
        static_cast<const int32_t*>(desc->topk_ids.data),
        static_cast<const float*>(desc->topk_weights.data),
        num_tokens,
        p.top_k,
        p.local_expert_offset,
        p.local_num_experts,
        ws.indptr,
        ws.write_offsets,
        ws.route_rows,
        ws.row_tokens,
        ws.row_scales
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    const uint64_t hidden_work = static_cast<uint64_t>(max_routes) * p.hidden_size;
    const uint32_t hidden_blocks = static_cast<uint32_t>((hidden_work + kThreads - 1) / kThreads);
    scatter_hidden_kernel<<<hidden_blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(desc->hidden.data),
        ws.route_rows,
        num_tokens,
        p.top_k,
        p.hidden_size,
        ws.expanded_hidden
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    const size_t gemm1_bytes
        = static_cast<size_t>(max_routes) * 2u * p.intermediate_size * sizeof(__nv_bfloat16);
    err = cudaMemsetAsync(ws.gemm1_out, 0, gemm1_bytes, stream);
    if (err != cudaSuccess)
        return err;
    prepare_group_gemm_args_kernel<<<
        (local_experts + kThreads - 1) / kThreads,
        kThreads,
        0,
        stream>>>(
        ws.indptr,
        local_experts,
        p.hidden_size,
        2u * p.intermediate_size,
        ws.expanded_hidden,
        static_cast<const __nv_bfloat16*>(desc->gate_up_weight.data),
        ws.gemm1_out,
        ws.problems,
        ws.x_ptrs,
        ws.w_ptrs,
        ws.y_ptrs,
        ws.x_ld,
        ws.w_ld,
        ws.y_ld
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;
    err = flashinfer::group_gemm::CutlassSegmentGEMMRun<cutlass::bfloat16_t>(
        desc->workspace.data,
        static_cast<size_t>(desc->workspace.shape[0]),
        ws.problems,
        local_experts,
        ws.x_ptrs,
        ws.w_ptrs,
        ws.y_ptrs,
        ws.x_ld,
        ws.w_ld,
        ws.y_ld,
        true,
        stream
    );
    if (err != cudaSuccess)
        return err;

    const uint64_t act_work = static_cast<uint64_t>(max_routes) * p.intermediate_size;
    const uint32_t act_blocks = static_cast<uint32_t>((act_work + kThreads - 1) / kThreads);
    swiglu_kernel<<<act_blocks, kThreads, 0, stream>>>(
        ws.gemm1_out,
        max_routes,
        p.intermediate_size,
        ws.act
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    const size_t gemm2_bytes
        = static_cast<size_t>(max_routes) * p.hidden_size * sizeof(__nv_bfloat16);
    err = cudaMemsetAsync(ws.gemm2_out, 0, gemm2_bytes, stream);
    if (err != cudaSuccess)
        return err;
    prepare_group_gemm_args_kernel<<<
        (local_experts + kThreads - 1) / kThreads,
        kThreads,
        0,
        stream>>>(
        ws.indptr,
        local_experts,
        p.intermediate_size,
        p.hidden_size,
        ws.act,
        static_cast<const __nv_bfloat16*>(desc->down_weight.data),
        ws.gemm2_out,
        ws.problems,
        ws.x_ptrs,
        ws.w_ptrs,
        ws.y_ptrs,
        ws.x_ld,
        ws.w_ld,
        ws.y_ld
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;
    err = flashinfer::group_gemm::CutlassSegmentGEMMRun<cutlass::bfloat16_t>(
        desc->workspace.data,
        static_cast<size_t>(desc->workspace.shape[0]),
        ws.problems,
        local_experts,
        ws.x_ptrs,
        ws.w_ptrs,
        ws.y_ptrs,
        ws.x_ld,
        ws.w_ld,
        ws.y_ld,
        true,
        stream
    );
    if (err != cudaSuccess)
        return err;

    const uint64_t final_work = static_cast<uint64_t>(num_tokens) * p.hidden_size;
    const uint32_t final_blocks = static_cast<uint32_t>((final_work + kThreads - 1) / kThreads);
    finalize_kernel<<<final_blocks, kThreads, 0, stream>>>(
        ws.route_rows,
        ws.row_scales,
        ws.gemm2_out,
        num_tokens,
        p.top_k,
        p.hidden_size,
        static_cast<__nv_bfloat16*>(desc->out.data)
    );
    return cudaGetLastError();
}

} // namespace

extern "C" {

qsfi_status
qsfi_moe_plan_create(qsfi_context* ctx, const qsfi_moe_plan_desc* desc, qsfi_moe_plan** out)
{
    if (out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    qsfi_status status = validate_plan_desc(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;
    qsfi_moe_plan* plan = new (std::nothrow) qsfi_moe_plan;
    if (plan == nullptr)
        return set_out_of_memory(ctx, "allocating MoE plan");
    plan->desc = *desc;
    *out = plan;
    return QSFI_STATUS_OK;
}

void qsfi_moe_plan_destroy(qsfi_moe_plan* plan)
{
    delete plan;
}

qsfi_status qsfi_moe_workspace_size(
    qsfi_context* ctx, const qsfi_moe_plan* plan, uint32_t num_tokens, size_t* device_bytes
)
{
    if (device_bytes == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *device_bytes = 0;
    if (ctx == nullptr || plan == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (plan->desc.backend != QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16) {
        return set_unsupported(ctx, "MoE workspace query is only implemented for staged BF16");
    }
    moe_workspace layout {};
    qsfi_status status = compute_workspace_layout(ctx, plan->desc, num_tokens, nullptr, &layout);
    if (status != QSFI_STATUS_OK)
        return status;
    *device_bytes = layout.bytes;
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_moe_execute_bf16(
    qsfi_context* ctx, const qsfi_moe_plan* plan, const qsfi_moe_bf16_execute_desc* desc
)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;

    moe_workspace layout {};
    status = validate_bf16_execute(ctx, plan, desc, &layout);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_moe_routes(
        ctx,
        plan->desc,
        desc,
        static_cast<uint32_t>(static_cast<uint64_t>(desc->num_tokens) * plan->desc.top_k)
    );
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        cudaError_t err = launch_bf16_moe(ctx, plan->desc, desc, layout);
        return set_cuda_error(ctx, err, "qsfi_moe_execute_bf16");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "qsfi_moe_execute_bf16", ex);
    }
}

qsfi_status qsfi_moe_execute_nvfp4(
    qsfi_context* ctx, const qsfi_moe_plan* plan, const qsfi_moe_nvfp4_execute_desc* desc
)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (plan == nullptr || desc == nullptr) {
        return set_invalid_arg(ctx, "plan/desc is null");
    }
    if (plan->desc.backend != QSFI_MOE_BACKEND_FLASHINFER_NVFP4) {
        return set_invalid_arg(ctx, "qsfi_moe_execute_nvfp4 requires an NVFP4 MoE plan");
    }
    return set_unsupported(ctx, "NVFP4 MoE execution is declared but not implemented yet");
}

} // extern "C"
