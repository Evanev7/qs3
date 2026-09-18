// FlashInfer-owned dense W4A4 lane. No Python, JIT or device memory ownership.
#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"
#include "qsfi_internal.h"
#include "qsfi_macros.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <climits>
#include <memory>
#include <new>

namespace flashinfer::gemm {
#define QSFI_INSTANTIATE_NVFP4(N)                                                                  \
    INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, N, 128, 1, 1, 1, _1SM, true)
QSFI_NVFP4_TILES(QSFI_INSTANTIATE_NVFP4)
#undef QSFI_INSTANTIATE_NVFP4
}

using Fp4Launch = decltype(&flashinfer::gemm::genericFp4GemmKernelLauncher<
                           __nv_bfloat16,
                           cute::Int<128>,
                           cute::Int<32>,
                           cute::Int<128>,
                           cute::Int<1>,
                           cute::Int<1>,
                           cute::Int<1>,
                           flashinfer::gemm::_1SM,
                           true>);

struct qsfi_nvfp4_plan {
    qsfi_context* owner;
    qsfi_nvfp4_plan_desc shape;
    Fp4Launch launch;
    flashinfer::gemm::CutlassGemmConfig config;
    size_t workspace_bytes;
};

namespace qsfi_nvfp4_detail {
bool aligned(const void* p, size_t alignment)
{
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % alignment == 0;
}

int64_t scale_count(int64_t rows, int64_t k)
{
    return ((rows + 127) / 128) * 128 * (k / 16);
}

bool valid_dimensions(int64_t rows, int64_t k)
{
    // The upstream quantizer rounds row counts in signed int arithmetic.
    return rows > 0 && rows <= INT_MAX - 127 && k > 0 && k <= INT_MAX && k % 128 == 0;
}

qsfi_status
matrix(qsfi_context* ctx, const qsfi_tensor2& t, qsfi_dtype dtype, int64_t rows, int64_t cols)
{
    auto status = validate_tensor(ctx, t, "NVFP4 matrix", dtype, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    if (t.shape[0] != rows || t.shape[1] != cols || t.stride[0] != cols || t.stride[1] != 1
        || !aligned(t.data, 16))
        return set_invalid_arg(ctx, "NVFP4 matrix shape/contiguity/16-byte alignment mismatch");
    return QSFI_STATUS_OK;
}

qsfi_status
vector(qsfi_context* ctx, const qsfi_tensor1& t, qsfi_dtype dtype, int64_t count, size_t alignment)
{
    auto status = validate_tensor(ctx, t, "NVFP4 scale storage", dtype, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    if (t.shape[0] != count || t.stride[0] != 1 || !aligned(t.data, alignment))
        return set_invalid_arg(ctx, "NVFP4 scale extent/stride/alignment mismatch");
    return QSFI_STATUS_OK;
}

__global__ void
swizzle_scales(const uint8_t* src, uint8_t* dst, int64_t rows, int64_t cols, size_t count)
{
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        const size_t r = i / cols, c = i % cols;
        // [R/128,4,32,C/4,4] -> [R/128,C/4,32,4,4].
        const size_t offset = (r / 128) * (128 * cols) + (c / 4) * 512 + (r % 32) * 16
            + ((r % 128) / 32) * 4 + c % 4;
        dst[offset] = r < size_t(rows) ? src[i] : 0;
    }
}
} // namespace qsfi_nvfp4_detail

using namespace qsfi_nvfp4_detail;

extern "C" {
qsfi_status qsfi_nvfp4_plan_create(
    qsfi_context* ctx,
    const qsfi_nvfp4_plan_desc* desc,
    qsfi_nvfp4_plan** out,
    size_t* workspace_bytes
)
{
    if (out == nullptr || workspace_bytes == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    *workspace_bytes = 0;
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (desc == nullptr || !valid_dimensions(desc->rows, desc->in_features)
        || !valid_dimensions(desc->out_features, desc->in_features) || desc->out_features % 8)
        return set_invalid_arg(ctx, "invalid NVFP4 dimensions or uncompiled tactic");
    auto status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        using namespace flashinfer::gemm;
        std::unique_ptr<qsfi_nvfp4_plan> p(new qsfi_nvfp4_plan {});
        p->owner = ctx;
        p->shape = *desc;
#define QSFI_SELECT_NVFP4(ID, N, STREAM_K, LAUNCHER)                                               \
    case ID:                                                                                       \
        p->config = CutlassGemmConfig(                                                             \
            CutlassTileConfigSM120::CtaShape128x##N##x64B,                                         \
            MainloopScheduleType::AUTO,                                                            \
            EpilogueScheduleType::AUTO,                                                            \
            ClusterShape::ClusterShape_1x1x1,                                                      \
            true,                                                                                  \
            STREAM_K                                                                               \
        );                                                                                         \
        p->launch = &LAUNCHER<                                                                     \
            __nv_bfloat16,                                                                         \
            cute::Int<128>,                                                                        \
            cute::Int<N>,                                                                          \
            cute::Int<128>,                                                                        \
            cute::Int<1>,                                                                          \
            cute::Int<1>,                                                                          \
            cute::Int<1>,                                                                          \
            _1SM,                                                                                  \
            true>;                                                                                 \
        break;
        switch (desc->tactic) {
            QSFI_NVFP4_TACTICS(QSFI_SELECT_NVFP4)
        default:
            return set_invalid_arg(ctx, "NVFP4 tactic is not enabled in this build");
        }
#undef QSFI_SELECT_NVFP4
        p->workspace_bytes = p->launch(
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            desc->rows,
            desc->out_features,
            desc->in_features,
            1,
            p->config,
            nullptr,
            0,
            ctx->stream,
            nullptr
        );
        *workspace_bytes = p->workspace_bytes;
        *out = p.release();
        return QSFI_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return set_out_of_memory(ctx, "NVFP4 plan allocation");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "NVFP4 plan preparation", ex);
    }
}

void qsfi_nvfp4_plan_destroy(qsfi_nvfp4_plan* plan)
{
    delete plan;
}

qsfi_status
qsfi_nvfp4_execute(qsfi_context* ctx, const qsfi_nvfp4_plan* plan, const qsfi_nvfp4_execute_desc* d)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (plan == nullptr || d == nullptr || plan->owner != ctx)
        return set_invalid_arg(ctx, "NVFP4 plan/descriptor/context mismatch");
    const auto s = plan->shape;
    auto status = matrix(ctx, d->x, QSFI_DTYPE_NVFP4_E2M1, s.rows, s.in_features);
    if (status != QSFI_STATUS_OK)
        return status;
    status = matrix(ctx, d->weight, QSFI_DTYPE_NVFP4_E2M1, s.out_features, s.in_features);
    if (status != QSFI_STATUS_OK)
        return status;
    status = matrix(ctx, d->out, QSFI_DTYPE_BF16, s.rows, s.out_features);
    if (status != QSFI_STATUS_OK)
        return status;
    status = vector(ctx, d->x_scales, QSFI_DTYPE_FP8_E4M3, scale_count(s.rows, s.in_features), 16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = vector(
        ctx,
        d->weight_scales,
        QSFI_DTYPE_FP8_E4M3,
        scale_count(s.out_features, s.in_features),
        16
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = vector(ctx, d->alpha, QSFI_DTYPE_F32, 1, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    if (d->workspace_bytes < plan->workspace_bytes
        || (d->workspace_bytes && d->workspace == nullptr)
        || (d->workspace != nullptr && !aligned(d->workspace, 256)))
        return set_invalid_arg(ctx, "NVFP4 workspace capacity/alignment mismatch");
    status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    try {
        plan->launch(
            d->out.data,
            d->x.data,
            d->weight.data,
            d->x_scales.data,
            d->weight_scales.data,
            static_cast<const float*>(d->alpha.data),
            s.rows,
            s.out_features,
            s.in_features,
            1,
            plan->config,
            static_cast<char*>(d->workspace),
            d->workspace_bytes,
            ctx->stream,
            nullptr
        );
        return set_cuda_error(ctx, cudaGetLastError(), "NVFP4 GEMM launch");
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "NVFP4 GEMM", ex);
    }
}

qsfi_status qsfi_nvfp4_quantize(qsfi_context* ctx, const qsfi_nvfp4_quantize_desc* d)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (d == nullptr || !valid_dimensions(d->x.shape[0], d->x.shape[1]))
        return set_invalid_arg(ctx, "invalid NVFP4 quantization dimensions");
    const int m = d->x.shape[0], k = d->x.shape[1];
    auto status = matrix(ctx, d->x, QSFI_DTYPE_BF16, m, k);
    if (status != QSFI_STATUS_OK)
        return status;
    status = matrix(ctx, d->out, QSFI_DTYPE_NVFP4_E2M1, m, k);
    if (status != QSFI_STATUS_OK)
        return status;
    status = vector(ctx, d->scales, QSFI_DTYPE_FP8_E4M3, scale_count(m, k), 16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = vector(ctx, d->quant_multiplier, QSFI_DTYPE_F32, 1, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    using namespace tensorrt_llm;
    using namespace tensorrt_llm::kernels;
    const int threads = std::min(k / CVT_FP16_TO_FP4_ELTS_PER_THREAD, 512);
    const int blocks = std::min((m + 127) / 128 * 128, 65535);
    quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, __nv_bfloat16, 16, false>
        <<<blocks, threads, 0, ctx->stream>>>(
            1,
            m,
            k,
            k,
            static_cast<const __nv_bfloat16*>(d->x.data),
            static_cast<const float*>(d->quant_multiplier.data),
            d->out.data,
            static_cast<uint32_t*>(d->scales.data),
            QuantizationSFLayout::SWIZZLED_128x4
        );
    return set_cuda_error(ctx, cudaGetLastError(), "NVFP4 quantization");
}

qsfi_status
qsfi_nvfp4_swizzle_scales(qsfi_context* ctx, const qsfi_tensor2* scales, const qsfi_tensor1* out)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (scales == nullptr || out == nullptr || scales->shape[1] > INT_MAX / 16
        || scales->shape[1] <= 0 || !valid_dimensions(scales->shape[0], scales->shape[1] * 16))
        return set_invalid_arg(ctx, "invalid NVFP4 scale dimensions");
    const int64_t rows = scales->shape[0], cols = scales->shape[1];
    auto status = matrix(ctx, *scales, QSFI_DTYPE_FP8_E4M3, rows, cols);
    if (status != QSFI_STATUS_OK)
        return status;
    const size_t count = scale_count(rows, cols * 16);
    status = vector(ctx, *out, QSFI_DTYPE_FP8_E4M3, count, 16);
    if (status != QSFI_STATUS_OK)
        return status;
    if (scales->data == out->data)
        return set_invalid_arg(ctx, "scale swizzle cannot be in-place");
    status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    swizzle_scales<<<blocks, 256, 0, ctx->stream>>>(
        static_cast<const uint8_t*>(scales->data),
        static_cast<uint8_t*>(out->data),
        rows,
        cols,
        count
    );
    return set_cuda_error(ctx, cudaGetLastError(), "NVFP4 scale swizzle");
}
} // extern "C"
