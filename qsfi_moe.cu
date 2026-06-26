#include "qsfi_build_constants.h"
#include "qsfi_internal.h"
#include "qsfi_macros.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef QSFI_BUILD_TRTLLM_GEN_MOE
#include <cuda_bf16.h>
#include <flashinfer/trtllm/fused_moe/runner.h>
#include <tensorrt_llm/kernels/quantization.h>

namespace tensorrt_llm {
namespace kernels {

    template <>
    void invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
        uint32_t m,
        uint32_t n,
        const __nv_bfloat16* input,
        float globalScaleInv,
        int32_t* expandedIdxToPermutedIdx,
        uint8_t* weightOutput,
        uint8_t* scaleOutput,
        float* perTokenScaleOutput,
        QuantizationSFLayout sfLayout,
        cudaStream_t stream
    )
    {
        (void)m;
        (void)n;
        (void)input;
        (void)globalScaleInv;
        (void)expandedIdxToPermutedIdx;
        (void)weightOutput;
        (void)scaleOutput;
        (void)perTokenScaleOutput;
        (void)sfLayout;
        (void)stream;
        throw std::runtime_error("FlashInfer TRTLLM NVFP4 quantization is not linked in qsfi");
    }

} // namespace kernels
} // namespace tensorrt_llm
#endif

namespace {

template <typename Tensor> bool is_contiguous(const Tensor& tensor)
{
    constexpr uint32_t rank = sizeof(tensor.shape) / sizeof(tensor.shape[0]);
    int64_t expected = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        if (tensor.stride[i] != expected)
            return false;
        expected *= tensor.shape[i];
    }
    return true;
}

template <typename Tensor>
qsfi_status validate_contiguous(qsfi_context* ctx, const Tensor& tensor, const char* name)
{
    if (!is_contiguous(tensor)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s must be contiguous",
            name
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status
validate_shape1(qsfi_context* ctx, const qsfi_tensor1& tensor, const char* name, int64_t d0)
{
    if (tensor.shape[0] != d0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s shape mismatch",
            name
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_shape2(
    qsfi_context* ctx, const qsfi_tensor2& tensor, const char* name, int64_t d0, int64_t d1
)
{
    if (tensor.shape[0] != d0 || tensor.shape[1] != d1) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s shape mismatch",
            name
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_shape3(
    qsfi_context* ctx,
    const qsfi_tensor3& tensor,
    const char* name,
    int64_t d0,
    int64_t d1,
    int64_t d2
)
{
    if (tensor.shape[0] != d0 || tensor.shape[1] != d1 || tensor.shape[2] != d2) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s shape mismatch",
            name
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_shape4(
    qsfi_context* ctx,
    const qsfi_tensor4& tensor,
    const char* name,
    int64_t d0,
    int64_t d1,
    int64_t d2,
    int64_t d3
)
{
    if (tensor.shape[0] != d0 || tensor.shape[1] != d1 || tensor.shape[2] != d2
        || tensor.shape[3] != d3) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s shape mismatch",
            name
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_routing_config(qsfi_context* ctx, const qsfi_routing_config& routing)
{
    if (routing.num_experts == 0 || routing.top_k == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "routing num_experts and top_k must be non-zero"
        );
    }
    if (routing.top_k > routing.num_experts) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "routing top_k must be <= num_experts"
        );
    }
    switch (routing.method) {
    case QSFI_ROUTING_METHOD_DEFAULT:
    case QSFI_ROUTING_METHOD_RENORMALIZE:
    case QSFI_ROUTING_METHOD_RENORMALIZE_NAIVE:
    case QSFI_ROUTING_METHOD_TOPK:
        break;
    case QSFI_ROUTING_METHOD_DEEPSEEK_V3:
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "DeepSeekV3 routing requires routing bias, which qsfi MoE does not expose"
        );
    case QSFI_ROUTING_METHOD_LLAMA4:
        if (routing.top_k != 1) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "Llama4 routing requires top_k == 1"
            );
        }
        break;
    default:
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "unsupported FlashInfer MoE routing method"
        );
    }
    if (routing.apply_router_weight_on_input != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "applying router weights on input is not exposed in qsfi MoE"
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_execution_common(
    qsfi_context* ctx,
    uint32_t num_tokens,
    qsfi_activation activation,
    qsfi_dtype accum_dtype,
    const qsfi_routing_config& routing
)
{
    if (num_tokens == 0)
        return QSFI_STATUS_OK;
    if (activation != QSFI_ACTIVATION_SWIGLU) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qsfi MoE currently exposes only FlashInfer SwiGLU MoE"
        );
    }
    if (accum_dtype != QSFI_DTYPE_F32) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "qsfi MoE currently exposes f32 accumulation only"
        );
    }
    return validate_routing_config(ctx, routing);
}

qsfi_status validate_bf16_weights(qsfi_context* ctx, const qsfi_trtllm_bf16_moe_weights& weights)
{
    if (weights.hidden_size == 0 || weights.intermediate_size == 0
        || weights.local_num_experts == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "BF16 MoE hidden_size, intermediate_size, and local_num_experts must be non-zero"
        );
    }
    if (weights.local_expert_offset != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "partial local expert ranges are not exposed in qsfi MoE yet"
        );
    }
    if (weights.hidden_size % 64 != 0 || weights.intermediate_size % 64 != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "BF16 TRTLLM MoE weights require hidden_size and intermediate_size multiples of 64"
        );
    }

    qsfi_status status
        = validate_tensor(ctx, weights.gate_up_weight, "gate_up_weight", QSFI_DTYPE_BF16, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.gate_up_weight, "gate_up_weight");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape4(
        ctx,
        weights.gate_up_weight,
        "gate_up_weight",
        weights.local_num_experts,
        weights.hidden_size / 64,
        2 * weights.intermediate_size,
        64
    );
    if (status != QSFI_STATUS_OK)
        return status;

    status = validate_tensor(ctx, weights.down_weight, "down_weight", QSFI_DTYPE_BF16, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.down_weight, "down_weight");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape4(
        ctx,
        weights.down_weight,
        "down_weight",
        weights.local_num_experts,
        weights.intermediate_size / 64,
        weights.hidden_size,
        64
    );
}

qsfi_status validate_nvfp4_weights(qsfi_context* ctx, const qsfi_trtllm_nvfp4_moe_weights& weights)
{
    if (weights.hidden_size == 0 || weights.intermediate_size == 0
        || weights.local_num_experts == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "NVFP4 MoE hidden_size, intermediate_size, and local_num_experts must be non-zero"
        );
    }
    if (weights.local_expert_offset != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "partial local expert ranges are not exposed in qsfi MoE yet"
        );
    }
    if (weights.hidden_size % 16 != 0 || weights.intermediate_size % 16 != 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "NVFP4 TRTLLM MoE weights require hidden_size and intermediate_size multiples of 16"
        );
    }

    qsfi_status status
        = validate_tensor(ctx, weights.gate_up_weight, "gate_up_weight", QSFI_DTYPE_NVFP4_E2M1, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.gate_up_weight, "gate_up_weight");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape3(
        ctx,
        weights.gate_up_weight,
        "gate_up_weight",
        weights.local_num_experts,
        2 * weights.intermediate_size,
        weights.hidden_size / 2
    );
    if (status != QSFI_STATUS_OK)
        return status;

    status = validate_tensor(ctx, weights.down_weight, "down_weight", QSFI_DTYPE_NVFP4_E2M1, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.down_weight, "down_weight");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape3(
        ctx,
        weights.down_weight,
        "down_weight",
        weights.local_num_experts,
        weights.hidden_size,
        weights.intermediate_size / 2
    );
    if (status != QSFI_STATUS_OK)
        return status;

    status = validate_tensor(ctx, weights.gate_up_scale, "gate_up_scale", QSFI_DTYPE_FP8_E4M3, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.gate_up_scale, "gate_up_scale");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape3(
        ctx,
        weights.gate_up_scale,
        "gate_up_scale",
        weights.local_num_experts,
        2 * weights.intermediate_size,
        weights.hidden_size / 16
    );
    if (status != QSFI_STATUS_OK)
        return status;

    status = validate_tensor(ctx, weights.down_scale, "down_scale", QSFI_DTYPE_FP8_E4M3, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.down_scale, "down_scale");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape3(
        ctx,
        weights.down_scale,
        "down_scale",
        weights.local_num_experts,
        weights.hidden_size,
        weights.intermediate_size / 16
    );
    if (status != QSFI_STATUS_OK)
        return status;

    status = validate_tensor(ctx, weights.output1_scale, "output1_scale", QSFI_DTYPE_F32, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.output1_scale, "output1_scale");
    if (status != QSFI_STATUS_OK)
        return status;
    status
        = validate_shape1(ctx, weights.output1_scale, "output1_scale", weights.local_num_experts);
    if (status != QSFI_STATUS_OK)
        return status;
    status
        = validate_tensor(ctx, weights.output1_gate_scale, "output1_gate_scale", QSFI_DTYPE_F32, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.output1_gate_scale, "output1_gate_scale");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape1(
        ctx,
        weights.output1_gate_scale,
        "output1_gate_scale",
        weights.local_num_experts
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, weights.output2_scale, "output2_scale", QSFI_DTYPE_F32, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, weights.output2_scale, "output2_scale");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape1(ctx, weights.output2_scale, "output2_scale", weights.local_num_experts);
}

qsfi_status validate_bf16_io(
    qsfi_context* ctx,
    const qsfi_tensor2& hidden_states,
    const qsfi_tensor2& out,
    uint32_t num_tokens,
    uint32_t hidden_size
)
{
    qsfi_status status = validate_tensor(ctx, hidden_states, "hidden_states", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, hidden_states, "hidden_states");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape2(ctx, hidden_states, "hidden_states", num_tokens, hidden_size);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, out, "out", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, out, "out");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape2(ctx, out, "out", num_tokens, hidden_size);
}

qsfi_status validate_nvfp4_io(
    qsfi_context* ctx,
    const qsfi_tensor2& hidden_states,
    const qsfi_tensor2& hidden_states_scale,
    const qsfi_tensor2& out,
    uint32_t num_tokens,
    uint32_t hidden_size
)
{
    qsfi_status status
        = validate_tensor(ctx, hidden_states, "hidden_states", QSFI_DTYPE_NVFP4_E2M1, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, hidden_states, "hidden_states");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape2(ctx, hidden_states, "hidden_states", num_tokens, hidden_size / 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status
        = validate_tensor(ctx, hidden_states_scale, "hidden_states_scale", QSFI_DTYPE_FP8_E4M3, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, hidden_states_scale, "hidden_states_scale");
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape2(
        ctx,
        hidden_states_scale,
        "hidden_states_scale",
        num_tokens,
        hidden_size / 16
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, out, "out", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, out, "out");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape2(ctx, out, "out", num_tokens, hidden_size);
}

qsfi_status validate_logits(
    qsfi_context* ctx, const qsfi_tensor2& routing_logits, uint32_t num_tokens, uint32_t num_experts
)
{
    qsfi_status status = validate_tensor(ctx, routing_logits, "routing_logits", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, routing_logits, "routing_logits");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape2(ctx, routing_logits, "routing_logits", num_tokens, num_experts);
}

qsfi_status validate_packed_topk(
    qsfi_context* ctx, const qsfi_tensor2& packed_topk, uint32_t num_tokens, uint32_t top_k
)
{
    qsfi_status status = validate_tensor(ctx, packed_topk, "packed_topk", QSFI_DTYPE_I32, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_contiguous(ctx, packed_topk, "packed_topk");
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape2(ctx, packed_topk, "packed_topk", num_tokens, top_k);
}

#ifdef QSFI_BUILD_TRTLLM_GEN_MOE

namespace btg = batchedGemm::trtllm::gen;
namespace trtllm_moe = tensorrt_llm::kernels::trtllmgen_moe::MoE;
namespace trtllm_routing = tensorrt_llm::kernels::trtllmgen_moe::Routing;

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    ~DeviceBuffer()
    {
        release();
    }

    qsfi_status allocate(qsfi_context* ctx, size_t bytes, const char* name)
    {
        release();
        bytes_ = bytes;
        if (bytes == 0)
            return QSFI_STATUS_OK;
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess) {
            ptr_ = nullptr;
            bytes_ = 0;
            return set_cuda_error(ctx, err, name);
        }
        return QSFI_STATUS_OK;
    }

    void* data() const
    {
        return ptr_;
    }

    template <typename T> T* as() const
    {
        return static_cast<T*>(ptr_);
    }

private:
    void release()
    {
        if (ptr_ != nullptr)
            cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

int32_t next_power_of_two(float value)
{
    int32_t n = static_cast<int32_t>(std::ceil(value));
    if (n <= 1)
        return 1;
    if ((n & (n - 1)) == 0)
        return n;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

std::set<int32_t> selected_tile_sizes(
    const std::vector<int32_t>& supported_tile_sizes,
    int32_t num_tokens,
    int32_t top_k,
    int32_t local_num_experts
)
{
    const float avg_tokens_per_expert
        = static_cast<float>(num_tokens * top_k) / static_cast<float>(local_num_experts);
    const int32_t tile = std::clamp(
        next_power_of_two(avg_tokens_per_expert),
        supported_tile_sizes.front(),
        supported_tile_sizes.back()
    );
    auto it = std::find(supported_tile_sizes.begin(), supported_tile_sizes.end(), tile);
    if (it == supported_tile_sizes.end())
        it = supported_tile_sizes.begin();

    std::set<int32_t> selected;
    selected.insert(*it);
    if (std::next(it) != supported_tile_sizes.end()) {
        selected.insert(*std::next(it));
        if (std::next(std::next(it)) != supported_tile_sizes.end())
            selected.insert(*std::next(std::next(it)));
    }
    if (it != supported_tile_sizes.begin())
        selected.insert(*std::prev(it));
    return selected;
}

int32_t select_bf16_tile_size(int32_t num_tokens, int32_t top_k, int32_t local_num_experts)
{
    static const std::vector<int32_t> supported = { 8, 16, 32, 64, 128 };
    const std::set<int32_t> selected
        = selected_tile_sizes(supported, num_tokens, top_k, local_num_experts);
    return *selected.begin();
}

trtllm_routing::RoutingMethodType to_flashinfer_routing(qsfi_routing_method method)
{
    return static_cast<trtllm_routing::RoutingMethodType>(static_cast<int64_t>(method));
}

qsfi_status checked_i32(qsfi_context* ctx, uint32_t value, const char* name, int32_t* out)
{
    if (value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s exceeds int32 range",
            name
        );
    }
    *out = static_cast<int32_t>(value);
    return QSFI_STATUS_OK;
}

struct Bf16MoeRun {
    const qsfi_tensor2* hidden_states;
    const qsfi_tensor2* routing_logits;
    const qsfi_tensor2* packed_topk;
    const qsfi_tensor2* out;
    const qsfi_trtllm_bf16_moe_weights* weights;
    const qsfi_routing_config* routing;
    uint32_t num_tokens;
    float clamp_limit;
};

qsfi_status run_flashinfer_bf16_moe(qsfi_context* ctx, const Bf16MoeRun& run)
{
    int32_t num_tokens = 0;
    int32_t num_experts = 0;
    int32_t top_k = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t local_expert_offset = 0;
    int32_t local_num_experts = 0;
    qsfi_status status = checked_i32(ctx, run.num_tokens, "num_tokens", &num_tokens);
    if (status != QSFI_STATUS_OK)
        return status;
    status = checked_i32(ctx, run.routing->num_experts, "num_experts", &num_experts);
    if (status != QSFI_STATUS_OK)
        return status;
    status = checked_i32(ctx, run.routing->top_k, "top_k", &top_k);
    if (status != QSFI_STATUS_OK)
        return status;
    status = checked_i32(ctx, run.weights->hidden_size, "hidden_size", &hidden_size);
    if (status != QSFI_STATUS_OK)
        return status;
    status
        = checked_i32(ctx, run.weights->intermediate_size, "intermediate_size", &intermediate_size);
    if (status != QSFI_STATUS_OK)
        return status;
    status = checked_i32(
        ctx,
        run.weights->local_expert_offset,
        "local_expert_offset",
        &local_expert_offset
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status
        = checked_i32(ctx, run.weights->local_num_experts, "local_num_experts", &local_num_experts);
    if (status != QSFI_STATUS_OK)
        return status;

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
        return set_cuda_error(ctx, err, "cudaGetDevice");

    const int32_t tile_tokens_dim = select_bf16_tile_size(num_tokens, top_k, local_num_experts);
    const int32_t max_padded_tokens = trtllm_routing::getMaxPermutedPaddedCount(
        num_tokens,
        top_k,
        num_experts,
        tile_tokens_dim
    );
    const int32_t max_ctas
        = trtllm_routing::getMaxNumCtasInBatchDim(num_tokens, top_k, num_experts, tile_tokens_dim);
    const int32_t expert_count_histogram_len = std::max(num_experts * 2, 256 * 2);

    DeviceBuffer num_tokens_per_expert;
    DeviceBuffer total_num_padded_tokens;
    DeviceBuffer expanded_idx_to_permuted_idx;
    DeviceBuffer permuted_idx_to_token_idx;
    DeviceBuffer expert_weights;
    DeviceBuffer routing_expert_indexes;
    DeviceBuffer expert_count_histogram;
    DeviceBuffer cta_idx_xy_to_batch_idx;
    DeviceBuffer cta_idx_xy_to_mn_limit;
    DeviceBuffer num_non_exiting_ctas;
    DeviceBuffer gemm1_output;
    DeviceBuffer activation_output;
    DeviceBuffer gemm2_output;
    DeviceBuffer bmm1_workspace;
    DeviceBuffer bmm2_workspace;
    DeviceBuffer clamp_limit;

    status = num_tokens_per_expert.allocate(
        ctx,
        static_cast<size_t>(num_experts) * sizeof(int32_t),
        "cudaMalloc MoE num_tokens_per_expert"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = total_num_padded_tokens
                 .allocate(ctx, sizeof(int32_t), "cudaMalloc MoE total_num_padded_tokens");
    if (status != QSFI_STATUS_OK)
        return status;
    status = expanded_idx_to_permuted_idx.allocate(
        ctx,
        static_cast<size_t>(num_tokens) * top_k * sizeof(int32_t),
        "cudaMalloc MoE expanded_idx_to_permuted_idx"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = permuted_idx_to_token_idx.allocate(
        ctx,
        static_cast<size_t>(max_padded_tokens) * sizeof(int32_t),
        "cudaMalloc MoE permuted_idx_to_token_idx"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = expert_weights.allocate(
        ctx,
        static_cast<size_t>(num_tokens) * top_k * sizeof(uint16_t),
        "cudaMalloc MoE expert_weights"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = routing_expert_indexes.allocate(
        ctx,
        run.packed_topk == nullptr ? static_cast<size_t>(num_tokens) * top_k * sizeof(int32_t) : 0,
        "cudaMalloc MoE routing_expert_indexes"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = expert_count_histogram.allocate(
        ctx,
        static_cast<size_t>(expert_count_histogram_len) * sizeof(int32_t),
        "cudaMalloc MoE expert_count_histogram"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = cta_idx_xy_to_batch_idx.allocate(
        ctx,
        static_cast<size_t>(max_ctas) * sizeof(int32_t),
        "cudaMalloc MoE cta_idx_xy_to_batch_idx"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = cta_idx_xy_to_mn_limit.allocate(
        ctx,
        static_cast<size_t>(max_ctas) * sizeof(int32_t),
        "cudaMalloc MoE cta_idx_xy_to_mn_limit"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = num_non_exiting_ctas
                 .allocate(ctx, sizeof(int32_t), "cudaMalloc MoE num_non_exiting_ctas");
    if (status != QSFI_STATUS_OK)
        return status;
    status = gemm1_output.allocate(
        ctx,
        static_cast<size_t>(max_padded_tokens) * intermediate_size * sizeof(uint16_t),
        "cudaMalloc MoE gemm1_output"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = activation_output.allocate(
        ctx,
        static_cast<size_t>(max_padded_tokens) * intermediate_size * sizeof(uint16_t),
        "cudaMalloc MoE activation_output"
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = gemm2_output.allocate(
        ctx,
        static_cast<size_t>(max_padded_tokens) * hidden_size * sizeof(uint16_t),
        "cudaMalloc MoE gemm2_output"
    );
    if (status != QSFI_STATUS_OK)
        return status;

    if (run.clamp_limit > 0.0f) {
        status = clamp_limit.allocate(ctx, sizeof(float), "cudaMalloc MoE clamp_limit");
        if (status != QSFI_STATUS_OK)
            return status;
        err = cudaMemcpyAsync(
            clamp_limit.data(),
            &run.clamp_limit,
            sizeof(float),
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "cudaMemcpyAsync MoE clamp_limit");
    }

    trtllm_moe::MoERunnerArgs args;
    args.routing_logits = run.routing_logits != nullptr ? run.routing_logits->data : nullptr;
    args.hidden_states = run.hidden_states->data;
    args.gemm1_weights = run.weights->gate_up_weight.data;
    args.gemm2_weights = run.weights->down_weight.data;
    args.gemm1_clamp_limit = clamp_limit.as<float>();
    args.num_tokens = num_tokens;
    args.num_experts = num_experts;
    args.hidden_size = hidden_size;
    args.hidden_size_output = hidden_size;
    args.top_k = top_k;
    args.n_group = static_cast<int32_t>(run.routing->n_group);
    args.topk_group = static_cast<int32_t>(run.routing->topk_group);
    args.routed_scaling_factor = default_one(run.routing->routed_scaling_factor);
    args.intermediate_size = intermediate_size;
    args.local_expert_offset = local_expert_offset;
    args.local_num_experts = local_num_experts;
    args.mDtypeElt = btg::Dtype::Bfloat16;
    args.mDtypeExpW = btg::Dtype::Bfloat16;
    args.mDtypeOut = btg::Dtype::Bfloat16;
    args.mUseRoutingScalesOnInput = false;
    args.mUseDeepSeekFp8 = false;
    args.output = run.out->data;
    args.output_scale = nullptr;
    args.do_finalize = true;

    trtllm_moe::MoEWorkspace workspace;
    workspace.routing_expert_indexes = run.packed_topk != nullptr
        ? static_cast<int32_t*>(run.packed_topk->data)
        : routing_expert_indexes.as<int32_t>();
    workspace.permuted_idx_size = total_num_padded_tokens.as<int32_t>();
    workspace.total_num_padded_tokens = total_num_padded_tokens.as<int32_t>();
    workspace.total_max_padded_tokens = max_padded_tokens;
    workspace.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.as<int32_t>();
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx.as<int32_t>();
    workspace.permuted_idx_to_expanded_idx = nullptr;
    workspace.expert_weights = expert_weights.data();
    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx.as<int32_t>();
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit.as<int32_t>();
    workspace.num_non_exiting_ctas = num_non_exiting_ctas.as<int32_t>();
    workspace.gemm1_output = gemm1_output.data();
    workspace.gemm1_output_scale = nullptr;
    workspace.activation_output = activation_output.data();
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data();
    workspace.gemm2_output_scale = nullptr;
    workspace.ProjUpTileN = tile_tokens_dim;

    try {
        trtllm_routing::Runner routing_runner(tile_tokens_dim);
        routing_runner.run(
            run.packed_topk != nullptr ? nullptr : args.routing_logits,
            nullptr,
            num_tokens,
            num_experts,
            top_k,
            args.n_group,
            args.topk_group,
            local_expert_offset,
            local_num_experts,
            args.routed_scaling_factor,
            workspace.routing_expert_indexes,
            expert_count_histogram.as<int32_t>(),
            total_num_padded_tokens.as<int32_t>(),
            expanded_idx_to_permuted_idx.as<int32_t>(),
            nullptr,
            permuted_idx_to_token_idx.as<int32_t>(),
            nullptr,
            workspace.expert_weights,
            num_tokens_per_expert.as<int32_t>(),
            cta_idx_xy_to_batch_idx.as<int32_t>(),
            cta_idx_xy_to_mn_limit.as<int32_t>(),
            num_non_exiting_ctas.as<int32_t>(),
            args.mDtypeElt,
            btg::Dtype::Bfloat16,
            false,
            false,
            to_flashinfer_routing(run.routing->method),
            ctx->stream,
            btg::Dtype::Bfloat16,
            run.routing->renormalize != 0,
            nullptr
        );

        trtllm_moe::Runner moe_runner(
            btg::Dtype::Bfloat16,
            btg::Dtype::Bfloat16,
            false,
            tile_tokens_dim,
            trtllm_moe::ActivationType::Swiglu,
            true,
            batchedGemm::gemm::MatrixLayout::BlockMajorK,
            batchedGemm::gemm::BiasType::None
        );
        const int64_t config_index = moe_runner.getDefaultValidConfigIndex(
            top_k,
            hidden_size,
            intermediate_size,
            local_num_experts,
            num_tokens
        );
        const auto workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args, config_index);
        status = bmm1_workspace.allocate(
            ctx,
            static_cast<size_t>(std::get<0>(workspace_sizes)),
            "cudaMalloc MoE bmm1 workspace"
        );
        if (status != QSFI_STATUS_OK)
            return status;
        status = bmm2_workspace.allocate(
            ctx,
            static_cast<size_t>(std::get<1>(workspace_sizes)),
            "cudaMalloc MoE bmm2 workspace"
        );
        if (status != QSFI_STATUS_OK)
            return status;
        workspace.bmm1_workspace = bmm1_workspace.data();
        workspace.bmm2_workspace = bmm2_workspace.data();

        moe_runner.run(args, workspace, device, ctx->stream, config_index, QSFI_ENABLE_PDL != 0);
    } catch (const std::exception& ex) {
        return set_flashinfer_error(ctx, "FlashInfer TRTLLM BF16 MoE", ex);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
        return set_cuda_error(ctx, err, "FlashInfer TRTLLM BF16 MoE launch");
    return QSFI_STATUS_OK;
}

#endif

qsfi_status validate_desc(qsfi_context* ctx, const qsfi_trtllm_bf16_moe_desc* desc)
{
    if (desc == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "bf16_moe desc must not be null"
        );
    }
    qsfi_status status = validate_execution_common(
        ctx,
        desc->num_tokens,
        desc->activation,
        desc->accum_dtype,
        desc->routing
    );
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
    if (desc->weights.local_num_experts != desc->routing.num_experts) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "BF16 MoE currently requires local_num_experts == num_experts"
        );
    }
    status = validate_bf16_weights(ctx, desc->weights);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_bf16_io(
        ctx,
        desc->hidden_states,
        desc->out,
        desc->num_tokens,
        desc->weights.hidden_size
    );
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_logits(ctx, desc->routing_logits, desc->num_tokens, desc->routing.num_experts);
}

qsfi_status validate_desc(qsfi_context* ctx, const qsfi_trtllm_bf16_routed_moe_desc* desc)
{
    if (desc == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "bf16_routed_moe desc must not be null"
        );
    }
    qsfi_status status = validate_execution_common(
        ctx,
        desc->num_tokens,
        desc->activation,
        desc->accum_dtype,
        desc->routing
    );
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
    if (desc->weights.local_num_experts != desc->routing.num_experts) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "BF16 MoE currently requires local_num_experts == num_experts"
        );
    }
    status = validate_bf16_weights(ctx, desc->weights);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_bf16_io(
        ctx,
        desc->hidden_states,
        desc->out,
        desc->num_tokens,
        desc->weights.hidden_size
    );
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_packed_topk(ctx, desc->packed_topk, desc->num_tokens, desc->routing.top_k);
}

qsfi_status validate_desc(qsfi_context* ctx, const qsfi_trtllm_nvfp4_moe_desc* desc)
{
    if (desc == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "nvfp4_moe desc must not be null"
        );
    }
    qsfi_status status = validate_execution_common(
        ctx,
        desc->num_tokens,
        desc->activation,
        desc->accum_dtype,
        desc->routing
    );
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
    if (desc->weights.local_num_experts != desc->routing.num_experts) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "NVFP4 MoE currently requires local_num_experts == num_experts"
        );
    }
    status = validate_nvfp4_weights(ctx, desc->weights);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_nvfp4_io(
        ctx,
        desc->hidden_states,
        desc->hidden_states_scale,
        desc->out,
        desc->num_tokens,
        desc->weights.hidden_size
    );
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_logits(ctx, desc->routing_logits, desc->num_tokens, desc->routing.num_experts);
}

qsfi_status validate_desc(qsfi_context* ctx, const qsfi_trtllm_nvfp4_routed_moe_desc* desc)
{
    if (desc == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "nvfp4_routed_moe desc must not be null"
        );
    }
    qsfi_status status = validate_execution_common(
        ctx,
        desc->num_tokens,
        desc->activation,
        desc->accum_dtype,
        desc->routing
    );
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
    if (desc->weights.local_num_experts != desc->routing.num_experts) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "NVFP4 MoE currently requires local_num_experts == num_experts"
        );
    }
    status = validate_nvfp4_weights(ctx, desc->weights);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_nvfp4_io(
        ctx,
        desc->hidden_states,
        desc->hidden_states_scale,
        desc->out,
        desc->num_tokens,
        desc->weights.hidden_size
    );
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_packed_topk(ctx, desc->packed_topk, desc->num_tokens, desc->routing.top_k);
}

qsfi_status begin_call(qsfi_context* ctx)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    return activate_context(ctx);
}

qsfi_status unsupported_flashinfer_moe(qsfi_context* ctx, const char* dtype_name)
{
    return set_error(
        ctx,
        QSFI_STATUS_UNSUPPORTED,
        QSFI_ERROR_SOURCE_FLASHINFER,
        0,
        "FlashInfer TRTLLM %s MoE runner is not linked into qsfi yet",
        dtype_name
    );
}

#ifndef QSFI_BUILD_TRTLLM_GEN_MOE
qsfi_status trtllm_moe_not_built(qsfi_context* ctx, const char* dtype_name)
{
    return set_error(
        ctx,
        QSFI_STATUS_UNSUPPORTED,
        QSFI_ERROR_SOURCE_QSFI,
        0,
        "qsfi was not built with FlashInfer TRTLLM %s MoE support",
        dtype_name
    );
}
#endif

} // namespace

extern "C" {

qsfi_status qsfi_trtllm_bf16_moe(qsfi_context* ctx, const qsfi_trtllm_bf16_moe_desc* desc)
{
    qsfi_status status = begin_call(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_desc(ctx, desc);
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
#ifdef QSFI_BUILD_TRTLLM_GEN_MOE
    const Bf16MoeRun run {
        &desc->hidden_states, &desc->routing_logits, nullptr,          &desc->out,
        &desc->weights,       &desc->routing,        desc->num_tokens, desc->clamp_limit,
    };
    return run_flashinfer_bf16_moe(ctx, run);
#else
    return trtllm_moe_not_built(ctx, "BF16");
#endif
}

qsfi_status
qsfi_trtllm_bf16_routed_moe(qsfi_context* ctx, const qsfi_trtllm_bf16_routed_moe_desc* desc)
{
    qsfi_status status = begin_call(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_desc(ctx, desc);
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
#ifdef QSFI_BUILD_TRTLLM_GEN_MOE
    const Bf16MoeRun run {
        &desc->hidden_states, nullptr,        &desc->packed_topk, &desc->out,
        &desc->weights,       &desc->routing, desc->num_tokens,   desc->clamp_limit,
    };
    return run_flashinfer_bf16_moe(ctx, run);
#else
    return trtllm_moe_not_built(ctx, "BF16 routed");
#endif
}

qsfi_status qsfi_trtllm_nvfp4_moe(qsfi_context* ctx, const qsfi_trtllm_nvfp4_moe_desc* desc)
{
    qsfi_status status = begin_call(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_desc(ctx, desc);
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
    return unsupported_flashinfer_moe(ctx, "NVFP4");
}

qsfi_status
qsfi_trtllm_nvfp4_routed_moe(qsfi_context* ctx, const qsfi_trtllm_nvfp4_routed_moe_desc* desc)
{
    qsfi_status status = begin_call(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_desc(ctx, desc);
    if (status != QSFI_STATUS_OK || desc->num_tokens == 0)
        return status;
    return unsupported_flashinfer_moe(ctx, "NVFP4 routed");
}

} // extern "C"
