#include "qscu.h"

#include "qsfi_macros.h"
#include "qsfi_native_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kQwen36GdnNumQHeads = QSFI_QWEN36_GDN_NUM_Q_HEADS;
constexpr uint32_t kQwen36GdnNumKHeads = QSFI_QWEN36_GDN_NUM_K_HEADS;
constexpr uint32_t kQwen36GdnNumVHeads = QSFI_QWEN36_GDN_NUM_V_HEADS;
constexpr uint32_t kQwen36GdnKeyDim = QSFI_QWEN36_GDN_KEY_DIM;
constexpr uint32_t kQwen36GdnValueDim = QSFI_QWEN36_GDN_VALUE_DIM;
constexpr uint32_t kQwen36GdnConvWidth = 4;
constexpr uint32_t kQwen36GdnConvState = kQwen36GdnConvWidth - 1;
constexpr uint32_t kQwen36GdnPackedDim
    = 2 * kQwen36GdnNumKHeads * kQwen36GdnKeyDim + kQwen36GdnNumVHeads * kQwen36GdnValueDim;
constexpr uint32_t kQwen36FullAttentionQHidden = 4096;
constexpr uint32_t kQwen36GdnThreads = QSFI_QWEN36_GDN_THREADS;
constexpr uint32_t kElementwiseThreads = 256;
constexpr uint32_t kArgmaxThreads = 256;
constexpr uint32_t kRouterMaxTopK = 16;
constexpr uint32_t kRouterMaxExperts = 4096;
constexpr float kRouterNegInf = -3.4028234663852886e38f;

static_assert(kQwen36GdnNumQHeads == kQwen36GdnNumKHeads, "qwen3.6 GDN maps q/k heads");
static_assert(kQwen36GdnKeyDim == kQwen36GdnThreads, "one thread per GDN head dim");
static_assert(kQwen36GdnValueDim == kQwen36GdnThreads, "one thread per GDN value dim");

template <typename Tensor> bool tensor_present(const Tensor& tensor)
{
    return tensor.data != nullptr;
}

template <typename Tensor> qsfi_status validate_tensor(const Tensor& tensor, qsfi_dtype dtype)
{
    if (tensor.data == nullptr || tensor.dtype != dtype)
        return QSFI_STATUS_INVALID_ARGUMENT;

    constexpr uint32_t rank = sizeof(tensor.shape) / sizeof(tensor.shape[0]);
    for (uint32_t i = 0; i < rank; ++i) {
        if (tensor.shape[i] <= 0 || tensor.stride[i] <= 0)
            return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

template <typename Tensor>
qsfi_status validate_optional_tensor(const Tensor& tensor, qsfi_dtype dtype)
{
    if (tensor.data == nullptr)
        return QSFI_STATUS_OK;
    return validate_tensor(tensor, dtype);
}

template <typename Tensor>
qsfi_status validate_optional_tensor2(const Tensor& tensor, qsfi_dtype a, qsfi_dtype b)
{
    if (tensor.data == nullptr)
        return QSFI_STATUS_OK;
    if (tensor.dtype != a && tensor.dtype != b)
        return QSFI_STATUS_INVALID_ARGUMENT;
    return validate_tensor(tensor, tensor.dtype);
}

template <typename Tensor>
qsfi_status validate_tensor2(const Tensor& tensor, qsfi_dtype a, qsfi_dtype b)
{
    if (tensor.dtype != a && tensor.dtype != b)
        return QSFI_STATUS_INVALID_ARGUMENT;
    return validate_tensor(tensor, tensor.dtype);
}

bool contiguous1(const qsfi_tensor1& tensor)
{
    return tensor.stride[0] == 1;
}

bool contiguous2(const qsfi_tensor2& tensor)
{
    return tensor.stride[1] == 1 && tensor.stride[0] == tensor.shape[1];
}

bool contiguous3(const qsfi_tensor3& tensor)
{
    return tensor.stride[2] == 1 && tensor.stride[1] == tensor.shape[2]
        && tensor.stride[0] == tensor.shape[1] * tensor.shape[2];
}

bool contiguous_bf16_ranges_overlap(const qsfi_tensor2& a, const qsfi_tensor2& b)
{
    const uint64_t elem_count
        = static_cast<uint64_t>(a.shape[0]) * static_cast<uint64_t>(a.shape[1]);
    constexpr uint64_t max_addr = std::numeric_limits<uintptr_t>::max();
    if (elem_count > max_addr / sizeof(__nv_bfloat16))
        return true;
    const uintptr_t byte_count = static_cast<uintptr_t>(elem_count * sizeof(__nv_bfloat16));
    const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a.data);
    const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b.data);
    if (max_addr - a_begin < byte_count || max_addr - b_begin < byte_count)
        return true;
    const uintptr_t a_end = a_begin + byte_count;
    const uintptr_t b_end = b_begin + byte_count;
    return a_begin < b_end && b_begin < a_end;
}

qsfi_status validate_cuda(cudaError_t err)
{
    if (err == cudaSuccess)
        return QSFI_STATUS_OK;
    if (err == cudaErrorMemoryAllocation)
        return QSFI_STATUS_OUT_OF_MEMORY;
    return QSFI_STATUS_CUDA_ERROR;
}

#if QSFI_ENABLE_CHECKED_VALIDATION
struct qscu_status_reporter {
    qsfi_status cuda_error(cudaError_t err, const char*) const
    {
        return validate_cuda(err);
    }

    template <typename... Args> qsfi_status invalid_arg(const char*, Args...) const
    {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
};
#endif

qsfi_status checked_grid(uint64_t items, uint32_t threads, uint32_t* blocks)
{
    if (items == 0 || threads == 0 || blocks == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    const uint64_t needed = (items + threads - 1) / threads;
    if (needed > std::numeric_limits<uint32_t>::max())
        return QSFI_STATUS_UNSUPPORTED;
    *blocks = static_cast<uint32_t>(needed);
    return QSFI_STATUS_OK;
}

__device__ float load_bf16(const __nv_bfloat16* ptr)
{
    return __bfloat162float(*ptr);
}

__device__ void store_bf16(__nv_bfloat16* ptr, float value)
{
    *ptr = __float2bfloat16(value);
}

#if QSFI_ENABLE_CHECKED_VALIDATION
struct finite_tensor2_params {
    const __nv_bfloat16* bf16;
    const float* f32;
    int64_t stride0;
    int64_t stride1;
    uint32_t cols;
    int* error;
};

__device__ float
load_finite_tensor2_value(const finite_tensor2_params& p, uint32_t row, uint32_t col)
{
    const int64_t offset
        = static_cast<int64_t>(row) * p.stride0 + static_cast<int64_t>(col) * p.stride1;
    if (p.f32 != nullptr)
        return p.f32[offset];
    return load_bf16(p.bf16 + offset);
}

__global__ void validate_finite_tensor2_kernel(finite_tensor2_params p)
{
    const uint32_t row = blockIdx.x;
    for (uint32_t col = threadIdx.x; col < p.cols; col += blockDim.x) {
        if (!isfinite(load_finite_tensor2_value(p, row, col)))
            atomicExch(p.error, 1);
    }
}

qsfi_status validate_finite_tensor2_contents(
    const qsfi_tensor2& tensor,
    uint32_t rows,
    uint32_t cols,
    cudaStream_t stream,
    const char* invalid_message
)
{
    qscu_status_reporter errors;
    qsfi_checked_validation_flag validation(stream);
    qsfi_status status = validation.reset(
        errors,
        "cudaMalloc finite tensor validation flag",
        "cudaMemsetAsync finite tensor validation flag"
    );
    if (status != QSFI_STATUS_OK)
        return status;

    finite_tensor2_params params {};
    if (tensor.dtype == QSFI_DTYPE_F32)
        params.f32 = static_cast<const float*>(tensor.data);
    else
        params.bf16 = static_cast<const __nv_bfloat16*>(tensor.data);
    params.stride0 = tensor.stride[0];
    params.stride1 = tensor.stride[1];
    params.cols = cols;
    params.error = validation.device_ptr();

    validate_finite_tensor2_kernel<<<rows, kElementwiseThreads, 0, stream>>>(params);
    status = validation.check_launch(errors, "launch finite tensor validation");
    if (status != QSFI_STATUS_OK)
        return status;
    return validation.finish(
        errors,
        "copy finite tensor validation flag",
        "cudaFree finite tensor validation flag",
        invalid_message
    );
}
#endif

template <typename T> __device__ float load_state(const T* ptr)
{
    return static_cast<float>(*ptr);
}

template <> __device__ float load_state(const __nv_bfloat16* ptr)
{
    return __bfloat162float(*ptr);
}

template <typename T> __device__ T make_state(float value)
{
    return static_cast<T>(value);
}

template <> __device__ __nv_bfloat16 make_state(float value)
{
    return __float2bfloat16(value);
}

__device__ float silu(float value)
{
    return value / (1.0f + __expf(-value));
}

__device__ float sigmoid(float value)
{
    return 1.0f / (1.0f + __expf(-value));
}

__device__ float apply_activation(float value, qscu_activation activation)
{
    if (activation == QSCU_ACTIVATION_SILU)
        return silu(value);
    if (activation == QSCU_ACTIVATION_SIGMOID)
        return sigmoid(value);
    return value;
}

struct silu_and_mul_params {
    const __nv_bfloat16* gate;
    int64_t gate_stride0;
    int64_t gate_stride1;
    const __nv_bfloat16* up;
    int64_t up_stride0;
    int64_t up_stride1;
    __nv_bfloat16* out;
    int64_t out_stride0;
    int64_t out_stride1;
    uint32_t intermediate_size;
    uint64_t total_elements;
};

struct qwen36_shared_expert_gate_add_params {
    const __nv_bfloat16* gate_bf16;
    const float* gate_f32;
    int64_t gate_stride0;
    const __nv_bfloat16* shared;
    int64_t shared_stride0;
    int64_t shared_stride1;
    __nv_bfloat16* out;
    int64_t out_stride0;
    int64_t out_stride1;
    uint32_t hidden_size;
    uint64_t total_elements;
};

struct qwen36_full_attention_output_gate_params {
    const __nv_bfloat16* gate;
    int64_t gate_stride0;
    int64_t gate_stride1;
    __nv_bfloat16* out;
    int64_t out_stride0;
    int64_t out_stride1;
    uint32_t q_hidden;
    uint64_t total_elements;
};

__global__ void silu_and_mul_bf16_kernel(silu_and_mul_params p)
{
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= p.total_elements)
        return;

    const uint32_t token = static_cast<uint32_t>(linear / p.intermediate_size);
    const uint32_t dim
        = static_cast<uint32_t>(linear - static_cast<uint64_t>(token) * p.intermediate_size);
    const float gate = load_bf16(
        p.gate + static_cast<int64_t>(token) * p.gate_stride0
        + static_cast<int64_t>(dim) * p.gate_stride1
    );
    const float up = load_bf16(
        p.up + static_cast<int64_t>(token) * p.up_stride0 + static_cast<int64_t>(dim) * p.up_stride1
    );
    store_bf16(
        p.out + static_cast<int64_t>(token) * p.out_stride0
            + static_cast<int64_t>(dim) * p.out_stride1,
        silu(gate) * up
    );
}

__device__ float
load_qwen36_shared_expert_gate(const qwen36_shared_expert_gate_add_params& p, uint32_t row)
{
    const int64_t offset = static_cast<int64_t>(row) * p.gate_stride0;
    if (p.gate_f32 != nullptr)
        return p.gate_f32[offset];
    return load_bf16(p.gate_bf16 + offset);
}

__global__ void qwen36_shared_expert_gate_add_bf16_kernel(qwen36_shared_expert_gate_add_params p)
{
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= p.total_elements)
        return;

    const uint32_t row = static_cast<uint32_t>(linear / p.hidden_size);
    const uint32_t dim = static_cast<uint32_t>(linear - static_cast<uint64_t>(row) * p.hidden_size);
    const float gate = sigmoid(load_qwen36_shared_expert_gate(p, row));
    const int64_t offset
        = static_cast<int64_t>(row) * p.out_stride0 + static_cast<int64_t>(dim) * p.out_stride1;
    const float shared = load_bf16(
        p.shared + static_cast<int64_t>(row) * p.shared_stride0
        + static_cast<int64_t>(dim) * p.shared_stride1
    );
    const float current = load_bf16(p.out + offset);
    store_bf16(p.out + offset, current + gate * shared);
}

__global__ void
qwen36_full_attention_output_gate_bf16_kernel(qwen36_full_attention_output_gate_params p)
{
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= p.total_elements)
        return;

    const uint32_t row = static_cast<uint32_t>(linear / p.q_hidden);
    const uint32_t dim = static_cast<uint32_t>(linear - static_cast<uint64_t>(row) * p.q_hidden);
    const int64_t gate_offset
        = static_cast<int64_t>(row) * p.gate_stride0 + static_cast<int64_t>(dim) * p.gate_stride1;
    const int64_t out_offset
        = static_cast<int64_t>(row) * p.out_stride0 + static_cast<int64_t>(dim) * p.out_stride1;
    const float gate = load_bf16(p.gate + gate_offset);
    const float current = load_bf16(p.out + out_offset);
    store_bf16(p.out + out_offset, current * sigmoid(gate));
}

template <typename TokenT> __device__ bool is_padding_token(TokenT token, int32_t padding_token_id)
{
    return padding_token_id >= 0
        && static_cast<uint64_t>(token) == static_cast<uint32_t>(padding_token_id);
}

template <> __device__ bool is_padding_token<int32_t>(int32_t token, int32_t padding_token_id)
{
    return padding_token_id >= 0 && token == padding_token_id;
}

template <typename TokenT> __device__ bool token_in_vocab(TokenT token, uint32_t vocab_size)
{
    return static_cast<uint64_t>(token) < static_cast<uint64_t>(vocab_size);
}

template <> __device__ bool token_in_vocab<int32_t>(int32_t token, uint32_t vocab_size)
{
    return token >= 0 && static_cast<uint32_t>(token) < vocab_size;
}

template <typename TokenT> struct embedding_gather_params {
    const TokenT* token_ids;
    int64_t token_stride0;
    const __nv_bfloat16* embedding;
    int64_t embedding_stride0;
    int64_t embedding_stride1;
    __nv_bfloat16* out;
    int64_t out_stride0;
    int64_t out_stride1;
    uint32_t hidden_size;
    uint32_t vocab_size;
    int32_t padding_token_id;
    int* invalid_token;
    uint64_t total_elements;
};

template <typename TokenT>
__global__ void embedding_gather_bf16_kernel(embedding_gather_params<TokenT> p)
{
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= p.total_elements)
        return;

    const uint32_t token_pos = static_cast<uint32_t>(linear / p.hidden_size);
    const uint32_t dim
        = static_cast<uint32_t>(linear - static_cast<uint64_t>(token_pos) * p.hidden_size);
    const TokenT token_id = p.token_ids[static_cast<int64_t>(token_pos) * p.token_stride0];
    const bool padding = is_padding_token(token_id, p.padding_token_id);
    const bool valid = padding || token_in_vocab(token_id, p.vocab_size);
    __nv_bfloat16 value = __float2bfloat16(0.0f);

    if (!valid) {
        if (p.invalid_token != nullptr)
            atomicExch(p.invalid_token, 1);
    } else if (!padding) {
        const uint32_t vocab_row = static_cast<uint32_t>(token_id);
        value = p.embedding
                    [static_cast<int64_t>(vocab_row) * p.embedding_stride0
                     + static_cast<int64_t>(dim) * p.embedding_stride1];
    }

    p.out
        [static_cast<int64_t>(token_pos) * p.out_stride0
         + static_cast<int64_t>(dim) * p.out_stride1]
        = value;
}

struct logits_soft_cap_params {
    float* logits;
    int64_t stride0;
    int64_t stride1;
    uint32_t vocab_size;
    float soft_cap;
    uint64_t total_elements;
};

__global__ void logits_soft_cap_f32_kernel(logits_soft_cap_params p)
{
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= p.total_elements)
        return;

    const uint32_t row = static_cast<uint32_t>(linear / p.vocab_size);
    const uint32_t col = static_cast<uint32_t>(linear - static_cast<uint64_t>(row) * p.vocab_size);
    float* slot
        = p.logits + static_cast<int64_t>(row) * p.stride0 + static_cast<int64_t>(col) * p.stride1;
    *slot = p.soft_cap * tanhf(*slot / p.soft_cap);
}

struct greedy_argmax_params {
    const float* logits;
    int64_t logits_stride0;
    int64_t logits_stride1;
    int32_t* out_i32;
    uint32_t* out_u32;
    int64_t out_stride0;
    uint32_t vocab_size;
};

__device__ bool
better_argmax_candidate(float score, uint32_t token, float best, uint32_t best_token)
{
    return score > best || (score == best && token < best_token);
}

__global__ void greedy_argmax_f32_kernel(greedy_argmax_params p)
{
    const uint32_t row = blockIdx.x;
    float best_score = kRouterNegInf;
    uint32_t best_token = UINT_MAX;

    for (uint32_t token = threadIdx.x; token < p.vocab_size; token += blockDim.x) {
        float score = p.logits
                          [static_cast<int64_t>(row) * p.logits_stride0
                           + static_cast<int64_t>(token) * p.logits_stride1];
        if (better_argmax_candidate(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }

    __shared__ float scores[kArgmaxThreads];
    __shared__ uint32_t tokens[kArgmaxThreads];
    scores[threadIdx.x] = best_score;
    tokens[threadIdx.x] = best_token;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride
            && better_argmax_candidate(
                scores[threadIdx.x + stride],
                tokens[threadIdx.x + stride],
                scores[threadIdx.x],
                tokens[threadIdx.x]
            )) {
            scores[threadIdx.x] = scores[threadIdx.x + stride];
            tokens[threadIdx.x] = tokens[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (p.out_i32 != nullptr)
            p.out_i32[static_cast<int64_t>(row) * p.out_stride0] = static_cast<int32_t>(tokens[0]);
        else
            p.out_u32[static_cast<int64_t>(row) * p.out_stride0] = tokens[0];
    }
}

__device__ float
load_optional_bias(const __nv_bfloat16* bias_bf16, const float* bias_f32, uint32_t dim)
{
    if (bias_f32 != nullptr)
        return bias_f32[dim];
    if (bias_bf16 != nullptr)
        return load_bf16(bias_bf16 + dim);
    return 0.0f;
}

__device__ float
load_optional_weight(const __nv_bfloat16* weight_bf16, const float* weight_f32, uint32_t dim)
{
    if (weight_f32 != nullptr)
        return weight_f32[dim];
    return load_bf16(weight_bf16 + dim);
}

struct conv1d_params {
    const __nv_bfloat16* x;
    int64_t x_stride0;
    int64_t x_stride1;
    const __nv_bfloat16* weight;
    int64_t weight_stride0;
    int64_t weight_stride1;
    const __nv_bfloat16* bias_bf16;
    const float* bias_f32;
    int64_t bias_stride0;
    int64_t state_stride0;
    int64_t state_stride1;
    int64_t state_stride2;
    const int32_t* read_indices;
    const int32_t* write_indices;
    const int32_t* seq_indptr;
    __nv_bfloat16* out;
    int64_t out_stride0;
    int64_t out_stride1;
    uint32_t conv_dim;
    qscu_activation activation;
    uint32_t update_state;
};

template <typename StateT>
__global__ void qwen36_gdn_causal_conv1d_kernel(conv1d_params p, StateT* state)
{
    const uint32_t seq = blockIdx.x;
    const int32_t token_begin
        = p.seq_indptr == nullptr ? static_cast<int32_t>(seq) : p.seq_indptr[seq];
    const int32_t token_end
        = p.seq_indptr == nullptr ? static_cast<int32_t>(seq + 1) : p.seq_indptr[seq + 1];
    const int32_t fallback_slot = p.write_indices == nullptr ? -1 : p.write_indices[seq];
    const int32_t read_slot = p.read_indices == nullptr ? fallback_slot : p.read_indices[seq];
    const int32_t write_slot = p.update_state == 0
        ? -1
        : (p.write_indices == nullptr ? read_slot : p.write_indices[seq]);

    for (uint32_t dim = threadIdx.x; dim < p.conv_dim; dim += blockDim.x) {
        float h0 = 0.0f;
        float h1 = 0.0f;
        float h2 = 0.0f;
        if (read_slot >= 0) {
            const int64_t base = static_cast<int64_t>(read_slot) * p.state_stride0
                + static_cast<int64_t>(dim) * p.state_stride1;
            h0 = load_state(state + base);
            h1 = load_state(state + base + p.state_stride2);
            h2 = load_state(state + base + 2 * p.state_stride2);
        }

        const float w0 = load_bf16(p.weight + static_cast<int64_t>(dim) * p.weight_stride0);
        const float w1
            = load_bf16(p.weight + static_cast<int64_t>(dim) * p.weight_stride0 + p.weight_stride1);
        const float w2 = load_bf16(
            p.weight + static_cast<int64_t>(dim) * p.weight_stride0 + 2 * p.weight_stride1
        );
        const float w3 = load_bf16(
            p.weight + static_cast<int64_t>(dim) * p.weight_stride0 + 3 * p.weight_stride1
        );
        const float bias = load_optional_bias(
            p.bias_bf16 == nullptr ? nullptr
                                   : p.bias_bf16 + static_cast<int64_t>(dim) * p.bias_stride0,
            p.bias_f32 == nullptr ? nullptr
                                  : p.bias_f32 + static_cast<int64_t>(dim) * p.bias_stride0,
            0
        );

        for (int32_t token = token_begin; token < token_end; ++token) {
            const float x_value = load_bf16(
                p.x + static_cast<int64_t>(token) * p.x_stride0
                + static_cast<int64_t>(dim) * p.x_stride1
            );
            float out_value = h0 * w0 + h1 * w1 + h2 * w2 + x_value * w3 + bias;
            out_value = apply_activation(out_value, p.activation);
            store_bf16(
                p.out + static_cast<int64_t>(token) * p.out_stride0
                    + static_cast<int64_t>(dim) * p.out_stride1,
                out_value
            );
            h0 = h1;
            h1 = h2;
            h2 = x_value;
        }

        if (write_slot >= 0) {
            const int64_t base = static_cast<int64_t>(write_slot) * p.state_stride0
                + static_cast<int64_t>(dim) * p.state_stride1;
            state[base] = make_state<StateT>(h0);
            state[base + p.state_stride2] = make_state<StateT>(h1);
            state[base + 2 * p.state_stride2] = make_state<StateT>(h2);
        }
    }
}

#if QSFI_ENABLE_CHECKED_VALIDATION
__device__ bool qscu_invalid_state_slot(int32_t slot, int32_t state_pool)
{
    return slot >= state_pool;
}

__global__ void validate_conv1d_metadata_kernel(
    const int32_t* seq_indptr,
    const int32_t* read_indices,
    const int32_t* write_indices,
    uint32_t batch_size,
    uint32_t num_tokens,
    int32_t state_pool,
    int* error
)
{
    const uint32_t seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch_size)
        return;
    const int32_t token_begin = seq_indptr == nullptr ? static_cast<int32_t>(seq) : seq_indptr[seq];
    const int32_t token_end
        = seq_indptr == nullptr ? static_cast<int32_t>(seq + 1) : seq_indptr[seq + 1];
    if (token_begin < 0 || token_end < token_begin || token_end > static_cast<int32_t>(num_tokens)
        || (seq_indptr != nullptr
            && ((seq == 0 && token_begin != 0)
                || (seq + 1 == batch_size && token_end != static_cast<int32_t>(num_tokens))))) {
        atomicExch(error, 1);
    }

    const int32_t fallback_slot = write_indices == nullptr ? -1 : write_indices[seq];
    const int32_t read_slot = read_indices == nullptr ? fallback_slot : read_indices[seq];
    const int32_t write_slot = write_indices == nullptr ? read_slot : write_indices[seq];
    if (qscu_invalid_state_slot(read_slot, state_pool)
        || qscu_invalid_state_slot(write_slot, state_pool)) {
        atomicExch(error, 1);
    }
}
#endif

struct post_conv_params {
    const __nv_bfloat16* conv_out;
    int64_t conv_stride0;
    int64_t conv_stride1;
    const __nv_bfloat16* a;
    int64_t a_stride0;
    int64_t a_stride1;
    const __nv_bfloat16* b;
    int64_t b_stride0;
    int64_t b_stride1;
    const float* a_log;
    int64_t a_log_stride0;
    const float* dt_bias;
    int64_t dt_bias_stride0;
    __nv_bfloat16* q;
    int64_t q_stride0;
    int64_t q_stride1;
    int64_t q_stride2;
    __nv_bfloat16* k;
    int64_t k_stride0;
    int64_t k_stride1;
    int64_t k_stride2;
    __nv_bfloat16* v;
    int64_t v_stride0;
    int64_t v_stride1;
    int64_t v_stride2;
    float* g_out;
    int64_t g_stride0;
    int64_t g_stride1;
    float* beta_out;
    int64_t beta_stride0;
    int64_t beta_stride1;
    float l2norm_eps;
    qscu_gdn_forget_gate_output forget_gate_output;
    uint32_t apply_qk_l2norm;
};

__device__ float block_sum_128(float value)
{
    __shared__ float scratch[kQwen36GdnThreads];
    const uint32_t tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (uint32_t stride = kQwen36GdnThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__global__ void qwen36_gdn_post_conv_prepare_kernel(post_conv_params p)
{
    const uint32_t head_slot = blockIdx.x % (kQwen36GdnNumKHeads + kQwen36GdnNumVHeads);
    const uint32_t token = blockIdx.x / (kQwen36GdnNumKHeads + kQwen36GdnNumVHeads);
    const uint32_t tid = threadIdx.x;

    if (head_slot < kQwen36GdnNumKHeads) {
        const uint32_t head = head_slot;
        const int64_t q_offset = static_cast<int64_t>(token) * p.conv_stride0
            + static_cast<int64_t>(head * kQwen36GdnKeyDim + tid) * p.conv_stride1;
        const int64_t k_offset = static_cast<int64_t>(token) * p.conv_stride0
            + static_cast<int64_t>(
                  kQwen36GdnNumKHeads * kQwen36GdnKeyDim + head * kQwen36GdnKeyDim + tid
              ) * p.conv_stride1;

        float q_value = load_bf16(p.conv_out + q_offset);
        float k_value = load_bf16(p.conv_out + k_offset);
        if (p.apply_qk_l2norm != 0) {
            const float q_norm2 = block_sum_128(q_value * q_value);
            const float k_norm2 = block_sum_128(k_value * k_value);
            q_value *= rsqrtf(q_norm2 + p.l2norm_eps);
            k_value *= rsqrtf(k_norm2 + p.l2norm_eps);
        }

        store_bf16(
            p.q + static_cast<int64_t>(token) * p.q_stride0
                + static_cast<int64_t>(head) * p.q_stride1
                + static_cast<int64_t>(tid) * p.q_stride2,
            q_value
        );
        store_bf16(
            p.k + static_cast<int64_t>(token) * p.k_stride0
                + static_cast<int64_t>(head) * p.k_stride1
                + static_cast<int64_t>(tid) * p.k_stride2,
            k_value
        );
        return;
    }

    const uint32_t v_head = head_slot - kQwen36GdnNumKHeads;
    const int64_t v_offset = static_cast<int64_t>(token) * p.conv_stride0
        + static_cast<int64_t>(
              2 * kQwen36GdnNumKHeads * kQwen36GdnKeyDim + v_head * kQwen36GdnValueDim + tid
          ) * p.conv_stride1;
    const float v_value = load_bf16(p.conv_out + v_offset);
    store_bf16(
        p.v + static_cast<int64_t>(token) * p.v_stride0 + static_cast<int64_t>(v_head) * p.v_stride1
            + static_cast<int64_t>(tid) * p.v_stride2,
        v_value
    );

    if (tid == 0 && (p.g_out != nullptr || p.beta_out != nullptr)) {
        const float a_value = load_bf16(
            p.a + static_cast<int64_t>(token) * p.a_stride0
            + static_cast<int64_t>(v_head) * p.a_stride1
        );
        const float b_value = load_bf16(
            p.b + static_cast<int64_t>(token) * p.b_stride0
            + static_cast<int64_t>(v_head) * p.b_stride1
        );
        const float x = a_value + p.dt_bias[static_cast<int64_t>(v_head) * p.dt_bias_stride0];
        const float softplus_x = x <= QSFI_QWEN36_GDN_SOFTPLUS_THRESHOLD ? log1pf(expf(x)) : x;
        float g_value = -expf(p.a_log[static_cast<int64_t>(v_head) * p.a_log_stride0]) * softplus_x;
        if (p.forget_gate_output == QSCU_GDN_FORGET_LINEAR_ALPHA)
            g_value = expf(g_value);
        if (p.g_out != nullptr) {
            p.g_out
                [static_cast<int64_t>(token) * p.g_stride0
                 + static_cast<int64_t>(v_head) * p.g_stride1]
                = g_value;
        }
        if (p.beta_out != nullptr) {
            p.beta_out
                [static_cast<int64_t>(token) * p.beta_stride0
                 + static_cast<int64_t>(v_head) * p.beta_stride1]
                = sigmoid(b_value);
        }
    }
}

struct rmsnorm_gated_params {
    const __nv_bfloat16* x;
    int64_t x_stride0;
    int64_t x_stride1;
    int64_t x_stride2;
    const __nv_bfloat16* gate;
    int64_t gate_stride0;
    int64_t gate_stride1;
    int64_t gate_stride2;
    const __nv_bfloat16* weight_bf16;
    const float* weight_f32;
    int64_t weight_stride0;
    __nv_bfloat16* out;
    int64_t out_stride0;
    int64_t out_stride1;
    int64_t out_stride2;
    float eps;
    qscu_activation gate_activation;
};

__global__ void qwen36_gdn_rmsnorm_gated_kernel(rmsnorm_gated_params p)
{
    const uint32_t token = blockIdx.x / kQwen36GdnNumVHeads;
    const uint32_t v_head = blockIdx.x % kQwen36GdnNumVHeads;
    const uint32_t tid = threadIdx.x;
    const float x_value = load_bf16(
        p.x + static_cast<int64_t>(token) * p.x_stride0 + static_cast<int64_t>(v_head) * p.x_stride1
        + static_cast<int64_t>(tid) * p.x_stride2
    );
    const float sum = block_sum_128(x_value * x_value);
    const float rstd = rsqrtf(sum / static_cast<float>(kQwen36GdnValueDim) + p.eps);
    const float weight = load_optional_weight(
        p.weight_bf16 == nullptr ? nullptr
                                 : p.weight_bf16 + static_cast<int64_t>(tid) * p.weight_stride0,
        p.weight_f32 == nullptr ? nullptr
                                : p.weight_f32 + static_cast<int64_t>(tid) * p.weight_stride0,
        0
    );
    const float gate_value = load_bf16(
        p.gate + static_cast<int64_t>(token) * p.gate_stride0
        + static_cast<int64_t>(v_head) * p.gate_stride1 + static_cast<int64_t>(tid) * p.gate_stride2
    );
    const float out_value
        = x_value * rstd * weight * apply_activation(gate_value, p.gate_activation);
    store_bf16(
        p.out + static_cast<int64_t>(token) * p.out_stride0
            + static_cast<int64_t>(v_head) * p.out_stride1
            + static_cast<int64_t>(tid) * p.out_stride2,
        out_value
    );
}

struct router_topk_params {
    const __nv_bfloat16* logits_bf16;
    const float* logits_f32;
    int64_t logits_stride0;
    int64_t logits_stride1;
    int32_t* topk_ids;
    int64_t ids_stride0;
    int64_t ids_stride1;
    float* topk_weights;
    int64_t weights_stride0;
    int64_t weights_stride1;
    uint32_t num_experts;
    uint32_t top_k;
    qscu_router_score score;
    uint32_t renormalize;
    float routed_scaling_factor;
};

__device__ float load_router_logit(const router_topk_params& p, uint32_t token, uint32_t expert)
{
    const int64_t offset = static_cast<int64_t>(token) * p.logits_stride0
        + static_cast<int64_t>(expert) * p.logits_stride1;
    if (p.logits_f32 != nullptr)
        return p.logits_f32[offset];
    return load_bf16(p.logits_bf16 + offset);
}

__device__ bool
better_router_candidate(float score, int32_t expert, float best, int32_t best_expert)
{
    return score > best || (score == best && expert < best_expert);
}

__global__ void router_topk_kernel(router_topk_params p)
{
    const uint32_t token = blockIdx.x;
    float best_scores[kRouterMaxTopK];
    int32_t best_ids[kRouterMaxTopK];
    for (uint32_t i = 0; i < kRouterMaxTopK; ++i) {
        best_scores[i] = kRouterNegInf;
        best_ids[i] = INT_MAX;
    }

    float max_logit = kRouterNegInf;
    if (p.score == QSCU_ROUTER_SCORE_SOFTMAX) {
        for (uint32_t expert = 0; expert < p.num_experts; ++expert) {
            const float logit = load_router_logit(p, token, expert);
            if (logit > max_logit)
                max_logit = logit;
        }
    }

    float softmax_denom = 0.0f;
    if (p.score == QSCU_ROUTER_SCORE_SOFTMAX) {
        for (uint32_t expert = 0; expert < p.num_experts; ++expert) {
            const float logit = load_router_logit(p, token, expert);
            softmax_denom += expf(logit - max_logit);
        }
    }

    for (uint32_t expert = 0; expert < p.num_experts; ++expert) {
        const float logit = load_router_logit(p, token, expert);
        const float select_score = p.score == QSCU_ROUTER_SCORE_SOFTMAX ? logit : sigmoid(logit);
        for (uint32_t pos = 0; pos < p.top_k; ++pos) {
            if (!better_router_candidate(
                    select_score,
                    static_cast<int32_t>(expert),
                    best_scores[pos],
                    best_ids[pos]
                ))
                continue;
            for (uint32_t move = p.top_k - 1; move > pos; --move) {
                best_scores[move] = best_scores[move - 1];
                best_ids[move] = best_ids[move - 1];
            }
            best_scores[pos] = select_score;
            best_ids[pos] = static_cast<int32_t>(expert);
            break;
        }
    }

    float selected_sum = 0.0f;
    float weights[kRouterMaxTopK];
    for (uint32_t pos = 0; pos < p.top_k; ++pos) {
        if (best_ids[pos] == INT_MAX) {
            best_ids[pos] = static_cast<int32_t>(pos);
            weights[pos] = 0.0f;
            continue;
        }
        const float logit = load_router_logit(p, token, static_cast<uint32_t>(best_ids[pos]));
        float weight = 0.0f;
        if (p.score == QSCU_ROUTER_SCORE_SOFTMAX) {
            weight = softmax_denom > 0.0f ? expf(logit - max_logit) / softmax_denom : 0.0f;
        } else {
            weight = sigmoid(logit);
        }
        weights[pos] = weight;
        selected_sum += weight;
    }
    const float renorm = p.renormalize == 0 ? 1.0f : 1.0f / fmaxf(selected_sum, 1.0e-20f);
    for (uint32_t pos = 0; pos < p.top_k; ++pos) {
        p.topk_ids
            [static_cast<int64_t>(token) * p.ids_stride0
             + static_cast<int64_t>(pos) * p.ids_stride1]
            = best_ids[pos];
        p.topk_weights
            [static_cast<int64_t>(token) * p.weights_stride0
             + static_cast<int64_t>(pos) * p.weights_stride1]
            = weights[pos] * renorm * p.routed_scaling_factor;
    }
}

qsfi_status validate_conv_desc(const qscu_qwen36_gdn_causal_conv1d_desc* desc)
{
    if (desc == nullptr || desc->num_tokens == 0 || desc->batch_size == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->num_tokens > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        return QSFI_STATUS_UNSUPPORTED;
    if (desc->seq_indptr == nullptr && desc->batch_size != desc->num_tokens)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->activation != QSCU_ACTIVATION_NONE && desc->activation != QSCU_ACTIVATION_SILU)
        return QSFI_STATUS_UNSUPPORTED;

    qsfi_status status = validate_tensor(desc->x, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->weight, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_tensor2(desc->bias, QSFI_DTYPE_BF16, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor2(desc->state, QSFI_DTYPE_BF16, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_tensor(desc->state_read_indices, QSFI_DTYPE_I32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_tensor(desc->state_write_indices, QSFI_DTYPE_I32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous2(desc->x) || !contiguous2(desc->weight) || !contiguous3(desc->state)
        || !contiguous2(desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (tensor_present(desc->bias) && !contiguous1(desc->bias))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (tensor_present(desc->state_read_indices) && !contiguous1(desc->state_read_indices))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (tensor_present(desc->state_write_indices) && !contiguous1(desc->state_write_indices))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (!tensor_present(desc->state_read_indices) && !tensor_present(desc->state_write_indices))
        return QSFI_STATUS_INVALID_ARGUMENT;

    if (desc->x.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->x.shape[1] != kQwen36GdnPackedDim || desc->weight.shape[0] != kQwen36GdnPackedDim
        || desc->weight.shape[1] != kQwen36GdnConvWidth
        || (tensor_present(desc->bias) && desc->bias.shape[0] != kQwen36GdnPackedDim)
        || desc->state.shape[1] != kQwen36GdnPackedDim
        || desc->state.shape[2] != kQwen36GdnConvState
        || desc->out.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->out.shape[1] != kQwen36GdnPackedDim
        || (tensor_present(desc->state_read_indices)
            && desc->state_read_indices.shape[0] != static_cast<int64_t>(desc->batch_size))
        || (tensor_present(desc->state_write_indices)
            && desc->state_write_indices.shape[0] != static_cast<int64_t>(desc->batch_size))) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status
validate_conv1d_metadata(const qscu_qwen36_gdn_causal_conv1d_desc* desc, cudaStream_t stream)
{
    if (desc->state.shape[0] > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return QSFI_STATUS_UNSUPPORTED;

#if !QSFI_ENABLE_CHECKED_VALIDATION
    (void)stream;
    return QSFI_STATUS_OK;
#else
    qscu_status_reporter errors;
    qsfi_checked_validation_flag validation(stream);
    qsfi_status status = validation.reset(
        errors,
        "cudaMalloc conv1d metadata validation flag",
        "cudaMemsetAsync conv1d metadata validation flag"
    );
    if (status != QSFI_STATUS_OK)
        return status;

    constexpr uint32_t threads = 256;
    const uint32_t blocks = (desc->batch_size + threads - 1) / threads;
    validate_conv1d_metadata_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const int32_t*>(desc->seq_indptr),
        static_cast<const int32_t*>(desc->state_read_indices.data),
        static_cast<const int32_t*>(desc->state_write_indices.data),
        desc->batch_size,
        desc->num_tokens,
        static_cast<int32_t>(desc->state.shape[0]),
        validation.device_ptr()
    );
    status = validation.check_launch(errors, "launch conv1d metadata validation");
    if (status != QSFI_STATUS_OK)
        return status;
    return validation.finish(
        errors,
        "copy conv1d metadata validation flag",
        "cudaFree conv1d metadata validation flag",
        "conv1d seq_indptr/state indices are out of range"
    );
#endif
}

qsfi_status validate_post_conv_desc(const qscu_qwen36_gdn_post_conv_prepare_desc* desc)
{
    if (desc == nullptr || desc->num_tokens == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    // Host descriptor scalar: invalid eps would poison every normalization denominator.
    if (desc->l2norm_eps <= 0.0f || !std::isfinite(desc->l2norm_eps))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->forget_gate_output != QSCU_GDN_FORGET_LOG_DECAY
        && desc->forget_gate_output != QSCU_GDN_FORGET_LINEAR_ALPHA)
        return QSFI_STATUS_UNSUPPORTED;

    qsfi_status status = validate_tensor(desc->conv_out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->a, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->b, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->a_log, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->dt_bias, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->q, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->k, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->v, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_tensor(desc->g_out, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_tensor(desc->beta_out, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous2(desc->conv_out) || !contiguous2(desc->a) || !contiguous2(desc->b)
        || !contiguous1(desc->a_log) || !contiguous1(desc->dt_bias) || !contiguous3(desc->q)
        || !contiguous3(desc->k) || !contiguous3(desc->v))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (tensor_present(desc->g_out) && !contiguous2(desc->g_out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (tensor_present(desc->beta_out) && !contiguous2(desc->beta_out))
        return QSFI_STATUS_INVALID_ARGUMENT;

    if (desc->conv_out.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->conv_out.shape[1] != kQwen36GdnPackedDim
        || desc->a.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->a.shape[1] != kQwen36GdnNumVHeads
        || desc->b.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->b.shape[1] != kQwen36GdnNumVHeads || desc->a_log.shape[0] != kQwen36GdnNumVHeads
        || desc->dt_bias.shape[0] != kQwen36GdnNumVHeads
        || desc->q.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->q.shape[1] != kQwen36GdnNumQHeads || desc->q.shape[2] != kQwen36GdnKeyDim
        || desc->k.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->k.shape[1] != kQwen36GdnNumKHeads || desc->k.shape[2] != kQwen36GdnKeyDim
        || desc->v.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->v.shape[1] != kQwen36GdnNumVHeads || desc->v.shape[2] != kQwen36GdnValueDim
        || (tensor_present(desc->g_out)
            && (desc->g_out.shape[0] != static_cast<int64_t>(desc->num_tokens)
                || desc->g_out.shape[1] != kQwen36GdnNumVHeads))
        || (tensor_present(desc->beta_out)
            && (desc->beta_out.shape[0] != static_cast<int64_t>(desc->num_tokens)
                || desc->beta_out.shape[1] != kQwen36GdnNumVHeads))) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_rmsnorm_gated_desc(const qscu_qwen36_gdn_rmsnorm_gated_desc* desc)
{
    if (desc == nullptr || desc->num_tokens == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    // Host descriptor scalar: invalid eps would poison every normalization denominator.
    if (desc->eps <= 0.0f || !std::isfinite(desc->eps))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->gate_activation != QSCU_ACTIVATION_SILU
        && desc->gate_activation != QSCU_ACTIVATION_SIGMOID)
        return QSFI_STATUS_UNSUPPORTED;
    qsfi_status status = validate_tensor(desc->x, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->gate, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor2(desc->weight, QSFI_DTYPE_BF16, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    if (!contiguous3(desc->x) || !contiguous3(desc->gate) || !contiguous1(desc->weight)
        || !contiguous3(desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->x.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->x.shape[1] != kQwen36GdnNumVHeads || desc->x.shape[2] != kQwen36GdnValueDim
        || desc->gate.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->gate.shape[1] != kQwen36GdnNumVHeads || desc->gate.shape[2] != kQwen36GdnValueDim
        || desc->weight.shape[0] != kQwen36GdnValueDim
        || desc->out.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->out.shape[1] != kQwen36GdnNumVHeads || desc->out.shape[2] != kQwen36GdnValueDim) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_router_desc(const qscu_router_topk_desc* desc)
{
    if (desc == nullptr || desc->num_tokens == 0 || desc->num_experts == 0 || desc->top_k == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->top_k > kRouterMaxTopK || desc->num_experts > kRouterMaxExperts
        || desc->top_k > desc->num_experts)
        return QSFI_STATUS_UNSUPPORTED;
    if (desc->score != QSCU_ROUTER_SCORE_SOFTMAX && desc->score != QSCU_ROUTER_SCORE_SIGMOID)
        return QSFI_STATUS_UNSUPPORTED;
    // Host descriptor scalar: the kernel multiplies selected weights by this value.
    if (!std::isfinite(desc->routed_scaling_factor) || desc->routed_scaling_factor <= 0.0f)
        return QSFI_STATUS_INVALID_ARGUMENT;

    qsfi_status status = validate_tensor2(desc->logits, QSFI_DTYPE_BF16, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->topk_ids, QSFI_DTYPE_I32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->topk_weights, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    if (!contiguous2(desc->logits) || !contiguous2(desc->topk_ids)
        || !contiguous2(desc->topk_weights))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->logits.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->logits.shape[1] != static_cast<int64_t>(desc->num_experts)
        || desc->topk_ids.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->topk_ids.shape[1] != static_cast<int64_t>(desc->top_k)
        || desc->topk_weights.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->topk_weights.shape[1] != static_cast<int64_t>(desc->top_k)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_silu_and_mul_desc(const qscu_silu_and_mul_desc* desc)
{
    if (desc == nullptr || desc->num_tokens == 0 || desc->intermediate_size == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;

    qsfi_status status = validate_tensor(desc->gate, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->up, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous2(desc->gate) || !contiguous2(desc->up) || !contiguous2(desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->gate.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->gate.shape[1] != static_cast<int64_t>(desc->intermediate_size)
        || desc->up.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->up.shape[1] != static_cast<int64_t>(desc->intermediate_size)
        || desc->out.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->out.shape[1] != static_cast<int64_t>(desc->intermediate_size)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status
validate_qwen36_shared_expert_gate_add_desc(const qscu_qwen36_shared_expert_gate_add_desc* desc)
{
    if (desc == nullptr || desc->num_tokens == 0 || desc->hidden_size == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;

    qsfi_status status = validate_tensor2(desc->gate_logits, QSFI_DTYPE_BF16, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->shared, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous2(desc->gate_logits) || !contiguous2(desc->shared) || !contiguous2(desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->gate_logits.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->gate_logits.shape[1] != 1
        || desc->shared.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->shared.shape[1] != static_cast<int64_t>(desc->hidden_size)
        || desc->out.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->out.shape[1] != static_cast<int64_t>(desc->hidden_size)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_qwen36_full_attention_output_gate_desc(
    const qscu_qwen36_full_attention_output_gate_desc* desc
)
{
    if (desc == nullptr || desc->num_tokens == 0 || desc->q_hidden != kQwen36FullAttentionQHidden)
        return QSFI_STATUS_INVALID_ARGUMENT;

    qsfi_status status = validate_tensor(desc->gate, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous2(desc->gate) || !contiguous2(desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->gate.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->gate.shape[1] != static_cast<int64_t>(desc->q_hidden)
        || desc->out.shape[0] != static_cast<int64_t>(desc->num_tokens)
        || desc->out.shape[1] != static_cast<int64_t>(desc->q_hidden)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    if (contiguous_bf16_ranges_overlap(desc->gate, desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    return QSFI_STATUS_OK;
}

qsfi_status validate_embedding_gather_desc(const qscu_embedding_gather_desc* desc)
{
    if (desc == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;

    qsfi_status status = validate_tensor2(desc->token_ids, QSFI_DTYPE_I32, QSFI_DTYPE_U32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->embedding, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(desc->out, QSFI_DTYPE_BF16);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous1(desc->token_ids) || !contiguous2(desc->embedding) || !contiguous2(desc->out))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->out.shape[0] != desc->token_ids.shape[0]
        || desc->out.shape[1] != desc->embedding.shape[1]) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    if (desc->token_ids.shape[0] > std::numeric_limits<uint32_t>::max()
        || desc->embedding.shape[0] > std::numeric_limits<uint32_t>::max()
        || desc->embedding.shape[1] > std::numeric_limits<uint32_t>::max()) {
        return QSFI_STATUS_UNSUPPORTED;
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_logits_soft_cap_desc(
    const qsfi_tensor2* logits, uint32_t rows, uint32_t vocab_size, float soft_cap
)
{
    if (logits == nullptr || rows == 0 || vocab_size == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    // Host descriptor scalar: positive soft caps are used as a divisor and scale.
    if (!(soft_cap <= 0.0f) && !std::isfinite(soft_cap))
        return QSFI_STATUS_INVALID_ARGUMENT;

    qsfi_status status = validate_tensor(*logits, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    if (!contiguous2(*logits))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (logits->shape[0] != static_cast<int64_t>(rows)
        || logits->shape[1] != static_cast<int64_t>(vocab_size)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_greedy_argmax_desc(const qscu_sampling_desc* desc)
{
    if (desc == nullptr || desc->batch_size == 0 || desc->vocab_size == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    // Host descriptor scalars: NaN would silently bypass the unsupported sampling-mode checks.
    if (std::isnan(desc->temperature) || std::isnan(desc->top_p) || std::isnan(desc->min_p))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->temperature > 0.0f || desc->top_k != 0 || (desc->top_p > 0.0f && desc->top_p < 1.0f)
        || desc->min_p > 0.0f) {
        return QSFI_STATUS_UNSUPPORTED;
    }
    if (tensor_present(desc->selected_logprobs) || tensor_present(desc->selected_probs))
        return QSFI_STATUS_UNSUPPORTED;

    qsfi_status status = validate_tensor(desc->logits, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_tensor(desc->uniform_samples, QSFI_DTYPE_F32);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor2(desc->next_token_ids, QSFI_DTYPE_I32, QSFI_DTYPE_U32);
    if (status != QSFI_STATUS_OK)
        return status;

    if (!contiguous2(desc->logits) || !contiguous1(desc->next_token_ids))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (tensor_present(desc->uniform_samples) && !contiguous1(desc->uniform_samples))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (desc->logits.shape[0] != static_cast<int64_t>(desc->batch_size)
        || desc->logits.shape[1] != static_cast<int64_t>(desc->vocab_size)
        || desc->next_token_ids.shape[0] != static_cast<int64_t>(desc->batch_size)
        || (tensor_present(desc->uniform_samples)
            && desc->uniform_samples.shape[0] != static_cast<int64_t>(desc->batch_size))) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    if (desc->next_token_ids.dtype == QSFI_DTYPE_I32
        && desc->vocab_size > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        return QSFI_STATUS_UNSUPPORTED;
    }
    return QSFI_STATUS_OK;
}

} // namespace

extern "C" {

qsfi_status qscu_silu_and_mul_bf16(const qscu_silu_and_mul_desc* desc, qsfi_cuda_stream stream)
{
    qsfi_status status = validate_silu_and_mul_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;

    const uint64_t items
        = static_cast<uint64_t>(desc->num_tokens) * static_cast<uint64_t>(desc->intermediate_size);
    uint32_t blocks = 0;
    status = checked_grid(items, kElementwiseThreads, &blocks);
    if (status != QSFI_STATUS_OK)
        return status;

    silu_and_mul_params params {};
    params.gate = static_cast<const __nv_bfloat16*>(desc->gate.data);
    params.gate_stride0 = desc->gate.stride[0];
    params.gate_stride1 = desc->gate.stride[1];
    params.up = static_cast<const __nv_bfloat16*>(desc->up.data);
    params.up_stride0 = desc->up.stride[0];
    params.up_stride1 = desc->up.stride[1];
    params.out = static_cast<__nv_bfloat16*>(desc->out.data);
    params.out_stride0 = desc->out.stride[0];
    params.out_stride1 = desc->out.stride[1];
    params.intermediate_size = desc->intermediate_size;
    params.total_elements = items;

    silu_and_mul_bf16_kernel<<<blocks, kElementwiseThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        params
    );
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_qwen36_shared_expert_gate_add_bf16(
    const qscu_qwen36_shared_expert_gate_add_desc* desc, qsfi_cuda_stream stream
)
{
    qsfi_status status = validate_qwen36_shared_expert_gate_add_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;

    const uint64_t items
        = static_cast<uint64_t>(desc->num_tokens) * static_cast<uint64_t>(desc->hidden_size);
    uint32_t blocks = 0;
    status = checked_grid(items, kElementwiseThreads, &blocks);
    if (status != QSFI_STATUS_OK)
        return status;

    qwen36_shared_expert_gate_add_params params {};
    if (desc->gate_logits.dtype == QSFI_DTYPE_F32)
        params.gate_f32 = static_cast<const float*>(desc->gate_logits.data);
    else
        params.gate_bf16 = static_cast<const __nv_bfloat16*>(desc->gate_logits.data);
    params.gate_stride0 = desc->gate_logits.stride[0];
    params.shared = static_cast<const __nv_bfloat16*>(desc->shared.data);
    params.shared_stride0 = desc->shared.stride[0];
    params.shared_stride1 = desc->shared.stride[1];
    params.out = static_cast<__nv_bfloat16*>(desc->out.data);
    params.out_stride0 = desc->out.stride[0];
    params.out_stride1 = desc->out.stride[1];
    params.hidden_size = desc->hidden_size;
    params.total_elements = items;

    qwen36_shared_expert_gate_add_bf16_kernel<<<
        blocks,
        kElementwiseThreads,
        0,
        static_cast<cudaStream_t>(stream)>>>(params);
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_qwen36_full_attention_output_gate_bf16(
    const qscu_qwen36_full_attention_output_gate_desc* desc, qsfi_cuda_stream stream
)
{
    qsfi_status status = validate_qwen36_full_attention_output_gate_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;

    const uint64_t items
        = static_cast<uint64_t>(desc->num_tokens) * static_cast<uint64_t>(desc->q_hidden);
    uint32_t blocks = 0;
    status = checked_grid(items, kElementwiseThreads, &blocks);
    if (status != QSFI_STATUS_OK)
        return status;

    qwen36_full_attention_output_gate_params params {};
    params.gate = static_cast<const __nv_bfloat16*>(desc->gate.data);
    params.gate_stride0 = desc->gate.stride[0];
    params.gate_stride1 = desc->gate.stride[1];
    params.out = static_cast<__nv_bfloat16*>(desc->out.data);
    params.out_stride0 = desc->out.stride[0];
    params.out_stride1 = desc->out.stride[1];
    params.q_hidden = desc->q_hidden;
    params.total_elements = items;

    qwen36_full_attention_output_gate_bf16_kernel<<<
        blocks,
        kElementwiseThreads,
        0,
        static_cast<cudaStream_t>(stream)>>>(params);
    return validate_cuda(cudaGetLastError());
}

qsfi_status
qscu_embedding_gather_bf16(const qscu_embedding_gather_desc* desc, qsfi_cuda_stream stream)
{
    qsfi_status status = validate_embedding_gather_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;

    const uint32_t num_tokens = static_cast<uint32_t>(desc->token_ids.shape[0]);
    const uint32_t vocab_size = static_cast<uint32_t>(desc->embedding.shape[0]);
    const uint32_t hidden_size = static_cast<uint32_t>(desc->embedding.shape[1]);
    const uint64_t items = static_cast<uint64_t>(num_tokens) * static_cast<uint64_t>(hidden_size);
    uint32_t blocks = 0;
    status = checked_grid(items, kElementwiseThreads, &blocks);
    if (status != QSFI_STATUS_OK)
        return status;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int* device_invalid_token = nullptr;
#if QSFI_ENABLE_CHECKED_VALIDATION
    qscu_status_reporter errors;
    qsfi_checked_validation_flag invalid_token(cuda_stream);
    if (desc->validate_token_ids != 0) {
        status = invalid_token.reset(
            errors,
            "cudaMalloc embedding token validation flag",
            "cudaMemsetAsync embedding token validation flag"
        );
        if (status != QSFI_STATUS_OK)
            return status;
        device_invalid_token = invalid_token.device_ptr();
    }
#endif

    if (desc->token_ids.dtype == QSFI_DTYPE_I32) {
        embedding_gather_params<int32_t> params {};
        params.token_ids = static_cast<const int32_t*>(desc->token_ids.data);
        params.token_stride0 = desc->token_ids.stride[0];
        params.embedding = static_cast<const __nv_bfloat16*>(desc->embedding.data);
        params.embedding_stride0 = desc->embedding.stride[0];
        params.embedding_stride1 = desc->embedding.stride[1];
        params.out = static_cast<__nv_bfloat16*>(desc->out.data);
        params.out_stride0 = desc->out.stride[0];
        params.out_stride1 = desc->out.stride[1];
        params.hidden_size = hidden_size;
        params.vocab_size = vocab_size;
        params.padding_token_id = desc->padding_token_id;
        params.invalid_token = device_invalid_token;
        params.total_elements = items;
        embedding_gather_bf16_kernel<<<blocks, kElementwiseThreads, 0, cuda_stream>>>(params);
    } else {
        embedding_gather_params<uint32_t> params {};
        params.token_ids = static_cast<const uint32_t*>(desc->token_ids.data);
        params.token_stride0 = desc->token_ids.stride[0];
        params.embedding = static_cast<const __nv_bfloat16*>(desc->embedding.data);
        params.embedding_stride0 = desc->embedding.stride[0];
        params.embedding_stride1 = desc->embedding.stride[1];
        params.out = static_cast<__nv_bfloat16*>(desc->out.data);
        params.out_stride0 = desc->out.stride[0];
        params.out_stride1 = desc->out.stride[1];
        params.hidden_size = hidden_size;
        params.vocab_size = vocab_size;
        params.padding_token_id = desc->padding_token_id;
        params.invalid_token = device_invalid_token;
        params.total_elements = items;
        embedding_gather_bf16_kernel<<<blocks, kElementwiseThreads, 0, cuda_stream>>>(params);
    }

    status = validate_cuda(cudaGetLastError());
#if QSFI_ENABLE_CHECKED_VALIDATION
    if (device_invalid_token == nullptr || status != QSFI_STATUS_OK) {
        return status;
    }

    return invalid_token.finish(
        errors,
        "copy embedding token validation flag",
        "cudaFree embedding token validation flag",
        "embedding token ids are out of range"
    );
#else
    return status;
#endif
}

qsfi_status qscu_logits_soft_cap_f32(
    const qsfi_tensor2* logits,
    uint32_t rows,
    uint32_t vocab_size,
    float soft_cap,
    qsfi_cuda_stream stream
)
{
    qsfi_status status = validate_logits_soft_cap_desc(logits, rows, vocab_size, soft_cap);
    if (status != QSFI_STATUS_OK)
        return status;
    if (soft_cap <= 0.0f)
        return QSFI_STATUS_OK;

    const uint64_t items = static_cast<uint64_t>(rows) * static_cast<uint64_t>(vocab_size);
    uint32_t blocks = 0;
    status = checked_grid(items, kElementwiseThreads, &blocks);
    if (status != QSFI_STATUS_OK)
        return status;

    logits_soft_cap_params params {};
    params.logits = static_cast<float*>(logits->data);
    params.stride0 = logits->stride[0];
    params.stride1 = logits->stride[1];
    params.vocab_size = vocab_size;
    params.soft_cap = soft_cap;
    params.total_elements = items;

    logits_soft_cap_f32_kernel<<<
        blocks,
        kElementwiseThreads,
        0,
        static_cast<cudaStream_t>(stream)>>>(params);
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_greedy_argmax_f32(const qscu_sampling_desc* desc, qsfi_cuda_stream stream)
{
    qsfi_status status = validate_greedy_argmax_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

#if QSFI_ENABLE_CHECKED_VALIDATION
    status = validate_finite_tensor2_contents(
        desc->logits,
        desc->batch_size,
        desc->vocab_size,
        cuda_stream,
        "qscu greedy argmax logits contain non-finite values"
    );
    if (status != QSFI_STATUS_OK)
        return status;
#endif

    greedy_argmax_params params {};
    params.logits = static_cast<const float*>(desc->logits.data);
    params.logits_stride0 = desc->logits.stride[0];
    params.logits_stride1 = desc->logits.stride[1];
    if (desc->next_token_ids.dtype == QSFI_DTYPE_I32)
        params.out_i32 = static_cast<int32_t*>(desc->next_token_ids.data);
    else
        params.out_u32 = static_cast<uint32_t*>(desc->next_token_ids.data);
    params.out_stride0 = desc->next_token_ids.stride[0];
    params.vocab_size = desc->vocab_size;

    greedy_argmax_f32_kernel<<<desc->batch_size, kArgmaxThreads, 0, cuda_stream>>>(params);
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_qwen36_gdn_causal_conv1d_bf16(
    const qscu_qwen36_gdn_causal_conv1d_desc* desc, qsfi_cuda_stream stream
)
{
    qsfi_status status = validate_conv_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    status = validate_conv1d_metadata(desc, cuda_stream);
    if (status != QSFI_STATUS_OK)
        return status;

    conv1d_params params {};
    params.x = static_cast<const __nv_bfloat16*>(desc->x.data);
    params.x_stride0 = desc->x.stride[0];
    params.x_stride1 = desc->x.stride[1];
    params.weight = static_cast<const __nv_bfloat16*>(desc->weight.data);
    params.weight_stride0 = desc->weight.stride[0];
    params.weight_stride1 = desc->weight.stride[1];
    if (tensor_present(desc->bias)) {
        if (desc->bias.dtype == QSFI_DTYPE_BF16)
            params.bias_bf16 = static_cast<const __nv_bfloat16*>(desc->bias.data);
        else
            params.bias_f32 = static_cast<const float*>(desc->bias.data);
        params.bias_stride0 = desc->bias.stride[0];
    }
    params.state_stride0 = desc->state.stride[0];
    params.state_stride1 = desc->state.stride[1];
    params.state_stride2 = desc->state.stride[2];
    params.read_indices = static_cast<const int32_t*>(desc->state_read_indices.data);
    params.write_indices = static_cast<const int32_t*>(desc->state_write_indices.data);
    params.seq_indptr = static_cast<const int32_t*>(desc->seq_indptr);
    params.out = static_cast<__nv_bfloat16*>(desc->out.data);
    params.out_stride0 = desc->out.stride[0];
    params.out_stride1 = desc->out.stride[1];
    params.conv_dim = kQwen36GdnPackedDim;
    params.activation = desc->activation;
    params.update_state = desc->update_state != 0 ? 1u : 0u;

    if (desc->state.dtype == QSFI_DTYPE_BF16) {
        qwen36_gdn_causal_conv1d_kernel<<<desc->batch_size, 256, 0, cuda_stream>>>(
            params,
            static_cast<__nv_bfloat16*>(desc->state.data)
        );
    } else {
        qwen36_gdn_causal_conv1d_kernel<<<desc->batch_size, 256, 0, cuda_stream>>>(
            params,
            static_cast<float*>(desc->state.data)
        );
    }
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_qwen36_gdn_post_conv_prepare_bf16(
    const qscu_qwen36_gdn_post_conv_prepare_desc* desc, qsfi_cuda_stream stream
)
{
    qsfi_status status = validate_post_conv_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;

    post_conv_params params {};
    params.conv_out = static_cast<const __nv_bfloat16*>(desc->conv_out.data);
    params.conv_stride0 = desc->conv_out.stride[0];
    params.conv_stride1 = desc->conv_out.stride[1];
    params.a = static_cast<const __nv_bfloat16*>(desc->a.data);
    params.a_stride0 = desc->a.stride[0];
    params.a_stride1 = desc->a.stride[1];
    params.b = static_cast<const __nv_bfloat16*>(desc->b.data);
    params.b_stride0 = desc->b.stride[0];
    params.b_stride1 = desc->b.stride[1];
    params.a_log = static_cast<const float*>(desc->a_log.data);
    params.a_log_stride0 = desc->a_log.stride[0];
    params.dt_bias = static_cast<const float*>(desc->dt_bias.data);
    params.dt_bias_stride0 = desc->dt_bias.stride[0];
    params.q = static_cast<__nv_bfloat16*>(desc->q.data);
    params.q_stride0 = desc->q.stride[0];
    params.q_stride1 = desc->q.stride[1];
    params.q_stride2 = desc->q.stride[2];
    params.k = static_cast<__nv_bfloat16*>(desc->k.data);
    params.k_stride0 = desc->k.stride[0];
    params.k_stride1 = desc->k.stride[1];
    params.k_stride2 = desc->k.stride[2];
    params.v = static_cast<__nv_bfloat16*>(desc->v.data);
    params.v_stride0 = desc->v.stride[0];
    params.v_stride1 = desc->v.stride[1];
    params.v_stride2 = desc->v.stride[2];
    if (tensor_present(desc->g_out)) {
        params.g_out = static_cast<float*>(desc->g_out.data);
        params.g_stride0 = desc->g_out.stride[0];
        params.g_stride1 = desc->g_out.stride[1];
    }
    if (tensor_present(desc->beta_out)) {
        params.beta_out = static_cast<float*>(desc->beta_out.data);
        params.beta_stride0 = desc->beta_out.stride[0];
        params.beta_stride1 = desc->beta_out.stride[1];
    }
    params.l2norm_eps = desc->l2norm_eps;
    params.forget_gate_output = desc->forget_gate_output;
    params.apply_qk_l2norm = desc->apply_qk_l2norm != 0 ? 1u : 0u;

    const uint64_t items
        = static_cast<uint64_t>(desc->num_tokens) * (kQwen36GdnNumKHeads + kQwen36GdnNumVHeads);
    if (items > std::numeric_limits<uint32_t>::max())
        return QSFI_STATUS_UNSUPPORTED;
    qwen36_gdn_post_conv_prepare_kernel<<<
        static_cast<uint32_t>(items),
        kQwen36GdnThreads,
        0,
        static_cast<cudaStream_t>(stream)>>>(params);
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_qwen36_gdn_rmsnorm_gated_bf16(
    const qscu_qwen36_gdn_rmsnorm_gated_desc* desc, qsfi_cuda_stream stream
)
{
    qsfi_status status = validate_rmsnorm_gated_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;

    rmsnorm_gated_params params {};
    params.x = static_cast<const __nv_bfloat16*>(desc->x.data);
    params.x_stride0 = desc->x.stride[0];
    params.x_stride1 = desc->x.stride[1];
    params.x_stride2 = desc->x.stride[2];
    params.gate = static_cast<const __nv_bfloat16*>(desc->gate.data);
    params.gate_stride0 = desc->gate.stride[0];
    params.gate_stride1 = desc->gate.stride[1];
    params.gate_stride2 = desc->gate.stride[2];
    if (desc->weight.dtype == QSFI_DTYPE_F32)
        params.weight_f32 = static_cast<const float*>(desc->weight.data);
    else
        params.weight_bf16 = static_cast<const __nv_bfloat16*>(desc->weight.data);
    params.weight_stride0 = desc->weight.stride[0];
    params.out = static_cast<__nv_bfloat16*>(desc->out.data);
    params.out_stride0 = desc->out.stride[0];
    params.out_stride1 = desc->out.stride[1];
    params.out_stride2 = desc->out.stride[2];
    params.eps = desc->eps;
    params.gate_activation = desc->gate_activation;

    const uint64_t items = static_cast<uint64_t>(desc->num_tokens) * kQwen36GdnNumVHeads;
    if (items > std::numeric_limits<uint32_t>::max())
        return QSFI_STATUS_UNSUPPORTED;
    qwen36_gdn_rmsnorm_gated_kernel<<<
        static_cast<uint32_t>(items),
        kQwen36GdnThreads,
        0,
        static_cast<cudaStream_t>(stream)>>>(params);
    return validate_cuda(cudaGetLastError());
}

qsfi_status qscu_router_topk(const qscu_router_topk_desc* desc, qsfi_cuda_stream stream)
{
    qsfi_status status = validate_router_desc(desc);
    if (status != QSFI_STATUS_OK)
        return status;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

#if QSFI_ENABLE_CHECKED_VALIDATION
    status = validate_finite_tensor2_contents(
        desc->logits,
        desc->num_tokens,
        desc->num_experts,
        cuda_stream,
        "qscu router logits contain non-finite values"
    );
    if (status != QSFI_STATUS_OK)
        return status;
#endif

    router_topk_params params {};
    if (desc->logits.dtype == QSFI_DTYPE_F32)
        params.logits_f32 = static_cast<const float*>(desc->logits.data);
    else
        params.logits_bf16 = static_cast<const __nv_bfloat16*>(desc->logits.data);
    params.logits_stride0 = desc->logits.stride[0];
    params.logits_stride1 = desc->logits.stride[1];
    params.topk_ids = static_cast<int32_t*>(desc->topk_ids.data);
    params.ids_stride0 = desc->topk_ids.stride[0];
    params.ids_stride1 = desc->topk_ids.stride[1];
    params.topk_weights = static_cast<float*>(desc->topk_weights.data);
    params.weights_stride0 = desc->topk_weights.stride[0];
    params.weights_stride1 = desc->topk_weights.stride[1];
    params.num_experts = desc->num_experts;
    params.top_k = desc->top_k;
    params.score = desc->score;
    params.renormalize = desc->renormalize != 0 ? 1u : 0u;
    params.routed_scaling_factor = desc->routed_scaling_factor;

    router_topk_kernel<<<desc->num_tokens, 1, 0, cuda_stream>>>(params);
    return validate_cuda(cudaGetLastError());
}

} // extern "C"
