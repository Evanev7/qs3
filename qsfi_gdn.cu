#include "qsfi_internal.h"
#include "qsfi_macros.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kDefaultNumQHeads = QSFI_QWEN36_GDN_NUM_Q_HEADS;
constexpr uint32_t kDefaultNumKHeads = QSFI_QWEN36_GDN_NUM_K_HEADS;
constexpr uint32_t kDefaultNumVHeads = QSFI_QWEN36_GDN_NUM_V_HEADS;
constexpr uint32_t kDefaultKeyDim = QSFI_QWEN36_GDN_KEY_DIM;
constexpr uint32_t kDefaultValueDim = QSFI_QWEN36_GDN_VALUE_DIM;
constexpr uint32_t kThreads = QSFI_QWEN36_GDN_THREADS;
static_assert(kThreads == kDefaultKeyDim, "GDN row kernel maps one thread to one K element");

struct gdn_shape {
    uint32_t num_q_heads;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    float scale;
    bool use_qk_l2norm;
    bool disable_state_update;
};

struct tensor1_bf16_view {
    const __nv_bfloat16* data;
    int64_t stride0;
};

struct tensor1_f32_view {
    const float* data;
    int64_t stride0;
};

struct tensor2_bf16_view {
    const __nv_bfloat16* data;
    int64_t stride0;
    int64_t stride1;
};

struct tensor3_bf16_view {
    const __nv_bfloat16* data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
};

struct tensor3_bf16_out_view {
    __nv_bfloat16* data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
};

struct state_view {
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    int64_t stride3;
};

struct gdn_kernel_params {
    tensor3_bf16_view q;
    tensor3_bf16_view k;
    tensor3_bf16_view v;
    tensor2_bf16_view a;
    tensor2_bf16_view b;
    tensor1_f32_view a_log;
    tensor1_f32_view dt_bias;
    state_view state;
    const int32_t* seq_indptr;
    const int32_t* state_indices;
    const int32_t* state_out_indices;
    tensor3_bf16_out_view out;
    uint32_t outer_count;
    uint32_t num_q_heads;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    float scale;
    uint32_t use_qk_l2norm;
    uint32_t disable_state_update;
};

template <typename Desc> gdn_shape shape_from_desc(const Desc* desc)
{
    gdn_shape shape {};
    shape.num_q_heads = desc->num_q_heads;
    shape.num_k_heads = desc->num_k_heads;
    shape.num_v_heads = desc->num_v_heads;
    shape.key_dim = desc->key_dim;
    shape.value_dim = desc->value_dim;
    shape.scale = desc->scale;
    shape.use_qk_l2norm = desc->use_qk_l2norm != 0;
    shape.disable_state_update = desc->disable_state_update != 0;
    return shape;
}

tensor3_bf16_view tensor3_in(const qsfi_tensor3& tensor)
{
    return {
        static_cast<const __nv_bfloat16*>(tensor.data),
        tensor.stride[0],
        tensor.stride[1],
        tensor.stride[2],
    };
}

tensor3_bf16_out_view tensor3_out(const qsfi_tensor3& tensor)
{
    return {
        static_cast<__nv_bfloat16*>(tensor.data),
        tensor.stride[0],
        tensor.stride[1],
        tensor.stride[2],
    };
}

tensor2_bf16_view tensor2_in(const qsfi_tensor2& tensor)
{
    return { static_cast<const __nv_bfloat16*>(tensor.data), tensor.stride[0], tensor.stride[1] };
}

tensor1_f32_view tensor1_f32_in(const qsfi_tensor1& tensor)
{
    return { static_cast<const float*>(tensor.data), tensor.stride[0] };
}

state_view state_inout(const qsfi_tensor4& tensor)
{
    return { tensor.stride[0], tensor.stride[1], tensor.stride[2], tensor.stride[3] };
}

template <typename T> __device__ float load_state_value(const T* ptr)
{
    return static_cast<float>(*ptr);
}

template <> __device__ float load_state_value(const __nv_bfloat16* ptr)
{
    return __bfloat162float(*ptr);
}

template <typename T> __device__ T make_state_value(float value)
{
    return static_cast<T>(value);
}

template <> __device__ __nv_bfloat16 make_state_value(float value)
{
    return __float2bfloat16(value);
}

__device__ float load_bf16(const __nv_bfloat16* ptr)
{
    return __bfloat162float(*ptr);
}

__device__ void store_bf16(__nv_bfloat16* ptr, float value)
{
    *ptr = __float2bfloat16(value);
}

__device__ float block_sum(float value)
{
    __shared__ float scratch[kThreads];
    const uint32_t tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float sum = scratch[0];
    __syncthreads();
    return sum;
}

__device__ float softplus(float x, float beta, float threshold)
{
    const float beta_x = beta * x;
    if (beta_x <= threshold)
        return log1pf(expf(beta_x)) / beta;
    return x;
}

__device__ uint32_t mapped_head(uint32_t v_head, uint32_t v_heads, uint32_t mapped_heads)
{
    return v_head / (v_heads / mapped_heads);
}

template <typename StateT>
__device__ void run_gdn_sequence_row(
    const gdn_kernel_params& p,
    StateT* state,
    uint32_t sequence_or_token,
    uint32_t v_head,
    uint32_t value_dim,
    int32_t token_begin,
    int32_t token_end,
    int32_t read_slot,
    int32_t write_slot
)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t q_head = mapped_head(v_head, p.num_v_heads, p.num_q_heads);
    const uint32_t k_head = mapped_head(v_head, p.num_v_heads, p.num_k_heads);

    if (read_slot < 0) {
        if (tid == 0) {
            for (int32_t token = token_begin; token < token_end; ++token) {
                store_bf16(
                    p.out.data + token * p.out.stride0 + v_head * p.out.stride1
                        + value_dim * p.out.stride2,
                    0.0f
                );
            }
        }
        return;
    }

    const int64_t read_base = static_cast<int64_t>(read_slot) * p.state.stride0
        + static_cast<int64_t>(v_head) * p.state.stride1
        + static_cast<int64_t>(value_dim) * p.state.stride2;
    float h = load_state_value(state + read_base + static_cast<int64_t>(tid) * p.state.stride3);

    for (int32_t token = token_begin; token < token_end; ++token) {
        const float q_raw = load_bf16(
            p.q.data + static_cast<int64_t>(token) * p.q.stride0
            + static_cast<int64_t>(q_head) * p.q.stride1 + tid * p.q.stride2
        );
        const float k_raw = load_bf16(
            p.k.data + static_cast<int64_t>(token) * p.k.stride0
            + static_cast<int64_t>(k_head) * p.k.stride1 + tid * p.k.stride2
        );

        float q_factor = p.scale;
        float k_factor = 1.0f;
        if (p.use_qk_l2norm != 0) {
            const float q_norm2 = block_sum(q_raw * q_raw);
            const float k_norm2 = block_sum(k_raw * k_raw);
            q_factor *= rsqrtf(q_norm2 + 1.0e-8f);
            k_factor *= rsqrtf(k_norm2 + 1.0e-8f);
        }
        const float q_value = q_raw * q_factor;
        const float k_value = k_raw * k_factor;

        const float a_value = load_bf16(
            p.a.data + static_cast<int64_t>(token) * p.a.stride0
            + static_cast<int64_t>(v_head) * p.a.stride1
        );
        const float b_value = load_bf16(
            p.b.data + static_cast<int64_t>(token) * p.b.stride0
            + static_cast<int64_t>(v_head) * p.b.stride1
        );
        const float a_log = p.a_log.data[static_cast<int64_t>(v_head) * p.a_log.stride0];
        const float dt_bias = p.dt_bias.data[static_cast<int64_t>(v_head) * p.dt_bias.stride0];
        const float g = -expf(a_log)
            * softplus(
                a_value + dt_bias,
                QSFI_QWEN36_GDN_SOFTPLUS_BETA,
                QSFI_QWEN36_GDN_SOFTPLUS_THRESHOLD
            );
        const float beta_gate = 1.0f / (1.0f + expf(-b_value));

        h *= expf(g);
        const float kv_dot = block_sum(k_value * h);
        const float v_value = load_bf16(
            p.v.data + static_cast<int64_t>(token) * p.v.stride0
            + static_cast<int64_t>(v_head) * p.v.stride1
            + static_cast<int64_t>(value_dim) * p.v.stride2
        );
        const float delta_v = (v_value - kv_dot) * beta_gate;
        h += k_value * delta_v;

        const float out_value = block_sum(q_value * h);
        if (tid == 0) {
            store_bf16(
                p.out.data + static_cast<int64_t>(token) * p.out.stride0
                    + static_cast<int64_t>(v_head) * p.out.stride1
                    + static_cast<int64_t>(value_dim) * p.out.stride2,
                out_value
            );
        }
    }

    if (p.disable_state_update == 0 && write_slot >= 0) {
        const int64_t write_base = static_cast<int64_t>(write_slot) * p.state.stride0
            + static_cast<int64_t>(v_head) * p.state.stride1
            + static_cast<int64_t>(value_dim) * p.state.stride2;
        state[write_base + static_cast<int64_t>(tid) * p.state.stride3]
            = make_state_value<StateT>(h);
    }
}

template <typename StateT> __global__ void gdn_decode_kernel(gdn_kernel_params p, StateT* state)
{
    const uint64_t linear = blockIdx.x;
    const uint32_t value_dim = static_cast<uint32_t>(linear % p.value_dim);
    const uint32_t v_head = static_cast<uint32_t>((linear / p.value_dim) % p.num_v_heads);
    const uint32_t token = static_cast<uint32_t>(linear / (p.value_dim * p.num_v_heads));
    const int32_t read_slot = p.state_indices[token];
    const int32_t write_slot
        = p.state_out_indices == nullptr ? read_slot : p.state_out_indices[token];
    run_gdn_sequence_row(
        p,
        state,
        token,
        v_head,
        value_dim,
        static_cast<int32_t>(token),
        static_cast<int32_t>(token + 1),
        read_slot,
        write_slot
    );
}

template <typename StateT> __global__ void gdn_prefill_kernel(gdn_kernel_params p, StateT* state)
{
    const uint64_t linear = blockIdx.x;
    const uint32_t value_dim = static_cast<uint32_t>(linear % p.value_dim);
    const uint32_t v_head = static_cast<uint32_t>((linear / p.value_dim) % p.num_v_heads);
    const uint32_t sequence = static_cast<uint32_t>(linear / (p.value_dim * p.num_v_heads));
    const int32_t token_begin = p.seq_indptr[sequence];
    const int32_t token_end = p.seq_indptr[sequence + 1];
    const int32_t read_slot = p.state_indices[sequence];
    const int32_t write_slot
        = p.state_out_indices == nullptr ? read_slot : p.state_out_indices[sequence];
    run_gdn_sequence_row(
        p,
        state,
        sequence,
        v_head,
        value_dim,
        token_begin,
        token_end,
        read_slot,
        write_slot
    );
}

qsfi_status require_exact_shape(qsfi_context* ctx, const gdn_shape& shape)
{
    if (shape.num_q_heads != kDefaultNumQHeads || shape.num_k_heads != kDefaultNumKHeads
        || shape.num_v_heads != kDefaultNumVHeads || shape.key_dim != kDefaultKeyDim
        || shape.value_dim != kDefaultValueDim) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "only qwen3.6 GDN shape q=%u k=%u v=%u key_dim=%u value_dim=%u is wired",
            kDefaultNumQHeads,
            kDefaultNumKHeads,
            kDefaultNumVHeads,
            kDefaultKeyDim,
            kDefaultValueDim
        );
    }
    return QSFI_STATUS_OK;
}

template <typename Desc> qsfi_status require_explicit_gdn_desc(qsfi_context* ctx, const Desc* desc)
{
    const struct {
        const char* name;
        bool is_set;
    } fields[] = {
        { "gdn.num_q_heads", desc->num_q_heads != 0 },
        { "gdn.num_k_heads", desc->num_k_heads != 0 },
        { "gdn.num_v_heads", desc->num_v_heads != 0 },
        { "gdn.key_dim", desc->key_dim != 0 },
        { "gdn.value_dim", desc->value_dim != 0 },
        { "gdn.scale", desc->scale != 0.0f },
    };

    for (const auto& field : fields) {
        if (!field.is_set) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "%s must not be zero",
                field.name
            );
        }
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_index_tensor(
    qsfi_context* ctx, const qsfi_tensor1& tensor, const char* name, uint32_t expected
)
{
    qsfi_status status = validate_tensor(ctx, tensor, name, QSFI_DTYPE_I32, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    if (tensor.shape[0] != static_cast<int64_t>(expected)) {
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

qsfi_status validate_optional_index_tensor(
    qsfi_context* ctx, const qsfi_tensor1& tensor, const char* name, uint32_t expected
)
{
    if (tensor.data == nullptr)
        return QSFI_STATUS_OK;
    return validate_index_tensor(ctx, tensor, name, expected);
}

qsfi_status validate_gdn_tensors(
    qsfi_context* ctx,
    const gdn_shape& shape,
    const qsfi_tensor3& q,
    const qsfi_tensor3& k,
    const qsfi_tensor3& v,
    const qsfi_tensor2& a,
    const qsfi_tensor2& b,
    const qsfi_tensor1& a_log,
    const qsfi_tensor1& dt_bias,
    const qsfi_tensor4& state,
    const qsfi_tensor3& out,
    uint32_t total_tokens
)
{
    qsfi_status status = validate_tensor(ctx, q, "gdn.q", QSFI_DTYPE_BF16, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, k, "gdn.k", QSFI_DTYPE_BF16, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, v, "gdn.v", QSFI_DTYPE_BF16, 3);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, a, "gdn.a", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, b, "gdn.b", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, a_log, "gdn.a_log", QSFI_DTYPE_F32, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, dt_bias, "gdn.dt_bias", QSFI_DTYPE_F32, 1);
    if (status != QSFI_STATUS_OK)
        return status;
    if (state.dtype != QSFI_DTYPE_BF16 && state.dtype != QSFI_DTYPE_F32) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "gdn.state must be bf16 or f32"
        );
    }
    status = validate_tensor(ctx, state, "gdn.state", state.dtype, 4);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_tensor(ctx, out, "gdn.out", QSFI_DTYPE_BF16, 3);
    if (status != QSFI_STATUS_OK)
        return status;

    if (q.shape[0] != static_cast<int64_t>(total_tokens)
        || q.shape[1] != static_cast<int64_t>(shape.num_q_heads)
        || q.shape[2] != static_cast<int64_t>(shape.key_dim)
        || k.shape[0] != static_cast<int64_t>(total_tokens)
        || k.shape[1] != static_cast<int64_t>(shape.num_k_heads)
        || k.shape[2] != static_cast<int64_t>(shape.key_dim)
        || v.shape[0] != static_cast<int64_t>(total_tokens)
        || v.shape[1] != static_cast<int64_t>(shape.num_v_heads)
        || v.shape[2] != static_cast<int64_t>(shape.value_dim)
        || a.shape[0] != static_cast<int64_t>(total_tokens)
        || a.shape[1] != static_cast<int64_t>(shape.num_v_heads)
        || b.shape[0] != static_cast<int64_t>(total_tokens)
        || b.shape[1] != static_cast<int64_t>(shape.num_v_heads)
        || a_log.shape[0] != static_cast<int64_t>(shape.num_v_heads)
        || dt_bias.shape[0] != static_cast<int64_t>(shape.num_v_heads)
        || state.shape[1] != static_cast<int64_t>(shape.num_v_heads)
        || state.shape[2] != static_cast<int64_t>(shape.value_dim)
        || state.shape[3] != static_cast<int64_t>(shape.key_dim)
        || out.shape[0] != static_cast<int64_t>(total_tokens)
        || out.shape[1] != static_cast<int64_t>(shape.num_v_heads)
        || out.shape[2] != static_cast<int64_t>(shape.value_dim)) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "gdn tensor shape mismatch"
        );
    }
    return QSFI_STATUS_OK;
}

template <typename Desc>
gdn_kernel_params make_params(
    const Desc* desc, const gdn_shape& shape, uint32_t outer_count, const int32_t* seq_indptr
)
{
    gdn_kernel_params params {};
    params.q = tensor3_in(desc->q);
    params.k = tensor3_in(desc->k);
    params.v = tensor3_in(desc->v);
    params.a = tensor2_in(desc->a);
    params.b = tensor2_in(desc->b);
    params.a_log = tensor1_f32_in(desc->a_log);
    params.dt_bias = tensor1_f32_in(desc->dt_bias);
    params.state = state_inout(desc->state);
    params.seq_indptr = seq_indptr;
    params.state_indices = static_cast<const int32_t*>(desc->state_indices.data);
    params.state_out_indices = static_cast<const int32_t*>(desc->state_out_indices.data);
    params.out = tensor3_out(desc->out);
    params.outer_count = outer_count;
    params.num_q_heads = shape.num_q_heads;
    params.num_k_heads = shape.num_k_heads;
    params.num_v_heads = shape.num_v_heads;
    params.key_dim = shape.key_dim;
    params.value_dim = shape.value_dim;
    params.scale = shape.scale;
    params.use_qk_l2norm = shape.use_qk_l2norm ? 1u : 0u;
    params.disable_state_update = shape.disable_state_update ? 1u : 0u;
    return params;
}

uint64_t work_items(uint32_t outer_count, const gdn_shape& shape)
{
    return static_cast<uint64_t>(outer_count) * shape.num_v_heads * shape.value_dim;
}

qsfi_status check_work_items(qsfi_context* ctx, uint64_t items)
{
    if (items > std::numeric_limits<uint32_t>::max()) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "gdn launch grid is too large"
        );
    }
    return QSFI_STATUS_OK;
}

template <typename StateT>
cudaError_t
launch_gdn_decode(qsfi_context* ctx, const qsfi_gdn_decode_desc* desc, const gdn_shape& shape)
{
    const uint64_t items = work_items(desc->num_tokens, shape);
    gdn_kernel_params params = make_params(desc, shape, desc->num_tokens, nullptr);
    gdn_decode_kernel<StateT><<<static_cast<uint32_t>(items), kThreads, 0, ctx->stream>>>(
        params,
        static_cast<StateT*>(desc->state.data)
    );
    return cudaGetLastError();
}

template <typename StateT>
cudaError_t
launch_gdn_prefill(qsfi_context* ctx, const qsfi_gdn_prefill_desc* desc, const gdn_shape& shape)
{
    const uint64_t items = work_items(desc->batch_size, shape);
    gdn_kernel_params params
        = make_params(desc, shape, desc->batch_size, static_cast<const int32_t*>(desc->seq_indptr));
    gdn_prefill_kernel<StateT><<<static_cast<uint32_t>(items), kThreads, 0, ctx->stream>>>(
        params,
        static_cast<StateT*>(desc->state.data)
    );
    return cudaGetLastError();
}

} // namespace

extern "C" {

qsfi_status qsfi_gdn_decode(qsfi_context* ctx, const qsfi_gdn_decode_desc* desc)
{
    if (ctx == nullptr || desc == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    if (desc->num_tokens == 0) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "gdn decode num_tokens must be non-zero"
        );
    }
    if (desc->state_layout != QSFI_GDN_STATE_LAYOUT_VK) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "only VK GDN state layout is wired"
        );
    }

    status = require_explicit_gdn_desc(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;

    const gdn_shape shape = shape_from_desc(desc);
    status = require_exact_shape(ctx, shape);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_gdn_tensors(
        ctx,
        shape,
        desc->q,
        desc->k,
        desc->v,
        desc->a,
        desc->b,
        desc->a_log,
        desc->dt_bias,
        desc->state,
        desc->out,
        desc->num_tokens
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_index_tensor(ctx, desc->state_indices, "gdn.state_indices", desc->num_tokens);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_index_tensor(
        ctx,
        desc->state_out_indices,
        "gdn.state_out_indices",
        desc->num_tokens
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = check_work_items(ctx, work_items(desc->num_tokens, shape));
    if (status != QSFI_STATUS_OK)
        return status;

    cudaError_t err = cudaSuccess;
    if (desc->state.dtype == QSFI_DTYPE_BF16) {
        err = launch_gdn_decode<__nv_bfloat16>(ctx, desc, shape);
    } else {
        err = launch_gdn_decode<float>(ctx, desc, shape);
    }
    return set_cuda_error(ctx, err, "qsfi_gdn_decode");
}

qsfi_status qsfi_gdn_prefill(qsfi_context* ctx, const qsfi_gdn_prefill_desc* desc)
{
    if (ctx == nullptr || desc == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    if (desc->batch_size == 0 || desc->total_tokens == 0 || desc->seq_indptr == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "gdn prefill batch_size, total_tokens, and seq_indptr must be set"
        );
    }
    if (desc->state_layout != QSFI_GDN_STATE_LAYOUT_VK) {
        return set_error(
            ctx,
            QSFI_STATUS_UNSUPPORTED,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "only VK GDN state layout is wired"
        );
    }

    status = require_explicit_gdn_desc(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;

    const gdn_shape shape = shape_from_desc(desc);
    status = require_exact_shape(ctx, shape);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_gdn_tensors(
        ctx,
        shape,
        desc->q,
        desc->k,
        desc->v,
        desc->a,
        desc->b,
        desc->a_log,
        desc->dt_bias,
        desc->state,
        desc->out,
        desc->total_tokens
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_index_tensor(ctx, desc->state_indices, "gdn.state_indices", desc->batch_size);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_optional_index_tensor(
        ctx,
        desc->state_out_indices,
        "gdn.state_out_indices",
        desc->batch_size
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = check_work_items(ctx, work_items(desc->batch_size, shape));
    if (status != QSFI_STATUS_OK)
        return status;

    cudaError_t err = cudaSuccess;
    if (desc->state.dtype == QSFI_DTYPE_BF16) {
        err = launch_gdn_prefill<__nv_bfloat16>(ctx, desc, shape);
    } else {
        err = launch_gdn_prefill<float>(ctx, desc, shape);
    }
    return set_cuda_error(ctx, err, "qsfi_gdn_prefill");
}

} // extern "C"
