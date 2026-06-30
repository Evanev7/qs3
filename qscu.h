#ifndef QSCU_H
#define QSCU_H

#include "qs_info.h"
#include "qs_tensor.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qsfi_context qsfi_context;

/*
 * qscu is a small local CUDA helper layer for qwen3.6 kernels that do not map
 * cleanly onto a packed FlashInfer API. Helpers launch on the supplied stream;
 * callers are responsible for setting the CUDA device and for mapping the
 * returned status into qsfi_context error state.
 */

/*
 * Split-pointer SwiGLU: out = silu(gate) * up for BF16 row-major tensors.
 * Requires gate/up/out shapes to match desc->num_tokens x desc->intermediate_size.
 * Inner stride must be 1.
 */
typedef struct {
    qsfi_tensor2 gate;
    qsfi_tensor2 up;
    qsfi_tensor2 out;
    uint32_t num_tokens;
    uint32_t intermediate_size;
} qscu_silu_and_mul_desc;

qsfi_status qscu_silu_and_mul_bf16(const qscu_silu_and_mul_desc* desc, qsfi_cuda_stream stream);

/*
 * Qwen3.6 shared expert combine: out += sigmoid(gate_logits[row, 0]) * shared.
 * gate_logits may be bf16/f32 [num_tokens, 1]. shared/out are bf16
 * [num_tokens, hidden_size]. All tensors must be contiguous row-major.
 */
typedef struct {
    qsfi_tensor2 gate_logits;
    qsfi_tensor2 shared;
    qsfi_tensor2 out;
    uint32_t num_tokens;
    uint32_t hidden_size;
} qscu_qwen36_shared_expert_gate_add_desc;

qsfi_status qscu_qwen36_shared_expert_gate_add_bf16(
    const qscu_qwen36_shared_expert_gate_add_desc* desc, qsfi_cuda_stream stream
);

/*
 * BF16 embedding row gather. token_ids may be i32 or u32. In checked native
 * validation builds, validate_token_ids synchronizes the stream to report
 * out-of-range non-padding ids as QSFI_STATUS_INVALID_ARGUMENT. Unchecked
 * release builds still zero invalid rows in-kernel but do not wait for the
 * validation flag.
 */
typedef struct {
    qsfi_tensor1 token_ids; /* i32/u32 [num_tokens]. */
    qsfi_tensor2 embedding; /* bf16 [vocab_size, hidden_size]. */
    qsfi_tensor2 out; /* bf16 [num_tokens, hidden_size]. */
    int32_t padding_token_id; /* < 0 means no padding token. */
    uint32_t validate_token_ids;
} qscu_embedding_gather_desc;

qsfi_status
qscu_embedding_gather_bf16(const qscu_embedding_gather_desc* desc, qsfi_cuda_stream stream);

/* In-place F32 logits soft cap: logits = cap * tanh(logits / cap). cap <= 0 is a no-op. */
qsfi_status qscu_logits_soft_cap_f32(
    const qsfi_tensor2* logits,
    uint32_t rows,
    uint32_t vocab_size,
    float soft_cap,
    qsfi_cuda_stream stream
);

/*
 * Greedy F32 argmax only: temperature <= 0, no top-k/top-p/min-p, no logprobs.
 * Ties pick lowest id. Checked native validation builds reject non-finite
 * logits. Release builds do not scan logits: NaN comparisons do not win,
 * infinities follow CUDA floating-point comparisons, and an all-NaN row writes
 * the internal UINT_MAX sentinel cast to the selected output dtype (-1 for i32,
 * UINT_MAX for u32).
 */
typedef struct {
    qsfi_tensor2 logits; /* f32 [batch, vocab_size]. */
    qsfi_tensor1 uniform_samples; /* optional f32 [batch], ignored by greedy argmax. */
    qsfi_tensor1 next_token_ids; /* i32/u32 [batch]. */
    qsfi_tensor1 selected_logprobs; /* unsupported. */
    qsfi_tensor1 selected_probs; /* unsupported. */
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t top_k; /* must be 0. */
    float top_p; /* <= 0 or >= 1 means disabled. */
    float min_p; /* <= 0 means disabled. */
    float temperature; /* must be <= 0. */
} qscu_sampling_desc;

qsfi_status qscu_greedy_argmax_f32(const qscu_sampling_desc* desc, qsfi_cuda_stream stream);

typedef enum {
    QSCU_ACTIVATION_NONE = 0,
    QSCU_ACTIVATION_SILU = 1,
    QSCU_ACTIVATION_SIGMOID = 2
} qscu_activation;

typedef enum {
    /* g_out stores g = -exp(a_log) * softplus(a + dt_bias). */
    QSCU_GDN_FORGET_LOG_DECAY = 0,
    /* g_out stores exp(g), the linear alpha expected by FlashInfer chunk GDN. */
    QSCU_GDN_FORGET_LINEAR_ALPHA = 1
} qscu_gdn_forget_gate_output;

/*
 * Qwen3.6 Gated Delta Net recurrence.
 *
 * This is the local CUDA recurrence fallback used after the caller has loaded
 * weights, run the input projections, applied the causal conv over the packed
 * q/k/v stream, and prepared q/k/v/a/b. State is v-major / K-last:
 * [state_pool, num_v_heads, value_dim, key_dim].
 *
 * state_out_indices is optional. When omitted, state_indices is used for both
 * read and write. A negative state index skips that sequence/token and writes a
 * zero output row.
 */

typedef enum { QSCU_GDN_STATE_LAYOUT_VK = 0 } qscu_gdn_state_layout;

typedef struct {
    qsfi_tensor3 q; /* bf16 [num_tokens, num_q_heads, key_dim]. */
    qsfi_tensor3 k; /* bf16 [num_tokens, num_k_heads, key_dim]. */
    qsfi_tensor3 v; /* bf16 [num_tokens, num_v_heads, value_dim]. */
    qsfi_tensor2 a; /* bf16 [num_tokens, num_v_heads]. */
    qsfi_tensor2 b; /* bf16 [num_tokens, num_v_heads]. */
    qsfi_tensor1 a_log; /* f32 [num_v_heads]. */
    qsfi_tensor1 dt_bias; /* f32 [num_v_heads]. */
    qsfi_tensor4 state; /* bf16/f32 [state_pool, num_v_heads, value_dim, key_dim]. */
    qsfi_tensor1 state_indices; /* i32 [num_tokens]. */
    qsfi_tensor1 state_out_indices; /* optional i32 [num_tokens]. */
    qsfi_tensor3 out; /* bf16 [num_tokens, num_v_heads, value_dim]. */
    uint32_t num_tokens;
    uint32_t num_q_heads;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    qscu_gdn_state_layout state_layout;
    float scale;
    uint32_t use_qk_l2norm;
    uint32_t disable_state_update;
} qscu_gdn_decode_desc;

typedef struct {
    qsfi_tensor3 q; /* bf16 [total_tokens, num_q_heads, key_dim]. */
    qsfi_tensor3 k; /* bf16 [total_tokens, num_k_heads, key_dim]. */
    qsfi_tensor3 v; /* bf16 [total_tokens, num_v_heads, value_dim]. */
    qsfi_tensor2 a; /* bf16 [total_tokens, num_v_heads]. */
    qsfi_tensor2 b; /* bf16 [total_tokens, num_v_heads]. */
    qsfi_tensor1 a_log; /* f32 [num_v_heads]. */
    qsfi_tensor1 dt_bias; /* f32 [num_v_heads]. */
    qsfi_tensor4 state; /* bf16/f32 [state_pool, num_v_heads, value_dim, key_dim]. */
    qsfi_device_ptr seq_indptr; /* i32 [batch_size + 1], device pointer. */
    qsfi_tensor1 state_indices; /* i32 [batch_size]. */
    qsfi_tensor1 state_out_indices; /* optional i32 [batch_size]. */
    qsfi_tensor3 out; /* bf16 [total_tokens, num_v_heads, value_dim]. */
    uint32_t batch_size;
    uint32_t total_tokens;
    uint32_t num_q_heads;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    qscu_gdn_state_layout state_layout;
    float scale;
    uint32_t use_qk_l2norm;
    uint32_t disable_state_update;
} qscu_gdn_prefill_desc;

typedef enum { QSCU_ROUTER_SCORE_SOFTMAX = 0, QSCU_ROUTER_SCORE_SIGMOID = 1 } qscu_router_score;

typedef struct {
    qsfi_tensor2 x; /* bf16 [num_tokens, 8192], packed q/k/v projection. */
    qsfi_tensor2 weight; /* bf16 [8192, 4]. */
    qsfi_tensor1 bias; /* optional bf16/f32 [8192]. */
    qsfi_tensor3 state; /* bf16/f32 [state_pool, 8192, 3], old-to-new. */
    qsfi_tensor1 state_read_indices; /* optional i32 [batch_size], negative means zero state. */
    qsfi_tensor1 state_write_indices; /* optional i32 [batch_size], negative skips writeback. */
    qsfi_device_ptr seq_indptr; /* optional i32 [batch_size + 1]; null means one token per row. */
    qsfi_tensor2 out; /* bf16 [num_tokens, 8192], may alias x. */
    uint32_t num_tokens;
    uint32_t batch_size;
    qscu_activation activation; /* none or silu for qwen3.6 GDN conv. */
    uint32_t update_state;
} qscu_qwen36_gdn_causal_conv1d_desc;

typedef struct {
    qsfi_tensor2 conv_out; /* bf16 [num_tokens, 8192]. */
    qsfi_tensor2 a; /* bf16 [num_tokens, 32]. */
    qsfi_tensor2 b; /* bf16 [num_tokens, 32]. */
    qsfi_tensor1 a_log; /* f32 [32]. */
    qsfi_tensor1 dt_bias; /* f32 [32]. */
    qsfi_tensor3 q; /* bf16 [num_tokens, 16, 128]. */
    qsfi_tensor3 k; /* bf16 [num_tokens, 16, 128]. */
    qsfi_tensor3 v; /* bf16 [num_tokens, 32, 128]. */
    qsfi_tensor2 g_out; /* optional f32 [num_tokens, 32]. */
    qsfi_tensor2 beta_out; /* optional f32 [num_tokens, 32]. */
    uint32_t num_tokens;
    uint32_t apply_qk_l2norm;
    float l2norm_eps;
    qscu_gdn_forget_gate_output forget_gate_output;
} qscu_qwen36_gdn_post_conv_prepare_desc;

typedef struct {
    qsfi_tensor3 x; /* bf16 [num_tokens, 32, 128]. */
    qsfi_tensor3 gate; /* bf16 [num_tokens, 32, 128]. */
    qsfi_tensor1 weight; /* bf16/f32 [128]. */
    qsfi_tensor3 out; /* bf16 [num_tokens, 32, 128]. */
    uint32_t num_tokens;
    float eps;
    qscu_activation gate_activation; /* silu/swish or sigmoid. */
} qscu_qwen36_gdn_rmsnorm_gated_desc;

typedef struct {
    qsfi_tensor2 logits; /* bf16/f32 [num_tokens, num_experts]. */
    qsfi_tensor2 topk_ids; /* i32 [num_tokens, top_k]. */
    qsfi_tensor2 topk_weights; /* f32 [num_tokens, top_k]. */
    uint32_t num_tokens;
    uint32_t num_experts;
    uint32_t top_k;
    qscu_router_score score;
    uint32_t renormalize;
    float routed_scaling_factor;
} qscu_router_topk_desc;

qsfi_status qscu_qwen36_gdn_causal_conv1d_bf16(
    const qscu_qwen36_gdn_causal_conv1d_desc* desc, qsfi_cuda_stream stream
);

qsfi_status qscu_qwen36_gdn_post_conv_prepare_bf16(
    const qscu_qwen36_gdn_post_conv_prepare_desc* desc, qsfi_cuda_stream stream
);

qsfi_status qscu_qwen36_gdn_rmsnorm_gated_bf16(
    const qscu_qwen36_gdn_rmsnorm_gated_desc* desc, qsfi_cuda_stream stream
);

qsfi_status qscu_gdn_decode(qsfi_context* ctx, const qscu_gdn_decode_desc* desc);
qsfi_status qscu_gdn_prefill(qsfi_context* ctx, const qscu_gdn_prefill_desc* desc);

/*
 * Checked native validation builds reject non-finite router logits before
 * launching the router. Release builds do not scan logits; CUDA math and
 * comparison behavior applies, so non-finite inputs may propagate to weights.
 */
qsfi_status qscu_router_topk(const qscu_router_topk_desc* desc, qsfi_cuda_stream stream);

#ifdef __cplusplus
}
#endif

#endif
