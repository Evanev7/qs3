#ifndef QSCU_H
#define QSCU_H

#include "qs_info.h"
#include "qs_tensor.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * qscu is a small local CUDA helper layer for qwen3.6 kernels that do not map
 * cleanly onto a packed FlashInfer API. Helpers launch on the supplied stream;
 * callers are responsible for setting the CUDA device and for mapping the
 * returned status into qsfi_context error state.
 */

/*
 * Split-pointer SwiGLU: out = silu(gate) * up for BF16 row-major tensors.
 * Requires gate/up/out shapes to match desc->num_tokens x desc->intermediate_size.
 * Inner stride must be 1. clamp_limit > 0 is intentionally not implemented yet.
 */
typedef struct {
    qsfi_tensor2 gate;
    qsfi_tensor2 up;
    qsfi_tensor2 out;
    uint32_t num_tokens;
    uint32_t intermediate_size;
    float clamp_limit; /* <= 0 means unclamped. */
} qscu_silu_and_mul_desc;

qsfi_status qscu_silu_and_mul_bf16(const qscu_silu_and_mul_desc* desc, qsfi_cuda_stream stream);

/*
 * BF16 embedding row gather. token_ids may be i32 or u32. When
 * validate_token_ids is non-zero, this helper synchronizes the stream to report
 * out-of-range non-padding ids as QSFI_STATUS_INVALID_ARGUMENT.
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
 * Ties pick lowest id.
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

qsfi_status qscu_router_topk(const qscu_router_topk_desc* desc, qsfi_cuda_stream stream);

#ifdef __cplusplus
}
#endif

#endif
