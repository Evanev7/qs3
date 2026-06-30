#ifndef QS_FLASHINFER_H
#define QS_FLASHINFER_H

#include "qs_info.h"
#include "qs_tensor.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * qsfi is a qwen3.6 runtime kernel boundary, not a blanket mirror of
 * every FlashInfer Python API. Optional tensors are represented by data == NULL.
 * All execution functions run on the stream owned by ctx.
 */

typedef struct qsfi_context qsfi_context;
typedef struct qsfi_batch_decode_plan qsfi_batch_decode_plan;
typedef struct qsfi_batch_prefill_plan qsfi_batch_prefill_plan;
typedef struct qsfi_moe_plan qsfi_moe_plan;

typedef enum {
    /* K/V shape: [num_pages, page_size, num_kv_heads, head_dim]. */
    QSFI_KV_LAYOUT_NHD = 0,
    /* K/V shape: [num_pages, num_kv_heads, page_size, head_dim]. */
    QSFI_KV_LAYOUT_HND = 1
} qsfi_kv_layout;

typedef enum { QSFI_POS_ENCODING_NONE = 0, QSFI_POS_ENCODING_ROPE_LLAMA = 1 } qsfi_pos_encoding;

typedef enum { QSFI_MASK_MODE_NONE = 0, QSFI_MASK_MODE_CAUSAL = 1 } qsfi_mask_mode;

typedef struct {
    int32_t device_ordinal;
    qsfi_cuda_stream stream;
} qsfi_context_desc;

typedef struct {
    uint32_t target_sm;
    uint32_t target_compute_capability_major;
    uint32_t target_compute_capability_minor;
    uint32_t assume_fp8;
    uint32_t assume_fp4;
    uint32_t assume_pdl;
    uint32_t gemm_backend;
} qsfi_build_config;

typedef struct {
    uint32_t runtime_compute_capability_major;
    uint32_t runtime_compute_capability_minor;
} qsfi_context_info;

/*
 * Attention and paged KV cache.
 *
 * The compiled attention dispatch is intentionally narrow: head_dim_qk and
 * head_dim_vo must both be 64, and num_qo_heads must equal num_kv_heads. Wider
 * head dimensions and GQA are valid future API shapes but are not wired yet.
 */

typedef struct {
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim_qk;
    uint32_t head_dim_vo;
    uint32_t page_size;
    qsfi_dtype q_dtype;
    qsfi_dtype kv_dtype;
    qsfi_dtype o_dtype;
    qsfi_kv_layout kv_layout;
    qsfi_pos_encoding pos_encoding;
    qsfi_mask_mode mask_mode;
    int32_t window_left;
    int32_t fixed_split_size;
    float sm_scale;
    float logits_soft_cap;
    float rope_scale;
    float rope_theta;
    uint32_t disable_split_kv;
    uint32_t use_fp16_qk_reduction;
} qsfi_attention_desc;

typedef struct {
    qsfi_tensor4 k;
    qsfi_tensor4 v;
    qsfi_tensor4 k_scale;
    qsfi_tensor4 v_scale;
} qsfi_paged_kv_cache;

typedef struct {
    const int32_t* indptr;
    const int32_t* indices;
    const int32_t* last_page_len;
    uint32_t batch_size;
    uint32_t num_indices;
} qsfi_paged_kv_plan;

typedef struct {
    const int32_t* indptr;
    uint32_t batch_size;
    uint32_t total_tokens;
} qsfi_qo_plan;

typedef struct {
    qsfi_device_ptr indptr;
    qsfi_device_ptr indices;
    qsfi_device_ptr last_page_len;
    qsfi_device_ptr rope_pos_offset;
    uint32_t batch_size;
    uint32_t num_indices;
} qsfi_paged_kv_table;

typedef struct {
    qsfi_tensor3 q;
    qsfi_device_ptr q_rope_offset;
    qsfi_tensor3 o;
    qsfi_device_ptr lse;
    qsfi_paged_kv_cache kv_cache;
    qsfi_paged_kv_table page_table;
    float q_scale;
    float k_scale;
    float v_scale;
} qsfi_batch_decode_execute_desc;

typedef struct {
    qsfi_tensor3 q;
    qsfi_device_ptr q_rope_offset;
    qsfi_tensor3 o;
    qsfi_device_ptr lse;
    qsfi_device_ptr qo_indptr;
    qsfi_paged_kv_cache kv_cache;
    qsfi_paged_kv_table page_table;
    float q_scale;
    float k_scale;
    float v_scale;
} qsfi_batch_prefill_execute_desc;

typedef struct {
    qsfi_tensor3 k;
    qsfi_tensor3 v;
    qsfi_paged_kv_cache kv_cache;
    qsfi_paged_kv_table page_table;
} qsfi_append_decode_desc;

typedef struct {
    qsfi_tensor3 k;
    qsfi_tensor3 v;
    qsfi_device_ptr batch_indices;
    qsfi_device_ptr positions;
    qsfi_paged_kv_cache kv_cache;
    qsfi_paged_kv_table page_table;
    uint32_t num_tokens;
} qsfi_append_prefill_desc;

/*
 * FlashInfer-backed non-attention transformer helpers.
 */

typedef struct {
    qsfi_tensor2 x; /* bf16/f32 [rows, hidden_size]. */
    qsfi_tensor1 weight; /* same dtype [hidden_size]. */
    qsfi_tensor2 out; /* same dtype [rows, hidden_size]. */
    uint32_t hidden_size;
    float eps;
} qsfi_rmsnorm_desc;

typedef struct {
    qsfi_tensor2 x; /* bf16/f32 [rows, hidden_size], overwritten with normalized output. */
    qsfi_tensor2 residual_inout; /* same dtype [rows, hidden_size], updated to residual + x. */
    qsfi_tensor1 weight; /* same dtype [hidden_size]. */
    qsfi_tensor2 out; /* must alias x; documents the normalized output tensor. */
    uint32_t hidden_size;
    float eps;
} qsfi_fused_add_rmsnorm_desc;

typedef struct {
    qsfi_tensor3 q; /* bf16/f32 [num_tokens, num_qo_heads, head_dim]. */
    qsfi_tensor3 k; /* bf16/f32 [num_tokens, num_kv_heads, head_dim]. */
    qsfi_tensor3 q_out; /* same dtype/shape as q; may alias q. */
    qsfi_tensor3 k_out; /* same dtype/shape as k; may alias k. */
    qsfi_tensor1 positions; /* i32/u32 [num_tokens]. */
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim; /* full-head rotary dim for qwen3.6. */
    float rope_scale; /* 0 means 1. */
    float rope_theta; /* 0 means 10000. */
    uint32_t interleave; /* must be 0: NeoX/Llama non-interleaved layout. */
} qsfi_rope_apply_desc;

/*
 * FlashInfer-backed Qwen3.6 routed SwiGLU MoE.
 *
 * qsfi does not load, own, or repack model weights. The caller supplies tensors
 * in the layout below and keeps them alive until execution on ctx->stream has
 * completed.
 *
 * The initial BF16 backend is a staged path using FlashInfer grouped GEMM for
 * expert projections. Routing is precomputed by the caller: topk_ids contains
 * global expert ids and topk_weights contains the final per-route scale after
 * any softmax/top-k renormalization. Expert parallel fields are part of the
 * ABI, but the initial staged backend only accepts a full local expert set.
 *
 * NVFP4 surfaces are declared now so the weight/activation layout is fixed for
 * the future static fused path. Packed FP4 tensors use physical byte shapes,
 * never sub-byte tensor strides.
 */

typedef enum {
    QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16 = 1,
    QSFI_MOE_BACKEND_FLASHINFER_FUSED_BF16 = 2,
    QSFI_MOE_BACKEND_FLASHINFER_NVFP4 = 3
} qsfi_moe_backend;

typedef enum {
    QSFI_MOE_ROUTE_PRECOMPUTED_TOPK = 0,
    QSFI_MOE_ROUTE_ROUTER_LOGITS = 1
} qsfi_moe_route_mode;

typedef struct {
    qsfi_moe_backend backend;
    qsfi_moe_route_mode route_mode;
    uint32_t max_num_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t local_expert_offset;
    uint32_t local_num_experts;
    qsfi_dtype activation_dtype;
    qsfi_dtype weight_dtype;
    qsfi_dtype output_dtype;
    uint32_t reserved0;
} qsfi_moe_plan_desc;

typedef struct {
    qsfi_tensor2 hidden; /* bf16 [num_tokens, hidden_size]. */
    qsfi_tensor2 topk_ids; /* i32 [num_tokens, top_k], global expert ids. */
    qsfi_tensor2 topk_weights; /* f32 [num_tokens, top_k]. */
    qsfi_tensor3 gate_up_weight; /* bf16 [local_num_experts, 2*intermediate_size, hidden_size]. */
    qsfi_tensor3 down_weight; /* bf16 [local_num_experts, hidden_size, intermediate_size]. */
    qsfi_tensor2 out; /* bf16 [num_tokens, hidden_size]. */
    qsfi_tensor1 workspace; /* u8/i8 [qsfi_moe_workspace_size(...)] device scratch. */
    uint32_t num_tokens;
} qsfi_moe_bf16_execute_desc;

typedef struct {
    qsfi_tensor2 hidden_packed; /* u8 [num_tokens, hidden_size / 2]. */
    qsfi_tensor2 hidden_scale; /* fp8_e4m3 [num_tokens, hidden_size / 16]. */
    qsfi_tensor2 topk_ids; /* i32 [num_tokens, top_k], global expert ids. */
    qsfi_tensor2 topk_weights; /* f32 [num_tokens, top_k]. */
    qsfi_tensor3
        gate_up_weight_packed; /* u8 [local_num_experts, 2*intermediate_size, hidden_size / 2]. */
    qsfi_tensor3 gate_up_weight_scale; /* fp8_e4m3 [local_num_experts, 2*intermediate_size,
                                          hidden_size / 16]. */
    qsfi_tensor3
        down_weight_packed; /* u8 [local_num_experts, hidden_size, intermediate_size / 2]. */
    qsfi_tensor3
        down_weight_scale; /* fp8_e4m3 [local_num_experts, hidden_size, intermediate_size / 16]. */
    qsfi_tensor1 expert_output_scale; /* optional f32 [local_num_experts]. */
    qsfi_tensor2 out; /* bf16 [num_tokens, hidden_size]. */
    qsfi_tensor1 workspace; /* u8/i8 [qsfi_moe_workspace_size(...)] device scratch. */
    uint32_t num_tokens;
} qsfi_moe_nvfp4_execute_desc;

qsfi_status qsfi_context_create(const qsfi_context_desc* desc, qsfi_context** out);
void qsfi_context_destroy(qsfi_context* ctx);

qsfi_status qsfi_context_set_stream(qsfi_context* ctx, qsfi_cuda_stream stream);
qsfi_status qsfi_get_build_config(qsfi_build_config* out);
qsfi_status qsfi_context_get_info(qsfi_context* ctx, qsfi_context_info* out);
qsfi_status qsfi_context_validate_target(qsfi_context* ctx);
/*
 * Reserves coarse context-owned scratch used by module implementations.
 * Shape-specific scratch should be validated by the operation that needs it;
 * add an operation-specific workspace query later only if a backend cannot use
 * fixed or caller-provided scratch cleanly.
 */
qsfi_status qsfi_context_reserve_workspace(
    qsfi_context* ctx,
    size_t float_workspace_bytes,
    size_t int_workspace_bytes,
    size_t host_int_workspace_bytes
);
qsfi_status qsfi_context_get_last_error(const qsfi_context* ctx, qsfi_error_info* out);
void qsfi_context_clear_last_error(qsfi_context* ctx);

/*
 * Only FlashInfer interfaces with an explicit plan/run split get qsfi plan
 * handles. Paged attention plans snapshot host-side schedule inputs. Segment
 * GEMM is a direct run surface for now; if a backend later needs persistent
 * tactic/schedule state, add a module-specific plan type then.
 */
qsfi_status qsfi_batch_decode_plan_create(
    qsfi_context* ctx,
    const qsfi_attention_desc* attention,
    const qsfi_paged_kv_plan* page_table,
    qsfi_batch_decode_plan** out
);
qsfi_status qsfi_batch_decode_execute(
    qsfi_context* ctx,
    const qsfi_batch_decode_plan* plan,
    const qsfi_batch_decode_execute_desc* desc
);
void qsfi_batch_decode_plan_destroy(qsfi_batch_decode_plan* plan);

qsfi_status qsfi_batch_prefill_plan_create(
    qsfi_context* ctx,
    const qsfi_attention_desc* attention,
    const qsfi_qo_plan* qo,
    const qsfi_paged_kv_plan* page_table,
    qsfi_batch_prefill_plan** out
);
qsfi_status qsfi_batch_prefill_execute(
    qsfi_context* ctx,
    const qsfi_batch_prefill_plan* plan,
    const qsfi_batch_prefill_execute_desc* desc
);
void qsfi_batch_prefill_plan_destroy(qsfi_batch_prefill_plan* plan);

qsfi_status qsfi_append_paged_kv_decode(
    qsfi_context* ctx, const qsfi_attention_desc* attention, const qsfi_append_decode_desc* desc
);
qsfi_status qsfi_append_paged_kv_prefill(
    qsfi_context* ctx, const qsfi_attention_desc* attention, const qsfi_append_prefill_desc* desc
);

qsfi_status qsfi_rmsnorm(qsfi_context* ctx, const qsfi_rmsnorm_desc* desc);
qsfi_status qsfi_fused_add_rmsnorm(qsfi_context* ctx, const qsfi_fused_add_rmsnorm_desc* desc);
qsfi_status qsfi_rope_apply(qsfi_context* ctx, const qsfi_rope_apply_desc* desc);
qsfi_status
qsfi_moe_plan_create(qsfi_context* ctx, const qsfi_moe_plan_desc* desc, qsfi_moe_plan** out);
void qsfi_moe_plan_destroy(qsfi_moe_plan* plan);
qsfi_status qsfi_moe_workspace_size(
    qsfi_context* ctx, const qsfi_moe_plan* plan, uint32_t num_tokens, size_t* device_bytes
);
qsfi_status qsfi_moe_execute_bf16(
    qsfi_context* ctx, const qsfi_moe_plan* plan, const qsfi_moe_bf16_execute_desc* desc
);
qsfi_status qsfi_moe_execute_nvfp4(
    qsfi_context* ctx, const qsfi_moe_plan* plan, const qsfi_moe_nvfp4_execute_desc* desc
);

#ifdef __cplusplus
}
#endif

#endif
