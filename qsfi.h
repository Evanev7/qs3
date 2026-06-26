#ifndef QS_FLASHINFER_H
#define QS_FLASHINFER_H

#include "qsfi_build_constants.h"

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

#define QSFI_ERROR_MESSAGE_BYTES 512u

typedef struct qsfi_context qsfi_context;
typedef struct qsfi_batch_decode_plan qsfi_batch_decode_plan;
typedef struct qsfi_batch_prefill_plan qsfi_batch_prefill_plan;

typedef void* qsfi_cuda_stream;
typedef void* qsfi_device_ptr;

typedef enum {
    QSFI_STATUS_OK = 0,
    QSFI_STATUS_INVALID_ARGUMENT = 1,
    QSFI_STATUS_UNSUPPORTED = 2,
    QSFI_STATUS_OUT_OF_MEMORY = 3,
    QSFI_STATUS_CUDA_ERROR = 4,
    QSFI_STATUS_BACKEND_ERROR = 5,
    QSFI_STATUS_INTERNAL_ERROR = 6
} qsfi_status;

typedef enum {
    QSFI_ERROR_SOURCE_NONE = 0,
    QSFI_ERROR_SOURCE_QSFI = 1,
    QSFI_ERROR_SOURCE_CUDA = 2,
    QSFI_ERROR_SOURCE_FLASHINFER = 3,
    QSFI_ERROR_SOURCE_CUBLASLT = 4
} qsfi_error_source;

typedef enum {
    QSFI_DTYPE_INVALID = 0,
    QSFI_DTYPE_F32 = 1,
    QSFI_DTYPE_F16 = 2,
    QSFI_DTYPE_BF16 = 3,
    QSFI_DTYPE_FP8_E4M3 = 4,
    QSFI_DTYPE_FP8_E5M2 = 5,
    QSFI_DTYPE_NVFP4_E2M1 = 6,
    QSFI_DTYPE_MXFP4_E2M1 = 7,
    QSFI_DTYPE_MXFP8_E4M3 = 8,
    QSFI_DTYPE_I32 = 9,
    QSFI_DTYPE_U32 = 10,
    QSFI_DTYPE_I8 = 11,
    QSFI_DTYPE_U8 = 12
} qsfi_dtype;

typedef enum {
    /* K/V shape: [num_pages, page_size, num_kv_heads, head_dim]. */
    QSFI_KV_LAYOUT_NHD = 0,
    /* K/V shape: [num_pages, num_kv_heads, page_size, head_dim]. */
    QSFI_KV_LAYOUT_HND = 1
} qsfi_kv_layout;

typedef enum {
    QSFI_POS_ENCODING_NONE = 0,
    QSFI_POS_ENCODING_ROPE_LLAMA = 1
} qsfi_pos_encoding;

typedef enum {
    QSFI_MASK_MODE_NONE = 0,
    QSFI_MASK_MODE_CAUSAL = 1
} qsfi_mask_mode;

typedef enum {
    QSFI_ACTIVATION_NONE = 0,
    QSFI_ACTIVATION_SWIGLU = 1
} qsfi_activation;

typedef enum {
    QSFI_ROUTING_METHOD_DEFAULT = 0,
    QSFI_ROUTING_METHOD_DEEPSEEK_V3 = 1,
    QSFI_ROUTING_METHOD_LLAMA4 = 2
} qsfi_routing_method;

typedef struct {
    int32_t device_ordinal;
    qsfi_cuda_stream stream;
} qsfi_context_desc;

typedef struct {
    qsfi_status status;
    qsfi_error_source source;
    int32_t native_code;
    char message[QSFI_ERROR_MESSAGE_BYTES];
} qsfi_error_info;

/*
 * Tensor strides are element strides, not byte strides. Rank is part of the
 * type name so public calls stay narrow and validation can focus on the qwen3.6
 * shape contract for that operation.
 */
typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[1];
    int64_t stride[1];
} qsfi_tensor1;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[2];
    int64_t stride[2];
} qsfi_tensor2;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[3];
    int64_t stride[3];
} qsfi_tensor3;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[4];
    int64_t stride[4];
} qsfi_tensor4;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[5];
    int64_t stride[5];
} qsfi_tensor5;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[6];
    int64_t stride[6];
} qsfi_tensor6;

typedef struct {
    uint32_t target_sm;
    uint32_t target_compute_capability_major;
    uint32_t target_compute_capability_minor;
    uint32_t assume_fp8;
    uint32_t assume_fp4;
    uint32_t assume_pdl;
    uint32_t gemm_backend;
    uint32_t moe_backend;
} qsfi_build_config;

typedef struct {
    uint32_t runtime_compute_capability_major;
    uint32_t runtime_compute_capability_minor;
} qsfi_context_info;

/*
 * Attention and paged KV cache.
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
 * Non-attention transformer kernels.
 */

typedef struct {
    qsfi_tensor1 token_ids; /* i32/u32 [num_tokens]. */
    qsfi_tensor2 embedding; /* [vocab_size, hidden_size]. */
    qsfi_tensor2 out;       /* [num_tokens, hidden_size]. */
    int32_t padding_token_id;      /* < 0 means no padding token. */
    uint32_t validate_token_ids;
} qsfi_embedding_gather_desc;

typedef struct {
    qsfi_tensor2 x;
    qsfi_tensor1 weight;
    qsfi_tensor2 out;
    uint32_t hidden_size;
    float eps;
} qsfi_rmsnorm_desc;

typedef struct {
    qsfi_tensor2 x;
    qsfi_tensor2 residual_inout;
    qsfi_tensor1 weight;
    qsfi_tensor2 out;
    uint32_t hidden_size;
    float eps;
} qsfi_fused_add_rmsnorm_desc;

typedef struct {
    qsfi_tensor3 q;
    qsfi_tensor3 k;
    qsfi_tensor3 q_out;
    qsfi_tensor3 k_out;
    qsfi_tensor1 positions; /* i32/u32 [num_tokens]. */
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    float rope_scale;
    float rope_theta;
    uint32_t interleave;
} qsfi_rope_apply_desc;

typedef struct {
    qsfi_tensor2 x;             /* [rows, in_features]. */
    qsfi_tensor2 weight;        /* [out_features, in_features]. */
    qsfi_tensor1 bias;          /* optional [out_features]. */
    qsfi_tensor1 weight_scale;  /* optional quant scales. */
    qsfi_tensor1 weight_zero;   /* optional quant zeros. */
    qsfi_tensor2 out;           /* [rows, out_features]. */
    uint32_t rows;
    uint32_t in_features;
    uint32_t out_features;
    qsfi_dtype accum_dtype;
    float alpha;
    float beta;
} qsfi_linear_desc;

typedef struct {
    qsfi_tensor2 x;        /* [num_tokens, hidden_size]. */
    qsfi_tensor2 q_weight; /* [num_qo_heads * head_dim, hidden_size]. */
    qsfi_tensor2 k_weight; /* [num_kv_heads * head_dim, hidden_size]. */
    qsfi_tensor2 v_weight; /* [num_kv_heads * head_dim, hidden_size]. */
    qsfi_tensor1 q_bias;   /* optional [num_qo_heads * head_dim]. */
    qsfi_tensor1 k_bias;   /* optional [num_kv_heads * head_dim]. */
    qsfi_tensor1 v_bias;   /* optional [num_kv_heads * head_dim]. */
    qsfi_tensor3 q_out;    /* [num_tokens, num_qo_heads, head_dim]. */
    qsfi_tensor3 k_out;    /* [num_tokens, num_kv_heads, head_dim]. */
    qsfi_tensor3 v_out;    /* [num_tokens, num_kv_heads, head_dim]. */
    uint32_t num_tokens;
    uint32_t hidden_size;
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    qsfi_dtype accum_dtype;
} qsfi_qkv_projection_desc;

typedef struct {
    qsfi_tensor2 gate;
    qsfi_tensor2 up;
    qsfi_tensor2 out;
    uint32_t num_tokens;
    uint32_t intermediate_size;
    float clamp_limit; /* <= 0 means unclamped. */
} qsfi_silu_and_mul_desc;

typedef struct {
    qsfi_tensor2 x;
    qsfi_tensor2 gate_weight;
    qsfi_tensor2 up_weight;
    qsfi_tensor2 gate_up_weight; /* optional packed [2*intermediate, hidden]. */
    qsfi_tensor2 down_weight;
    qsfi_tensor1 gate_bias;
    qsfi_tensor1 up_bias;
    qsfi_tensor1 down_bias;
    qsfi_tensor2 tmp_gate;
    qsfi_tensor2 tmp_up;
    qsfi_tensor2 tmp_act;
    qsfi_tensor2 out;
    uint32_t num_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    qsfi_dtype accum_dtype;
    float clamp_limit;
} qsfi_dense_swiglu_mlp_desc;

typedef struct {
    qsfi_tensor2 x;      /* [rows, hidden_size]. */
    qsfi_tensor2 weight; /* [vocab_size, hidden_size]. */
    qsfi_tensor1 bias;   /* optional [vocab_size]. */
    qsfi_tensor2 logits; /* [rows, vocab_size]. */
    uint32_t rows;
    uint32_t hidden_size;
    uint32_t vocab_size;
    qsfi_dtype accum_dtype;
    float logits_soft_cap; /* <= 0 means no cap. */
} qsfi_lm_head_desc;

/*
 * Sampling. For deterministic GPU sampling, pass uniform_samples [batch] f32.
 * temperature <= 0 means greedy argmax.
 */

typedef struct {
    qsfi_tensor2 logits;             /* [batch, vocab_size]. */
    qsfi_tensor1 uniform_samples;    /* optional f32 [batch]. */
    qsfi_tensor1 next_token_ids;     /* i32/u32 [batch]. */
    qsfi_tensor1 selected_logprobs;  /* optional f32 [batch]. */
    qsfi_tensor1 selected_probs;     /* optional f32 [batch]. */
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t top_k; /* 0 means disabled. */
    float top_p;    /* <= 0 or >= 1 means disabled. */
    float min_p;    /* <= 0 means disabled. */
    float temperature;
} qsfi_sampling_desc;

/*
 * Router and MoE. selected_experts is int32 [num_tokens, top_k],
 * selected_weights is f32 [num_tokens, top_k].
 */

typedef struct {
    qsfi_routing_method method;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t n_group;
    uint32_t topk_group;
    float routed_scaling_factor;
    uint32_t renormalize;
    uint32_t apply_router_weight_on_input;
} qsfi_routing_config;

typedef struct {
    qsfi_tensor2 routing_logits;   /* [num_tokens, num_experts]. */
    qsfi_tensor1 routing_bias;     /* optional [num_experts]. */
    qsfi_tensor2 selected_experts; /* i32 [num_tokens, top_k]. */
    qsfi_tensor2 selected_weights; /* f32 [num_tokens, top_k]. */
    qsfi_routing_config routing;
    uint32_t num_tokens;
} qsfi_router_topk_desc;

typedef struct {
    qsfi_tensor3 gate_up_weight; /* [local_experts, 2*intermediate, hidden]. */
    qsfi_tensor3 down_weight;    /* [local_experts, hidden, intermediate]. */
    qsfi_tensor3 gate_up_scale;  /* optional quant scales with expert axis. */
    qsfi_tensor3 down_scale;     /* optional quant scales with expert axis. */
    qsfi_tensor2 gate_up_bias;   /* optional [local_experts, 2*intermediate]. */
    qsfi_tensor2 down_bias;      /* optional [local_experts, hidden]. */
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t local_expert_offset;
    uint32_t local_num_experts;
} qsfi_moe_weights_desc;

typedef struct {
    qsfi_tensor2 hidden_states;    /* [num_tokens, hidden_size]. */
    qsfi_tensor2 routing_logits;   /* optional if selected_* are supplied. */
    qsfi_tensor2 selected_experts; /* optional i32 [num_tokens, top_k]. */
    qsfi_tensor2 selected_weights; /* optional f32 [num_tokens, top_k]. */
    qsfi_tensor2 out;              /* [num_tokens, hidden_size]. */
    qsfi_moe_weights_desc weights;
    qsfi_routing_config routing;
    qsfi_activation activation;
    qsfi_dtype accum_dtype;
    uint32_t num_tokens;
    float clamp_limit;
} qsfi_routed_moe_desc;

typedef struct {
    qsfi_tensor2 src; /* [2*intermediate, hidden]. */
    qsfi_tensor2 dst; /* [2*intermediate, hidden]. */
    uint32_t intermediate_size;
    uint32_t hidden_size;
} qsfi_gated_weight_reorder_desc;

typedef struct {
    qsfi_tensor2 src;
    qsfi_tensor2 dst;
    uint32_t block_size;
} qsfi_block_scale_interleave_desc;

const char* qsfi_status_string(qsfi_status status);

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
 * GEMM and routed MoE are direct run surfaces for now; if a backend later needs
 * persistent tactic/schedule state, add a module-specific plan type then.
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

qsfi_status qsfi_embedding_gather(qsfi_context* ctx, const qsfi_embedding_gather_desc* desc);
qsfi_status qsfi_rmsnorm(qsfi_context* ctx, const qsfi_rmsnorm_desc* desc);
qsfi_status qsfi_fused_add_rmsnorm(qsfi_context* ctx, const qsfi_fused_add_rmsnorm_desc* desc);
qsfi_status qsfi_rope_apply(qsfi_context* ctx, const qsfi_rope_apply_desc* desc);
qsfi_status qsfi_linear(qsfi_context* ctx, const qsfi_linear_desc* desc);
qsfi_status qsfi_qkv_projection(qsfi_context* ctx, const qsfi_qkv_projection_desc* desc);
qsfi_status qsfi_silu_and_mul(qsfi_context* ctx, const qsfi_silu_and_mul_desc* desc);
qsfi_status qsfi_dense_swiglu_mlp(qsfi_context* ctx, const qsfi_dense_swiglu_mlp_desc* desc);
qsfi_status qsfi_lm_head(qsfi_context* ctx, const qsfi_lm_head_desc* desc);
qsfi_status qsfi_sample(qsfi_context* ctx, const qsfi_sampling_desc* desc);
qsfi_status qsfi_router_topk(qsfi_context* ctx, const qsfi_router_topk_desc* desc);
/*
 * If selected_experts and selected_weights are present, routed_moe uses those
 * precomputed routing decisions. Otherwise routing_logits must be present and
 * the implementation computes routing internally with desc->routing.
 */
qsfi_status qsfi_routed_moe(qsfi_context* ctx, const qsfi_routed_moe_desc* desc);

qsfi_status qsfi_reorder_gated_act_weight(
    qsfi_context* ctx, const qsfi_gated_weight_reorder_desc* desc
);
qsfi_status qsfi_block_scale_interleave(
    qsfi_context* ctx, const qsfi_block_scale_interleave_desc* desc
);

#ifdef __cplusplus
}
#endif

#endif
