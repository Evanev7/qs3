#ifndef QS_FLASHINFER_H
#define QS_FLASHINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QSFI_ERROR_MESSAGE_BYTES 512u
#define QSFI_MAX_TENSOR_DIMS 5u

typedef struct qsfi_context qsfi_context_t;
typedef struct qsfi_plan qsfi_plan_t;

typedef void* qsfi_cuda_stream_t;
typedef void* qsfi_device_ptr_t;
typedef uint32_t qsfi_kernel_flags_t;

typedef enum qsfi_status {
    QSFI_STATUS_OK = 0,
    QSFI_STATUS_INVALID_ARGUMENT = 1,
    QSFI_STATUS_UNSUPPORTED = 2,
    QSFI_STATUS_OUT_OF_MEMORY = 3,
    QSFI_STATUS_CUDA_ERROR = 4,
    QSFI_STATUS_BACKEND_ERROR = 5,
    QSFI_STATUS_INTERNAL_ERROR = 6
} qsfi_status_t;

typedef enum qsfi_error_source {
    QSFI_ERROR_SOURCE_NONE = 0,
    QSFI_ERROR_SOURCE_QSFI = 1,
    QSFI_ERROR_SOURCE_CUDA = 2,
    QSFI_ERROR_SOURCE_FLASHINFER = 3
} qsfi_error_source_t;

typedef enum qsfi_dtype {
    QSFI_DTYPE_F16 = 0,
    QSFI_DTYPE_BF16 = 1,
    QSFI_DTYPE_FP8_E4M3 = 2,
    QSFI_DTYPE_FP8_E5M2 = 3,
    QSFI_DTYPE_NVFP4_E2M1 = 4
} qsfi_dtype_t;

typedef enum qsfi_kv_layout {
    /* K/V shape: [num_pages, page_size, num_kv_heads, head_dim]. */
    QSFI_KV_LAYOUT_NHD = 0,
    /* K/V shape: [num_pages, num_kv_heads, page_size, head_dim]. */
    QSFI_KV_LAYOUT_HND = 1
} qsfi_kv_layout_t;

typedef enum qsfi_pos_encoding {
    QSFI_POS_ENCODING_NONE = 0,
    QSFI_POS_ENCODING_ROPE_LLAMA = 1
} qsfi_pos_encoding_t;

typedef enum qsfi_mask_mode { QSFI_MASK_MODE_NONE = 0, QSFI_MASK_MODE_CAUSAL = 1 } qsfi_mask_mode_t;

typedef enum qsfi_plan_kind {
    QSFI_PLAN_BATCH_DECODE = 0,
    QSFI_PLAN_BATCH_PREFILL = 1
} qsfi_plan_kind_t;

typedef enum qsfi_kernel_module {
    QSFI_KERNEL_MODULE_NONE = 0,
    QSFI_KERNEL_MODULE_ATTENTION = 1u << 0,
    QSFI_KERNEL_MODULE_KV_CACHE = 1u << 1,
    QSFI_KERNEL_MODULE_ALL = (1u << 2) - 1u
} qsfi_kernel_module_t;

typedef struct qsfi_context_desc {
    int32_t device_ordinal;
    qsfi_cuda_stream_t stream;
} qsfi_context_desc_t;

typedef struct qsfi_error_info {
    qsfi_status_t status;
    qsfi_error_source_t source;
    int32_t native_code;
    char message[QSFI_ERROR_MESSAGE_BYTES];
} qsfi_error_info_t;

typedef struct qsfi_tensor_desc {
    qsfi_device_ptr_t data;
    qsfi_dtype_t dtype;
    uint32_t ndim;
    int64_t shape[QSFI_MAX_TENSOR_DIMS];
    int64_t stride[QSFI_MAX_TENSOR_DIMS];
} qsfi_tensor_desc_t;

typedef struct qsfi_attention_desc {
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim_qk;
    uint32_t head_dim_vo;
    uint32_t page_size;
    qsfi_dtype_t q_dtype;
    qsfi_dtype_t kv_dtype;
    qsfi_dtype_t o_dtype;
    qsfi_kv_layout_t kv_layout;
    qsfi_pos_encoding_t pos_encoding;
    qsfi_mask_mode_t mask_mode;
    int32_t window_left;
    int32_t fixed_split_size;
    float sm_scale;
    float logits_soft_cap;
    float rope_scale;
    float rope_theta;
    uint32_t disable_split_kv;
    uint32_t use_fp16_qk_reduction;
} qsfi_attention_desc_t;

typedef struct qsfi_paged_kv_cache {
    qsfi_tensor_desc_t k;
    qsfi_tensor_desc_t v;
    qsfi_tensor_desc_t k_scale;
    qsfi_tensor_desc_t v_scale;
} qsfi_paged_kv_cache_t;

typedef struct qsfi_paged_kv_plan {
    const int32_t* indptr;
    const int32_t* indices;
    const int32_t* last_page_len;
    uint32_t batch_size;
    uint32_t num_indices;
} qsfi_paged_kv_plan_t;

typedef struct qsfi_qo_plan {
    const int32_t* indptr;
    uint32_t batch_size;
    uint32_t total_tokens;
} qsfi_qo_plan_t;

typedef struct qsfi_paged_kv_table {
    qsfi_device_ptr_t indptr;
    qsfi_device_ptr_t indices;
    qsfi_device_ptr_t last_page_len;
    qsfi_device_ptr_t rope_pos_offset;
    uint32_t batch_size;
    uint32_t num_indices;
} qsfi_paged_kv_table_t;

typedef struct qsfi_batch_decode_execute_desc {
    qsfi_tensor_desc_t q;
    qsfi_tensor_desc_t o;
    qsfi_device_ptr_t lse;
    qsfi_paged_kv_cache_t kv_cache;
    qsfi_paged_kv_table_t page_table;
    float q_scale;
    float k_scale;
    float v_scale;
    uint32_t enable_pdl;
} qsfi_batch_decode_execute_desc_t;

typedef struct qsfi_batch_prefill_execute_desc {
    qsfi_tensor_desc_t q;
    qsfi_tensor_desc_t o;
    qsfi_device_ptr_t lse;
    qsfi_device_ptr_t qo_indptr;
    qsfi_paged_kv_cache_t kv_cache;
    qsfi_paged_kv_table_t page_table;
    float q_scale;
    float k_scale;
    float v_scale;
    uint32_t enable_pdl;
} qsfi_batch_prefill_execute_desc_t;

typedef struct qsfi_append_decode {
    qsfi_tensor_desc_t k;
    qsfi_tensor_desc_t v;
    qsfi_paged_kv_cache_t kv_cache;
    qsfi_paged_kv_table_t page_table;
} qsfi_append_decode_t;

typedef struct qsfi_append_prefill {
    qsfi_tensor_desc_t k;
    qsfi_tensor_desc_t v;
    qsfi_device_ptr_t batch_indices;
    qsfi_device_ptr_t positions;
    qsfi_paged_kv_cache_t kv_cache;
    qsfi_paged_kv_table_t page_table;
    uint32_t num_tokens;
} qsfi_append_prefill_t;

const char* qsfi_status_string(qsfi_status_t status);

qsfi_status_t qsfi_context_create(const qsfi_context_desc_t* desc, qsfi_context_t** out);
void qsfi_context_destroy(qsfi_context_t* ctx);

qsfi_status_t qsfi_context_set_stream(qsfi_context_t* ctx, qsfi_cuda_stream_t stream);
qsfi_status_t qsfi_context_reserve_scratch(
    qsfi_context_t* ctx,
    size_t float_workspace_bytes,
    size_t int_workspace_bytes,
    size_t host_int_workspace_bytes
);
qsfi_status_t qsfi_context_get_last_error(const qsfi_context_t* ctx, qsfi_error_info_t* out);
void qsfi_context_clear_last_error(qsfi_context_t* ctx);

qsfi_status_t qsfi_load_kernels(qsfi_context_t* ctx, qsfi_kernel_flags_t modules);

qsfi_status_t qsfi_batch_decode_plan_create(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_paged_kv_plan_t* page_table,
    qsfi_plan_t** out
);

qsfi_status_t qsfi_batch_decode_execute(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_decode_execute_desc_t* desc
);

qsfi_status_t qsfi_batch_prefill_plan_create(
    qsfi_context_t* ctx,
    const qsfi_attention_desc_t* attention,
    const qsfi_qo_plan_t* qo,
    const qsfi_paged_kv_plan_t* page_table,
    qsfi_plan_t** out
);

qsfi_status_t qsfi_batch_prefill_execute(
    qsfi_context_t* ctx, const qsfi_plan_t* plan, const qsfi_batch_prefill_execute_desc_t* desc
);

qsfi_status_t qsfi_plan_kind(const qsfi_plan_t* plan, qsfi_plan_kind_t* out);
void qsfi_plan_destroy(qsfi_plan_t* plan);

qsfi_status_t qsfi_append_paged_kv_decode(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_decode_t* append
);

qsfi_status_t qsfi_append_paged_kv_prefill(
    qsfi_context_t* ctx, const qsfi_attention_desc_t* attention, const qsfi_append_prefill_t* append
);

#ifdef __cplusplus
}
#endif

#endif
