#include "session.h"

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct qs_i32_vec {
    int32_t* data;
    size_t len;
    size_t cap;
} qs_i32_vec_t;

typedef struct qs_u32_vec {
    uint32_t* data;
    size_t len;
    size_t cap;
} qs_u32_vec_t;

typedef struct qs_request_id_vec {
    qs_session_request_id_t* data;
    size_t len;
    size_t cap;
} qs_request_id_vec_t;

typedef struct qs_device_i32_buffer {
    int32_t* data;
    size_t cap;
} qs_device_i32_buffer_t;

typedef struct qs_session_request {
    qs_session_request_id_t id;
    uint32_t seq_len;
    qs_i32_vec_t pages;
} qs_session_request_t;

typedef struct qs_request_vec {
    qs_session_request_t* data;
    size_t len;
    size_t cap;
} qs_request_vec_t;

typedef struct qs_session_staged_row {
    qs_session_request_id_t id;
    int32_t request_index;
    uint32_t old_seq_len;
    uint32_t old_page_count;
    uint32_t token_count;
    qs_i32_vec_t pages;
} qs_session_staged_row_t;

typedef struct qs_staged_row_vec {
    qs_session_staged_row_t* data;
    size_t len;
    size_t cap;
} qs_staged_row_vec_t;

typedef struct qs_session_layer_cache {
    void* k;
    void* v;
} qs_session_layer_cache_t;

typedef struct qs_session_plan_cache {
    qsfi_plan_t* plan;
    uint32_t batch_size;
    uint32_t num_indices;
    uint32_t total_tokens;
    qs_i32_vec_t qo_indptr;
    qs_i32_vec_t kv_indptr;
    bool valid;
} qs_session_plan_cache_t;

struct qs_session {
    qs_session_config_t config;
    cudaStream_t stream;
    qsfi_context_t* ctx;
    qsfi_attention_desc_t append_attention;
    qsfi_attention_desc_t decode_attention;

    qs_request_vec_t requests;
    qs_i32_vec_t free_pages;
    qs_session_layer_cache_t* layer_caches;
    uint32_t layer_cache_count;

    qs_request_id_vec_t live_request_ids;
    qs_i32_vec_t live_seq_lens;
    qs_i32_vec_t live_kv_indptr;
    qs_i32_vec_t live_kv_indices;
    qs_i32_vec_t live_last_page_len;

    qs_session_batch_kind_t batch_kind;
    uint32_t batch_size;
    uint32_t batch_token_count;
    qs_staged_row_vec_t staged_rows;

    qs_request_id_vec_t batch_request_ids;
    qs_i32_vec_t batch_tokens;
    qs_i32_vec_t batch_qo_indptr;
    qs_i32_vec_t batch_kv_indptr;
    qs_i32_vec_t batch_kv_indices;
    qs_i32_vec_t batch_last_page_len;
    qs_i32_vec_t batch_rope_pos_offset;
    qs_i32_vec_t batch_append_batch_indices;
    qs_i32_vec_t batch_append_positions;

    qs_device_i32_buffer_t d_batch_tokens;
    qs_device_i32_buffer_t d_batch_qo_indptr;
    qs_device_i32_buffer_t d_batch_kv_indptr;
    qs_device_i32_buffer_t d_batch_kv_indices;
    qs_device_i32_buffer_t d_batch_last_page_len;
    qs_device_i32_buffer_t d_batch_rope_pos_offset;
    qs_device_i32_buffer_t d_batch_append_batch_indices;
    qs_device_i32_buffer_t d_batch_append_positions;

    qs_session_plan_cache_t append_plan;
    qs_session_plan_cache_t decode_plan;
};

enum { QS_NO_REQUEST_INDEX = -1 };

static uint32_t qs_ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static uint32_t qs_page_count_for_len(uint32_t seq_len, uint32_t page_size)
{
    return seq_len == 0 ? 0 : qs_ceil_div_u32(seq_len, page_size);
}

static int32_t qs_last_page_len_for_seq(uint32_t seq_len, uint32_t page_size)
{
    if (seq_len == 0)
        return 0;
    {
        uint32_t rem = seq_len % page_size;
        return (int32_t)(rem == 0 ? page_size : rem);
    }
}

static size_t qs_dtype_size(qsfi_dtype_t dtype)
{
    switch (dtype) {
    case QSFI_DTYPE_F16:
    case QSFI_DTYPE_BF16:
        return 2;
    default:
        return 0;
    }
}

static qsfi_status_t qs_cuda_status(cudaError_t err)
{
    if (err == cudaSuccess)
        return QSFI_STATUS_OK;
    return err == cudaErrorMemoryAllocation ? QSFI_STATUS_OUT_OF_MEMORY : QSFI_STATUS_CUDA_ERROR;
}

static bool qs_pointer_host_readable(const void* ptr)
{
    cudaError_t err;
    struct cudaPointerAttributes attr;
    if (ptr == NULL)
        return false;
    memset(&attr, 0, sizeof(attr));
    err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return true;
    }
#if CUDART_VERSION >= 10000
    return attr.type != cudaMemoryTypeDevice;
#else
    return attr.memoryType == cudaMemoryTypeHost;
#endif
}

static bool qs_valid_dtype(qsfi_dtype_t dtype)
{
    return dtype == QSFI_DTYPE_F16 || dtype == QSFI_DTYPE_BF16;
}

static bool qs_valid_kv_layout(qsfi_kv_layout_t layout)
{
    return layout == QSFI_KV_LAYOUT_NHD || layout == QSFI_KV_LAYOUT_HND;
}

static qsfi_status_t qs_i32_reserve(qs_i32_vec_t* vec, size_t cap)
{
    int32_t* next;
    if (vec->cap >= cap)
        return QSFI_STATUS_OK;
    next = (int32_t*)realloc(vec->data, cap * sizeof(*next));
    if (next == NULL)
        return QSFI_STATUS_OUT_OF_MEMORY;
    vec->data = next;
    vec->cap = cap;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_i32_push(qs_i32_vec_t* vec, int32_t value)
{
    qsfi_status_t status;
    if (vec->len == vec->cap) {
        size_t next_cap = vec->cap == 0 ? 8 : vec->cap * 2;
        status = qs_i32_reserve(vec, next_cap);
        if (status != QSFI_STATUS_OK)
            return status;
    }
    vec->data[vec->len++] = value;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_i32_append(qs_i32_vec_t* vec, const int32_t* values, size_t count)
{
    qsfi_status_t status;
    if (count == 0)
        return QSFI_STATUS_OK;
    status = qs_i32_reserve(vec, vec->len + count);
    if (status != QSFI_STATUS_OK)
        return status;
    memcpy(vec->data + vec->len, values, count * sizeof(*values));
    vec->len += count;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_i32_assign(qs_i32_vec_t* vec, const int32_t* values, size_t count)
{
    qsfi_status_t status = qs_i32_reserve(vec, count);
    if (status != QSFI_STATUS_OK)
        return status;
    if (count != 0 && values != NULL)
        memcpy(vec->data, values, count * sizeof(*values));
    vec->len = count;
    return QSFI_STATUS_OK;
}

static void qs_i32_clear(qs_i32_vec_t* vec)
{
    vec->len = 0;
}

static void qs_i32_free(qs_i32_vec_t* vec)
{
    free(vec->data);
    vec->data = NULL;
    vec->len = 0;
    vec->cap = 0;
}

static qsfi_status_t qs_u32_assign(qs_u32_vec_t* vec, const uint32_t* values, size_t count)
{
    uint32_t* next;
    if (vec->cap < count) {
        next = (uint32_t*)realloc(vec->data, count * sizeof(*next));
        if (next == NULL)
            return QSFI_STATUS_OUT_OF_MEMORY;
        vec->data = next;
        vec->cap = count;
    }
    if (count != 0 && values != NULL)
        memcpy(vec->data, values, count * sizeof(*values));
    vec->len = count;
    return QSFI_STATUS_OK;
}

static void qs_u32_free(qs_u32_vec_t* vec)
{
    free(vec->data);
    vec->data = NULL;
    vec->len = 0;
    vec->cap = 0;
}

static qsfi_status_t qs_request_id_reserve(qs_request_id_vec_t* vec, size_t cap)
{
    qs_session_request_id_t* next;
    if (vec->cap >= cap)
        return QSFI_STATUS_OK;
    next = (qs_session_request_id_t*)realloc(vec->data, cap * sizeof(*next));
    if (next == NULL)
        return QSFI_STATUS_OUT_OF_MEMORY;
    vec->data = next;
    vec->cap = cap;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_request_id_push(qs_request_id_vec_t* vec, qs_session_request_id_t value)
{
    qsfi_status_t status;
    if (vec->len == vec->cap) {
        size_t next_cap = vec->cap == 0 ? 8 : vec->cap * 2;
        status = qs_request_id_reserve(vec, next_cap);
        if (status != QSFI_STATUS_OK)
            return status;
    }
    vec->data[vec->len++] = value;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_request_id_assign(
    qs_request_id_vec_t* vec, const qs_session_request_id_t* values, size_t count
)
{
    qsfi_status_t status = qs_request_id_reserve(vec, count);
    if (status != QSFI_STATUS_OK)
        return status;
    if (count != 0)
        memcpy(vec->data, values, count * sizeof(*values));
    vec->len = count;
    return QSFI_STATUS_OK;
}

static void qs_request_id_clear(qs_request_id_vec_t* vec)
{
    vec->len = 0;
}

static void qs_request_id_free(qs_request_id_vec_t* vec)
{
    free(vec->data);
    vec->data = NULL;
    vec->len = 0;
    vec->cap = 0;
}

static qsfi_status_t qs_request_reserve(qs_request_vec_t* vec, size_t cap)
{
    qs_session_request_t* next;
    if (vec->cap >= cap)
        return QSFI_STATUS_OK;
    next = (qs_session_request_t*)realloc(vec->data, cap * sizeof(*next));
    if (next == NULL)
        return QSFI_STATUS_OUT_OF_MEMORY;
    memset(next + vec->cap, 0, (cap - vec->cap) * sizeof(*next));
    vec->data = next;
    vec->cap = cap;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_request_push(qs_request_vec_t* vec, const qs_session_request_t* req)
{
    qsfi_status_t status;
    if (vec->len == vec->cap) {
        size_t next_cap = vec->cap == 0 ? 8 : vec->cap * 2;
        status = qs_request_reserve(vec, next_cap);
        if (status != QSFI_STATUS_OK)
            return status;
    }
    vec->data[vec->len++] = *req;
    return QSFI_STATUS_OK;
}

static void qs_request_free_members(qs_session_request_t* req)
{
    qs_i32_free(&req->pages);
}

static void qs_request_remove(qs_request_vec_t* vec, size_t idx)
{
    qs_request_free_members(&vec->data[idx]);
    if (idx + 1 < vec->len)
        memmove(vec->data + idx, vec->data + idx + 1, (vec->len - idx - 1) * sizeof(vec->data[0]));
    vec->len -= 1;
    memset(vec->data + vec->len, 0, sizeof(vec->data[0]));
}

static void qs_request_clear(qs_request_vec_t* vec)
{
    size_t i;
    for (i = 0; i < vec->len; ++i)
        qs_request_free_members(&vec->data[i]);
    vec->len = 0;
}

static void qs_request_free(qs_request_vec_t* vec)
{
    qs_request_clear(vec);
    free(vec->data);
    vec->data = NULL;
    vec->cap = 0;
}

static qsfi_status_t qs_staged_reserve(qs_staged_row_vec_t* vec, size_t cap)
{
    qs_session_staged_row_t* next;
    if (vec->cap >= cap)
        return QSFI_STATUS_OK;
    next = (qs_session_staged_row_t*)realloc(vec->data, cap * sizeof(*next));
    if (next == NULL)
        return QSFI_STATUS_OUT_OF_MEMORY;
    memset(next + vec->cap, 0, (cap - vec->cap) * sizeof(*next));
    vec->data = next;
    vec->cap = cap;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_staged_push(qs_staged_row_vec_t* vec, const qs_session_staged_row_t* row)
{
    qsfi_status_t status;
    if (vec->len == vec->cap) {
        size_t next_cap = vec->cap == 0 ? 8 : vec->cap * 2;
        status = qs_staged_reserve(vec, next_cap);
        if (status != QSFI_STATUS_OK)
            return status;
    }
    vec->data[vec->len++] = *row;
    return QSFI_STATUS_OK;
}

static void qs_staged_clear(qs_staged_row_vec_t* vec)
{
    size_t i;
    for (i = 0; i < vec->len; ++i)
        qs_i32_free(&vec->data[i].pages);
    vec->len = 0;
}

static void qs_staged_free(qs_staged_row_vec_t* vec)
{
    qs_staged_clear(vec);
    free(vec->data);
    vec->data = NULL;
    vec->cap = 0;
}

static qsfi_status_t qs_device_i32_ensure(qs_device_i32_buffer_t* buffer, size_t count)
{
    int32_t* next;
    cudaError_t err;
    if (count == 0 || buffer->cap >= count)
        return QSFI_STATUS_OK;
    next = NULL;
    err = cudaMalloc((void**)&next, count * sizeof(*next));
    if (err != cudaSuccess)
        return qs_cuda_status(err);
    if (buffer->data != NULL)
        cudaFree(buffer->data);
    buffer->data = next;
    buffer->cap = count;
    return QSFI_STATUS_OK;
}

static void qs_device_i32_free(qs_device_i32_buffer_t* buffer)
{
    if (buffer->data != NULL)
        cudaFree(buffer->data);
    buffer->data = NULL;
    buffer->cap = 0;
}

static qsfi_status_t qs_upload_i32_vec(
    qs_session_t* session, const qs_i32_vec_t* values, qs_device_i32_buffer_t* buffer
)
{
    qsfi_status_t status;
    if (values->len == 0)
        return QSFI_STATUS_OK;
    status = qs_device_i32_ensure(buffer, values->len);
    if (status != QSFI_STATUS_OK)
        return status;
    return qs_cuda_status(cudaMemcpyAsync(
        buffer->data,
        values->data,
        values->len * sizeof(values->data[0]),
        cudaMemcpyHostToDevice,
        session->stream
    ));
}

static void qs_plan_cache_init(qs_session_plan_cache_t* cache)
{
    memset(cache, 0, sizeof(*cache));
}

static void qs_plan_cache_destroy(qs_session_plan_cache_t* cache)
{
    if (cache->plan != NULL)
        qsfi_plan_destroy(cache->plan);
    cache->plan = NULL;
    qs_i32_free(&cache->qo_indptr);
    qs_i32_free(&cache->kv_indptr);
    cache->valid = false;
}

static qsfi_status_t qs_validate_config(const qs_session_config_t* config)
{
    if (config == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (config->num_layers == 0 || config->max_live_requests == 0 || config->max_batch_size == 0
        || config->max_seq_len == 0 || config->max_pages == 0 || config->page_size == 0) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    if (config->num_q_heads == 0 || config->num_kv_heads == 0 || config->head_dim == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (config->num_q_heads % config->num_kv_heads != 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (!qs_valid_dtype(config->activation_dtype) || !qs_valid_dtype(config->kv_dtype))
        return QSFI_STATUS_UNSUPPORTED;
    if (config->activation_dtype != config->kv_dtype)
        return QSFI_STATUS_UNSUPPORTED;
    if (!qs_valid_kv_layout(config->kv_layout))
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (config->max_seq_len > config->max_pages * config->page_size)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (config->qsfi_float_workspace_bytes == 0 || config->qsfi_int_workspace_bytes == 0
        || config->qsfi_host_int_workspace_bytes == 0) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    return QSFI_STATUS_OK;
}

static qsfi_attention_desc_t qs_make_attention(
    const qs_session_config_t* config, qsfi_mask_mode_t mask_mode
)
{
    qsfi_attention_desc_t attention;
    memset(&attention, 0, sizeof(attention));
    attention.num_qo_heads = config->num_q_heads;
    attention.num_kv_heads = config->num_kv_heads;
    attention.head_dim_qk = config->head_dim;
    attention.head_dim_vo = config->head_dim;
    attention.page_size = config->page_size;
    attention.q_dtype = config->activation_dtype;
    attention.kv_dtype = config->kv_dtype;
    attention.o_dtype = config->activation_dtype;
    attention.kv_layout = config->kv_layout;
    attention.pos_encoding = QSFI_POS_ENCODING_ROPE_LLAMA;
    attention.mask_mode = mask_mode;
    attention.window_left = -1;
    attention.logits_soft_cap = config->logits_soft_cap;
    attention.rope_scale = config->rope_scale == 0.0f ? 1.0f : config->rope_scale;
    attention.rope_theta = config->rope_theta == 0.0f ? 10000.0f : config->rope_theta;
    return attention;
}

static qsfi_tensor_desc_t qs_make_cache_tensor(const qs_session_t* session, void* data)
{
    qsfi_tensor_desc_t tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.data = data;
    tensor.dtype = session->config.kv_dtype;
    tensor.ndim = 4;
    tensor.shape[0] = session->config.max_pages;
    tensor.shape[3] = session->config.head_dim;
    tensor.stride[3] = 1;
    if (session->config.kv_layout == QSFI_KV_LAYOUT_NHD) {
        tensor.shape[1] = session->config.page_size;
        tensor.shape[2] = session->config.num_kv_heads;
        tensor.stride[0]
            = (int64_t)session->config.page_size * session->config.num_kv_heads
            * session->config.head_dim;
        tensor.stride[1] = (int64_t)session->config.num_kv_heads * session->config.head_dim;
        tensor.stride[2] = session->config.head_dim;
    } else {
        tensor.shape[1] = session->config.num_kv_heads;
        tensor.shape[2] = session->config.page_size;
        tensor.stride[0]
            = (int64_t)session->config.num_kv_heads * session->config.page_size
            * session->config.head_dim;
        tensor.stride[1] = (int64_t)session->config.page_size * session->config.head_dim;
        tensor.stride[2] = session->config.head_dim;
    }
    return tensor;
}

static qsfi_paged_kv_cache_t qs_make_kv_cache(const qs_session_t* session, uint32_t layer_idx)
{
    qsfi_paged_kv_cache_t cache;
    memset(&cache, 0, sizeof(cache));
    cache.k = qs_make_cache_tensor(session, session->layer_caches[layer_idx].k);
    cache.v = qs_make_cache_tensor(session, session->layer_caches[layer_idx].v);
    return cache;
}

static qsfi_paged_kv_table_t qs_make_active_page_table(const qs_session_t* session)
{
    qsfi_paged_kv_table_t table;
    memset(&table, 0, sizeof(table));
    table.indptr = session->batch_kv_indptr.len == 0 ? NULL : session->d_batch_kv_indptr.data;
    table.indices = session->batch_kv_indices.len == 0 ? NULL : session->d_batch_kv_indices.data;
    table.last_page_len
        = session->batch_last_page_len.len == 0 ? NULL : session->d_batch_last_page_len.data;
    table.rope_pos_offset
        = session->batch_rope_pos_offset.len == 0 ? NULL : session->d_batch_rope_pos_offset.data;
    table.batch_size = session->batch_size;
    table.num_indices = (uint32_t)session->batch_kv_indices.len;
    return table;
}

static int32_t qs_find_request_index(const qs_session_t* session, qs_session_request_id_t id)
{
    size_t i;
    for (i = 0; i < session->requests.len; ++i) {
        if (session->requests.data[i].id == id)
            return (int32_t)i;
    }
    return QS_NO_REQUEST_INDEX;
}

static bool qs_id_seen_in_prefix(
    const qs_session_request_id_t* ids, uint32_t count, qs_session_request_id_t id
)
{
    uint32_t i;
    for (i = 0; i < count; ++i) {
        if (ids[i] == id)
            return true;
    }
    return false;
}

static qsfi_status_t qs_return_page(qs_session_t* session, int32_t page)
{
    return qs_i32_push(&session->free_pages, page);
}

static int32_t qs_take_page(qs_session_t* session)
{
    int32_t page = session->free_pages.data[session->free_pages.len - 1];
    session->free_pages.len -= 1;
    return page;
}

static qsfi_status_t qs_rollback_staged_pages(qs_session_t* session)
{
    size_t i;
    for (i = 0; i < session->staged_rows.len; ++i) {
        qs_session_staged_row_t* row = &session->staged_rows.data[i];
        size_t page_idx;
        for (page_idx = row->old_page_count; page_idx < row->pages.len; ++page_idx) {
            qsfi_status_t status = qs_return_page(session, row->pages.data[page_idx]);
            if (status != QSFI_STATUS_OK)
                return status;
        }
    }
    return QSFI_STATUS_OK;
}

static void qs_clear_active_batch(qs_session_t* session)
{
    session->batch_kind = QS_SESSION_BATCH_NONE;
    session->batch_size = 0;
    session->batch_token_count = 0;
    qs_staged_clear(&session->staged_rows);
    qs_request_id_clear(&session->batch_request_ids);
    qs_i32_clear(&session->batch_tokens);
    qs_i32_clear(&session->batch_qo_indptr);
    qs_i32_clear(&session->batch_kv_indptr);
    qs_i32_clear(&session->batch_kv_indices);
    qs_i32_clear(&session->batch_last_page_len);
    qs_i32_clear(&session->batch_rope_pos_offset);
    qs_i32_clear(&session->batch_append_batch_indices);
    qs_i32_clear(&session->batch_append_positions);
}

static void qs_clear_batch_views(qs_session_t* session)
{
    qs_request_id_clear(&session->batch_request_ids);
    qs_i32_clear(&session->batch_tokens);
    qs_i32_clear(&session->batch_qo_indptr);
    qs_i32_clear(&session->batch_kv_indptr);
    qs_i32_clear(&session->batch_kv_indices);
    qs_i32_clear(&session->batch_last_page_len);
    qs_i32_clear(&session->batch_rope_pos_offset);
    qs_i32_clear(&session->batch_append_batch_indices);
    qs_i32_clear(&session->batch_append_positions);
}

static qsfi_status_t qs_rebuild_live_views(qs_session_t* session)
{
    size_t i;
    qsfi_status_t status;
    qs_request_id_clear(&session->live_request_ids);
    qs_i32_clear(&session->live_seq_lens);
    qs_i32_clear(&session->live_kv_indptr);
    qs_i32_clear(&session->live_kv_indices);
    qs_i32_clear(&session->live_last_page_len);

    status = qs_i32_push(&session->live_kv_indptr, 0);
    if (status != QSFI_STATUS_OK)
        return status;

    for (i = 0; i < session->requests.len; ++i) {
        const qs_session_request_t* req = &session->requests.data[i];
        status = qs_request_id_push(&session->live_request_ids, req->id);
        if (status != QSFI_STATUS_OK)
            return status;
        status = qs_i32_push(&session->live_seq_lens, (int32_t)req->seq_len);
        if (status != QSFI_STATUS_OK)
            return status;
        status = qs_i32_append(&session->live_kv_indices, req->pages.data, req->pages.len);
        if (status != QSFI_STATUS_OK)
            return status;
        status = qs_i32_push(&session->live_kv_indptr, (int32_t)session->live_kv_indices.len);
        if (status != QSFI_STATUS_OK)
            return status;
        status = qs_i32_push(
            &session->live_last_page_len,
            qs_last_page_len_for_seq(req->seq_len, session->config.page_size)
        );
        if (status != QSFI_STATUS_OK)
            return status;
    }
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_upload_active_batch(qs_session_t* session)
{
    qsfi_status_t status;
    status = qs_upload_i32_vec(session, &session->batch_tokens, &session->d_batch_tokens);
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_upload_i32_vec(session, &session->batch_qo_indptr, &session->d_batch_qo_indptr);
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_upload_i32_vec(session, &session->batch_kv_indptr, &session->d_batch_kv_indptr);
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_upload_i32_vec(session, &session->batch_kv_indices, &session->d_batch_kv_indices);
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_upload_i32_vec(
        session,
        &session->batch_last_page_len,
        &session->d_batch_last_page_len
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_upload_i32_vec(
        session,
        &session->batch_rope_pos_offset,
        &session->d_batch_rope_pos_offset
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_upload_i32_vec(
        session,
        &session->batch_append_batch_indices,
        &session->d_batch_append_batch_indices
    );
    if (status != QSFI_STATUS_OK)
        return status;
    return qs_upload_i32_vec(
        session,
        &session->batch_append_positions,
        &session->d_batch_append_positions
    );
}

static bool qs_i32_equal(const qs_i32_vec_t* a, const qs_i32_vec_t* b)
{
    if (a->len != b->len)
        return false;
    if (a->len == 0)
        return true;
    return memcmp(a->data, b->data, a->len * sizeof(a->data[0])) == 0;
}

static bool qs_plan_cache_matches(
    const qs_session_plan_cache_t* cache,
    uint32_t batch_size,
    uint32_t num_indices,
    uint32_t total_tokens,
    const qs_i32_vec_t* qo_indptr,
    const qs_i32_vec_t* kv_indptr
)
{
    return cache->valid && cache->batch_size == batch_size && cache->num_indices == num_indices
        && cache->total_tokens == total_tokens && qs_i32_equal(&cache->qo_indptr, qo_indptr)
        && qs_i32_equal(&cache->kv_indptr, kv_indptr);
}

static qsfi_status_t qs_ensure_append_plan(qs_session_t* session)
{
    qsfi_status_t status;
    qsfi_qo_plan_t qo;
    qsfi_paged_kv_plan_t page_table;
    qsfi_plan_t* plan;
    uint32_t num_indices = (uint32_t)session->batch_kv_indices.len;
    if (qs_plan_cache_matches(
            &session->append_plan,
            session->batch_size,
            num_indices,
            session->batch_token_count,
            &session->batch_qo_indptr,
            &session->batch_kv_indptr
        )) {
        return QSFI_STATUS_OK;
    }

    qs_plan_cache_destroy(&session->append_plan);
    qs_plan_cache_init(&session->append_plan);
    memset(&qo, 0, sizeof(qo));
    qo.indptr = session->batch_qo_indptr.data;
    qo.batch_size = session->batch_size;
    qo.total_tokens = session->batch_token_count;

    memset(&page_table, 0, sizeof(page_table));
    page_table.indptr = session->batch_kv_indptr.data;
    page_table.indices = session->batch_kv_indices.data;
    page_table.last_page_len = session->batch_last_page_len.data;
    page_table.batch_size = session->batch_size;
    page_table.num_indices = num_indices;

    plan = NULL;
    status = qsfi_batch_prefill_plan_create(
        session->ctx,
        &session->append_attention,
        &qo,
        &page_table,
        &plan
    );
    if (status != QSFI_STATUS_OK)
        return status;

    session->append_plan.plan = plan;
    session->append_plan.batch_size = session->batch_size;
    session->append_plan.num_indices = num_indices;
    session->append_plan.total_tokens = session->batch_token_count;
    status = qs_i32_assign(
        &session->append_plan.qo_indptr,
        session->batch_qo_indptr.data,
        session->batch_qo_indptr.len
    );
    if (status != QSFI_STATUS_OK)
        return status;
    status = qs_i32_assign(
        &session->append_plan.kv_indptr,
        session->batch_kv_indptr.data,
        session->batch_kv_indptr.len
    );
    if (status != QSFI_STATUS_OK)
        return status;
    session->append_plan.valid = true;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_ensure_decode_plan(qs_session_t* session)
{
    qsfi_status_t status;
    qs_i32_vec_t empty_qo;
    qsfi_paged_kv_plan_t page_table;
    qsfi_plan_t* plan;
    uint32_t num_indices = (uint32_t)session->batch_kv_indices.len;
    memset(&empty_qo, 0, sizeof(empty_qo));

    if (qs_plan_cache_matches(
            &session->decode_plan,
            session->batch_size,
            num_indices,
            session->batch_size,
            &empty_qo,
            &session->batch_kv_indptr
        )) {
        return QSFI_STATUS_OK;
    }

    qs_plan_cache_destroy(&session->decode_plan);
    qs_plan_cache_init(&session->decode_plan);
    memset(&page_table, 0, sizeof(page_table));
    page_table.indptr = session->batch_kv_indptr.data;
    page_table.indices = session->batch_kv_indices.data;
    page_table.last_page_len = session->batch_last_page_len.data;
    page_table.batch_size = session->batch_size;
    page_table.num_indices = num_indices;

    plan = NULL;
    status = qsfi_batch_decode_plan_create(
        session->ctx,
        &session->decode_attention,
        &page_table,
        &plan
    );
    if (status != QSFI_STATUS_OK)
        return status;
    session->decode_plan.plan = plan;
    session->decode_plan.batch_size = session->batch_size;
    session->decode_plan.num_indices = num_indices;
    session->decode_plan.total_tokens = session->batch_size;
    status = qs_i32_assign(
        &session->decode_plan.kv_indptr,
        session->batch_kv_indptr.data,
        session->batch_kv_indptr.len
    );
    if (status != QSFI_STATUS_OK)
        return status;
    session->decode_plan.valid = true;
    return QSFI_STATUS_OK;
}

static qsfi_status_t qs_validate_layer_common(
    const qs_session_t* session, const qs_session_layer_t* layer
)
{
    if (session == NULL || layer == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (session->batch_kind == QS_SESSION_BATCH_NONE)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (layer->layer_idx >= session->config.num_layers)
        return QSFI_STATUS_INVALID_ARGUMENT;
    return QSFI_STATUS_OK;
}

qsfi_status_t qs_session_create(const qs_session_config_t* config, qs_session_t** out)
{
    qs_session_t* session;
    qsfi_context_desc_t ctx_desc;
    qsfi_status_t status;
    size_t elems;
    size_t bytes;
    uint32_t i;

    if (out == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = NULL;

    status = qs_validate_config(config);
    if (status != QSFI_STATUS_OK)
        return status;

    session = (qs_session_t*)calloc(1, sizeof(*session));
    if (session == NULL)
        return QSFI_STATUS_OUT_OF_MEMORY;
    session->config = *config;
    session->stream = (cudaStream_t)config->stream;
    session->batch_kind = QS_SESSION_BATCH_NONE;
    qs_plan_cache_init(&session->append_plan);
    qs_plan_cache_init(&session->decode_plan);
    session->append_attention = qs_make_attention(config, QSFI_MASK_MODE_CAUSAL);
    session->decode_attention = qs_make_attention(config, QSFI_MASK_MODE_NONE);

    memset(&ctx_desc, 0, sizeof(ctx_desc));
    ctx_desc.device_ordinal = config->device_ordinal;
    ctx_desc.stream = config->stream;
    status = qsfi_context_create(&ctx_desc, &session->ctx);
    if (status != QSFI_STATUS_OK) {
        free(session);
        return status;
    }
    status = qsfi_context_reserve_scratch(
        session->ctx,
        config->qsfi_float_workspace_bytes,
        config->qsfi_int_workspace_bytes,
        config->qsfi_host_int_workspace_bytes
    );
    if (status != QSFI_STATUS_OK) {
        qs_session_destroy(session);
        return status;
    }

    status = qs_i32_reserve(&session->free_pages, config->max_pages);
    if (status != QSFI_STATUS_OK) {
        qs_session_destroy(session);
        return status;
    }
    for (i = config->max_pages; i > 0; --i) {
        status = qs_i32_push(&session->free_pages, (int32_t)(i - 1u));
        if (status != QSFI_STATUS_OK) {
            qs_session_destroy(session);
            return status;
        }
    }

    elems = (size_t)config->max_pages * config->page_size * config->num_kv_heads * config->head_dim;
    bytes = elems * qs_dtype_size(config->kv_dtype);
    session->layer_caches
        = (qs_session_layer_cache_t*)calloc(config->num_layers, sizeof(session->layer_caches[0]));
    if (session->layer_caches == NULL) {
        qs_session_destroy(session);
        return QSFI_STATUS_OUT_OF_MEMORY;
    }
    session->layer_cache_count = config->num_layers;
    for (i = 0; i < config->num_layers; ++i) {
        cudaError_t err = cudaMalloc(&session->layer_caches[i].k, bytes);
        if (err != cudaSuccess) {
            qs_session_destroy(session);
            return qs_cuda_status(err);
        }
        err = cudaMalloc(&session->layer_caches[i].v, bytes);
        if (err != cudaSuccess) {
            qs_session_destroy(session);
            return qs_cuda_status(err);
        }
    }

    status = qs_rebuild_live_views(session);
    if (status != QSFI_STATUS_OK) {
        qs_session_destroy(session);
        return status;
    }
    *out = session;
    return QSFI_STATUS_OK;
}

void qs_session_destroy(qs_session_t* session)
{
    uint32_t i;
    if (session == NULL)
        return;
    qs_plan_cache_destroy(&session->append_plan);
    qs_plan_cache_destroy(&session->decode_plan);
    for (i = 0; i < session->layer_cache_count; ++i) {
        if (session->layer_caches[i].k != NULL)
            cudaFree(session->layer_caches[i].k);
        if (session->layer_caches[i].v != NULL)
            cudaFree(session->layer_caches[i].v);
    }
    free(session->layer_caches);
    qs_device_i32_free(&session->d_batch_tokens);
    qs_device_i32_free(&session->d_batch_qo_indptr);
    qs_device_i32_free(&session->d_batch_kv_indptr);
    qs_device_i32_free(&session->d_batch_kv_indices);
    qs_device_i32_free(&session->d_batch_last_page_len);
    qs_device_i32_free(&session->d_batch_rope_pos_offset);
    qs_device_i32_free(&session->d_batch_append_batch_indices);
    qs_device_i32_free(&session->d_batch_append_positions);
    qsfi_context_destroy(session->ctx);

    qs_request_free(&session->requests);
    qs_i32_free(&session->free_pages);
    qs_request_id_free(&session->live_request_ids);
    qs_i32_free(&session->live_seq_lens);
    qs_i32_free(&session->live_kv_indptr);
    qs_i32_free(&session->live_kv_indices);
    qs_i32_free(&session->live_last_page_len);
    qs_staged_free(&session->staged_rows);
    qs_request_id_free(&session->batch_request_ids);
    qs_i32_free(&session->batch_tokens);
    qs_i32_free(&session->batch_qo_indptr);
    qs_i32_free(&session->batch_kv_indptr);
    qs_i32_free(&session->batch_kv_indices);
    qs_i32_free(&session->batch_last_page_len);
    qs_i32_free(&session->batch_rope_pos_offset);
    qs_i32_free(&session->batch_append_batch_indices);
    qs_i32_free(&session->batch_append_positions);
    free(session);
}

qsfi_status_t qs_session_reset(qs_session_t* session)
{
    uint32_t i;
    qsfi_status_t status;
    if (session == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qs_request_clear(&session->requests);
    qs_i32_clear(&session->free_pages);
    for (i = session->config.max_pages; i > 0; --i) {
        status = qs_i32_push(&session->free_pages, (int32_t)(i - 1u));
        if (status != QSFI_STATUS_OK)
            return status;
    }
    qs_clear_active_batch(session);
    return qs_rebuild_live_views(session);
}

qsfi_status_t qs_session_release_requests(
    qs_session_t* session, const qs_session_request_id_t* request_ids, uint32_t request_count
)
{
    uint32_t i;
    qsfi_status_t status;
    if (session == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (session->batch_kind != QS_SESSION_BATCH_NONE)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (request_count == 0)
        return QSFI_STATUS_OK;
    if (!qs_pointer_host_readable(request_ids))
        return QSFI_STATUS_INVALID_ARGUMENT;

    for (i = 0; i < request_count; ++i) {
        int32_t idx = qs_find_request_index(session, request_ids[i]);
        if (idx == QS_NO_REQUEST_INDEX)
            continue;
        {
            qs_session_request_t* req = &session->requests.data[idx];
            size_t page_idx;
            for (page_idx = 0; page_idx < req->pages.len; ++page_idx) {
                status = qs_return_page(session, req->pages.data[page_idx]);
                if (status != QSFI_STATUS_OK)
                    return status;
            }
        }
        qs_request_remove(&session->requests, (size_t)idx);
    }
    return qs_rebuild_live_views(session);
}

qsfi_status_t qs_session_get_state(const qs_session_t* session, qs_session_state_t* out)
{
    if (session == NULL || out == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    out->batch_kind = session->batch_kind;
    out->live_request_count = (uint32_t)session->requests.len;
    out->batch_size = session->batch_size;
    out->batch_token_count = session->batch_token_count;
    out->live_num_indices = (uint32_t)session->live_kv_indices.len;
    out->allocated_pages = session->config.max_pages - (uint32_t)session->free_pages.len;
    out->free_page_count = (uint32_t)session->free_pages.len;
    out->max_pages = session->config.max_pages;
    out->page_size = session->config.page_size;

    out->live_request_ids = session->live_request_ids.len == 0 ? NULL : session->live_request_ids.data;
    out->live_seq_lens = session->live_seq_lens.len == 0 ? NULL : session->live_seq_lens.data;
    out->live_kv_indptr = session->live_kv_indptr.len == 0 ? NULL : session->live_kv_indptr.data;
    out->live_kv_indices = session->live_kv_indices.len == 0 ? NULL : session->live_kv_indices.data;
    out->live_last_page_len
        = session->live_last_page_len.len == 0 ? NULL : session->live_last_page_len.data;
    out->free_pages = session->free_pages.len == 0 ? NULL : session->free_pages.data;

    out->batch_request_ids
        = session->batch_request_ids.len == 0 ? NULL : session->batch_request_ids.data;
    out->batch_tokens = session->batch_tokens.len == 0 ? NULL : session->batch_tokens.data;
    out->batch_qo_indptr
        = session->batch_qo_indptr.len == 0 ? NULL : session->batch_qo_indptr.data;
    out->batch_kv_indptr
        = session->batch_kv_indptr.len == 0 ? NULL : session->batch_kv_indptr.data;
    out->batch_kv_indices
        = session->batch_kv_indices.len == 0 ? NULL : session->batch_kv_indices.data;
    out->batch_last_page_len
        = session->batch_last_page_len.len == 0 ? NULL : session->batch_last_page_len.data;
    out->batch_rope_pos_offset
        = session->batch_rope_pos_offset.len == 0 ? NULL : session->batch_rope_pos_offset.data;
    out->batch_append_batch_indices = session->batch_append_batch_indices.len == 0
        ? NULL
        : session->batch_append_batch_indices.data;
    out->batch_append_positions
        = session->batch_append_positions.len == 0 ? NULL : session->batch_append_positions.data;

    out->d_batch_tokens = session->batch_tokens.len == 0 ? NULL : session->d_batch_tokens.data;
    out->d_batch_qo_indptr
        = session->batch_qo_indptr.len == 0 ? NULL : session->d_batch_qo_indptr.data;
    out->d_batch_kv_indptr
        = session->batch_kv_indptr.len == 0 ? NULL : session->d_batch_kv_indptr.data;
    out->d_batch_kv_indices
        = session->batch_kv_indices.len == 0 ? NULL : session->d_batch_kv_indices.data;
    out->d_batch_last_page_len
        = session->batch_last_page_len.len == 0 ? NULL : session->d_batch_last_page_len.data;
    out->d_batch_rope_pos_offset
        = session->batch_rope_pos_offset.len == 0 ? NULL : session->d_batch_rope_pos_offset.data;
    out->d_batch_append_batch_indices = session->batch_append_batch_indices.len == 0
        ? NULL
        : session->d_batch_append_batch_indices.data;
    out->d_batch_append_positions = session->batch_append_positions.len == 0
        ? NULL
        : session->d_batch_append_positions.data;
    return QSFI_STATUS_OK;
}

qsfi_status_t qs_session_begin_append(qs_session_t* session, const qs_session_append_batch_t* batch)
{
    uint32_t i;
    uint32_t new_request_count;
    uint32_t extra_pages;
    qsfi_status_t status;

    if (session == NULL || batch == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (session->batch_kind != QS_SESSION_BATCH_NONE)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (batch->batch_size == 0 || batch->batch_size > session->config.max_batch_size)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (batch->token_count == 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (!qs_pointer_host_readable(batch->request_ids) || !qs_pointer_host_readable(batch->token_indptr)
        || !qs_pointer_host_readable(batch->tokens)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }
    if (batch->token_indptr[0] != 0)
        return QSFI_STATUS_INVALID_ARGUMENT;
    for (i = 0; i < batch->batch_size; ++i) {
        if (batch->token_indptr[i] < 0 || batch->token_indptr[i + 1] < batch->token_indptr[i])
            return QSFI_STATUS_INVALID_ARGUMENT;
        if (qs_id_seen_in_prefix(batch->request_ids, i, batch->request_ids[i]))
            return QSFI_STATUS_INVALID_ARGUMENT;
    }
    if (batch->token_indptr[batch->batch_size] != (int32_t)batch->token_count)
        return QSFI_STATUS_INVALID_ARGUMENT;

    new_request_count = 0;
    extra_pages = 0;
    qs_staged_clear(&session->staged_rows);
    status = qs_staged_reserve(&session->staged_rows, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        return status;

    for (i = 0; i < batch->batch_size; ++i) {
        uint32_t token_begin = (uint32_t)batch->token_indptr[i];
        uint32_t token_end = (uint32_t)batch->token_indptr[i + 1];
        uint32_t token_count = token_end - token_begin;
        int32_t request_index = qs_find_request_index(session, batch->request_ids[i]);
        uint32_t old_seq_len = request_index == QS_NO_REQUEST_INDEX
            ? 0
            : session->requests.data[request_index].seq_len;
        uint32_t new_seq_len = old_seq_len + token_count;
        uint32_t old_page_count;
        qs_session_staged_row_t row;
        memset(&row, 0, sizeof(row));
        if (new_seq_len > session->config.max_seq_len)
            return QSFI_STATUS_INVALID_ARGUMENT;
        if (request_index == QS_NO_REQUEST_INDEX && token_count != 0)
            new_request_count += 1;
        row.id = batch->request_ids[i];
        row.request_index = request_index;
        row.old_seq_len = old_seq_len;
        row.token_count = token_count;
        old_page_count = request_index == QS_NO_REQUEST_INDEX
            ? 0
            : (uint32_t)session->requests.data[request_index].pages.len;
        row.old_page_count = old_page_count;
        if (request_index != QS_NO_REQUEST_INDEX) {
            status = qs_i32_assign(
                &row.pages,
                session->requests.data[request_index].pages.data,
                session->requests.data[request_index].pages.len
            );
            if (status != QSFI_STATUS_OK) {
                qs_i32_free(&row.pages);
                return status;
            }
        }
        extra_pages += qs_page_count_for_len(new_seq_len, session->config.page_size) - old_page_count;
        status = qs_staged_push(&session->staged_rows, &row);
        if (status != QSFI_STATUS_OK) {
            qs_i32_free(&row.pages);
            return status;
        }
    }

    if (session->requests.len + new_request_count > session->config.max_live_requests)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (extra_pages > session->free_pages.len)
        return QSFI_STATUS_OUT_OF_MEMORY;

    qs_clear_batch_views(session);
    session->batch_kind = QS_SESSION_BATCH_APPEND;
    session->batch_size = batch->batch_size;
    session->batch_token_count = batch->token_count;
    status = qs_request_id_assign(&session->batch_request_ids, batch->request_ids, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_assign(&session->batch_tokens, batch->tokens, batch->token_count);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_assign(&session->batch_qo_indptr, batch->token_indptr, batch->batch_size + 1u);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_reserve(&session->batch_kv_indptr, batch->batch_size + 1u);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_reserve(&session->batch_last_page_len, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_reserve(&session->batch_append_batch_indices, batch->token_count);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_reserve(&session->batch_append_positions, batch->token_count);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_push(&session->batch_kv_indptr, 0);
    if (status != QSFI_STATUS_OK)
        goto fail;
    for (i = 0; i < batch->batch_size; ++i) {
        qs_session_staged_row_t* row = &session->staged_rows.data[i];
        uint32_t new_seq_len = row->old_seq_len + row->token_count;
        uint32_t needed_pages = qs_page_count_for_len(new_seq_len, session->config.page_size);
        uint32_t j;
        while (row->pages.len < needed_pages) {
            status = qs_i32_push(&row->pages, qs_take_page(session));
            if (status != QSFI_STATUS_OK)
                goto fail;
        }
        status = qs_i32_append(&session->batch_kv_indices, row->pages.data, row->pages.len);
        if (status != QSFI_STATUS_OK)
            goto fail;
        status = qs_i32_push(&session->batch_kv_indptr, (int32_t)session->batch_kv_indices.len);
        if (status != QSFI_STATUS_OK)
            goto fail;
        status = qs_i32_push(
            &session->batch_last_page_len,
            qs_last_page_len_for_seq(new_seq_len, session->config.page_size)
        );
        if (status != QSFI_STATUS_OK)
            goto fail;
        status = qs_i32_push(&session->batch_rope_pos_offset, 0);
        if (status != QSFI_STATUS_OK)
            goto fail;
        for (j = 0; j < row->token_count; ++j) {
            status = qs_i32_push(&session->batch_append_batch_indices, (int32_t)i);
            if (status != QSFI_STATUS_OK)
                goto fail;
            status = qs_i32_push(&session->batch_append_positions, (int32_t)(row->old_seq_len + j));
            if (status != QSFI_STATUS_OK)
                goto fail;
        }
    }

    status = qs_upload_active_batch(session);
    if (status == QSFI_STATUS_OK)
        status = qs_ensure_append_plan(session);
    if (status != QSFI_STATUS_OK)
        goto fail;
    return QSFI_STATUS_OK;

fail:
    (void)qs_rollback_staged_pages(session);
    qs_clear_active_batch(session);
    return status;
}

qsfi_status_t qs_session_append_layer(qs_session_t* session, const qs_session_append_layer_t* layer)
{
    qsfi_status_t status;
    qsfi_paged_kv_table_t page_table;
    qsfi_paged_kv_cache_t kv_cache;
    qsfi_append_prefill_t append;
    qsfi_batch_prefill_execute_desc_t execute;

    status = qs_validate_layer_common(session, layer);
    if (status != QSFI_STATUS_OK)
        return status;
    if (session->batch_kind != QS_SESSION_BATCH_APPEND || session->append_plan.plan == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;

    page_table = qs_make_active_page_table(session);
    kv_cache = qs_make_kv_cache(session, layer->layer_idx);

    memset(&append, 0, sizeof(append));
    append.k = layer->k;
    append.v = layer->v;
    append.batch_indices = session->d_batch_append_batch_indices.data;
    append.positions = session->d_batch_append_positions.data;
    append.kv_cache = kv_cache;
    append.page_table = page_table;
    append.num_tokens = session->batch_token_count;
    status = qsfi_append_paged_kv_prefill(session->ctx, &session->append_attention, &append);
    if (status != QSFI_STATUS_OK)
        return status;

    memset(&execute, 0, sizeof(execute));
    execute.q = layer->q;
    execute.q_rope_offset = layer->q_rope_offset;
    execute.o = layer->o;
    execute.lse = layer->lse;
    execute.qo_indptr = session->d_batch_qo_indptr.data;
    execute.kv_cache = kv_cache;
    execute.page_table = page_table;
    execute.q_scale = layer->q_scale;
    execute.k_scale = layer->k_scale;
    execute.v_scale = layer->v_scale;
    execute.enable_pdl = layer->enable_pdl;
    return qsfi_batch_prefill_execute(session->ctx, session->append_plan.plan, &execute);
}

qsfi_status_t qs_session_begin_decode(qs_session_t* session, const qs_session_decode_batch_t* batch)
{
    uint32_t i;
    uint32_t extra_pages;
    qsfi_status_t status;

    if (session == NULL || batch == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (session->batch_kind != QS_SESSION_BATCH_NONE)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (batch->batch_size == 0 || batch->batch_size > session->config.max_batch_size)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (!qs_pointer_host_readable(batch->request_ids) || !qs_pointer_host_readable(batch->tokens))
        return QSFI_STATUS_INVALID_ARGUMENT;
    for (i = 0; i < batch->batch_size; ++i) {
        if (qs_id_seen_in_prefix(batch->request_ids, i, batch->request_ids[i]))
            return QSFI_STATUS_INVALID_ARGUMENT;
    }

    extra_pages = 0;
    qs_staged_clear(&session->staged_rows);
    status = qs_staged_reserve(&session->staged_rows, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        return status;

    for (i = 0; i < batch->batch_size; ++i) {
        int32_t request_index = qs_find_request_index(session, batch->request_ids[i]);
        const qs_session_request_t* req;
        qs_session_staged_row_t row;
        uint32_t needed_pages;
        if (request_index == QS_NO_REQUEST_INDEX)
            return QSFI_STATUS_INVALID_ARGUMENT;
        req = &session->requests.data[request_index];
        if (req->seq_len + 1u > session->config.max_seq_len)
            return QSFI_STATUS_INVALID_ARGUMENT;
        memset(&row, 0, sizeof(row));
        row.id = batch->request_ids[i];
        row.request_index = request_index;
        row.old_seq_len = req->seq_len;
        row.old_page_count = (uint32_t)req->pages.len;
        row.token_count = 1;
        status = qs_i32_assign(&row.pages, req->pages.data, req->pages.len);
        if (status != QSFI_STATUS_OK) {
            qs_i32_free(&row.pages);
            return status;
        }
        needed_pages = qs_page_count_for_len(req->seq_len + 1u, session->config.page_size);
        extra_pages += needed_pages - row.old_page_count;
        status = qs_staged_push(&session->staged_rows, &row);
        if (status != QSFI_STATUS_OK) {
            qs_i32_free(&row.pages);
            return status;
        }
    }

    if (extra_pages > session->free_pages.len)
        return QSFI_STATUS_OUT_OF_MEMORY;

    qs_clear_batch_views(session);
    session->batch_kind = QS_SESSION_BATCH_DECODE;
    session->batch_size = batch->batch_size;
    session->batch_token_count = batch->batch_size;
    status = qs_request_id_assign(&session->batch_request_ids, batch->request_ids, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_assign(&session->batch_tokens, batch->tokens, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_reserve(&session->batch_kv_indptr, batch->batch_size + 1u);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_reserve(&session->batch_last_page_len, batch->batch_size);
    if (status != QSFI_STATUS_OK)
        goto fail;
    status = qs_i32_push(&session->batch_kv_indptr, 0);
    if (status != QSFI_STATUS_OK)
        goto fail;

    for (i = 0; i < batch->batch_size; ++i) {
        qs_session_staged_row_t* row = &session->staged_rows.data[i];
        uint32_t new_seq_len = row->old_seq_len + 1u;
        uint32_t needed_pages = qs_page_count_for_len(new_seq_len, session->config.page_size);
        while (row->pages.len < needed_pages) {
            status = qs_i32_push(&row->pages, qs_take_page(session));
            if (status != QSFI_STATUS_OK)
                goto fail;
        }
        status = qs_i32_append(&session->batch_kv_indices, row->pages.data, row->pages.len);
        if (status != QSFI_STATUS_OK)
            goto fail;
        status = qs_i32_push(&session->batch_kv_indptr, (int32_t)session->batch_kv_indices.len);
        if (status != QSFI_STATUS_OK)
            goto fail;
        status = qs_i32_push(
            &session->batch_last_page_len,
            qs_last_page_len_for_seq(new_seq_len, session->config.page_size)
        );
        if (status != QSFI_STATUS_OK)
            goto fail;
        status = qs_i32_push(&session->batch_rope_pos_offset, 0);
        if (status != QSFI_STATUS_OK)
            goto fail;
    }

    status = qs_upload_active_batch(session);
    if (status == QSFI_STATUS_OK)
        status = qs_ensure_decode_plan(session);
    if (status != QSFI_STATUS_OK)
        goto fail;
    return QSFI_STATUS_OK;

fail:
    (void)qs_rollback_staged_pages(session);
    qs_clear_active_batch(session);
    return status;
}

qsfi_status_t qs_session_decode_layer(qs_session_t* session, const qs_session_decode_layer_t* layer)
{
    qsfi_status_t status;
    qsfi_paged_kv_table_t page_table;
    qsfi_paged_kv_cache_t kv_cache;
    qsfi_append_decode_t append;
    qsfi_batch_decode_execute_desc_t execute;

    status = qs_validate_layer_common(session, layer);
    if (status != QSFI_STATUS_OK)
        return status;
    if (session->batch_kind != QS_SESSION_BATCH_DECODE || session->decode_plan.plan == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;

    page_table = qs_make_active_page_table(session);
    kv_cache = qs_make_kv_cache(session, layer->layer_idx);

    memset(&append, 0, sizeof(append));
    append.k = layer->k;
    append.v = layer->v;
    append.kv_cache = kv_cache;
    append.page_table = page_table;
    status = qsfi_append_paged_kv_decode(session->ctx, &session->decode_attention, &append);
    if (status != QSFI_STATUS_OK)
        return status;

    memset(&execute, 0, sizeof(execute));
    execute.q = layer->q;
    execute.q_rope_offset = layer->q_rope_offset;
    execute.o = layer->o;
    execute.lse = layer->lse;
    execute.kv_cache = kv_cache;
    execute.page_table = page_table;
    execute.q_scale = layer->q_scale;
    execute.k_scale = layer->k_scale;
    execute.v_scale = layer->v_scale;
    execute.enable_pdl = layer->enable_pdl;
    return qsfi_batch_decode_execute(session->ctx, session->decode_plan.plan, &execute);
}

qsfi_status_t qs_session_commit_batch(qs_session_t* session, const qs_session_commit_t* commit)
{
    qs_u32_vec_t accepted;
    size_t i;
    qsfi_status_t status;

    if (session == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (session->batch_kind == QS_SESSION_BATCH_NONE)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (commit != NULL && commit->accepted_token_counts != NULL
        && !qs_pointer_host_readable(commit->accepted_token_counts)) {
        return QSFI_STATUS_INVALID_ARGUMENT;
    }

    memset(&accepted, 0, sizeof(accepted));
    status = qs_u32_assign(&accepted, NULL, session->staged_rows.len);
    if (status != QSFI_STATUS_OK)
        return status;
    for (i = 0; i < session->staged_rows.len; ++i) {
        uint32_t count = commit == NULL || commit->accepted_token_counts == NULL
            ? session->staged_rows.data[i].token_count
            : commit->accepted_token_counts[i];
        if (count > session->staged_rows.data[i].token_count) {
            qs_u32_free(&accepted);
            return QSFI_STATUS_INVALID_ARGUMENT;
        }
        if (session->batch_kind == QS_SESSION_BATCH_DECODE && count > 1u) {
            qs_u32_free(&accepted);
            return QSFI_STATUS_INVALID_ARGUMENT;
        }
        accepted.data[i] = count;
    }

    for (i = 0; i < session->staged_rows.len; ++i) {
        qs_session_staged_row_t* row = &session->staged_rows.data[i];
        uint32_t new_seq_len = row->old_seq_len + accepted.data[i];
        uint32_t needed_pages = qs_page_count_for_len(new_seq_len, session->config.page_size);
        size_t page_idx;
        if (row->request_index == QS_NO_REQUEST_INDEX) {
            if (accepted.data[i] != 0) {
                qs_session_request_t req;
                memset(&req, 0, sizeof(req));
                req.id = row->id;
                req.seq_len = new_seq_len;
                status = qs_i32_assign(&req.pages, row->pages.data, needed_pages);
                if (status != QSFI_STATUS_OK) {
                    qs_request_free_members(&req);
                    qs_u32_free(&accepted);
                    return status;
                }
                status = qs_request_push(&session->requests, &req);
                if (status != QSFI_STATUS_OK) {
                    qs_request_free_members(&req);
                    qs_u32_free(&accepted);
                    return status;
                }
            }
        } else {
            qs_session_request_t* req = &session->requests.data[row->request_index];
            req->seq_len = new_seq_len;
            status = qs_i32_assign(&req->pages, row->pages.data, needed_pages);
            if (status != QSFI_STATUS_OK) {
                qs_u32_free(&accepted);
                return status;
            }
        }
        for (page_idx = needed_pages; page_idx < row->pages.len; ++page_idx) {
            status = qs_return_page(session, row->pages.data[page_idx]);
            if (status != QSFI_STATUS_OK) {
                qs_u32_free(&accepted);
                return status;
            }
        }
    }

    qs_u32_free(&accepted);
    qs_clear_active_batch(session);
    return qs_rebuild_live_views(session);
}

qsfi_status_t qs_session_abort_batch(qs_session_t* session)
{
    qsfi_status_t status;
    if (session == NULL)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (session->batch_kind == QS_SESSION_BATCH_NONE)
        return QSFI_STATUS_OK;
    status = qs_rollback_staged_pages(session);
    qs_clear_active_batch(session);
    if (status != QSFI_STATUS_OK)
        return status;
    return qs_rebuild_live_views(session);
}
