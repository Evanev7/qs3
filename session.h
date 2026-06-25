#ifndef QS_SESSION_H
#define QS_SESSION_H

#include "qsfi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qs_session qs_session_t;
typedef uint64_t qs_session_request_id_t;

typedef enum qs_session_batch_kind {
    QS_SESSION_BATCH_NONE = 0,
    QS_SESSION_BATCH_APPEND = 1,
    QS_SESSION_BATCH_DECODE = 2
} qs_session_batch_kind_t;

typedef struct qs_session_config {
    int32_t device_ordinal;
    qsfi_cuda_stream_t stream;

    uint32_t num_layers;
    uint32_t max_live_requests;
    uint32_t max_batch_size;
    uint32_t max_seq_len;
    uint32_t max_pages;
    uint32_t page_size;

    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t vocab_size;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    qsfi_dtype_t activation_dtype;
    qsfi_dtype_t kv_dtype;
    qsfi_kv_layout_t kv_layout;
    float rope_theta;
    float rope_scale;
    float logits_soft_cap;

    size_t qsfi_float_workspace_bytes;
    size_t qsfi_int_workspace_bytes;
    size_t qsfi_host_int_workspace_bytes;
} qs_session_config_t;

typedef struct qs_session_state {
    qs_session_batch_kind_t batch_kind;
    uint32_t live_request_count;
    uint32_t batch_size;
    uint32_t batch_token_count;
    uint32_t live_num_indices;
    uint32_t allocated_pages;
    uint32_t free_page_count;
    uint32_t max_pages;
    uint32_t page_size;

    /*
     * Live request tables are session-owned and remain valid until the next
     * reset/release/begin/commit/abort call. live_seq_lens are post-commit
     * sequence lengths.
     */
    const qs_session_request_id_t* live_request_ids;
    const int32_t* live_seq_lens;
    const int32_t* live_kv_indptr;
    const int32_t* live_kv_indices;
    const int32_t* live_last_page_len;
    const int32_t* free_pages;

    /*
     * Active batch tables are rebuilt by qs_session_begin_append/decode.
     * For append, batch_tokens is a flat token array and batch_qo_indptr has
     * batch_size + 1 entries. For decode, batch_tokens has batch_size entries
     * and batch_qo_indptr is null. Page tables and last_page_len describe the
     * speculative post-append KV state used by attention and append kernels.
     */
    const qs_session_request_id_t* batch_request_ids;
    const int32_t* batch_tokens;
    const int32_t* batch_qo_indptr;
    const int32_t* batch_kv_indptr;
    const int32_t* batch_kv_indices;
    const int32_t* batch_last_page_len;
    const int32_t* batch_rope_pos_offset;
    const int32_t* batch_append_batch_indices;
    const int32_t* batch_append_positions;

    qsfi_device_ptr_t d_batch_tokens;
    qsfi_device_ptr_t d_batch_qo_indptr;
    qsfi_device_ptr_t d_batch_kv_indptr;
    qsfi_device_ptr_t d_batch_kv_indices;
    qsfi_device_ptr_t d_batch_last_page_len;
    qsfi_device_ptr_t d_batch_rope_pos_offset;
    qsfi_device_ptr_t d_batch_append_batch_indices;
    qsfi_device_ptr_t d_batch_append_positions;
} qs_session_state_t;

typedef struct qs_session_append_batch {
    /*
     * Variable-length append against each request's current accepted prefix.
     *
     * Unknown request IDs create new request state and append from position 0.
     * Live request IDs append from their current committed sequence length.
     * Prompt rewrites are represented by release_requests() followed by a new
     * append, not by per-row reset flags.
     *
     * token_indptr is host-readable, has batch_size + 1 entries, starts at 0,
     * and ends at token_count. tokens is the flat concatenation of per-request
     * suffixes. This shape is used for cold prompt prefill, resumed suffix
     * prefill, and target-model MTP verification.
     */
    const qs_session_request_id_t* request_ids;
    const int32_t* token_indptr;
    const int32_t* tokens;
    uint32_t batch_size;
    uint32_t token_count;
} qs_session_append_batch_t;

typedef struct qs_session_decode_batch {
    /*
     * One next token per active request row. Use this for ordinary generation
     * and for each draft-model MTP iteration.
     */
    const qs_session_request_id_t* request_ids;
    const int32_t* tokens;
    uint32_t batch_size;
} qs_session_decode_batch_t;

typedef struct qs_session_commit {
    /*
     * Null commits the whole active batch. For append batches, accepted_token_counts
     * may instead point to batch_size entries, each in [0, row_token_count], so a
     * verifier can keep only the accepted prefix of every speculative suffix.
     * For decode batches, entries are in [0, 1].
     */
    const uint32_t* accepted_token_counts;
} qs_session_commit_t;

typedef struct qs_session_layer {
    uint32_t layer_idx;
    qsfi_tensor_desc_t q;
    qsfi_tensor_desc_t k;
    qsfi_tensor_desc_t v;
    qsfi_tensor_desc_t o;
    qsfi_device_ptr_t q_rope_offset;
    qsfi_device_ptr_t lse;
    float q_scale;
    float k_scale;
    float v_scale;
    uint32_t enable_pdl;
} qs_session_layer_t;

typedef qs_session_layer_t qs_session_append_layer_t;
typedef qs_session_layer_t qs_session_decode_layer_t;

qsfi_status_t qs_session_create(const qs_session_config_t* config, qs_session_t** out);
void qs_session_destroy(qs_session_t* session);

qsfi_status_t qs_session_reset(qs_session_t* session);
qsfi_status_t qs_session_release_requests(
    qs_session_t* session, const qs_session_request_id_t* request_ids, uint32_t request_count
);
qsfi_status_t qs_session_get_state(const qs_session_t* session, qs_session_state_t* out);

qsfi_status_t
qs_session_begin_append(qs_session_t* session, const qs_session_append_batch_t* batch);
qsfi_status_t
qs_session_append_layer(qs_session_t* session, const qs_session_append_layer_t* layer);

qsfi_status_t qs_session_begin_decode(qs_session_t* session, const qs_session_decode_batch_t* batch);
qsfi_status_t
qs_session_decode_layer(qs_session_t* session, const qs_session_decode_layer_t* layer);

qsfi_status_t qs_session_commit_batch(qs_session_t* session, const qs_session_commit_t* commit);
qsfi_status_t qs_session_abort_batch(qs_session_t* session);

#ifdef __cplusplus
}
#endif

#endif
