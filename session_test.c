#include "session.h"

#include <cuda_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static int check_status(qsfi_status_t got, qsfi_status_t want, const char* what)
{
    if (got == want)
        return 1;
    fprintf(
        stderr,
        "FAIL: %s: got %s (%d), want %s (%d)\n",
        what,
        qsfi_status_string(got),
        (int)got,
        qsfi_status_string(want),
        (int)want
    );
    failures += 1;
    return 0;
}

static int check_cuda(cudaError_t got, const char* what)
{
    if (got == cudaSuccess)
        return 1;
    fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(got));
    failures += 1;
    return 0;
}

static void check_u32(uint32_t got, uint32_t want, const char* what)
{
    if (got == want)
        return;
    fprintf(stderr, "FAIL: %s: got %u, want %u\n", what, got, want);
    failures += 1;
}

static void check_i32(int32_t got, int32_t want, const char* what)
{
    if (got == want)
        return;
    fprintf(stderr, "FAIL: %s: got %d, want %d\n", what, got, want);
    failures += 1;
}

static void check_id(qs_session_request_id_t got, qs_session_request_id_t want, const char* what)
{
    if (got == want)
        return;
    fprintf(
        stderr,
        "FAIL: %s: got %llu, want %llu\n",
        what,
        (unsigned long long)got,
        (unsigned long long)want
    );
    failures += 1;
}

static qsfi_tensor_desc_t tensor3(
    void* data, qsfi_dtype_t dtype, int64_t n, int64_t heads, int64_t head_dim
)
{
    qsfi_tensor_desc_t tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.ndim = 3;
    tensor.shape[0] = n;
    tensor.shape[1] = heads;
    tensor.shape[2] = head_dim;
    tensor.stride[0] = heads * head_dim;
    tensor.stride[1] = head_dim;
    tensor.stride[2] = 1;
    return tensor;
}

static int alloc_device(void** out, size_t bytes, const char* what)
{
    *out = NULL;
    return check_cuda(cudaMalloc(out, bytes), what);
}

static int copy_i32_to_device(int32_t** out, const int32_t* values, size_t count, const char* what)
{
    if (!alloc_device((void**)out, count * sizeof(values[0]), what))
        return 0;
    return check_cuda(cudaMemcpy(*out, values, count * sizeof(values[0]), cudaMemcpyHostToDevice), what);
}

static int check_device_u16_zero(const uint16_t* device, size_t count, const char* what)
{
    uint16_t* host = NULL;
    size_t i;
    int ok = 1;
    host = (uint16_t*)malloc(count * sizeof(host[0]));
    if (host == NULL) {
        fprintf(stderr, "FAIL: %s: host allocation failed\n", what);
        failures += 1;
        return 0;
    }
    if (!check_cuda(cudaMemcpy(host, device, count * sizeof(host[0]), cudaMemcpyDeviceToHost), what)) {
        free(host);
        return 0;
    }
    for (i = 0; i < count; ++i) {
        if (host[i] != 0) {
            fprintf(stderr, "FAIL: %s: element %zu is 0x%04x, want zero\n", what, i, host[i]);
            failures += 1;
            ok = 0;
            break;
        }
    }
    free(host);
    return ok;
}

static qs_session_config_t tiny_config(void)
{
    qs_session_config_t config;
    memset(&config, 0, sizeof(config));
    config.device_ordinal = 0;
    config.stream = NULL;
    config.num_layers = 1;
    config.max_live_requests = 4;
    config.max_batch_size = 3;
    config.max_seq_len = 8;
    config.max_pages = 8;
    config.page_size = 4;
    config.hidden_size = 128;
    config.num_q_heads = 2;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.activation_dtype = QSFI_DTYPE_F16;
    config.kv_dtype = QSFI_DTYPE_F16;
    config.kv_layout = QSFI_KV_LAYOUT_NHD;
    config.rope_theta = 10000.0f;
    config.rope_scale = 1.0f;
    config.qsfi_float_workspace_bytes = 64u << 20;
    config.qsfi_int_workspace_bytes = 64u << 20;
    config.qsfi_host_int_workspace_bytes = 64u << 20;
    return config;
}

static int get_state(qs_session_t* session, qs_session_state_t* state, const char* what)
{
    return check_status(qs_session_get_state(session, state), QSFI_STATUS_OK, what);
}

static int begin_append_tokens(
    qs_session_t* session,
    qs_session_request_id_t request_id,
    const int32_t* tokens,
    uint32_t token_count,
    const char* what
)
{
    qs_session_append_batch_t append;
    int32_t indptr[2];
    memset(&append, 0, sizeof(append));
    indptr[0] = 0;
    indptr[1] = (int32_t)token_count;
    append.request_ids = &request_id;
    append.token_indptr = indptr;
    append.tokens = tokens;
    append.batch_size = 1;
    append.token_count = token_count;
    return check_status(qs_session_begin_append(session, &append), QSFI_STATUS_OK, what);
}

static int commit_full(qs_session_t* session, const char* what)
{
    return check_status(qs_session_commit_batch(session, NULL), QSFI_STATUS_OK, what);
}

static void test_append_commit_decode_release(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t reqs[] = { 10, 11 };
    int32_t indptr[] = { 0, 5, 6 };
    int32_t tokens[] = { 100, 101, 102, 103, 104, 200 };
    qs_session_append_batch_t append;
    qs_session_state_t state;

    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create session"))
        return;

    memset(&append, 0, sizeof(append));
    append.request_ids = reqs;
    append.token_indptr = indptr;
    append.tokens = tokens;
    append.batch_size = 2;
    append.token_count = 6;

    check_status(qs_session_begin_append(session, &append), QSFI_STATUS_OK, "begin first append");

    memset(&state, 0, sizeof(state));
    if (get_state(session, &state, "state after first begin")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_APPEND, "first begin kind");
        check_u32(state.batch_size, 2, "first begin batch size");
        check_u32(state.batch_token_count, 6, "first begin token count");
        check_i32(state.batch_qo_indptr[0], 0, "first qo[0]");
        check_i32(state.batch_qo_indptr[1], 5, "first qo[1]");
        check_i32(state.batch_qo_indptr[2], 6, "first qo[2]");
        check_i32(state.batch_kv_indptr[0], 0, "first kv indptr[0]");
        check_i32(state.batch_kv_indptr[1], 2, "first kv indptr[1]");
        check_i32(state.batch_kv_indptr[2], 3, "first kv indptr[2]");
        check_i32(state.batch_last_page_len[0], 1, "first last len row 0");
        check_i32(state.batch_last_page_len[1], 1, "first last len row 1");
        check_i32(state.batch_append_batch_indices[4], 0, "first append batch idx row 0");
        check_i32(state.batch_append_batch_indices[5], 1, "first append batch idx row 1");
        check_i32(state.batch_append_positions[0], 0, "first append pos 0");
        check_i32(state.batch_append_positions[4], 4, "first append pos 4");
        check_i32(state.batch_append_positions[5], 0, "first append pos 5");
        check_u32(state.free_page_count, 5, "first begin free pages");
    }

    check_status(qs_session_commit_batch(session, NULL), QSFI_STATUS_OK, "commit first append");
    if (get_state(session, &state, "state after first commit")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_NONE, "first commit clears batch");
        check_u32(state.live_request_count, 2, "first commit live count");
        check_id(state.live_request_ids[0], 10, "first live id 0");
        check_id(state.live_request_ids[1], 11, "first live id 1");
        check_i32(state.live_seq_lens[0], 5, "first live seq 0");
        check_i32(state.live_seq_lens[1], 1, "first live seq 1");
        check_i32(state.live_kv_indptr[0], 0, "first live kv indptr 0");
        check_i32(state.live_kv_indptr[1], 2, "first live kv indptr 1");
        check_i32(state.live_kv_indptr[2], 3, "first live kv indptr 2");
        check_u32(state.free_page_count, 5, "first commit free pages");
    }

    {
        qs_session_request_id_t reqs2[] = { 10, 12 };
        int32_t indptr2[] = { 0, 2, 6 };
        int32_t tokens2[] = { 105, 106, 300, 301, 302, 303 };
        uint32_t accepted_append[] = { 1, 0 };
        qs_session_commit_t commit;
        append.request_ids = reqs2;
        append.token_indptr = indptr2;
        append.tokens = tokens2;
        append.batch_size = 2;
        append.token_count = 6;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_OK,
            "begin speculative append"
        );
        memset(&commit, 0, sizeof(commit));
        commit.accepted_token_counts = accepted_append;
        check_status(
            qs_session_commit_batch(session, &commit),
            QSFI_STATUS_OK,
            "partial append commit"
        );
    }
    if (get_state(session, &state, "state after partial append")) {
        check_u32(state.live_request_count, 2, "partial append live count");
        check_i32(state.live_seq_lens[0], 6, "partial append req 10 seq");
        check_i32(state.live_seq_lens[1], 1, "partial append req 11 seq");
        check_u32(state.free_page_count, 5, "partial append frees rejected pages");
    }

    {
        int32_t decode_tokens[] = { 107, 201 };
        uint32_t accepted_decode[] = { 0, 1 };
        qs_session_decode_batch_t decode;
        qs_session_commit_t commit;
        memset(&decode, 0, sizeof(decode));
        decode.request_ids = reqs;
        decode.tokens = decode_tokens;
        decode.batch_size = 2;
        check_status(qs_session_begin_decode(session, &decode), QSFI_STATUS_OK, "begin decode");
        memset(&commit, 0, sizeof(commit));
        commit.accepted_token_counts = accepted_decode;
        check_status(
            qs_session_commit_batch(session, &commit),
            QSFI_STATUS_OK,
            "partial decode commit"
        );
    }
    if (get_state(session, &state, "state after partial decode")) {
        check_i32(state.live_seq_lens[0], 6, "partial decode req 10 seq");
        check_i32(state.live_seq_lens[1], 2, "partial decode req 11 seq");
        check_u32(state.free_page_count, 5, "partial decode free pages");
    }

    {
        qs_session_request_id_t release_ids[] = { 10 };
        check_status(
            qs_session_release_requests(session, release_ids, 1),
            QSFI_STATUS_OK,
            "release req 10"
        );
    }
    if (get_state(session, &state, "state after release")) {
        check_u32(state.live_request_count, 1, "release live count");
        check_id(state.live_request_ids[0], 11, "release remaining id");
        check_i32(state.live_seq_lens[0], 2, "release remaining seq");
        check_u32(state.free_page_count, 7, "release frees req 10 pages");
    }

    qs_session_destroy(session);
}

static void test_reject_duplicate_batch_ids(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t reqs[] = { 1, 1 };
    int32_t indptr[] = { 0, 1, 2 };
    int32_t tokens[] = { 10, 11 };
    qs_session_append_batch_t append;

    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create duplicate test"))
        return;

    memset(&append, 0, sizeof(append));
    append.request_ids = reqs;
    append.token_indptr = indptr;
    append.tokens = tokens;
    append.batch_size = 2;
    append.token_count = 2;

    check_status(
        qs_session_begin_append(session, &append),
        QSFI_STATUS_INVALID_ARGUMENT,
        "duplicate append ids"
    );
    qs_session_destroy(session);
}

static void test_abort_and_reset_restore_pages(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t request_id = 31;
    int32_t tokens[] = { 1, 2, 3, 4 };
    qs_session_state_t state;

    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create abort test"))
        return;

    if (begin_append_tokens(session, request_id, tokens, 4, "begin abort append")) {
        qs_session_request_id_t release_ids[] = { request_id };
        qs_session_decode_batch_t decode;
        int32_t decode_token = 5;

        check_status(
            qs_session_release_requests(session, release_ids, 1),
            QSFI_STATUS_INVALID_ARGUMENT,
            "release while active is rejected"
        );

        memset(&decode, 0, sizeof(decode));
        decode.request_ids = &request_id;
        decode.tokens = &decode_token;
        decode.batch_size = 1;
        check_status(
            qs_session_begin_decode(session, &decode),
            QSFI_STATUS_INVALID_ARGUMENT,
            "nested begin is rejected"
        );

        check_status(qs_session_abort_batch(session), QSFI_STATUS_OK, "abort append");
    }

    if (get_state(session, &state, "state after abort")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_NONE, "abort clears batch");
        check_u32(state.live_request_count, 0, "abort keeps no live requests");
        check_u32(state.free_page_count, 8, "abort restores free pages");
    }

    if (begin_append_tokens(session, request_id, tokens, 4, "begin reset append"))
        check_status(qs_session_reset(session), QSFI_STATUS_OK, "reset while active");

    if (get_state(session, &state, "state after reset")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_NONE, "reset clears batch");
        check_u32(state.live_request_count, 0, "reset clears live requests");
        check_u32(state.free_page_count, 8, "reset restores all pages");
    }

    qs_session_destroy(session);
}

static void test_reject_bad_append_shapes(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t reqs[] = { 41, 42 };
    int32_t tokens[] = { 1, 2, 3 };
    qs_session_append_batch_t append;
    qs_session_state_t state;

    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create bad append test"))
        return;

    memset(&append, 0, sizeof(append));
    append.request_ids = reqs;
    append.tokens = tokens;
    append.batch_size = 2;

    {
        int32_t bad_start[] = { 1, 2, 3 };
        append.token_indptr = bad_start;
        append.token_count = 3;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_INVALID_ARGUMENT,
            "append indptr must start at zero"
        );
    }

    {
        int32_t nonmonotonic[] = { 0, 2, 1 };
        append.token_indptr = nonmonotonic;
        append.token_count = 1;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_INVALID_ARGUMENT,
            "append indptr must be monotonic"
        );
    }

    {
        int32_t wrong_total[] = { 0, 1, 2 };
        append.token_indptr = wrong_total;
        append.token_count = 3;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_INVALID_ARGUMENT,
            "append indptr total must match token_count"
        );
    }

    {
        qs_session_request_id_t dup_reqs[] = { 43, 43 };
        int32_t good_indptr[] = { 0, 1, 2 };
        append.request_ids = dup_reqs;
        append.token_indptr = good_indptr;
        append.token_count = 2;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_INVALID_ARGUMENT,
            "append duplicate request ids rejected"
        );
    }

    if (get_state(session, &state, "state after bad append cases")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_NONE, "bad append keeps batch clear");
        check_u32(state.live_request_count, 0, "bad append creates no live requests");
        check_u32(state.free_page_count, 8, "bad append does not allocate pages");
    }

    qs_session_destroy(session);
}

static void test_decode_validation_and_invalid_commit(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t request_id = 51;
    int32_t prompt[] = { 1 };
    qs_session_state_t state;

    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create decode test"))
        return;

    if (begin_append_tokens(session, request_id, prompt, 1, "seed decode request"))
        commit_full(session, "commit seeded request");

    {
        qs_session_request_id_t unknown_id = 52;
        int32_t token = 9;
        qs_session_decode_batch_t decode;
        memset(&decode, 0, sizeof(decode));
        decode.request_ids = &unknown_id;
        decode.tokens = &token;
        decode.batch_size = 1;
        check_status(
            qs_session_begin_decode(session, &decode),
            QSFI_STATUS_INVALID_ARGUMENT,
            "decode unknown request rejected"
        );
    }

    {
        qs_session_request_id_t dup_ids[] = { request_id, request_id };
        int32_t tokens[] = { 2, 3 };
        qs_session_decode_batch_t decode;
        memset(&decode, 0, sizeof(decode));
        decode.request_ids = dup_ids;
        decode.tokens = tokens;
        decode.batch_size = 2;
        check_status(
            qs_session_begin_decode(session, &decode),
            QSFI_STATUS_INVALID_ARGUMENT,
            "decode duplicate request ids rejected"
        );
    }

    {
        int32_t token = 4;
        uint32_t bad_accept[] = { 2 };
        qs_session_decode_batch_t decode;
        qs_session_commit_t commit;
        memset(&decode, 0, sizeof(decode));
        decode.request_ids = &request_id;
        decode.tokens = &token;
        decode.batch_size = 1;
        check_status(qs_session_begin_decode(session, &decode), QSFI_STATUS_OK, "begin decode commit test");

        memset(&commit, 0, sizeof(commit));
        commit.accepted_token_counts = bad_accept;
        check_status(
            qs_session_commit_batch(session, &commit),
            QSFI_STATUS_INVALID_ARGUMENT,
            "decode commit count must be zero or one"
        );

        if (get_state(session, &state, "state after rejected decode commit")) {
            check_u32(state.batch_kind, QS_SESSION_BATCH_DECODE, "bad decode commit keeps batch active");
            check_i32(state.live_seq_lens[0], 1, "bad decode commit leaves live seq unchanged");
        }
        check_status(qs_session_abort_batch(session), QSFI_STATUS_OK, "abort after bad decode commit");
    }

    if (get_state(session, &state, "state after decode validation")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_NONE, "decode validation ends clear");
        check_i32(state.live_seq_lens[0], 1, "decode validation live seq unchanged");
        check_u32(state.free_page_count, 7, "decode validation page count");
    }

    qs_session_destroy(session);
}

static void test_limits_and_release_noop(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t release_unknown[] = { 999 };
    qs_session_state_t state;

    config.max_live_requests = 1;
    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create limit test"))
        return;

    check_status(
        qs_session_release_requests(session, release_unknown, 1),
        QSFI_STATUS_OK,
        "release unknown is a noop"
    );

    {
        qs_session_request_id_t reqs[] = { 61, 62 };
        int32_t indptr[] = { 0, 1, 2 };
        int32_t tokens[] = { 1, 2 };
        qs_session_append_batch_t append;
        memset(&append, 0, sizeof(append));
        append.request_ids = reqs;
        append.token_indptr = indptr;
        append.tokens = tokens;
        append.batch_size = 2;
        append.token_count = 2;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_INVALID_ARGUMENT,
            "max live request limit enforced"
        );
    }

    {
        qs_session_request_id_t req = 63;
        int32_t indptr[] = { 0, 9 };
        int32_t tokens[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        qs_session_append_batch_t append;
        memset(&append, 0, sizeof(append));
        append.request_ids = &req;
        append.token_indptr = indptr;
        append.tokens = tokens;
        append.batch_size = 1;
        append.token_count = 9;
        check_status(
            qs_session_begin_append(session, &append),
            QSFI_STATUS_INVALID_ARGUMENT,
            "max sequence length enforced"
        );
    }

    if (get_state(session, &state, "state after limit cases")) {
        check_u32(state.batch_kind, QS_SESSION_BATCH_NONE, "limit cases keep batch clear");
        check_u32(state.live_request_count, 0, "limit cases create no requests");
        check_u32(state.free_page_count, 8, "limit cases do not allocate pages");
    }

    qs_session_destroy(session);
}

static void test_append_and_decode_layer_execute(void)
{
    qs_session_t* session = NULL;
    qs_session_config_t config = tiny_config();
    qs_session_request_id_t request_id = 71;
    int32_t prompt_tokens[] = { 10, 11, 12 };
    int32_t append_rope_offsets[] = { 0, 1, 2 };
    int32_t decode_rope_offsets[] = { 3 };
    qs_session_state_t state;
    uint16_t* d_append_q = NULL;
    uint16_t* d_append_k = NULL;
    uint16_t* d_append_v = NULL;
    uint16_t* d_append_o = NULL;
    uint16_t* d_decode_q = NULL;
    uint16_t* d_decode_k = NULL;
    uint16_t* d_decode_v = NULL;
    uint16_t* d_decode_o = NULL;
    int32_t* d_append_rope = NULL;
    int32_t* d_decode_rope = NULL;
    size_t append_q_elems;
    size_t append_kv_elems;
    size_t decode_q_elems;
    size_t decode_kv_elems;

    append_q_elems = 3u * config.num_q_heads * config.head_dim;
    append_kv_elems = 3u * config.num_kv_heads * config.head_dim;
    decode_q_elems = config.num_q_heads * config.head_dim;
    decode_kv_elems = config.num_kv_heads * config.head_dim;

    if (!check_status(qs_session_create(&config, &session), QSFI_STATUS_OK, "create layer test"))
        return;

    if (!alloc_device((void**)&d_append_q, append_q_elems * sizeof(d_append_q[0]), "alloc append q")
        || !alloc_device((void**)&d_append_k, append_kv_elems * sizeof(d_append_k[0]), "alloc append k")
        || !alloc_device((void**)&d_append_v, append_kv_elems * sizeof(d_append_v[0]), "alloc append v")
        || !alloc_device((void**)&d_append_o, append_q_elems * sizeof(d_append_o[0]), "alloc append o")
        || !alloc_device((void**)&d_decode_q, decode_q_elems * sizeof(d_decode_q[0]), "alloc decode q")
        || !alloc_device((void**)&d_decode_k, decode_kv_elems * sizeof(d_decode_k[0]), "alloc decode k")
        || !alloc_device((void**)&d_decode_v, decode_kv_elems * sizeof(d_decode_v[0]), "alloc decode v")
        || !alloc_device((void**)&d_decode_o, decode_q_elems * sizeof(d_decode_o[0]), "alloc decode o")
        || !copy_i32_to_device(&d_append_rope, append_rope_offsets, 3, "copy append rope offsets")
        || !copy_i32_to_device(&d_decode_rope, decode_rope_offsets, 1, "copy decode rope offsets")) {
        goto done;
    }

    if (!check_cuda(cudaMemset(d_append_q, 0, append_q_elems * sizeof(d_append_q[0])), "zero append q")
        || !check_cuda(cudaMemset(d_append_k, 0, append_kv_elems * sizeof(d_append_k[0])), "zero append k")
        || !check_cuda(cudaMemset(d_append_v, 0, append_kv_elems * sizeof(d_append_v[0])), "zero append v")
        || !check_cuda(cudaMemset(d_append_o, 0xA5, append_q_elems * sizeof(d_append_o[0])), "sentinel append o")
        || !check_cuda(cudaMemset(d_decode_q, 0, decode_q_elems * sizeof(d_decode_q[0])), "zero decode q")
        || !check_cuda(cudaMemset(d_decode_k, 0, decode_kv_elems * sizeof(d_decode_k[0])), "zero decode k")
        || !check_cuda(cudaMemset(d_decode_v, 0, decode_kv_elems * sizeof(d_decode_v[0])), "zero decode v")
        || !check_cuda(cudaMemset(d_decode_o, 0xA5, decode_q_elems * sizeof(d_decode_o[0])), "sentinel decode o")) {
        goto done;
    }

    if (begin_append_tokens(session, request_id, prompt_tokens, 3, "begin append layer execution")) {
        qs_session_append_layer_t layer;
        memset(&layer, 0, sizeof(layer));
        layer.layer_idx = 0;
        layer.q = tensor3(d_append_q, QSFI_DTYPE_F16, 3, config.num_q_heads, config.head_dim);
        layer.k = tensor3(d_append_k, QSFI_DTYPE_F16, 3, config.num_kv_heads, config.head_dim);
        layer.v = tensor3(d_append_v, QSFI_DTYPE_F16, 3, config.num_kv_heads, config.head_dim);
        layer.o = tensor3(d_append_o, QSFI_DTYPE_F16, 3, config.num_q_heads, config.head_dim);
        layer.q_rope_offset = d_append_rope;
        check_status(
            qs_session_append_layer(session, &layer),
            QSFI_STATUS_OK,
            "execute append layer"
        );
        check_cuda(cudaDeviceSynchronize(), "sync append layer");
        check_device_u16_zero(d_append_o, append_q_elems, "append layer zero output");
        commit_full(session, "commit append layer execution");
    }

    if (get_state(session, &state, "state after append layer execution"))
        check_i32(state.live_seq_lens[0], 3, "append layer live seq");

    {
        int32_t decode_token = 13;
        qs_session_decode_batch_t decode;
        memset(&decode, 0, sizeof(decode));
        decode.request_ids = &request_id;
        decode.tokens = &decode_token;
        decode.batch_size = 1;
        if (check_status(qs_session_begin_decode(session, &decode), QSFI_STATUS_OK, "begin decode layer execution")) {
            qs_session_decode_layer_t layer;
            memset(&layer, 0, sizeof(layer));
            layer.layer_idx = 0;
            layer.q = tensor3(d_decode_q, QSFI_DTYPE_F16, 1, config.num_q_heads, config.head_dim);
            layer.k = tensor3(d_decode_k, QSFI_DTYPE_F16, 1, config.num_kv_heads, config.head_dim);
            layer.v = tensor3(d_decode_v, QSFI_DTYPE_F16, 1, config.num_kv_heads, config.head_dim);
            layer.o = tensor3(d_decode_o, QSFI_DTYPE_F16, 1, config.num_q_heads, config.head_dim);
            layer.q_rope_offset = d_decode_rope;
            check_status(
                qs_session_decode_layer(session, &layer),
                QSFI_STATUS_OK,
                "execute decode layer"
            );
            check_cuda(cudaDeviceSynchronize(), "sync decode layer");
            check_device_u16_zero(d_decode_o, decode_q_elems, "decode layer zero output");
            commit_full(session, "commit decode layer execution");
        }
    }

    if (get_state(session, &state, "state after decode layer execution"))
        check_i32(state.live_seq_lens[0], 4, "decode layer live seq");

done:
    cudaFree(d_append_q);
    cudaFree(d_append_k);
    cudaFree(d_append_v);
    cudaFree(d_append_o);
    cudaFree(d_decode_q);
    cudaFree(d_decode_k);
    cudaFree(d_decode_v);
    cudaFree(d_decode_o);
    cudaFree(d_append_rope);
    cudaFree(d_decode_rope);
    qs_session_destroy(session);
}

int main(void)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("SKIP: no CUDA device available: %s\n", cudaGetErrorString(err));
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess)
        return 1;

    test_append_commit_decode_release();
    test_reject_duplicate_batch_ids();
    test_abort_and_reset_restore_pages();
    test_reject_bad_append_shapes();
    test_decode_validation_and_invalid_commit();
    test_limits_and_release_noop();
    test_append_and_decode_layer_execute();

    if (failures != 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    puts("session tests passed");
    return 0;
}
