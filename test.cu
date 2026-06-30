#include "qscb.h"
#include "qscu.h"
#include "qsfi.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kBatch = 2;
constexpr int kNumPages = 3;
constexpr int kPageSize = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kNumIndices = 3;
constexpr uint16_t kSentinel = 0xA5A5u;
constexpr size_t kAttentionWorkspaceBytes = 64ull << 20;

constexpr int kGdnQHeads = 16;
constexpr int kGdnKHeads = 16;
constexpr int kGdnVHeads = 32;
constexpr int kGdnKeyDim = 128;
constexpr int kGdnValueDim = 128;
constexpr int kGdnStateSlots = 1;
constexpr int kGdnActiveVHead = 3;
constexpr int kGdnActiveQKHead = kGdnActiveVHead / 2;
constexpr int kGdnActiveKeyDim = 7;
constexpr int kGdnActiveValueDim = 5;
constexpr uint16_t kBf16Zero = 0x0000u;
constexpr uint16_t kBf16One = 0x3F80u;
constexpr uint16_t kBf16OnePoint25 = 0x3FA0u;
constexpr uint16_t kBf16OnePoint5 = 0x3FC0u;
constexpr uint16_t kBf16Two = 0x4000u;
constexpr uint16_t kBf16Three = 0x4040u;

int failures = 0;

bool check_cuda(cudaError_t got, const char* what)
{
    if (got == cudaSuccess)
        return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(got));
    failures += 1;
    return false;
}

bool check_status(qsfi_status got, qsfi_status want, const char* what)
{
    if (got == want)
        return true;
    std::fprintf(
        stderr,
        "FAIL: %s: got %s (%d), want %s (%d)\n",
        what,
        qsfi_status_string(got),
        static_cast<int>(got),
        qsfi_status_string(want),
        static_cast<int>(want)
    );
    failures += 1;
    return false;
}

void check_status_message(
    qsfi_context* ctx,
    qsfi_status got,
    qsfi_status want,
    const char* message_fragment,
    const char* what
)
{
    if (!check_status(got, want, what))
        return;

    qsfi_error_info error {};
    if (!check_status(qsfi_context_get_last_error(ctx, &error), QSFI_STATUS_OK, what))
        return;
    if (std::strstr(error.message, message_fragment) == nullptr) {
        std::fprintf(
            stderr,
            "FAIL: %s: last error message \"%s\" does not contain \"%s\"\n",
            what,
            error.message,
            message_fragment
        );
        failures += 1;
    }
}

template <typename T> bool alloc_device(T** device, size_t count, const char* what)
{
    return check_cuda(cudaMalloc(reinterpret_cast<void**>(device), count * sizeof(T)), what);
}

uint16_t key_value(int item, int head, int dim)
{
    return static_cast<uint16_t>(0x1000u + item * 0x0100u + head * 0x0040u + dim);
}

uint16_t value_value(int item, int head, int dim)
{
    return static_cast<uint16_t>(0x4000u + item * 0x0100u + head * 0x0040u + dim);
}

size_t cache_offset(int page, int entry, int head, int dim)
{
    return ((static_cast<size_t>(page) * kPageSize + entry) * kKvHeads + head) * kHeadDim + dim;
}

size_t append_offset(int item, int head, int dim)
{
    return (static_cast<size_t>(item) * kKvHeads + head) * kHeadDim + dim;
}

size_t gdn_qk_offset(int token, int head, int dim)
{
    return (static_cast<size_t>(token) * kGdnQHeads + head) * kGdnKeyDim + dim;
}

size_t gdn_v_offset(int token, int head, int dim)
{
    return (static_cast<size_t>(token) * kGdnVHeads + head) * kGdnValueDim + dim;
}

size_t gdn_state_offset(int slot, int head, int value_dim, int key_dim)
{
    return (((static_cast<size_t>(slot) * kGdnVHeads + head) * kGdnValueDim + value_dim)
            * kGdnKeyDim)
        + key_dim;
}

float bf16_to_f32(uint16_t bits)
{
    const uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &f32_bits, sizeof(value));
    return value;
}

qsfi_tensor1 tensor1_u8(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_U8;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

qsfi_tensor3 tensor3(void* data, int64_t n)
{
    qsfi_tensor3 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F16;
    tensor.shape[0] = n;
    tensor.shape[1] = kKvHeads;
    tensor.shape[2] = kHeadDim;
    tensor.stride[0] = kKvHeads * kHeadDim;
    tensor.stride[1] = kHeadDim;
    tensor.stride[2] = 1;
    return tensor;
}

qsfi_tensor2 tensor2_i32(void* data, int64_t rows, int64_t cols)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_I32;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.stride[0] = cols;
    tensor.stride[1] = 1;
    return tensor;
}

qsfi_tensor2 tensor2_f32(void* data, int64_t rows, int64_t cols)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F32;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.stride[0] = cols;
    tensor.stride[1] = 1;
    return tensor;
}

qsfi_tensor2 tensor2_bf16(void* data, int64_t rows, int64_t cols)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.stride[0] = cols;
    tensor.stride[1] = 1;
    return tensor;
}

qsfi_tensor1 gdn_tensor1_i32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_I32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

qsfi_tensor1 gdn_tensor1_f32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

qsfi_tensor2 gdn_tensor2_bf16(void* data, int64_t n, int64_t heads)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = n;
    tensor.shape[1] = heads;
    tensor.stride[0] = heads;
    tensor.stride[1] = 1;
    return tensor;
}

qsfi_tensor3 tensor3_bf16(void* data, int64_t d0, int64_t d1, int64_t d2)
{
    qsfi_tensor3 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = d0;
    tensor.shape[1] = d1;
    tensor.shape[2] = d2;
    tensor.stride[0] = d1 * d2;
    tensor.stride[1] = d2;
    tensor.stride[2] = 1;
    return tensor;
}

qsfi_tensor3 gdn_tensor3_bf16(void* data, int64_t n, int64_t heads, int64_t dim)
{
    return tensor3_bf16(data, n, heads, dim);
}

qsfi_tensor4 gdn_state_tensor_bf16(void* data)
{
    qsfi_tensor4 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = kGdnStateSlots;
    tensor.shape[1] = kGdnVHeads;
    tensor.shape[2] = kGdnValueDim;
    tensor.shape[3] = kGdnKeyDim;
    tensor.stride[0] = kGdnVHeads * kGdnValueDim * kGdnKeyDim;
    tensor.stride[1] = kGdnValueDim * kGdnKeyDim;
    tensor.stride[2] = kGdnKeyDim;
    tensor.stride[3] = 1;
    return tensor;
}

qsfi_tensor4 cache_tensor(void* data)
{
    qsfi_tensor4 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F16;
    tensor.shape[0] = kNumPages;
    tensor.shape[1] = kPageSize;
    tensor.shape[2] = kKvHeads;
    tensor.shape[3] = kHeadDim;
    tensor.stride[0] = kPageSize * kKvHeads * kHeadDim;
    tensor.stride[1] = kKvHeads * kHeadDim;
    tensor.stride[2] = kHeadDim;
    tensor.stride[3] = 1;
    return tensor;
}

qsfi_tensor4 cache_tensor_bf16(void* data)
{
    qsfi_tensor4 tensor = cache_tensor(data);
    tensor.dtype = QSFI_DTYPE_BF16;
    return tensor;
}

qsfi_attention_desc attention_desc()
{
    qsfi_attention_desc attention {};
    attention.num_qo_heads = kKvHeads;
    attention.num_kv_heads = kKvHeads;
    attention.head_dim_qk = kHeadDim;
    attention.head_dim_vo = kHeadDim;
    attention.page_size = kPageSize;
    attention.q_dtype = QSFI_DTYPE_F16;
    attention.kv_dtype = QSFI_DTYPE_F16;
    attention.o_dtype = QSFI_DTYPE_F16;
    attention.kv_layout = QSFI_KV_LAYOUT_NHD;
    attention.pos_encoding = QSFI_POS_ENCODING_NONE;
    attention.mask_mode = QSFI_MASK_MODE_NONE;
    attention.window_left = -1;
    attention.rope_scale = 1.0f;
    attention.rope_theta = 10000.0f;
    return attention;
}

qsfi_paged_kv_cache cache_desc(void* k, void* v)
{
    qsfi_paged_kv_cache cache {};
    cache.k = cache_tensor(k);
    cache.v = cache_tensor(v);
    return cache;
}

qsfi_paged_kv_cache cache_desc_bf16(void* k, void* v)
{
    qsfi_paged_kv_cache cache {};
    cache.k = cache_tensor_bf16(k);
    cache.v = cache_tensor_bf16(v);
    return cache;
}

qsfi_paged_kv_table page_table_desc(void* indptr, void* indices, void* last_page_len)
{
    qsfi_paged_kv_table table {};
    table.indptr = indptr;
    table.indices = indices;
    table.last_page_len = last_page_len;
    table.batch_size = kBatch;
    table.num_indices = kNumIndices;
    return table;
}

bool make_context(qsfi_context** out)
{
    qsfi_context_desc desc {};
    desc.device_ordinal = 0;
    desc.stream = nullptr;
    return check_status(qsfi_context_create(&desc, out), QSFI_STATUS_OK, "create context");
}

bool reserve_attention_workspace(qsfi_context* ctx)
{
    return check_status(
        qsfi_context_reserve_workspace(
            ctx,
            kAttentionWorkspaceBytes,
            kAttentionWorkspaceBytes,
            kAttentionWorkspaceBytes
        ),
        QSFI_STATUS_OK,
        "reserve attention workspace"
    );
}

template <typename T> bool copy_to_device(T** device, const T* host, size_t count, const char* what)
{
    if (!alloc_device(device, count, what))
        return false;
    return check_cuda(cudaMemcpy(*device, host, count * sizeof(T), cudaMemcpyHostToDevice), what);
}

bool make_page_table(int32_t** d_indptr, int32_t** d_indices, int32_t** d_last_page_len)
{
    const int32_t indptr[] = { 0, 2, 3 };
    const int32_t indices[] = { 0, 1, 2 };
    const int32_t last_page_len[] = { 1, 4 };
    return copy_to_device(d_indptr, indptr, 3, "copy page indptr")
        && copy_to_device(d_indices, indices, 3, "copy page indices")
        && copy_to_device(d_last_page_len, last_page_len, 2, "copy last page lengths");
}

void fill_gdn_inputs(
    std::vector<uint16_t>& q,
    std::vector<uint16_t>& k,
    std::vector<uint16_t>& v,
    std::vector<uint16_t>& a,
    std::vector<uint16_t>& b,
    int tokens
)
{
    std::fill(q.begin(), q.end(), kBf16Zero);
    std::fill(k.begin(), k.end(), kBf16Zero);
    std::fill(v.begin(), v.end(), kBf16Zero);
    std::fill(a.begin(), a.end(), kBf16Zero);
    std::fill(b.begin(), b.end(), kBf16Zero);
    for (int token = 0; token < tokens; ++token) {
        q[gdn_qk_offset(token, kGdnActiveQKHead, kGdnActiveKeyDim)] = kBf16One;
        k[gdn_qk_offset(token, kGdnActiveQKHead, kGdnActiveKeyDim)] = kBf16One;
        v[gdn_v_offset(token, kGdnActiveVHead, kGdnActiveValueDim)] = kBf16Two;
    }
}

bool approx_equal(float got, float want, float abs_tol)
{
    return std::fabs(got - want) <= abs_tol;
}

void check_bf16_vector_close(
    const std::vector<uint16_t>& got_bits,
    const std::vector<float>& want,
    float abs_tol,
    const char* what
)
{
    if (got_bits.size() != want.size()) {
        std::fprintf(stderr, "FAIL: %s: size mismatch\n", what);
        failures += 1;
        return;
    }
    for (size_t i = 0; i < got_bits.size(); ++i) {
        const float got = bf16_to_f32(got_bits[i]);
        if (!approx_equal(got, want[i], abs_tol)) {
            std::fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.8f, want %.8f +/- %.8f\n",
                what,
                i,
                got,
                want[i],
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

int attention_seq_len(const int32_t* indptr, const int32_t* last_page_len, int request)
{
    const int pages = indptr[request + 1] - indptr[request];
    if (pages == 0)
        return 0;
    return (pages - 1) * kPageSize + last_page_len[request];
}

size_t attention_logical_cache_offset(
    const int32_t* indptr, const int32_t* indices, int request, int token, int head, int dim
)
{
    const int page = indices[indptr[request] + token / kPageSize];
    const int entry = token % kPageSize;
    return cache_offset(page, entry, head, dim);
}

uint16_t attention_key_pattern(int request, int token)
{
    static const uint16_t req0[] = { kBf16Zero, kBf16One, kBf16Two, kBf16Three, kBf16OnePoint5 };
    static const uint16_t req1[] = { kBf16Two, kBf16One, kBf16Zero, kBf16Three };
    return request == 0 ? req0[token] : req1[token];
}

uint16_t attention_value_pattern(int request, int token, int head)
{
    static const uint16_t req0[]
        = { kBf16One, kBf16Two, kBf16Three, kBf16OnePoint5, kBf16OnePoint25 };
    static const uint16_t req1[] = { kBf16OnePoint25, kBf16Three, kBf16One, kBf16Two };
    const uint16_t* values = request == 0 ? req0 : req1;
    const int len = request == 0 ? 5 : 4;
    return values[(token + head) % len];
}

void fill_attention_queries(std::vector<uint16_t>& q)
{
    std::fill(q.begin(), q.end(), kBf16Zero);
    const int queries = static_cast<int>(q.size() / (kKvHeads * kHeadDim));
    for (int query = 0; query < queries; ++query) {
        for (int head = 0; head < kKvHeads; ++head)
            q[append_offset(query, head, head)] = kBf16One;
    }
}

void fill_attention_cache_fixture(
    std::vector<uint16_t>& k_cache,
    std::vector<uint16_t>& v_cache,
    const int32_t* indptr,
    const int32_t* indices,
    const int32_t* last_page_len
)
{
    std::fill(k_cache.begin(), k_cache.end(), kBf16Zero);
    std::fill(v_cache.begin(), v_cache.end(), kBf16Zero);
    for (int request = 0; request < kBatch; ++request) {
        const int seq_len = attention_seq_len(indptr, last_page_len, request);
        for (int token = 0; token < seq_len; ++token) {
            for (int head = 0; head < kKvHeads; ++head) {
                k_cache[attention_logical_cache_offset(indptr, indices, request, token, head, head)]
                    = attention_key_pattern(request, token);
                v_cache[attention_logical_cache_offset(indptr, indices, request, token, head, head)]
                    = attention_value_pattern(request, token, head);
            }
        }
    }
}

void cpu_attention_reference(
    const std::vector<uint16_t>& q,
    const std::vector<uint16_t>& k_cache,
    const std::vector<uint16_t>& v_cache,
    const int* query_to_request,
    int num_queries,
    const int32_t* indptr,
    const int32_t* indices,
    const int32_t* last_page_len,
    std::vector<float>& out
)
{
    std::fill(out.begin(), out.end(), 0.0f);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (int query = 0; query < num_queries; ++query) {
        const int request = query_to_request[query];
        const int seq_len = attention_seq_len(indptr, last_page_len, request);
        for (int head = 0; head < kKvHeads; ++head) {
            std::vector<float> scores(seq_len, 0.0f);
            float max_score = -std::numeric_limits<float>::infinity();
            for (int token = 0; token < seq_len; ++token) {
                float dot = 0.0f;
                for (int dim = 0; dim < kHeadDim; ++dim) {
                    dot += bf16_to_f32(q[append_offset(query, head, dim)])
                        * bf16_to_f32(
                               k_cache[attention_logical_cache_offset(
                                   indptr,
                                   indices,
                                   request,
                                   token,
                                   head,
                                   dim
                               )]
                        );
                }
                scores[token] = dot * sm_scale;
                max_score = std::max(max_score, scores[token]);
            }

            float denom = 0.0f;
            for (int token = 0; token < seq_len; ++token) {
                scores[token] = std::exp(scores[token] - max_score);
                denom += scores[token];
            }
            for (int dim = 0; dim < kHeadDim; ++dim) {
                float acc = 0.0f;
                for (int token = 0; token < seq_len; ++token) {
                    const float weight = scores[token] / denom;
                    acc += weight
                        * bf16_to_f32(
                               v_cache[attention_logical_cache_offset(
                                   indptr,
                                   indices,
                                   request,
                                   token,
                                   head,
                                   dim
                               )]
                        );
                }
                out[append_offset(query, head, dim)] = acc;
            }
        }
    }
}

void check_moe_two_outputs(
    const std::vector<uint16_t>& out,
    int token0,
    int hidden0,
    float want0,
    int token1,
    int hidden1,
    float want1,
    float abs_tol,
    const char* what
)
{
    constexpr int kMoeHidden = 8;
    for (size_t i = 0; i < out.size(); ++i) {
        const int row = static_cast<int>(i / kMoeHidden);
        const int col = static_cast<int>(i % kMoeHidden);
        float expected = 0.0f;
        if (row == token0 && col == hidden0)
            expected = want0;
        else if (row == token1 && col == hidden1)
            expected = want1;
        const float got = bf16_to_f32(out[i]);
        if (!approx_equal(got, expected, abs_tol)) {
            std::fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.8f, want %.8f +/- %.8f\n",
                what,
                i,
                got,
                expected,
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

void check_bf16_single_nonzero(
    const std::vector<uint16_t>& values,
    size_t nonzero_offset,
    float nonzero_value,
    float abs_tol,
    const char* what
)
{
    for (size_t i = 0; i < values.size(); ++i) {
        const float want = i == nonzero_offset ? nonzero_value : 0.0f;
        const float got = bf16_to_f32(values[i]);
        if (!approx_equal(got, want, abs_tol)) {
            std::fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.8f, want %.8f +/- %.8f\n",
                what,
                i,
                got,
                want,
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

void check_gdn_prefill_output(const std::vector<uint16_t>& out, float abs_tol)
{
    const size_t token0 = gdn_v_offset(0, kGdnActiveVHead, kGdnActiveValueDim);
    const size_t token1 = gdn_v_offset(1, kGdnActiveVHead, kGdnActiveValueDim);
    for (size_t i = 0; i < out.size(); ++i) {
        float want = 0.0f;
        if (i == token0)
            want = 1.0f;
        else if (i == token1)
            want = 1.25f;
        const float got = bf16_to_f32(out[i]);
        if (!approx_equal(got, want, abs_tol)) {
            std::fprintf(
                stderr,
                "FAIL: gdn prefill out[%zu]: got %.8f, want %.8f +/- %.8f\n",
                i,
                got,
                want,
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

void fill_append(std::vector<uint16_t>& k, std::vector<uint16_t>& v)
{
    for (size_t item = 0; item < k.size() / (kKvHeads * kHeadDim); ++item) {
        for (int head = 0; head < kKvHeads; ++head) {
            for (int dim = 0; dim < kHeadDim; ++dim) {
                k[append_offset(static_cast<int>(item), head, dim)]
                    = key_value(static_cast<int>(item), head, dim);
                v[append_offset(static_cast<int>(item), head, dim)]
                    = value_value(static_cast<int>(item), head, dim);
            }
        }
    }
}

void check_cache_item(
    const std::vector<uint16_t>& cache,
    const std::vector<uint16_t>& append,
    int page,
    int entry,
    int item,
    const char* what
)
{
    for (int head = 0; head < kKvHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
            if (cache[cache_offset(page, entry, head, dim)]
                != append[append_offset(item, head, dim)]) {
                std::fprintf(
                    stderr,
                    "FAIL: %s page=%d entry=%d item=%d head=%d dim=%d\n",
                    what,
                    page,
                    entry,
                    item,
                    head,
                    dim
                );
                failures += 1;
                return;
            }
        }
    }
}

void check_cache_sentinel(const std::vector<uint16_t>& cache, int page, int entry, const char* what)
{
    for (int head = 0; head < kKvHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
            if (cache[cache_offset(page, entry, head, dim)] != kSentinel) {
                std::fprintf(
                    stderr,
                    "FAIL: %s page=%d entry=%d head=%d dim=%d\n",
                    what,
                    page,
                    entry,
                    head,
                    dim
                );
                failures += 1;
                return;
            }
        }
    }
}

void test_batch_decode_attention_matches_cpu_reference()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;
    if (!reserve_attention_workspace(ctx)) {
        qsfi_context_destroy(ctx);
        return;
    }

    const int32_t indptr[] = { 0, 2, 3 };
    const int32_t indices[] = { 0, 1, 2 };
    const int32_t last_page_len[] = { 1, 4 };
    const int query_to_request[] = { 0, 1 };
    constexpr size_t cache_elems = static_cast<size_t>(kNumPages) * kPageSize * kKvHeads * kHeadDim;
    constexpr size_t q_elems = static_cast<size_t>(kBatch) * kKvHeads * kHeadDim;

    std::vector<uint16_t> h_q(q_elems);
    std::vector<uint16_t> h_k_cache(cache_elems);
    std::vector<uint16_t> h_v_cache(cache_elems);
    std::vector<uint16_t> h_out(q_elems, kSentinel);
    std::vector<float> h_expected(q_elems, 0.0f);
    fill_attention_queries(h_q);
    fill_attention_cache_fixture(h_k_cache, h_v_cache, indptr, indices, last_page_len);
    cpu_attention_reference(
        h_q,
        h_k_cache,
        h_v_cache,
        query_to_request,
        kBatch,
        indptr,
        indices,
        last_page_len,
        h_expected
    );

    uint16_t* d_q = nullptr;
    uint16_t* d_k_cache = nullptr;
    uint16_t* d_v_cache = nullptr;
    uint16_t* d_out = nullptr;
    int32_t* d_indptr = nullptr;
    int32_t* d_indices = nullptr;
    int32_t* d_last_page_len = nullptr;
    qsfi_batch_decode_plan* plan = nullptr;

    bool ok = copy_to_device(&d_q, h_q.data(), h_q.size(), "copy decode attention q")
        && copy_to_device(
                  &d_k_cache,
                  h_k_cache.data(),
                  h_k_cache.size(),
                  "copy decode attention k cache"
        )
        && copy_to_device(
                  &d_v_cache,
                  h_v_cache.data(),
                  h_v_cache.size(),
                  "copy decode attention v cache"
        )
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy decode attention out")
        && make_page_table(&d_indptr, &d_indices, &d_last_page_len);

    qsfi_attention_desc attention = attention_desc();
    attention.q_dtype = QSFI_DTYPE_BF16;
    attention.kv_dtype = QSFI_DTYPE_BF16;
    attention.o_dtype = QSFI_DTYPE_BF16;

    qsfi_paged_kv_plan plan_table {};
    plan_table.indptr = indptr;
    plan_table.indices = indices;
    plan_table.last_page_len = last_page_len;
    plan_table.batch_size = kBatch;
    plan_table.num_indices = kNumIndices;
    if (ok) {
        ok = check_status(
            qsfi_batch_decode_plan_create(ctx, &attention, &plan_table, &plan),
            QSFI_STATUS_OK,
            "batch decode plan"
        );
    }
    if (ok) {
        qsfi_batch_decode_execute_desc desc {};
        desc.q = tensor3_bf16(d_q, kBatch, kKvHeads, kHeadDim);
        desc.o = tensor3_bf16(d_out, kBatch, kKvHeads, kHeadDim);
        desc.kv_cache = cache_desc_bf16(d_k_cache, d_v_cache);
        desc.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);

        check_status(
            qsfi_batch_decode_execute(ctx, plan, &desc),
            QSFI_STATUS_OK,
            "batch decode execute"
        );
        check_cuda(cudaDeviceSynchronize(), "sync batch decode execute");
        check_cuda(
            cudaMemcpy(
                h_out.data(),
                d_out,
                h_out.size() * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy batch decode out back"
        );
        check_bf16_vector_close(h_out, h_expected, 0.06f, "batch decode attention out");
    }

    qsfi_batch_decode_plan_destroy(plan);
    cudaFree(d_q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_out);
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_last_page_len);
    qsfi_context_destroy(ctx);
}

void test_batch_prefill_attention_matches_cpu_reference()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;
    if (!reserve_attention_workspace(ctx)) {
        qsfi_context_destroy(ctx);
        return;
    }

    constexpr int total_tokens = 3;
    const int32_t qo_indptr[] = { 0, 2, total_tokens };
    const int32_t indptr[] = { 0, 2, 3 };
    const int32_t indices[] = { 0, 1, 2 };
    const int32_t last_page_len[] = { 1, 4 };
    const int query_to_request[] = { 0, 0, 1 };
    constexpr size_t cache_elems = static_cast<size_t>(kNumPages) * kPageSize * kKvHeads * kHeadDim;
    constexpr size_t q_elems = static_cast<size_t>(total_tokens) * kKvHeads * kHeadDim;

    std::vector<uint16_t> h_q(q_elems);
    std::vector<uint16_t> h_k_cache(cache_elems);
    std::vector<uint16_t> h_v_cache(cache_elems);
    std::vector<uint16_t> h_out(q_elems, kSentinel);
    std::vector<float> h_expected(q_elems, 0.0f);
    fill_attention_queries(h_q);
    fill_attention_cache_fixture(h_k_cache, h_v_cache, indptr, indices, last_page_len);
    cpu_attention_reference(
        h_q,
        h_k_cache,
        h_v_cache,
        query_to_request,
        total_tokens,
        indptr,
        indices,
        last_page_len,
        h_expected
    );

    uint16_t* d_q = nullptr;
    uint16_t* d_k_cache = nullptr;
    uint16_t* d_v_cache = nullptr;
    uint16_t* d_out = nullptr;
    int32_t* d_qo_indptr = nullptr;
    int32_t* d_indptr = nullptr;
    int32_t* d_indices = nullptr;
    int32_t* d_last_page_len = nullptr;
    qsfi_batch_prefill_plan* plan = nullptr;

    bool ok = copy_to_device(&d_q, h_q.data(), h_q.size(), "copy prefill attention q")
        && copy_to_device(
                  &d_k_cache,
                  h_k_cache.data(),
                  h_k_cache.size(),
                  "copy prefill attention k cache"
        )
        && copy_to_device(
                  &d_v_cache,
                  h_v_cache.data(),
                  h_v_cache.size(),
                  "copy prefill attention v cache"
        )
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy prefill attention out")
        && copy_to_device(&d_qo_indptr, qo_indptr, 3, "copy prefill qo indptr")
        && make_page_table(&d_indptr, &d_indices, &d_last_page_len);

    qsfi_attention_desc attention = attention_desc();
    attention.q_dtype = QSFI_DTYPE_BF16;
    attention.kv_dtype = QSFI_DTYPE_BF16;
    attention.o_dtype = QSFI_DTYPE_BF16;

    qsfi_qo_plan qo_plan {};
    qo_plan.indptr = qo_indptr;
    qo_plan.batch_size = kBatch;
    qo_plan.total_tokens = total_tokens;
    qsfi_paged_kv_plan plan_table {};
    plan_table.indptr = indptr;
    plan_table.indices = indices;
    plan_table.last_page_len = last_page_len;
    plan_table.batch_size = kBatch;
    plan_table.num_indices = kNumIndices;
    if (ok) {
        ok = check_status(
            qsfi_batch_prefill_plan_create(ctx, &attention, &qo_plan, &plan_table, &plan),
            QSFI_STATUS_OK,
            "batch prefill plan"
        );
    }
    if (ok) {
        qsfi_batch_prefill_execute_desc desc {};
        desc.q = tensor3_bf16(d_q, total_tokens, kKvHeads, kHeadDim);
        desc.o = tensor3_bf16(d_out, total_tokens, kKvHeads, kHeadDim);
        desc.qo_indptr = d_qo_indptr;
        desc.kv_cache = cache_desc_bf16(d_k_cache, d_v_cache);
        desc.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);

        check_status(
            qsfi_batch_prefill_execute(ctx, plan, &desc),
            QSFI_STATUS_OK,
            "batch prefill execute"
        );
        check_cuda(cudaDeviceSynchronize(), "sync batch prefill execute");
        check_cuda(
            cudaMemcpy(
                h_out.data(),
                d_out,
                h_out.size() * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy batch prefill out back"
        );
        check_bf16_vector_close(h_out, h_expected, 0.06f, "batch prefill attention out");
    }

    qsfi_batch_prefill_plan_destroy(plan);
    cudaFree(d_q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_out);
    cudaFree(d_qo_indptr);
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_last_page_len);
    qsfi_context_destroy(ctx);
}

void test_decode_append_uses_post_append_last_page_len()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr size_t cache_elems = static_cast<size_t>(kNumPages) * kPageSize * kKvHeads * kHeadDim;
    constexpr size_t append_elems = static_cast<size_t>(kBatch) * kKvHeads * kHeadDim;

    std::vector<uint16_t> h_k_append(append_elems);
    std::vector<uint16_t> h_v_append(append_elems);
    std::vector<uint16_t> h_k_cache(cache_elems);
    std::vector<uint16_t> h_v_cache(cache_elems);
    fill_append(h_k_append, h_v_append);

    uint16_t* d_k_cache = nullptr;
    uint16_t* d_v_cache = nullptr;
    uint16_t* d_k_append = nullptr;
    uint16_t* d_v_append = nullptr;
    int32_t* d_indptr = nullptr;
    int32_t* d_indices = nullptr;
    int32_t* d_last_page_len = nullptr;

    if (!alloc_device(&d_k_cache, cache_elems, "alloc k cache")
        || !alloc_device(&d_v_cache, cache_elems, "alloc v cache")
        || !check_cuda(cudaMemset(d_k_cache, 0xA5, cache_elems * sizeof(uint16_t)), "clear k cache")
        || !check_cuda(cudaMemset(d_v_cache, 0xA5, cache_elems * sizeof(uint16_t)), "clear v cache")
        || !copy_to_device(&d_k_append, h_k_append.data(), h_k_append.size(), "copy decode k")
        || !copy_to_device(&d_v_append, h_v_append.data(), h_v_append.size(), "copy decode v")
        || !make_page_table(&d_indptr, &d_indices, &d_last_page_len)) {
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_append_decode_desc append {};
    append.k = tensor3(d_k_append, kBatch);
    append.v = tensor3(d_v_append, kBatch);
    append.kv_cache = cache_desc(d_k_cache, d_v_cache);
    append.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);

    const qsfi_attention_desc attention = attention_desc();
    check_status(
        qsfi_append_paged_kv_decode(ctx, &attention, &append),
        QSFI_STATUS_OK,
        "decode append"
    );
    check_cuda(cudaDeviceSynchronize(), "sync decode append");
    check_cuda(
        cudaMemcpy(
            h_k_cache.data(),
            d_k_cache,
            cache_elems * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy k cache back"
    );
    check_cuda(
        cudaMemcpy(
            h_v_cache.data(),
            d_v_cache,
            cache_elems * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy v cache back"
    );

    check_cache_item(h_k_cache, h_k_append, 1, 0, 0, "decode k writes batch 0 to new page");
    check_cache_item(h_v_cache, h_v_append, 1, 0, 0, "decode v writes batch 0 to new page");
    check_cache_item(h_k_cache, h_k_append, 2, 3, 1, "decode k writes batch 1 page tail");
    check_cache_item(h_v_cache, h_v_append, 2, 3, 1, "decode v writes batch 1 page tail");
    check_cache_sentinel(h_k_cache, 0, 0, "decode leaves unrelated page untouched");
    check_cache_sentinel(h_v_cache, 1, 1, "decode leaves following entry untouched");

    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_append);
    cudaFree(d_v_append);
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_last_page_len);
    qsfi_context_destroy(ctx);
}

void test_prefill_append_maps_positions_through_page_table()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int num_tokens = 3;
    constexpr size_t cache_elems = static_cast<size_t>(kNumPages) * kPageSize * kKvHeads * kHeadDim;
    constexpr size_t append_elems = static_cast<size_t>(num_tokens) * kKvHeads * kHeadDim;

    std::vector<uint16_t> h_k_append(append_elems);
    std::vector<uint16_t> h_v_append(append_elems);
    std::vector<uint16_t> h_k_cache(cache_elems);
    std::vector<uint16_t> h_v_cache(cache_elems);
    const int32_t batch_indices[] = { 0, 0, 1 };
    const int32_t positions[] = { 0, 4, 3 };
    fill_append(h_k_append, h_v_append);

    uint16_t* d_k_cache = nullptr;
    uint16_t* d_v_cache = nullptr;
    uint16_t* d_k_append = nullptr;
    uint16_t* d_v_append = nullptr;
    int32_t* d_indptr = nullptr;
    int32_t* d_indices = nullptr;
    int32_t* d_last_page_len = nullptr;
    int32_t* d_batch_indices = nullptr;
    int32_t* d_positions = nullptr;

    if (!alloc_device(&d_k_cache, cache_elems, "alloc prefill k cache")
        || !alloc_device(&d_v_cache, cache_elems, "alloc prefill v cache")
        || !check_cuda(
            cudaMemset(d_k_cache, 0xA5, cache_elems * sizeof(uint16_t)),
            "clear prefill k"
        )
        || !check_cuda(
            cudaMemset(d_v_cache, 0xA5, cache_elems * sizeof(uint16_t)),
            "clear prefill v"
        )
        || !copy_to_device(&d_k_append, h_k_append.data(), h_k_append.size(), "copy prefill k")
        || !copy_to_device(&d_v_append, h_v_append.data(), h_v_append.size(), "copy prefill v")
        || !copy_to_device(&d_batch_indices, batch_indices, 3, "copy prefill batch indices")
        || !copy_to_device(&d_positions, positions, 3, "copy prefill positions")
        || !make_page_table(&d_indptr, &d_indices, &d_last_page_len)) {
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_append_prefill_desc append {};
    append.k = tensor3(d_k_append, num_tokens);
    append.v = tensor3(d_v_append, num_tokens);
    append.batch_indices = d_batch_indices;
    append.positions = d_positions;
    append.kv_cache = cache_desc(d_k_cache, d_v_cache);
    append.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);
    append.num_tokens = num_tokens;

    const qsfi_attention_desc attention = attention_desc();
    check_status(
        qsfi_append_paged_kv_prefill(ctx, &attention, &append),
        QSFI_STATUS_OK,
        "prefill append"
    );
    check_cuda(cudaDeviceSynchronize(), "sync prefill append");
    check_cuda(
        cudaMemcpy(
            h_k_cache.data(),
            d_k_cache,
            cache_elems * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy prefill k cache back"
    );
    check_cuda(
        cudaMemcpy(
            h_v_cache.data(),
            d_v_cache,
            cache_elems * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy prefill v cache back"
    );

    check_cache_item(h_k_cache, h_k_append, 0, 0, 0, "prefill k maps position 0");
    check_cache_item(h_v_cache, h_v_append, 0, 0, 0, "prefill v maps position 0");
    check_cache_item(h_k_cache, h_k_append, 1, 0, 1, "prefill k maps position 4");
    check_cache_item(h_v_cache, h_v_append, 1, 0, 1, "prefill v maps position 4");
    check_cache_item(h_k_cache, h_k_append, 2, 3, 2, "prefill k maps batch 1 position 3");
    check_cache_item(h_v_cache, h_v_append, 2, 3, 2, "prefill v maps batch 1 position 3");
    check_cache_sentinel(h_k_cache, 0, 1, "prefill leaves unrelated entry untouched");
    check_cache_sentinel(h_v_cache, 1, 1, "prefill leaves following entry untouched");

    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_append);
    cudaFree(d_v_append);
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_last_page_len);
    cudaFree(d_batch_indices);
    cudaFree(d_positions);
    qsfi_context_destroy(ctx);
}

void test_gdn_decode_one_hot_recurrence()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 1;
    constexpr size_t qk_elems = static_cast<size_t>(tokens) * kGdnQHeads * kGdnKeyDim;
    constexpr size_t v_elems = static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim;
    constexpr size_t gate_elems = static_cast<size_t>(tokens) * kGdnVHeads;
    constexpr size_t state_elems
        = static_cast<size_t>(kGdnStateSlots) * kGdnVHeads * kGdnValueDim * kGdnKeyDim;

    std::vector<uint16_t> h_q(qk_elems);
    std::vector<uint16_t> h_k(qk_elems);
    std::vector<uint16_t> h_v(v_elems);
    std::vector<uint16_t> h_a(gate_elems);
    std::vector<uint16_t> h_b(gate_elems);
    std::vector<float> h_a_log(kGdnVHeads, 0.0f);
    std::vector<float> h_dt_bias(kGdnVHeads, 0.0f);
    std::vector<uint16_t> h_state(state_elems, kBf16Zero);
    std::vector<uint16_t> h_out(v_elems, kSentinel);
    const int32_t state_indices[] = { 0 };
    fill_gdn_inputs(h_q, h_k, h_v, h_a, h_b, tokens);

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_a = nullptr;
    uint16_t* d_b = nullptr;
    float* d_a_log = nullptr;
    float* d_dt_bias = nullptr;
    uint16_t* d_state = nullptr;
    int32_t* d_state_indices = nullptr;
    uint16_t* d_out = nullptr;

    if (!copy_to_device(&d_q, h_q.data(), h_q.size(), "copy gdn decode q")
        || !copy_to_device(&d_k, h_k.data(), h_k.size(), "copy gdn decode k")
        || !copy_to_device(&d_v, h_v.data(), h_v.size(), "copy gdn decode v")
        || !copy_to_device(&d_a, h_a.data(), h_a.size(), "copy gdn decode a")
        || !copy_to_device(&d_b, h_b.data(), h_b.size(), "copy gdn decode b")
        || !copy_to_device(&d_a_log, h_a_log.data(), h_a_log.size(), "copy gdn decode A_log")
        || !copy_to_device(
            &d_dt_bias,
            h_dt_bias.data(),
            h_dt_bias.size(),
            "copy gdn decode dt_bias"
        )
        || !copy_to_device(&d_state, h_state.data(), h_state.size(), "copy gdn decode state")
        || !copy_to_device(&d_state_indices, state_indices, 1, "copy gdn decode state indices")
        || !copy_to_device(&d_out, h_out.data(), h_out.size(), "copy gdn decode out")) {
        qsfi_context_destroy(ctx);
        return;
    }

    qscu_gdn_decode_desc desc {};
    desc.q = gdn_tensor3_bf16(d_q, tokens, kGdnQHeads, kGdnKeyDim);
    desc.k = gdn_tensor3_bf16(d_k, tokens, kGdnKHeads, kGdnKeyDim);
    desc.v = gdn_tensor3_bf16(d_v, tokens, kGdnVHeads, kGdnValueDim);
    desc.a = gdn_tensor2_bf16(d_a, tokens, kGdnVHeads);
    desc.b = gdn_tensor2_bf16(d_b, tokens, kGdnVHeads);
    desc.a_log = gdn_tensor1_f32(d_a_log, kGdnVHeads);
    desc.dt_bias = gdn_tensor1_f32(d_dt_bias, kGdnVHeads);
    desc.state = gdn_state_tensor_bf16(d_state);
    desc.state_indices = gdn_tensor1_i32(d_state_indices, tokens);
    desc.out = gdn_tensor3_bf16(d_out, tokens, kGdnVHeads, kGdnValueDim);
    desc.num_tokens = tokens;
    desc.num_q_heads = kGdnQHeads;
    desc.num_k_heads = kGdnKHeads;
    desc.num_v_heads = kGdnVHeads;
    desc.key_dim = kGdnKeyDim;
    desc.value_dim = kGdnValueDim;
    desc.state_layout = QSCU_GDN_STATE_LAYOUT_VK;
    desc.scale = 1.0f;

    check_status(qscu_gdn_decode(ctx, &desc), QSFI_STATUS_OK, "gdn decode");
    check_cuda(cudaDeviceSynchronize(), "sync gdn decode");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy gdn decode out back"
    );
    check_cuda(
        cudaMemcpy(
            h_state.data(),
            d_state,
            h_state.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy gdn decode state back"
    );

    check_bf16_single_nonzero(
        h_out,
        gdn_v_offset(0, kGdnActiveVHead, kGdnActiveValueDim),
        1.0f,
        0.0f,
        "gdn decode out"
    );
    check_bf16_single_nonzero(
        h_state,
        gdn_state_offset(0, kGdnActiveVHead, kGdnActiveValueDim, kGdnActiveKeyDim),
        1.0f,
        0.0f,
        "gdn decode state"
    );

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_log);
    cudaFree(d_dt_bias);
    cudaFree(d_state);
    cudaFree(d_state_indices);
    cudaFree(d_out);
    qsfi_context_destroy(ctx);
}

void test_gdn_prefill_two_token_recurrence()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 2;
    constexpr size_t qk_elems = static_cast<size_t>(tokens) * kGdnQHeads * kGdnKeyDim;
    constexpr size_t v_elems = static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim;
    constexpr size_t gate_elems = static_cast<size_t>(tokens) * kGdnVHeads;
    constexpr size_t state_elems
        = static_cast<size_t>(kGdnStateSlots) * kGdnVHeads * kGdnValueDim * kGdnKeyDim;

    std::vector<uint16_t> h_q(qk_elems);
    std::vector<uint16_t> h_k(qk_elems);
    std::vector<uint16_t> h_v(v_elems);
    std::vector<uint16_t> h_a(gate_elems);
    std::vector<uint16_t> h_b(gate_elems);
    std::vector<float> h_a_log(kGdnVHeads, 0.0f);
    std::vector<float> h_dt_bias(kGdnVHeads, 0.0f);
    std::vector<uint16_t> h_state(state_elems, kBf16Zero);
    std::vector<uint16_t> h_out(v_elems, kSentinel);
    const int32_t seq_indptr[] = { 0, tokens };
    const int32_t state_indices[] = { 0 };
    fill_gdn_inputs(h_q, h_k, h_v, h_a, h_b, tokens);

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_a = nullptr;
    uint16_t* d_b = nullptr;
    float* d_a_log = nullptr;
    float* d_dt_bias = nullptr;
    uint16_t* d_state = nullptr;
    int32_t* d_seq_indptr = nullptr;
    int32_t* d_state_indices = nullptr;
    uint16_t* d_out = nullptr;

    if (!copy_to_device(&d_q, h_q.data(), h_q.size(), "copy gdn prefill q")
        || !copy_to_device(&d_k, h_k.data(), h_k.size(), "copy gdn prefill k")
        || !copy_to_device(&d_v, h_v.data(), h_v.size(), "copy gdn prefill v")
        || !copy_to_device(&d_a, h_a.data(), h_a.size(), "copy gdn prefill a")
        || !copy_to_device(&d_b, h_b.data(), h_b.size(), "copy gdn prefill b")
        || !copy_to_device(&d_a_log, h_a_log.data(), h_a_log.size(), "copy gdn prefill A_log")
        || !copy_to_device(
            &d_dt_bias,
            h_dt_bias.data(),
            h_dt_bias.size(),
            "copy gdn prefill dt_bias"
        )
        || !copy_to_device(&d_state, h_state.data(), h_state.size(), "copy gdn prefill state")
        || !copy_to_device(&d_seq_indptr, seq_indptr, 2, "copy gdn prefill seq indptr")
        || !copy_to_device(&d_state_indices, state_indices, 1, "copy gdn prefill state indices")
        || !copy_to_device(&d_out, h_out.data(), h_out.size(), "copy gdn prefill out")) {
        qsfi_context_destroy(ctx);
        return;
    }

    qscu_gdn_prefill_desc desc {};
    desc.q = gdn_tensor3_bf16(d_q, tokens, kGdnQHeads, kGdnKeyDim);
    desc.k = gdn_tensor3_bf16(d_k, tokens, kGdnKHeads, kGdnKeyDim);
    desc.v = gdn_tensor3_bf16(d_v, tokens, kGdnVHeads, kGdnValueDim);
    desc.a = gdn_tensor2_bf16(d_a, tokens, kGdnVHeads);
    desc.b = gdn_tensor2_bf16(d_b, tokens, kGdnVHeads);
    desc.a_log = gdn_tensor1_f32(d_a_log, kGdnVHeads);
    desc.dt_bias = gdn_tensor1_f32(d_dt_bias, kGdnVHeads);
    desc.state = gdn_state_tensor_bf16(d_state);
    desc.seq_indptr = d_seq_indptr;
    desc.state_indices = gdn_tensor1_i32(d_state_indices, 1);
    desc.out = gdn_tensor3_bf16(d_out, tokens, kGdnVHeads, kGdnValueDim);
    desc.batch_size = 1;
    desc.total_tokens = tokens;
    desc.num_q_heads = kGdnQHeads;
    desc.num_k_heads = kGdnKHeads;
    desc.num_v_heads = kGdnVHeads;
    desc.key_dim = kGdnKeyDim;
    desc.value_dim = kGdnValueDim;
    desc.state_layout = QSCU_GDN_STATE_LAYOUT_VK;
    desc.scale = 1.0f;

    check_status(qscu_gdn_prefill(ctx, &desc), QSFI_STATUS_OK, "gdn prefill");
    check_cuda(cudaDeviceSynchronize(), "sync gdn prefill");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy gdn prefill out back"
    );
    check_cuda(
        cudaMemcpy(
            h_state.data(),
            d_state,
            h_state.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy gdn prefill state back"
    );

    check_gdn_prefill_output(h_out, 0.0f);
    check_bf16_single_nonzero(
        h_state,
        gdn_state_offset(0, kGdnActiveVHead, kGdnActiveValueDim, kGdnActiveKeyDim),
        bf16_to_f32(kBf16OnePoint25),
        0.0f,
        "gdn prefill state"
    );

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_log);
    cudaFree(d_dt_bias);
    cudaFree(d_state);
    cudaFree(d_seq_indptr);
    cudaFree(d_state_indices);
    cudaFree(d_out);
    qsfi_context_destroy(ctx);
}

void test_moe_bf16_staged_grouped_gemm()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 2;
    constexpr int hidden = 8;
    constexpr int intermediate = 8;
    constexpr int experts = 2;
    constexpr int top_k = 1;
    constexpr size_t hidden_elems = static_cast<size_t>(tokens) * hidden;
    constexpr size_t gate_up_elems = static_cast<size_t>(experts) * 2 * intermediate * hidden;
    constexpr size_t down_elems = static_cast<size_t>(experts) * hidden * intermediate;

    std::vector<uint16_t> h_hidden(hidden_elems, kBf16Zero);
    std::vector<int32_t> h_topk_ids = { 0, 1 };
    std::vector<float> h_topk_weights = { 1.0f, 1.0f };
    std::vector<uint16_t> h_gate_up(gate_up_elems, kBf16Zero);
    std::vector<uint16_t> h_down(down_elems, kBf16Zero);
    std::vector<uint16_t> h_out(hidden_elems, kSentinel);

    h_hidden[0 * hidden + 0] = kBf16One;
    h_hidden[1 * hidden + 1] = kBf16One;

    h_gate_up[(0 * 2 * intermediate + 0) * hidden + 0] = kBf16One;
    h_gate_up[(0 * 2 * intermediate + intermediate + 0) * hidden + 0] = kBf16Two;
    h_down[(0 * hidden + 0) * intermediate + 0] = kBf16Three;

    h_gate_up[(1 * 2 * intermediate + 0) * hidden + 1] = kBf16Two;
    h_gate_up[(1 * 2 * intermediate + intermediate + 0) * hidden + 1] = kBf16OnePoint5;
    h_down[(1 * hidden + 1) * intermediate + 0] = kBf16OnePoint25;

    uint16_t* d_hidden = nullptr;
    int32_t* d_topk_ids = nullptr;
    float* d_topk_weights = nullptr;
    uint16_t* d_gate_up = nullptr;
    uint16_t* d_down = nullptr;
    uint16_t* d_out = nullptr;
    uint8_t* d_workspace = nullptr;
    qsfi_moe_plan* plan = nullptr;

    if (!copy_to_device(&d_hidden, h_hidden.data(), h_hidden.size(), "copy moe hidden")
        || !copy_to_device(&d_topk_ids, h_topk_ids.data(), h_topk_ids.size(), "copy moe topk ids")
        || !copy_to_device(
            &d_topk_weights,
            h_topk_weights.data(),
            h_topk_weights.size(),
            "copy moe topk weights"
        )
        || !copy_to_device(&d_gate_up, h_gate_up.data(), h_gate_up.size(), "copy moe gate up")
        || !copy_to_device(&d_down, h_down.data(), h_down.size(), "copy moe down")
        || !copy_to_device(&d_out, h_out.data(), h_out.size(), "copy moe out")) {
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_moe_plan_desc plan_desc {};
    plan_desc.backend = QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16;
    plan_desc.route_mode = QSFI_MOE_ROUTE_PRECOMPUTED_TOPK;
    plan_desc.max_num_tokens = tokens;
    plan_desc.hidden_size = hidden;
    plan_desc.intermediate_size = intermediate;
    plan_desc.num_experts = experts;
    plan_desc.top_k = top_k;
    plan_desc.local_num_experts = experts;
    plan_desc.activation_dtype = QSFI_DTYPE_BF16;
    plan_desc.weight_dtype = QSFI_DTYPE_BF16;
    plan_desc.output_dtype = QSFI_DTYPE_BF16;

    if (!check_status(qsfi_moe_plan_create(ctx, &plan_desc, &plan), QSFI_STATUS_OK, "moe plan"))
        return;

    size_t workspace_bytes = 0;
    if (!check_status(
            qsfi_moe_workspace_size(ctx, plan, tokens, &workspace_bytes),
            QSFI_STATUS_OK,
            "moe workspace size"
        )
        || !alloc_device(&d_workspace, workspace_bytes, "alloc moe workspace")) {
        qsfi_moe_plan_destroy(plan);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_moe_bf16_execute_desc desc {};
    desc.hidden = tensor2_bf16(d_hidden, tokens, hidden);
    desc.topk_ids = tensor2_i32(d_topk_ids, tokens, top_k);
    desc.topk_weights = tensor2_f32(d_topk_weights, tokens, top_k);
    desc.gate_up_weight = tensor3_bf16(d_gate_up, experts, 2 * intermediate, hidden);
    desc.down_weight = tensor3_bf16(d_down, experts, hidden, intermediate);
    desc.out = tensor2_bf16(d_out, tokens, hidden);
    desc.workspace = tensor1_u8(d_workspace, static_cast<int64_t>(workspace_bytes));
    desc.num_tokens = tokens;

    check_status(qsfi_moe_execute_bf16(ctx, plan, &desc), QSFI_STATUS_OK, "moe bf16 execute");
    check_cuda(cudaDeviceSynchronize(), "sync moe bf16");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy moe out back"
    );

    const float expected0 = (1.0f / (1.0f + std::exp(-1.0f))) * 2.0f * 3.0f;
    const float expected1 = (2.0f / (1.0f + std::exp(-2.0f))) * 1.5f * 1.25f;
    check_moe_two_outputs(h_out, 0, 0, expected0, 1, 1, expected1, 0.08f, "moe out");

    cudaFree(d_hidden);
    cudaFree(d_topk_ids);
    cudaFree(d_topk_weights);
    cudaFree(d_gate_up);
    cudaFree(d_down);
    cudaFree(d_out);
    cudaFree(d_workspace);
    qsfi_moe_plan_destroy(plan);
    qsfi_context_destroy(ctx);
}

void test_moe_bf16_top2_weighted_accumulation()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 2;
    constexpr int hidden = 8;
    constexpr int intermediate = 8;
    constexpr int experts = 2;
    constexpr int top_k = 2;
    constexpr size_t hidden_elems = static_cast<size_t>(tokens) * hidden;
    constexpr size_t gate_up_elems = static_cast<size_t>(experts) * 2 * intermediate * hidden;
    constexpr size_t down_elems = static_cast<size_t>(experts) * hidden * intermediate;

    std::vector<uint16_t> h_hidden(hidden_elems, kBf16Zero);
    std::vector<int32_t> h_topk_ids = { 0, 0, 0, 1 };
    std::vector<float> h_topk_weights = { 0.25f, 0.75f, 0.5f, 0.25f };
    std::vector<uint16_t> h_gate_up(gate_up_elems, kBf16Zero);
    std::vector<uint16_t> h_down(down_elems, kBf16Zero);
    std::vector<uint16_t> h_out(hidden_elems, kSentinel);
    std::vector<float> h_expected(hidden_elems, 0.0f);

    h_hidden[0 * hidden + 0] = kBf16One;
    h_hidden[1 * hidden + 1] = kBf16One;

    h_gate_up[(0 * 2 * intermediate + 0) * hidden + 0] = kBf16One;
    h_gate_up[(0 * 2 * intermediate + intermediate + 0) * hidden + 0] = kBf16Two;
    h_down[(0 * hidden + 0) * intermediate + 0] = kBf16Three;

    h_gate_up[(0 * 2 * intermediate + 1) * hidden + 1] = kBf16OnePoint5;
    h_gate_up[(0 * 2 * intermediate + intermediate + 1) * hidden + 1] = kBf16OnePoint25;
    h_down[(0 * hidden + 2) * intermediate + 1] = kBf16Two;

    h_gate_up[(1 * 2 * intermediate + 0) * hidden + 1] = kBf16Two;
    h_gate_up[(1 * 2 * intermediate + intermediate + 0) * hidden + 1] = kBf16OnePoint5;
    h_down[(1 * hidden + 3) * intermediate + 0] = kBf16OnePoint25;

    const float expert0_token0 = (1.0f / (1.0f + std::exp(-1.0f))) * 2.0f * 3.0f;
    const float expert0_token1 = (1.5f / (1.0f + std::exp(-1.5f))) * 1.25f * 2.0f;
    const float expert1_token1 = (2.0f / (1.0f + std::exp(-2.0f))) * 1.5f * 1.25f;
    h_expected[0 * hidden + 0] = expert0_token0;
    h_expected[1 * hidden + 2] = 0.5f * expert0_token1;
    h_expected[1 * hidden + 3] = 0.25f * expert1_token1;

    uint16_t* d_hidden = nullptr;
    int32_t* d_topk_ids = nullptr;
    float* d_topk_weights = nullptr;
    uint16_t* d_gate_up = nullptr;
    uint16_t* d_down = nullptr;
    uint16_t* d_out = nullptr;
    uint8_t* d_workspace = nullptr;
    qsfi_moe_plan* plan = nullptr;

    bool ok = copy_to_device(&d_hidden, h_hidden.data(), h_hidden.size(), "copy moe top2 hidden")
        && copy_to_device(&d_topk_ids, h_topk_ids.data(), h_topk_ids.size(), "copy moe top2 ids")
        && copy_to_device(
                  &d_topk_weights,
                  h_topk_weights.data(),
                  h_topk_weights.size(),
                  "copy moe top2 weights"
        )
        && copy_to_device(&d_gate_up, h_gate_up.data(), h_gate_up.size(), "copy moe top2 gate")
        && copy_to_device(&d_down, h_down.data(), h_down.size(), "copy moe top2 down")
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy moe top2 out");

    qsfi_moe_plan_desc plan_desc {};
    plan_desc.backend = QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16;
    plan_desc.route_mode = QSFI_MOE_ROUTE_PRECOMPUTED_TOPK;
    plan_desc.max_num_tokens = tokens;
    plan_desc.hidden_size = hidden;
    plan_desc.intermediate_size = intermediate;
    plan_desc.num_experts = experts;
    plan_desc.top_k = top_k;
    plan_desc.local_num_experts = experts;
    plan_desc.activation_dtype = QSFI_DTYPE_BF16;
    plan_desc.weight_dtype = QSFI_DTYPE_BF16;
    plan_desc.output_dtype = QSFI_DTYPE_BF16;

    if (ok) {
        ok = check_status(
            qsfi_moe_plan_create(ctx, &plan_desc, &plan),
            QSFI_STATUS_OK,
            "moe top2 plan"
        );
    }

    size_t workspace_bytes = 0;
    if (ok) {
        ok = check_status(
                 qsfi_moe_workspace_size(ctx, plan, tokens, &workspace_bytes),
                 QSFI_STATUS_OK,
                 "moe top2 workspace size"
             )
            && alloc_device(&d_workspace, workspace_bytes, "alloc moe top2 workspace");
    }

    if (ok) {
        qsfi_moe_bf16_execute_desc desc {};
        desc.hidden = tensor2_bf16(d_hidden, tokens, hidden);
        desc.topk_ids = tensor2_i32(d_topk_ids, tokens, top_k);
        desc.topk_weights = tensor2_f32(d_topk_weights, tokens, top_k);
        desc.gate_up_weight = tensor3_bf16(d_gate_up, experts, 2 * intermediate, hidden);
        desc.down_weight = tensor3_bf16(d_down, experts, hidden, intermediate);
        desc.out = tensor2_bf16(d_out, tokens, hidden);
        desc.workspace = tensor1_u8(d_workspace, static_cast<int64_t>(workspace_bytes));
        desc.num_tokens = tokens;

        check_status(qsfi_moe_execute_bf16(ctx, plan, &desc), QSFI_STATUS_OK, "moe top2 execute");
        check_cuda(cudaDeviceSynchronize(), "sync moe top2");
        check_cuda(
            cudaMemcpy(
                h_out.data(),
                d_out,
                h_out.size() * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy moe top2 out back"
        );
        check_bf16_vector_close(h_out, h_expected, 0.08f, "moe top2 out");
    }

    cudaFree(d_hidden);
    cudaFree(d_topk_ids);
    cudaFree(d_topk_weights);
    cudaFree(d_gate_up);
    cudaFree(d_down);
    cudaFree(d_out);
    cudaFree(d_workspace);
    qsfi_moe_plan_destroy(plan);
    qsfi_context_destroy(ctx);
}

void test_moe_router_logits_plan_is_unsupported()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    qsfi_moe_plan_desc plan_desc {};
    plan_desc.backend = QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16;
    plan_desc.route_mode = QSFI_MOE_ROUTE_ROUTER_LOGITS;
    plan_desc.max_num_tokens = 1;
    plan_desc.hidden_size = 8;
    plan_desc.intermediate_size = 8;
    plan_desc.num_experts = 2;
    plan_desc.top_k = 1;
    plan_desc.local_num_experts = 2;
    plan_desc.activation_dtype = QSFI_DTYPE_BF16;
    plan_desc.weight_dtype = QSFI_DTYPE_BF16;
    plan_desc.output_dtype = QSFI_DTYPE_BF16;

    qsfi_moe_plan* plan = nullptr;
    check_status_message(
        ctx,
        qsfi_moe_plan_create(ctx, &plan_desc, &plan),
        QSFI_STATUS_UNSUPPORTED,
        "router-logits",
        "moe router logits plan unsupported"
    );
    qsfi_moe_plan_destroy(plan);
    qsfi_context_destroy(ctx);
}

void test_moe_nvfp4_execute_is_declared_unsupported()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    qsfi_moe_plan_desc plan_desc {};
    plan_desc.backend = QSFI_MOE_BACKEND_FLASHINFER_NVFP4;
    plan_desc.route_mode = QSFI_MOE_ROUTE_PRECOMPUTED_TOPK;
    plan_desc.max_num_tokens = 1;
    plan_desc.hidden_size = 16;
    plan_desc.intermediate_size = 16;
    plan_desc.num_experts = 1;
    plan_desc.top_k = 1;
    plan_desc.local_num_experts = 1;
    plan_desc.activation_dtype = QSFI_DTYPE_NVFP4_E2M1;
    plan_desc.weight_dtype = QSFI_DTYPE_NVFP4_E2M1;
    plan_desc.output_dtype = QSFI_DTYPE_BF16;

    qsfi_moe_plan* plan = nullptr;
    if (check_status(
            qsfi_moe_plan_create(ctx, &plan_desc, &plan),
            QSFI_STATUS_OK,
            "moe nvfp4 plan"
        )) {
        qsfi_moe_nvfp4_execute_desc desc {};
        check_status_message(
            ctx,
            qsfi_moe_execute_nvfp4(ctx, plan, &desc),
            QSFI_STATUS_UNSUPPORTED,
            "NVFP4 MoE execution",
            "moe nvfp4 execute unsupported"
        );
    }

    qsfi_moe_plan_destroy(plan);
    qsfi_context_destroy(ctx);
}

#if QSFI_ENABLE_CHECKED_VALIDATION

void check_cache_all_sentinel(const std::vector<uint16_t>& cache, const char* what)
{
    for (size_t i = 0; i < cache.size(); ++i) {
        if (cache[i] != kSentinel) {
            std::fprintf(stderr, "FAIL: %s[%zu]: got 0x%04x\n", what, i, cache[i]);
            failures += 1;
            return;
        }
    }
}

void check_invalid_arg_message(
    qsfi_context* ctx, qsfi_status got, const char* message_fragment, const char* what
)
{
    check_status_message(ctx, got, QSFI_STATUS_INVALID_ARGUMENT, message_fragment, what);
}

void test_checked_append_decode_rejects_invalid_page_id()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr size_t cache_elems = static_cast<size_t>(kNumPages) * kPageSize * kKvHeads * kHeadDim;
    constexpr size_t append_elems = static_cast<size_t>(kBatch) * kKvHeads * kHeadDim;

    std::vector<uint16_t> h_k_append(append_elems);
    std::vector<uint16_t> h_v_append(append_elems);
    std::vector<uint16_t> h_k_cache(cache_elems);
    std::vector<uint16_t> h_v_cache(cache_elems);
    fill_append(h_k_append, h_v_append);

    const int32_t indptr[] = { 0, 2, 3 };
    const int32_t indices[] = { 0, kNumPages, 2 };
    const int32_t last_page_len[] = { 1, 4 };

    uint16_t* d_k_cache = nullptr;
    uint16_t* d_v_cache = nullptr;
    uint16_t* d_k_append = nullptr;
    uint16_t* d_v_append = nullptr;
    int32_t* d_indptr = nullptr;
    int32_t* d_indices = nullptr;
    int32_t* d_last_page_len = nullptr;

    const bool ok = alloc_device(&d_k_cache, cache_elems, "alloc checked invalid page k cache")
        && alloc_device(&d_v_cache, cache_elems, "alloc checked invalid page v cache")
        && check_cuda(cudaMemset(d_k_cache, 0xA5, cache_elems * sizeof(uint16_t)),
                      "clear checked invalid page k cache")
        && check_cuda(cudaMemset(d_v_cache, 0xA5, cache_elems * sizeof(uint16_t)),
                      "clear checked invalid page v cache")
        && copy_to_device(
                        &d_k_append,
                        h_k_append.data(),
                        h_k_append.size(),
                        "copy checked invalid page k"
        )
        && copy_to_device(
                        &d_v_append,
                        h_v_append.data(),
                        h_v_append.size(),
                        "copy checked invalid page v"
        )
        && copy_to_device(&d_indptr, indptr, 3, "copy checked invalid page indptr")
        && copy_to_device(&d_indices, indices, 3, "copy checked invalid page indices")
        && copy_to_device(&d_last_page_len, last_page_len, 2, "copy checked invalid page lengths");

    if (ok) {
        qsfi_append_decode_desc append {};
        append.k = tensor3(d_k_append, kBatch);
        append.v = tensor3(d_v_append, kBatch);
        append.kv_cache = cache_desc(d_k_cache, d_v_cache);
        append.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);

        const qsfi_attention_desc attention = attention_desc();
        check_invalid_arg_message(
            ctx,
            qsfi_append_paged_kv_decode(ctx, &attention, &append),
            "page table indptr/last_page_len/indices",
            "checked append decode rejects invalid page id"
        );
        check_cuda(
            cudaMemcpy(
                h_k_cache.data(),
                d_k_cache,
                cache_elems * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy checked invalid page k cache back"
        );
        check_cuda(
            cudaMemcpy(
                h_v_cache.data(),
                d_v_cache,
                cache_elems * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy checked invalid page v cache back"
        );
        check_cache_all_sentinel(h_k_cache, "checked invalid page leaves k cache untouched");
        check_cache_all_sentinel(h_v_cache, "checked invalid page leaves v cache untouched");
    }

    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_append);
    cudaFree(d_v_append);
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_last_page_len);
    qsfi_context_destroy(ctx);
}

void test_checked_append_prefill_rejects_invalid_position()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int num_tokens = 1;
    constexpr size_t cache_elems = static_cast<size_t>(kNumPages) * kPageSize * kKvHeads * kHeadDim;
    constexpr size_t append_elems = static_cast<size_t>(num_tokens) * kKvHeads * kHeadDim;

    std::vector<uint16_t> h_k_append(append_elems);
    std::vector<uint16_t> h_v_append(append_elems);
    std::vector<uint16_t> h_k_cache(cache_elems);
    std::vector<uint16_t> h_v_cache(cache_elems);
    const int32_t batch_indices[] = { 0 };
    const int32_t positions[] = { 5 };
    fill_append(h_k_append, h_v_append);

    uint16_t* d_k_cache = nullptr;
    uint16_t* d_v_cache = nullptr;
    uint16_t* d_k_append = nullptr;
    uint16_t* d_v_append = nullptr;
    int32_t* d_indptr = nullptr;
    int32_t* d_indices = nullptr;
    int32_t* d_last_page_len = nullptr;
    int32_t* d_batch_indices = nullptr;
    int32_t* d_positions = nullptr;

    const bool ok = alloc_device(&d_k_cache, cache_elems, "alloc checked invalid pos k cache")
        && alloc_device(&d_v_cache, cache_elems, "alloc checked invalid pos v cache")
        && check_cuda(cudaMemset(d_k_cache, 0xA5, cache_elems * sizeof(uint16_t)),
                      "clear checked invalid pos k cache")
        && check_cuda(cudaMemset(d_v_cache, 0xA5, cache_elems * sizeof(uint16_t)),
                      "clear checked invalid pos v cache")
        && copy_to_device(
                        &d_k_append,
                        h_k_append.data(),
                        h_k_append.size(),
                        "copy checked invalid pos k"
        )
        && copy_to_device(
                        &d_v_append,
                        h_v_append.data(),
                        h_v_append.size(),
                        "copy checked invalid pos v"
        )
        && make_page_table(&d_indptr, &d_indices, &d_last_page_len)
        && copy_to_device(&d_batch_indices, batch_indices, 1, "copy checked invalid pos batch")
        && copy_to_device(&d_positions, positions, 1, "copy checked invalid pos positions");

    if (ok) {
        qsfi_append_prefill_desc append {};
        append.k = tensor3(d_k_append, num_tokens);
        append.v = tensor3(d_v_append, num_tokens);
        append.batch_indices = d_batch_indices;
        append.positions = d_positions;
        append.kv_cache = cache_desc(d_k_cache, d_v_cache);
        append.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);
        append.num_tokens = num_tokens;

        const qsfi_attention_desc attention = attention_desc();
        check_invalid_arg_message(
            ctx,
            qsfi_append_paged_kv_prefill(ctx, &attention, &append),
            "append batch_indices/positions",
            "checked append prefill rejects invalid position"
        );
        check_cuda(
            cudaMemcpy(
                h_k_cache.data(),
                d_k_cache,
                cache_elems * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy checked invalid pos k cache back"
        );
        check_cuda(
            cudaMemcpy(
                h_v_cache.data(),
                d_v_cache,
                cache_elems * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ),
            "copy checked invalid pos v cache back"
        );
        check_cache_all_sentinel(h_k_cache, "checked invalid pos leaves k cache untouched");
        check_cache_all_sentinel(h_v_cache, "checked invalid pos leaves v cache untouched");
    }

    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_append);
    cudaFree(d_v_append);
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_last_page_len);
    cudaFree(d_batch_indices);
    cudaFree(d_positions);
    qsfi_context_destroy(ctx);
}

struct checked_gdn_buffers {
    uint16_t* q;
    uint16_t* k;
    uint16_t* v;
    uint16_t* a;
    uint16_t* b;
    float* a_log;
    float* dt_bias;
    uint16_t* state;
    uint16_t* out;
};

void free_checked_gdn_buffers(checked_gdn_buffers& buffers)
{
    cudaFree(buffers.q);
    cudaFree(buffers.k);
    cudaFree(buffers.v);
    cudaFree(buffers.a);
    cudaFree(buffers.b);
    cudaFree(buffers.a_log);
    cudaFree(buffers.dt_bias);
    cudaFree(buffers.state);
    cudaFree(buffers.out);
}

bool make_checked_gdn_buffers(checked_gdn_buffers& buffers, int tokens)
{
    const size_t qk_elems = static_cast<size_t>(tokens) * kGdnQHeads * kGdnKeyDim;
    const size_t v_elems = static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim;
    const size_t gate_elems = static_cast<size_t>(tokens) * kGdnVHeads;
    const size_t state_elems
        = static_cast<size_t>(kGdnStateSlots) * kGdnVHeads * kGdnValueDim * kGdnKeyDim;

    std::vector<uint16_t> h_q(qk_elems);
    std::vector<uint16_t> h_k(qk_elems);
    std::vector<uint16_t> h_v(v_elems);
    std::vector<uint16_t> h_a(gate_elems);
    std::vector<uint16_t> h_b(gate_elems);
    std::vector<float> h_a_log(kGdnVHeads, 0.0f);
    std::vector<float> h_dt_bias(kGdnVHeads, 0.0f);
    std::vector<uint16_t> h_state(state_elems, kBf16Zero);
    std::vector<uint16_t> h_out(v_elems, kSentinel);
    fill_gdn_inputs(h_q, h_k, h_v, h_a, h_b, tokens);

    return copy_to_device(&buffers.q, h_q.data(), h_q.size(), "copy checked gdn q")
        && copy_to_device(&buffers.k, h_k.data(), h_k.size(), "copy checked gdn k")
        && copy_to_device(&buffers.v, h_v.data(), h_v.size(), "copy checked gdn v")
        && copy_to_device(&buffers.a, h_a.data(), h_a.size(), "copy checked gdn a")
        && copy_to_device(&buffers.b, h_b.data(), h_b.size(), "copy checked gdn b")
        && copy_to_device(&buffers.a_log, h_a_log.data(), h_a_log.size(), "copy checked gdn A_log")
        && copy_to_device(
               &buffers.dt_bias,
               h_dt_bias.data(),
               h_dt_bias.size(),
               "copy checked gdn dt_bias"
        )
        && copy_to_device(&buffers.state, h_state.data(), h_state.size(), "copy checked gdn state")
        && copy_to_device(&buffers.out, h_out.data(), h_out.size(), "copy checked gdn out");
}

void test_checked_gdn_decode_rejects_invalid_state_index()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 1;
    const int32_t state_indices[] = { kGdnStateSlots };
    checked_gdn_buffers buffers {};
    int32_t* d_state_indices = nullptr;

    const bool ok = make_checked_gdn_buffers(buffers, tokens)
        && copy_to_device(
                        &d_state_indices,
                        state_indices,
                        1,
                        "copy checked gdn invalid decode state indices"
        );

    if (ok) {
        qscu_gdn_decode_desc desc {};
        desc.q = gdn_tensor3_bf16(buffers.q, tokens, kGdnQHeads, kGdnKeyDim);
        desc.k = gdn_tensor3_bf16(buffers.k, tokens, kGdnKHeads, kGdnKeyDim);
        desc.v = gdn_tensor3_bf16(buffers.v, tokens, kGdnVHeads, kGdnValueDim);
        desc.a = gdn_tensor2_bf16(buffers.a, tokens, kGdnVHeads);
        desc.b = gdn_tensor2_bf16(buffers.b, tokens, kGdnVHeads);
        desc.a_log = gdn_tensor1_f32(buffers.a_log, kGdnVHeads);
        desc.dt_bias = gdn_tensor1_f32(buffers.dt_bias, kGdnVHeads);
        desc.state = gdn_state_tensor_bf16(buffers.state);
        desc.state_indices = gdn_tensor1_i32(d_state_indices, tokens);
        desc.out = gdn_tensor3_bf16(buffers.out, tokens, kGdnVHeads, kGdnValueDim);
        desc.num_tokens = tokens;
        desc.num_q_heads = kGdnQHeads;
        desc.num_k_heads = kGdnKHeads;
        desc.num_v_heads = kGdnVHeads;
        desc.key_dim = kGdnKeyDim;
        desc.value_dim = kGdnValueDim;
        desc.state_layout = QSCU_GDN_STATE_LAYOUT_VK;
        desc.scale = 1.0f;

        check_invalid_arg_message(
            ctx,
            qscu_gdn_decode(ctx, &desc),
            "gdn state indices",
            "checked gdn decode rejects invalid state index"
        );
    }

    cudaFree(d_state_indices);
    free_checked_gdn_buffers(buffers);
    qsfi_context_destroy(ctx);
}

void test_checked_gdn_prefill_rejects_invalid_seq_indptr()
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 2;
    const int32_t seq_indptr[] = { 0, tokens + 1 };
    const int32_t state_indices[] = { 0 };
    checked_gdn_buffers buffers {};
    int32_t* d_seq_indptr = nullptr;
    int32_t* d_state_indices = nullptr;

    const bool ok = make_checked_gdn_buffers(buffers, tokens)
        && copy_to_device(
                        &d_seq_indptr,
                        seq_indptr,
                        2,
                        "copy checked gdn invalid prefill seq indptr"
        )
        && copy_to_device(
                        &d_state_indices,
                        state_indices,
                        1,
                        "copy checked gdn prefill state indices"
        );

    if (ok) {
        qscu_gdn_prefill_desc desc {};
        desc.q = gdn_tensor3_bf16(buffers.q, tokens, kGdnQHeads, kGdnKeyDim);
        desc.k = gdn_tensor3_bf16(buffers.k, tokens, kGdnKHeads, kGdnKeyDim);
        desc.v = gdn_tensor3_bf16(buffers.v, tokens, kGdnVHeads, kGdnValueDim);
        desc.a = gdn_tensor2_bf16(buffers.a, tokens, kGdnVHeads);
        desc.b = gdn_tensor2_bf16(buffers.b, tokens, kGdnVHeads);
        desc.a_log = gdn_tensor1_f32(buffers.a_log, kGdnVHeads);
        desc.dt_bias = gdn_tensor1_f32(buffers.dt_bias, kGdnVHeads);
        desc.state = gdn_state_tensor_bf16(buffers.state);
        desc.seq_indptr = d_seq_indptr;
        desc.state_indices = gdn_tensor1_i32(d_state_indices, 1);
        desc.out = gdn_tensor3_bf16(buffers.out, tokens, kGdnVHeads, kGdnValueDim);
        desc.batch_size = 1;
        desc.total_tokens = tokens;
        desc.num_q_heads = kGdnQHeads;
        desc.num_k_heads = kGdnKHeads;
        desc.num_v_heads = kGdnVHeads;
        desc.key_dim = kGdnKeyDim;
        desc.value_dim = kGdnValueDim;
        desc.state_layout = QSCU_GDN_STATE_LAYOUT_VK;
        desc.scale = 1.0f;

        check_invalid_arg_message(
            ctx,
            qscu_gdn_prefill(ctx, &desc),
            "gdn seq_indptr",
            "checked gdn prefill rejects invalid seq_indptr"
        );
    }

    cudaFree(d_seq_indptr);
    cudaFree(d_state_indices);
    free_checked_gdn_buffers(buffers);
    qsfi_context_destroy(ctx);
}

void run_checked_moe_route_negative(int32_t route_id, float route_weight, const char* what)
{
    qsfi_context* ctx = nullptr;
    if (!make_context(&ctx))
        return;

    constexpr int tokens = 1;
    constexpr int hidden = 8;
    constexpr int intermediate = 8;
    constexpr int experts = 2;
    constexpr int top_k = 1;
    constexpr size_t hidden_elems = static_cast<size_t>(tokens) * hidden;
    constexpr size_t gate_up_elems = static_cast<size_t>(experts) * 2 * intermediate * hidden;
    constexpr size_t down_elems = static_cast<size_t>(experts) * hidden * intermediate;

    std::vector<uint16_t> h_hidden(hidden_elems, kBf16Zero);
    std::vector<int32_t> h_topk_ids = { route_id };
    std::vector<float> h_topk_weights = { route_weight };
    std::vector<uint16_t> h_gate_up(gate_up_elems, kBf16Zero);
    std::vector<uint16_t> h_down(down_elems, kBf16Zero);
    std::vector<uint16_t> h_out(hidden_elems, kSentinel);

    uint16_t* d_hidden = nullptr;
    int32_t* d_topk_ids = nullptr;
    float* d_topk_weights = nullptr;
    uint16_t* d_gate_up = nullptr;
    uint16_t* d_down = nullptr;
    uint16_t* d_out = nullptr;
    uint8_t* d_workspace = nullptr;
    qsfi_moe_plan* plan = nullptr;

    bool ok = copy_to_device(&d_hidden, h_hidden.data(), h_hidden.size(), "copy checked moe hidden")
        && copy_to_device(
                  &d_topk_ids,
                  h_topk_ids.data(),
                  h_topk_ids.size(),
                  "copy checked moe topk ids"
        )
        && copy_to_device(
                  &d_topk_weights,
                  h_topk_weights.data(),
                  h_topk_weights.size(),
                  "copy checked moe topk weights"
        )
        && copy_to_device(&d_gate_up, h_gate_up.data(), h_gate_up.size(), "copy checked moe gate")
        && copy_to_device(&d_down, h_down.data(), h_down.size(), "copy checked moe down")
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy checked moe out");

    qsfi_moe_plan_desc plan_desc {};
    plan_desc.backend = QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16;
    plan_desc.route_mode = QSFI_MOE_ROUTE_PRECOMPUTED_TOPK;
    plan_desc.max_num_tokens = tokens;
    plan_desc.hidden_size = hidden;
    plan_desc.intermediate_size = intermediate;
    plan_desc.num_experts = experts;
    plan_desc.top_k = top_k;
    plan_desc.local_num_experts = experts;
    plan_desc.activation_dtype = QSFI_DTYPE_BF16;
    plan_desc.weight_dtype = QSFI_DTYPE_BF16;
    plan_desc.output_dtype = QSFI_DTYPE_BF16;

    if (ok)
        ok = check_status(qsfi_moe_plan_create(ctx, &plan_desc, &plan), QSFI_STATUS_OK, what);

    size_t workspace_bytes = 0;
    if (ok) {
        ok = check_status(
                 qsfi_moe_workspace_size(ctx, plan, tokens, &workspace_bytes),
                 QSFI_STATUS_OK,
                 what
             )
            && alloc_device(&d_workspace, workspace_bytes, "alloc checked moe workspace");
    }

    if (ok) {
        qsfi_moe_bf16_execute_desc desc {};
        desc.hidden = tensor2_bf16(d_hidden, tokens, hidden);
        desc.topk_ids = tensor2_i32(d_topk_ids, tokens, top_k);
        desc.topk_weights = tensor2_f32(d_topk_weights, tokens, top_k);
        desc.gate_up_weight = tensor3_bf16(d_gate_up, experts, 2 * intermediate, hidden);
        desc.down_weight = tensor3_bf16(d_down, experts, hidden, intermediate);
        desc.out = tensor2_bf16(d_out, tokens, hidden);
        desc.workspace = tensor1_u8(d_workspace, static_cast<int64_t>(workspace_bytes));
        desc.num_tokens = tokens;

        check_invalid_arg_message(ctx, qsfi_moe_execute_bf16(ctx, plan, &desc), "MoE routes", what);
    }

    cudaFree(d_hidden);
    cudaFree(d_topk_ids);
    cudaFree(d_topk_weights);
    cudaFree(d_gate_up);
    cudaFree(d_down);
    cudaFree(d_out);
    cudaFree(d_workspace);
    qsfi_moe_plan_destroy(plan);
    qsfi_context_destroy(ctx);
}

void test_checked_moe_rejects_invalid_route_id()
{
    run_checked_moe_route_negative(
        static_cast<int32_t>(2),
        1.0f,
        "checked moe rejects invalid route id"
    );
}

void test_checked_moe_rejects_non_finite_route_weight()
{
    run_checked_moe_route_negative(
        1,
        std::numeric_limits<float>::quiet_NaN(),
        "checked moe rejects non-finite route weight"
    );
}

#endif

#include "tests_cuda_flashinfer_norm_rope.inc"
#include "tests_cuda_qscb_gemm.inc"
#include "tests_cuda_qscu_gdn_router.inc"
#include "tests_cuda_qscu_utils.inc"

} // namespace

int main()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no CUDA device available: %s\n", cudaGetErrorString(err));
        return 0;
    }
    if (!check_cuda(cudaSetDevice(0), "select device"))
        return 1;

    test_decode_append_uses_post_append_last_page_len();
    test_prefill_append_maps_positions_through_page_table();
    test_batch_decode_attention_matches_cpu_reference();
    test_batch_prefill_attention_matches_cpu_reference();
    test_gdn_decode_one_hot_recurrence();
    test_gdn_prefill_two_token_recurrence();
    test_moe_bf16_staged_grouped_gemm();
    test_moe_bf16_top2_weighted_accumulation();
    test_moe_router_logits_plan_is_unsupported();
    test_moe_nvfp4_execute_is_declared_unsupported();
#if QSFI_ENABLE_CHECKED_VALIDATION
    test_checked_append_decode_rejects_invalid_page_id();
    test_checked_append_prefill_rejects_invalid_position();
    test_checked_gdn_decode_rejects_invalid_state_index();
    test_checked_gdn_prefill_rejects_invalid_seq_indptr();
    test_checked_moe_rejects_invalid_route_id();
    test_checked_moe_rejects_non_finite_route_weight();
#endif
    test_qsfi_rmsnorm_bf16_matches_cpu();
    test_qsfi_fused_add_rmsnorm_bf16_inplace_updates_residual();
    test_qsfi_fused_add_rmsnorm_rejects_non_alias_out();
    test_qsfi_rope_apply_bf16_neox_full_head_matches_cpu();
    test_qscb_context_lifecycle();
    test_qscb_gemm_bf16_hidden_output_strided();
    test_qscb_gemm_bf16_output_beta();
    test_qscb_gemm_bf16_logits_f32_output_beta();
    test_qscu_utils_silu_and_mul_bf16();
    test_qscu_utils_qwen36_shared_expert_gate_add_bf16();
    test_qscu_utils_embedding_gather_bf16();
    test_qscu_utils_logits_soft_cap_f32();
    test_qscu_utils_greedy_argmax_f32();
    test_qscu_utils_negative_validation();
    test_qscu_gdn_router_helpers();

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("qsfi CUDA tests passed");
    return 0;
}
