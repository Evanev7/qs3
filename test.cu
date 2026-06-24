#include "flashinfer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kBatch = 2;
constexpr int kNumPages = 3;
constexpr int kPageSize = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kNumIndices = 3;
constexpr uint16_t kSentinel = 0xA5A5u;

int failures = 0;

bool check_cuda(cudaError_t got, const char* what)
{
    if (got == cudaSuccess)
        return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(got));
    failures += 1;
    return false;
}

bool check_status(qsfi_status_t got, qsfi_status_t want, const char* what)
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

qsfi_tensor_desc_t tensor3(void* data, int64_t n)
{
    qsfi_tensor_desc_t tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F16;
    tensor.ndim = 3;
    tensor.shape[0] = n;
    tensor.shape[1] = kKvHeads;
    tensor.shape[2] = kHeadDim;
    tensor.stride[0] = kKvHeads * kHeadDim;
    tensor.stride[1] = kHeadDim;
    tensor.stride[2] = 1;
    return tensor;
}

qsfi_tensor_desc_t cache_tensor(void* data)
{
    qsfi_tensor_desc_t tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F16;
    tensor.ndim = 4;
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

qsfi_attention_desc_t attention_desc()
{
    qsfi_attention_desc_t attention {};
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

qsfi_paged_kv_cache_t cache_desc(void* k, void* v)
{
    qsfi_paged_kv_cache_t cache {};
    cache.k = cache_tensor(k);
    cache.v = cache_tensor(v);
    return cache;
}

qsfi_paged_kv_table_t page_table_desc(void* indptr, void* indices, void* last_page_len)
{
    qsfi_paged_kv_table_t table {};
    table.indptr = indptr;
    table.indices = indices;
    table.last_page_len = last_page_len;
    table.batch_size = kBatch;
    table.num_indices = kNumIndices;
    return table;
}

bool make_context(qsfi_context_t** out)
{
    qsfi_context_desc_t desc {};
    desc.device_ordinal = 0;
    desc.stream = nullptr;
    return check_status(qsfi_context_create(&desc, out), QSFI_STATUS_OK, "create context");
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

void test_decode_append_uses_post_append_last_page_len()
{
    qsfi_context_t* ctx = nullptr;
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

    qsfi_append_decode_t append {};
    append.k = tensor3(d_k_append, kBatch);
    append.v = tensor3(d_v_append, kBatch);
    append.kv_cache = cache_desc(d_k_cache, d_v_cache);
    append.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);

    const qsfi_attention_desc_t attention = attention_desc();
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
    qsfi_context_t* ctx = nullptr;
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

    qsfi_append_prefill_t append {};
    append.k = tensor3(d_k_append, num_tokens);
    append.v = tensor3(d_v_append, num_tokens);
    append.batch_indices = d_batch_indices;
    append.positions = d_positions;
    append.kv_cache = cache_desc(d_k_cache, d_v_cache);
    append.page_table = page_table_desc(d_indptr, d_indices, d_last_page_len);
    append.num_tokens = num_tokens;

    const qsfi_attention_desc_t attention = attention_desc();
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

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("flashinfer append tests passed");
    return 0;
}
