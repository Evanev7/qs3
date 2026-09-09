#include "qscb.h"
#include "qscu.h"
#include "qsfi.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kFullAttentionQHidden = 4096;
constexpr uint32_t kAttentionQHeads = 16;
constexpr uint32_t kAttentionKvHeads = 2;
constexpr uint32_t kAttentionHeadDim = 256;
constexpr uint32_t kAttentionRotaryDim = 64;
constexpr uint32_t kAttentionPageSize = 4;
constexpr uint32_t kAttentionQHidden = kAttentionQHeads * kAttentionHeadDim;
constexpr uint32_t kAttentionKvHidden = kAttentionKvHeads * kAttentionHeadDim;
constexpr uint32_t kFullAttentionPackedQGateHidden = 2 * kFullAttentionQHidden;
constexpr size_t kAttentionWorkspaceBytes = 64ull << 20;
constexpr uint32_t kMoeExperts = 256;
constexpr uint32_t kMoeTopK = 8;
constexpr uint32_t kMoeIntermediate = 512;
constexpr uint32_t kLogitsSmokeVocab = 16;
constexpr float kLogitsSoftCap = 30.0f;
constexpr uint32_t kGdnQHeads = 16;
constexpr uint32_t kGdnKHeads = 16;
constexpr uint32_t kGdnVHeads = 32;
constexpr uint32_t kGdnKeyDim = 128;
constexpr uint32_t kGdnValueDim = 128;
constexpr uint32_t kGdnConvWidth = 4;
constexpr uint32_t kGdnConvState = kGdnConvWidth - 1;
constexpr uint32_t kGdnPackedDim = 2 * kGdnKHeads * kGdnKeyDim + kGdnVHeads * kGdnValueDim;
constexpr float kGdnRecurrenceScale = 0.08838834764831845f; // 1/sqrt(128).
constexpr size_t kLinearWorkspaceBytes = 64ull << 20;

struct Options {
    int warmups = 5;
    int iters = 20;
    std::vector<uint32_t> tokens { 1, 16, 128 };
};

struct BenchRow {
    const char* name;
    const char* api;
    uint32_t tokens;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t hidden;
    uint32_t heads;
    uint32_t dim;
    uint32_t experts;
    uint32_t top_k;
};

struct BenchState {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    qsfi_context* qsfi = nullptr;
    qscb_context* qscb = nullptr;
};

struct MoePlanGuard {
    qsfi_moe_plan* ptr = nullptr;

    MoePlanGuard() = default;
    MoePlanGuard(const MoePlanGuard&) = delete;
    MoePlanGuard& operator=(const MoePlanGuard&) = delete;

    ~MoePlanGuard()
    {
        if (ptr != nullptr)
            qsfi_moe_plan_destroy(ptr);
    }
};

struct AttentionPageTableHost {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<int32_t> last_page_len;
};

template <typename T> struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (ptr != nullptr)
            cudaFree(ptr);
    }

    bool alloc(size_t n, const char* label)
    {
        count = n;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T));
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc %s failed: %s\n", label, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool zero(cudaStream_t stream, const char* label)
    {
        cudaError_t err = cudaMemsetAsync(ptr, 0, count * sizeof(T), stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMemsetAsync %s failed: %s\n", label, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
};

bool check_cuda(cudaError_t err, const char* what)
{
    if (err == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

const char* status_name(qsfi_status status)
{
    const char* name = qsfi_status_string(status);
    return name == nullptr ? "QSFI_STATUS_UNKNOWN" : name;
}

void report_qsfi_error(qsfi_context* ctx)
{
    qsfi_error_info error {};
    if (ctx != nullptr && qsfi_context_get_last_error(ctx, &error) == QSFI_STATUS_OK
        && error.message[0] != '\0') {
        std::fprintf(stderr, "  qsfi: %s\n", error.message);
    }
}

void report_qscb_error(qscb_context* ctx)
{
    qsfi_error_info error {};
    if (ctx != nullptr && qscb_context_get_last_error(ctx, &error) == QSFI_STATUS_OK
        && error.message[0] != '\0') {
        std::fprintf(stderr, "  qscb: %s\n", error.message);
    }
}

bool parse_uint(const char* text, uint32_t* out, bool allow_zero = false)
{
    if (text == nullptr || text[0] == '\0')
        return false;
    errno = 0;
    char* end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || (!allow_zero && value == 0)
        || value > static_cast<unsigned long>(UINT32_MAX)) {
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_token_list(const char* text, std::vector<uint32_t>* out)
{
    std::vector<uint32_t> parsed;
    const char* begin = text;
    while (begin != nullptr && *begin != '\0') {
        const char* comma = std::strchr(begin, ',');
        std::string item = comma == nullptr ? std::string(begin) : std::string(begin, comma);
        uint32_t value = 0;
        if (!parse_uint(item.c_str(), &value))
            return false;
        parsed.push_back(value);
        begin = comma == nullptr ? nullptr : comma + 1;
    }
    if (parsed.empty())
        return false;
    *out = parsed;
    return true;
}

void print_usage(const char* argv0)
{
    std::fprintf(stderr, "usage: %s [--warmups N] [--iters N] [--tokens A,B,C]\n", argv0);
}

bool parse_options(int argc, char** argv, Options* options)
{
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--warmups") == 0 || std::strcmp(arg, "--iters") == 0
            || std::strcmp(arg, "--tokens") == 0) {
            if (i + 1 >= argc)
                return false;
            const char* value = argv[++i];
            if (std::strcmp(arg, "--tokens") == 0) {
                if (!parse_token_list(value, &options->tokens))
                    return false;
            } else {
                uint32_t parsed = 0;
                const bool allow_zero = std::strcmp(arg, "--warmups") == 0;
                if (!parse_uint(value, &parsed, allow_zero))
                    return false;
                if (std::strcmp(arg, "--warmups") == 0)
                    options->warmups = static_cast<int>(parsed);
                else
                    options->iters = static_cast<int>(parsed);
            }
        } else {
            return false;
        }
    }
    return options->iters > 0 && options->warmups >= 0;
}

qsfi_tensor1 tensor1(void* data, qsfi_dtype dtype, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

qsfi_tensor2 tensor2(void* data, qsfi_dtype dtype, int64_t rows, int64_t cols)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.stride[0] = cols;
    tensor.stride[1] = 1;
    return tensor;
}

qsfi_tensor3 tensor3(void* data, qsfi_dtype dtype, int64_t d0, int64_t d1, int64_t d2)
{
    qsfi_tensor3 tensor {};
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.shape[0] = d0;
    tensor.shape[1] = d1;
    tensor.shape[2] = d2;
    tensor.stride[0] = d1 * d2;
    tensor.stride[1] = d2;
    tensor.stride[2] = 1;
    return tensor;
}

qsfi_tensor4 tensor4(void* data, qsfi_dtype dtype, int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    qsfi_tensor4 tensor {};
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.shape[0] = d0;
    tensor.shape[1] = d1;
    tensor.shape[2] = d2;
    tensor.shape[3] = d3;
    tensor.stride[0] = d1 * d2 * d3;
    tensor.stride[1] = d2 * d3;
    tensor.stride[2] = d3;
    tensor.stride[3] = 1;
    return tensor;
}

qsfi_tensor4 gdn_state_tensor(void* data, qsfi_dtype dtype, int64_t state_slots)
{
    qsfi_tensor4 tensor {};
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.shape[0] = state_slots;
    tensor.shape[1] = kGdnVHeads;
    tensor.shape[2] = kGdnValueDim;
    tensor.shape[3] = kGdnKeyDim;
    tensor.stride[0] = kGdnVHeads * kGdnValueDim * kGdnKeyDim;
    tensor.stride[1] = kGdnValueDim * kGdnKeyDim;
    tensor.stride[2] = kGdnKeyDim;
    tensor.stride[3] = 1;
    return tensor;
}

uint32_t ceil_div(uint32_t n, uint32_t d)
{
    return (n + d - 1) / d;
}

int32_t last_page_len(uint32_t seq_len)
{
    return seq_len == 0 ? 0 : static_cast<int32_t>((seq_len - 1) % kAttentionPageSize + 1);
}

void append_attention_sequence(AttentionPageTableHost* table, uint32_t seq_len)
{
    const uint32_t pages = ceil_div(seq_len, kAttentionPageSize);
    table->indptr.push_back(static_cast<int32_t>(table->indices.size()));
    for (uint32_t i = 0; i < pages; ++i)
        table->indices.push_back(static_cast<int32_t>(table->indices.size()));
    table->last_page_len.push_back(last_page_len(seq_len));
}

AttentionPageTableHost make_single_sequence_attention_table(uint32_t seq_len)
{
    AttentionPageTableHost table {};
    append_attention_sequence(&table, seq_len);
    table.indptr.push_back(static_cast<int32_t>(table.indices.size()));
    return table;
}

AttentionPageTableHost make_decode_attention_table(uint32_t batch_size)
{
    AttentionPageTableHost table {};
    for (uint32_t i = 0; i < batch_size; ++i)
        append_attention_sequence(&table, i + 1);
    table.indptr.push_back(static_cast<int32_t>(table.indices.size()));
    return table;
}

std::vector<int32_t> make_ping_pong_slots(uint32_t count, uint32_t slot)
{
    std::vector<int32_t> indices(count);
    for (uint32_t i = 0; i < count; ++i)
        indices[i] = static_cast<int32_t>(i * 2 + slot);
    return indices;
}

qsfi_attention_desc qwen_attention_desc(qsfi_mask_mode mask_mode = QSFI_MASK_MODE_NONE)
{
    qsfi_attention_desc attention {};
    attention.num_qo_heads = kAttentionQHeads;
    attention.num_kv_heads = kAttentionKvHeads;
    attention.head_dim_qk = kAttentionHeadDim;
    attention.head_dim_vo = kAttentionHeadDim;
    attention.page_size = kAttentionPageSize;
    attention.q_dtype = QSFI_DTYPE_BF16;
    attention.kv_dtype = QSFI_DTYPE_BF16;
    attention.o_dtype = QSFI_DTYPE_BF16;
    attention.kv_layout = QSFI_KV_LAYOUT_NHD;
    attention.mask_mode = mask_mode;
    attention.window_left = -1;
    attention.rope_scale = 1.0f;
    attention.rope_theta = 10000.0f;
    return attention;
}

qsfi_paged_kv_cache attention_cache(void* k, void* v, uint32_t pages)
{
    qsfi_paged_kv_cache cache {};
    cache.k = tensor4(
        k,
        QSFI_DTYPE_BF16,
        pages,
        kAttentionPageSize,
        kAttentionKvHeads,
        kAttentionHeadDim
    );
    cache.v = tensor4(
        v,
        QSFI_DTYPE_BF16,
        pages,
        kAttentionPageSize,
        kAttentionKvHeads,
        kAttentionHeadDim
    );
    return cache;
}

qsfi_paged_kv_plan attention_plan_table(const AttentionPageTableHost& table)
{
    qsfi_paged_kv_plan plan {};
    plan.indptr = table.indptr.data();
    plan.indices = table.indices.data();
    plan.last_page_len = table.last_page_len.data();
    plan.batch_size = static_cast<uint32_t>(table.last_page_len.size());
    plan.num_indices = static_cast<uint32_t>(table.indices.size());
    return plan;
}

qsfi_paged_kv_table attention_exec_table(
    const DeviceBuffer<int32_t>& indptr,
    const DeviceBuffer<int32_t>& indices,
    const DeviceBuffer<int32_t>& last_page_len,
    const AttentionPageTableHost& table
)
{
    qsfi_paged_kv_table exec {};
    exec.indptr = indptr.ptr;
    exec.indices = indices.ptr;
    exec.last_page_len = last_page_len.ptr;
    exec.batch_size = static_cast<uint32_t>(table.last_page_len.size());
    exec.num_indices = static_cast<uint32_t>(table.indices.size());
    return exec;
}

bool copy_i32(
    DeviceBuffer<int32_t>* dst,
    const std::vector<int32_t>& src,
    cudaStream_t stream,
    const char* label
);

bool copy_f32(
    DeviceBuffer<float>* dst, const std::vector<float>& src, cudaStream_t stream, const char* label
);

bool copy_attention_table(
    DeviceBuffer<int32_t>* indptr,
    DeviceBuffer<int32_t>* indices,
    DeviceBuffer<int32_t>* last_page_len,
    const AttentionPageTableHost& table,
    cudaStream_t stream,
    const char* label
)
{
    std::string indptr_label = std::string(label) + " indptr";
    std::string indices_label = std::string(label) + " indices";
    std::string last_page_label = std::string(label) + " last_page_len";
    return copy_i32(indptr, table.indptr, stream, indptr_label.c_str())
        && copy_i32(indices, table.indices, stream, indices_label.c_str())
        && copy_i32(last_page_len, table.last_page_len, stream, last_page_label.c_str());
}

bool copy_i32(
    DeviceBuffer<int32_t>* dst,
    const std::vector<int32_t>& src,
    cudaStream_t stream,
    const char* label
)
{
    if (!dst->alloc(src.size(), label))
        return false;
    cudaError_t err = cudaMemcpyAsync(
        dst->ptr,
        src.data(),
        src.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream
    );
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyAsync %s failed: %s\n", label, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool copy_f32(
    DeviceBuffer<float>* dst, const std::vector<float>& src, cudaStream_t stream, const char* label
)
{
    if (!dst->alloc(src.size(), label))
        return false;
    cudaError_t err = cudaMemcpyAsync(
        dst->ptr,
        src.data(),
        src.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    );
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyAsync %s failed: %s\n", label, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool reserve_attention_workspace(BenchState& state)
{
    qsfi_status status = qsfi_context_reserve_workspace(
        state.qsfi,
        kAttentionWorkspaceBytes,
        kAttentionWorkspaceBytes,
        kAttentionWorkspaceBytes
    );
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "qsfi_context_reserve_workspace failed: %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        report_qsfi_error(state.qsfi);
        return false;
    }
    return true;
}

template <typename Fn, typename ErrorReporter>
bool run_timed(
    BenchState& state,
    const Options& options,
    const BenchRow& row,
    Fn&& fn,
    ErrorReporter&& report_error
)
{
    for (int i = 0; i < options.warmups; ++i) {
        qsfi_status status = fn();
        if (status != QSFI_STATUS_OK) {
            std::fprintf(
                stderr,
                "%s warmup failed with %s (%d)\n",
                row.name,
                status_name(status),
                static_cast<int>(status)
            );
            report_error();
            return false;
        }
    }

    if (!check_cuda(cudaEventRecord(state.start, state.stream), "cudaEventRecord start"))
        return false;
    for (int i = 0; i < options.iters; ++i) {
        qsfi_status status = fn();
        if (status != QSFI_STATUS_OK) {
            std::fprintf(
                stderr,
                "%s iteration failed with %s (%d)\n",
                row.name,
                status_name(status),
                static_cast<int>(status)
            );
            report_error();
            return false;
        }
    }
    if (!check_cuda(cudaEventRecord(state.stop, state.stream), "cudaEventRecord stop"))
        return false;
    if (!check_cuda(cudaEventSynchronize(state.stop), "cudaEventSynchronize stop"))
        return false;

    float elapsed_ms = 0.0f;
    if (!check_cuda(
            cudaEventElapsedTime(&elapsed_ms, state.start, state.stop),
            "cudaEventElapsedTime"
        ))
        return false;
    const double avg_us = static_cast<double>(elapsed_ms) * 1000.0 / options.iters;
    std::printf(
        "%s\t%s\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%d\t%d\t%.3f\n",
        row.name,
        row.api,
        row.tokens,
        row.m,
        row.n,
        row.k,
        row.hidden,
        row.heads,
        row.dim,
        row.experts,
        row.top_k,
        options.warmups,
        options.iters,
        avg_us
    );
    return true;
}

bool sync_setup(cudaStream_t stream)
{
    return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize setup");
}

struct PreparedLinear {
    qscb_linear_plan* plan = nullptr;
    ~PreparedLinear()
    {
        qscb_linear_plan_destroy(plan);
    }
};

bool bench_linear_bf16(
    BenchState& state,
    const Options& options,
    const char* name,
    uint32_t tokens,
    uint32_t n,
    uint32_t k
)
{
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> weight;
    DeviceBuffer<uint16_t> out;
    DeviceBuffer<uint8_t> workspace;
    if (!x.alloc(static_cast<size_t>(tokens) * k, "linear x")
        || !weight.alloc(static_cast<size_t>(n) * k, "linear weight")
        || !out.alloc(static_cast<size_t>(tokens) * n, "linear out")
        || !workspace.alloc(kLinearWorkspaceBytes, "linear workspace")
        || !x.zero(state.stream, "linear x") || !weight.zero(state.stream, "linear weight")
        || !out.zero(state.stream, "linear out") || !sync_setup(state.stream)) {
        return false;
    }

    qscb_linear_desc desc {};
    desc.x = tensor2(x.ptr, QSFI_DTYPE_BF16, tokens, k);
    desc.weight = tensor2(weight.ptr, QSFI_DTYPE_BF16, n, k);
    desc.out = tensor2(out.ptr, QSFI_DTYPE_BF16, tokens, n);
    desc.rows = tokens;
    desc.in_features = k;
    desc.out_features = n;
    desc.workspace = workspace.ptr;
    desc.workspace_bytes = workspace.count;

    PreparedLinear prepared;
    if (qscb_linear_plan_create(state.qscb, &desc, &prepared.plan) != QSFI_STATUS_OK) {
        report_qscb_error(state.qscb);
        return false;
    }
    BenchRow row { name, "qscb_linear_prepared", tokens, tokens, n, k, 0, 0, 0, 0, 0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscb_linear_execute(state.qscb, prepared.plan, &desc); },
        [&]() { report_qscb_error(state.qscb); }
    );
}

bool bench_linear_f32(
    BenchState& state,
    const Options& options,
    const char* name,
    uint32_t tokens,
    uint32_t n,
    uint32_t k,
    uint32_t hidden,
    uint32_t experts
)
{
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> weight;
    DeviceBuffer<float> out;
    DeviceBuffer<uint8_t> workspace;
    if (!x.alloc(static_cast<size_t>(tokens) * k, "linear_f32 x")
        || !weight.alloc(static_cast<size_t>(n) * k, "linear_f32 weight")
        || !out.alloc(static_cast<size_t>(tokens) * n, "linear_f32 out")
        || !workspace.alloc(kLinearWorkspaceBytes, "linear_f32 workspace")
        || !x.zero(state.stream, "linear_f32 x") || !weight.zero(state.stream, "linear_f32 weight")
        || !out.zero(state.stream, "linear_f32 out") || !sync_setup(state.stream)) {
        return false;
    }

    qscb_linear_desc desc {};
    desc.x = tensor2(x.ptr, QSFI_DTYPE_BF16, tokens, k);
    desc.weight = tensor2(weight.ptr, QSFI_DTYPE_BF16, n, k);
    desc.out = tensor2(out.ptr, QSFI_DTYPE_F32, tokens, n);
    desc.rows = tokens;
    desc.in_features = k;
    desc.out_features = n;
    desc.workspace = workspace.ptr;
    desc.workspace_bytes = workspace.count;

    PreparedLinear prepared;
    if (qscb_linear_plan_create(state.qscb, &desc, &prepared.plan) != QSFI_STATUS_OK) {
        report_qscb_error(state.qscb);
        return false;
    }
    BenchRow row { name, "qscb_linear_prepared", tokens, tokens, n, k, hidden, 0, 0, experts, 0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscb_linear_execute(state.qscb, prepared.plan, &desc); },
        [&]() { report_qscb_error(state.qscb); }
    );
}

bool bench_rmsnorm(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> weight;
    DeviceBuffer<uint16_t> out;
    if (!x.alloc(static_cast<size_t>(tokens) * kHidden, "rmsnorm x")
        || !weight.alloc(kHidden, "rmsnorm weight")
        || !out.alloc(static_cast<size_t>(tokens) * kHidden, "rmsnorm out")
        || !x.zero(state.stream, "rmsnorm x") || !weight.zero(state.stream, "rmsnorm weight")
        || !out.zero(state.stream, "rmsnorm out") || !sync_setup(state.stream)) {
        return false;
    }

    qsfi_rmsnorm_desc desc {};
    desc.x = tensor2(x.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.weight = tensor1(weight.ptr, QSFI_DTYPE_BF16, kHidden);
    desc.out = tensor2(out.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.hidden_size = kHidden;
    desc.eps = 1.0e-6f;

    BenchRow row {
        "rmsnorm_hidden2048", "qsfi_rmsnorm", tokens, tokens, 0, 0, kHidden, 0, 0, 0, 0
    };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_rmsnorm(state.qsfi, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_fused_add_rmsnorm(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> residual;
    DeviceBuffer<uint16_t> weight;
    if (!x.alloc(static_cast<size_t>(tokens) * kHidden, "fused_add_rmsnorm x")
        || !residual.alloc(static_cast<size_t>(tokens) * kHidden, "fused_add_rmsnorm residual")
        || !weight.alloc(kHidden, "fused_add_rmsnorm weight")
        || !x.zero(state.stream, "fused_add_rmsnorm x")
        || !residual.zero(state.stream, "fused_add_rmsnorm residual")
        || !weight.zero(state.stream, "fused_add_rmsnorm weight") || !sync_setup(state.stream)) {
        return false;
    }

    qsfi_fused_add_rmsnorm_desc desc {};
    desc.x = tensor2(x.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.residual_inout = tensor2(residual.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.weight = tensor1(weight.ptr, QSFI_DTYPE_BF16, kHidden);
    desc.out = desc.x;
    desc.hidden_size = kHidden;
    desc.eps = 1.0e-6f;

    BenchRow row { "fused_add_rmsnorm_hidden2048",
                   "qsfi_fused_add_rmsnorm",
                   tokens,
                   tokens,
                   0,
                   0,
                   kHidden,
                   0,
                   0,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_fused_add_rmsnorm(state.qsfi, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_full_attention_q_gate_split(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> packed;
    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> gate;
    if (!packed.alloc(
            static_cast<size_t>(tokens) * kFullAttentionPackedQGateHidden,
            "full_attention_q_gate_split packed"
        )
        || !q.alloc(
            static_cast<size_t>(tokens) * kFullAttentionQHidden,
            "full_attention_q_gate_split q"
        )
        || !gate.alloc(
            static_cast<size_t>(tokens) * kFullAttentionQHidden,
            "full_attention_q_gate_split gate"
        )
        || !packed.zero(state.stream, "full_attention_q_gate_split packed")
        || !q.zero(state.stream, "full_attention_q_gate_split q")
        || !gate.zero(state.stream, "full_attention_q_gate_split gate")
        || !sync_setup(state.stream)) {
        return false;
    }

    const size_t head_bytes = static_cast<size_t>(kAttentionHeadDim) * sizeof(uint16_t);
    const size_t packed_head_bytes = 2 * head_bytes;
    const size_t head_rows = static_cast<size_t>(tokens) * kAttentionQHeads;

    BenchRow row { "full_attention_q_gate_split_q8192",
                   "cudaMemcpy2DAsync_x2",
                   tokens,
                   tokens,
                   0,
                   0,
                   kFullAttentionPackedQGateHidden,
                   kAttentionQHeads,
                   kAttentionHeadDim,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() {
            cudaError_t err = cudaMemcpy2DAsync(
                q.ptr,
                head_bytes,
                packed.ptr,
                packed_head_bytes,
                head_bytes,
                head_rows,
                cudaMemcpyDeviceToDevice,
                state.stream
            );
            if (err != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "cudaMemcpy2DAsync full_attention_q_gate_split q failed: %s\n",
                    cudaGetErrorString(err)
                );
                return QSFI_STATUS_CUDA_ERROR;
            }
            err = cudaMemcpy2DAsync(
                gate.ptr,
                head_bytes,
                packed.ptr + kAttentionHeadDim,
                packed_head_bytes,
                head_bytes,
                head_rows,
                cudaMemcpyDeviceToDevice,
                state.stream
            );
            if (err != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "cudaMemcpy2DAsync full_attention_q_gate_split gate failed: %s\n",
                    cudaGetErrorString(err)
                );
                return QSFI_STATUS_CUDA_ERROR;
            }
            return QSFI_STATUS_OK;
        },
        []() {}
    );
}

bool bench_full_attention_head_rmsnorm(
    BenchState& state, const Options& options, const char* name, uint32_t tokens, uint32_t heads
)
{
    const uint32_t rows = tokens * heads;
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> weight;
    if (!x.alloc(static_cast<size_t>(rows) * kAttentionHeadDim, "full_attention_head_rmsnorm x")
        || !weight.alloc(kAttentionHeadDim, "full_attention_head_rmsnorm weight")
        || !x.zero(state.stream, "full_attention_head_rmsnorm x")
        || !weight.zero(state.stream, "full_attention_head_rmsnorm weight")
        || !sync_setup(state.stream)) {
        return false;
    }

    qsfi_rmsnorm_desc desc {};
    desc.x = tensor2(x.ptr, QSFI_DTYPE_BF16, rows, kAttentionHeadDim);
    desc.weight = tensor1(weight.ptr, QSFI_DTYPE_BF16, kAttentionHeadDim);
    desc.out = desc.x;
    desc.hidden_size = kAttentionHeadDim;
    desc.eps = 1.0e-6f;

    BenchRow row { name,  "qsfi_rmsnorm",    tokens, rows, 0, 0, kAttentionHeadDim,
                   heads, kAttentionHeadDim, 0,      0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_rmsnorm(state.qsfi, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_full_attention_rope(BenchState& state, const Options& options, uint32_t tokens)
{
    const size_t q_elems = static_cast<size_t>(tokens) * kAttentionQHeads * kAttentionHeadDim;
    const size_t k_elems = static_cast<size_t>(tokens) * kAttentionKvHeads * kAttentionHeadDim;
    std::vector<int32_t> h_positions(tokens);
    for (uint32_t i = 0; i < tokens; ++i)
        h_positions[i] = static_cast<int32_t>(i);

    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> k;
    DeviceBuffer<int32_t> positions;
    if (!q.alloc(q_elems, "full_attention_rope q") || !k.alloc(k_elems, "full_attention_rope k")
        || !copy_i32(&positions, h_positions, state.stream, "full_attention_rope positions")
        || !q.zero(state.stream, "full_attention_rope q")
        || !k.zero(state.stream, "full_attention_rope k") || !sync_setup(state.stream)) {
        return false;
    }

    qsfi_rope_apply_desc desc {};
    desc.q = tensor3(q.ptr, QSFI_DTYPE_BF16, tokens, kAttentionQHeads, kAttentionHeadDim);
    desc.k = tensor3(k.ptr, QSFI_DTYPE_BF16, tokens, kAttentionKvHeads, kAttentionHeadDim);
    desc.q_out = desc.q;
    desc.k_out = desc.k;
    desc.positions = tensor1(positions.ptr, QSFI_DTYPE_I32, tokens);
    desc.num_qo_heads = kAttentionQHeads;
    desc.num_kv_heads = kAttentionKvHeads;
    desc.head_dim = kAttentionHeadDim;
    desc.rotary_dim = kAttentionRotaryDim;
    desc.rope_scale = 1.0f;
    desc.rope_theta = 10000.0f;
    desc.interleave = 0;

    BenchRow row { "full_attention_qk_rope_rotary64_dim256",
                   "qsfi_rope_apply",
                   tokens,
                   tokens,
                   0,
                   0,
                   kAttentionQHidden + kAttentionKvHidden,
                   kAttentionQHeads + kAttentionKvHeads,
                   kAttentionHeadDim,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_rope_apply(state.qsfi, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_full_attention_gate(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> gate;
    DeviceBuffer<uint16_t> out;
    if (!gate.alloc(static_cast<size_t>(tokens) * kFullAttentionQHidden, "full_attention_gate gate")
        || !out.alloc(
            static_cast<size_t>(tokens) * kFullAttentionQHidden,
            "full_attention_gate out"
        )
        || !gate.zero(state.stream, "full_attention_gate gate")
        || !out.zero(state.stream, "full_attention_gate out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_qwen36_full_attention_output_gate_desc desc {};
    desc.gate = tensor2(gate.ptr, QSFI_DTYPE_BF16, tokens, kFullAttentionQHidden);
    desc.out = tensor2(out.ptr, QSFI_DTYPE_BF16, tokens, kFullAttentionQHidden);
    desc.num_tokens = tokens;
    desc.q_hidden = kFullAttentionQHidden;

    BenchRow row { "full_attention_output_gate_q4096",
                   "qscu_qwen36_full_attention_output_gate_bf16",
                   tokens,
                   tokens,
                   0,
                   0,
                   kFullAttentionQHidden,
                   0,
                   0,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscu_qwen36_full_attention_output_gate_bf16(&desc, state.stream); },
        []() {}
    );
}

bool bench_append_paged_kv_decode(BenchState& state, const Options& options, uint32_t tokens)
{
    AttentionPageTableHost table = make_decode_attention_table(tokens);
    const uint32_t pages = static_cast<uint32_t>(table.indices.size());
    const size_t cache_elems
        = static_cast<size_t>(pages) * kAttentionPageSize * kAttentionKvHeads * kAttentionHeadDim;
    const size_t append_elems = static_cast<size_t>(tokens) * kAttentionKvHeads * kAttentionHeadDim;

    DeviceBuffer<uint16_t> k_cache;
    DeviceBuffer<uint16_t> v_cache;
    DeviceBuffer<uint16_t> k_append;
    DeviceBuffer<uint16_t> v_append;
    DeviceBuffer<int32_t> indptr;
    DeviceBuffer<int32_t> indices;
    DeviceBuffer<int32_t> last_page;
    if (!k_cache.alloc(cache_elems, "attention append decode k cache")
        || !v_cache.alloc(cache_elems, "attention append decode v cache")
        || !k_append.alloc(append_elems, "attention append decode k")
        || !v_append.alloc(append_elems, "attention append decode v")
        || !copy_attention_table(
            &indptr,
            &indices,
            &last_page,
            table,
            state.stream,
            "attention append decode"
        )
        || !k_cache.zero(state.stream, "attention append decode k cache")
        || !v_cache.zero(state.stream, "attention append decode v cache")
        || !k_append.zero(state.stream, "attention append decode k")
        || !v_append.zero(state.stream, "attention append decode v") || !sync_setup(state.stream)) {
        return false;
    }

    const qsfi_attention_desc attention = qwen_attention_desc();
    qsfi_append_decode_desc desc {};
    desc.k = tensor3(k_append.ptr, QSFI_DTYPE_BF16, tokens, kAttentionKvHeads, kAttentionHeadDim);
    desc.v = tensor3(v_append.ptr, QSFI_DTYPE_BF16, tokens, kAttentionKvHeads, kAttentionHeadDim);
    desc.kv_cache = attention_cache(k_cache.ptr, v_cache.ptr, pages);
    desc.page_table = attention_exec_table(indptr, indices, last_page, table);

    BenchRow row { "attention_append_decode_paged_kv_ps4",
                   "qsfi_append_paged_kv_decode",
                   tokens,
                   tokens,
                   pages,
                   kAttentionPageSize,
                   kAttentionKvHidden,
                   kAttentionKvHeads,
                   kAttentionHeadDim,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_append_paged_kv_decode(state.qsfi, &attention, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_append_paged_kv_prefill(BenchState& state, const Options& options, uint32_t tokens)
{
    AttentionPageTableHost table = make_single_sequence_attention_table(tokens);
    const uint32_t pages = static_cast<uint32_t>(table.indices.size());
    const size_t cache_elems
        = static_cast<size_t>(pages) * kAttentionPageSize * kAttentionKvHeads * kAttentionHeadDim;
    const size_t append_elems = static_cast<size_t>(tokens) * kAttentionKvHeads * kAttentionHeadDim;
    std::vector<int32_t> h_batch_indices(tokens, 0);
    std::vector<int32_t> h_positions(tokens);
    for (uint32_t i = 0; i < tokens; ++i)
        h_positions[i] = static_cast<int32_t>(i);

    DeviceBuffer<uint16_t> k_cache;
    DeviceBuffer<uint16_t> v_cache;
    DeviceBuffer<uint16_t> k_append;
    DeviceBuffer<uint16_t> v_append;
    DeviceBuffer<int32_t> indptr;
    DeviceBuffer<int32_t> indices;
    DeviceBuffer<int32_t> last_page;
    DeviceBuffer<int32_t> batch_indices;
    DeviceBuffer<int32_t> positions;
    if (!k_cache.alloc(cache_elems, "attention append prefill k cache")
        || !v_cache.alloc(cache_elems, "attention append prefill v cache")
        || !k_append.alloc(append_elems, "attention append prefill k")
        || !v_append.alloc(append_elems, "attention append prefill v")
        || !copy_attention_table(
            &indptr,
            &indices,
            &last_page,
            table,
            state.stream,
            "attention append prefill"
        )
        || !copy_i32(
            &batch_indices,
            h_batch_indices,
            state.stream,
            "attention append prefill batch"
        )
        || !copy_i32(&positions, h_positions, state.stream, "attention append prefill positions")
        || !k_cache.zero(state.stream, "attention append prefill k cache")
        || !v_cache.zero(state.stream, "attention append prefill v cache")
        || !k_append.zero(state.stream, "attention append prefill k")
        || !v_append.zero(state.stream, "attention append prefill v")
        || !sync_setup(state.stream)) {
        return false;
    }

    const qsfi_attention_desc attention = qwen_attention_desc();
    qsfi_append_prefill_desc desc {};
    desc.k = tensor3(k_append.ptr, QSFI_DTYPE_BF16, tokens, kAttentionKvHeads, kAttentionHeadDim);
    desc.v = tensor3(v_append.ptr, QSFI_DTYPE_BF16, tokens, kAttentionKvHeads, kAttentionHeadDim);
    desc.batch_indices = batch_indices.ptr;
    desc.positions = positions.ptr;
    desc.kv_cache = attention_cache(k_cache.ptr, v_cache.ptr, pages);
    desc.page_table = attention_exec_table(indptr, indices, last_page, table);
    desc.num_tokens = tokens;

    BenchRow row { "attention_append_prefill_paged_kv_ps4",
                   "qsfi_append_paged_kv_prefill",
                   tokens,
                   1,
                   pages,
                   kAttentionPageSize,
                   kAttentionKvHidden,
                   kAttentionKvHeads,
                   kAttentionHeadDim,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_append_paged_kv_prefill(state.qsfi, &attention, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_batch_decode_execute(BenchState& state, const Options& options, uint32_t tokens)
{
    AttentionPageTableHost table = make_decode_attention_table(tokens);
    const uint32_t pages = static_cast<uint32_t>(table.indices.size());
    const size_t cache_elems
        = static_cast<size_t>(pages) * kAttentionPageSize * kAttentionKvHeads * kAttentionHeadDim;
    const size_t q_elems = static_cast<size_t>(tokens) * kAttentionQHeads * kAttentionHeadDim;

    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> k_cache;
    DeviceBuffer<uint16_t> v_cache;
    DeviceBuffer<uint16_t> out;
    DeviceBuffer<int32_t> indptr;
    DeviceBuffer<int32_t> indices;
    DeviceBuffer<int32_t> last_page;
    if (!q.alloc(q_elems, "attention decode q")
        || !k_cache.alloc(cache_elems, "attention decode k cache")
        || !v_cache.alloc(cache_elems, "attention decode v cache")
        || !out.alloc(q_elems, "attention decode out")
        || !copy_attention_table(
            &indptr,
            &indices,
            &last_page,
            table,
            state.stream,
            "attention decode"
        )
        || !q.zero(state.stream, "attention decode q")
        || !k_cache.zero(state.stream, "attention decode k cache")
        || !v_cache.zero(state.stream, "attention decode v cache")
        || !out.zero(state.stream, "attention decode out")) {
        return false;
    }

    const qsfi_attention_desc attention = qwen_attention_desc();
    const qsfi_paged_kv_plan plan_table = attention_plan_table(table);
    qsfi_batch_decode_plan* plan = nullptr;
    qsfi_status status = qsfi_batch_decode_plan_prepare(state.qsfi, &attention, &plan_table, &plan);
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "batch decode plan failed with %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        report_qsfi_error(state.qsfi);
        return false;
    }
    if (!sync_setup(state.stream)) {
        qsfi_batch_decode_plan_destroy(plan);
        return false;
    }

    qsfi_batch_decode_execute_desc desc {};
    desc.q = tensor3(q.ptr, QSFI_DTYPE_BF16, tokens, kAttentionQHeads, kAttentionHeadDim);
    desc.o = tensor3(out.ptr, QSFI_DTYPE_BF16, tokens, kAttentionQHeads, kAttentionHeadDim);
    desc.kv_cache = attention_cache(k_cache.ptr, v_cache.ptr, pages);
    desc.page_table = attention_exec_table(indptr, indices, last_page, table);

    BenchRow row { "attention_batch_decode_execute_hot_plan",
                   "qsfi_batch_decode_execute",
                   tokens,
                   tokens,
                   pages,
                   kAttentionPageSize,
                   kAttentionQHidden,
                   kAttentionQHeads,
                   kAttentionHeadDim,
                   0,
                   0 };
    const bool ok = run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_batch_decode_execute(state.qsfi, plan, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
    qsfi_batch_decode_plan_destroy(plan);
    return ok;
}

bool bench_batch_prefill_execute(BenchState& state, const Options& options, uint32_t tokens)
{
    AttentionPageTableHost table = make_single_sequence_attention_table(tokens);
    const uint32_t pages = static_cast<uint32_t>(table.indices.size());
    const size_t cache_elems
        = static_cast<size_t>(pages) * kAttentionPageSize * kAttentionKvHeads * kAttentionHeadDim;
    const size_t q_elems = static_cast<size_t>(tokens) * kAttentionQHeads * kAttentionHeadDim;
    std::vector<int32_t> h_qo_indptr { 0, static_cast<int32_t>(tokens) };

    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> k_cache;
    DeviceBuffer<uint16_t> v_cache;
    DeviceBuffer<uint16_t> out;
    DeviceBuffer<int32_t> qo_indptr;
    DeviceBuffer<int32_t> indptr;
    DeviceBuffer<int32_t> indices;
    DeviceBuffer<int32_t> last_page;
    if (!q.alloc(q_elems, "attention prefill q")
        || !k_cache.alloc(cache_elems, "attention prefill k cache")
        || !v_cache.alloc(cache_elems, "attention prefill v cache")
        || !out.alloc(q_elems, "attention prefill out")
        || !copy_i32(&qo_indptr, h_qo_indptr, state.stream, "attention prefill qo indptr")
        || !copy_attention_table(
            &indptr,
            &indices,
            &last_page,
            table,
            state.stream,
            "attention prefill"
        )
        || !q.zero(state.stream, "attention prefill q")
        || !k_cache.zero(state.stream, "attention prefill k cache")
        || !v_cache.zero(state.stream, "attention prefill v cache")
        || !out.zero(state.stream, "attention prefill out")) {
        return false;
    }

    const qsfi_attention_desc attention = qwen_attention_desc(QSFI_MASK_MODE_CAUSAL);
    qsfi_qo_plan qo_plan {};
    qo_plan.indptr = h_qo_indptr.data();
    qo_plan.batch_size = 1;
    qo_plan.total_tokens = tokens;
    const qsfi_paged_kv_plan plan_table = attention_plan_table(table);
    qsfi_batch_prefill_plan* plan = nullptr;
    qsfi_status status
        = qsfi_batch_prefill_plan_prepare(state.qsfi, &attention, &qo_plan, &plan_table, &plan);
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "batch prefill plan failed with %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        report_qsfi_error(state.qsfi);
        return false;
    }
    if (!sync_setup(state.stream)) {
        qsfi_batch_prefill_plan_destroy(plan);
        return false;
    }

    qsfi_batch_prefill_execute_desc desc {};
    desc.q = tensor3(q.ptr, QSFI_DTYPE_BF16, tokens, kAttentionQHeads, kAttentionHeadDim);
    desc.o = tensor3(out.ptr, QSFI_DTYPE_BF16, tokens, kAttentionQHeads, kAttentionHeadDim);
    desc.qo_indptr = qo_indptr.ptr;
    desc.kv_cache = attention_cache(k_cache.ptr, v_cache.ptr, pages);
    desc.page_table = attention_exec_table(indptr, indices, last_page, table);

    BenchRow row { "attention_batch_prefill_execute_hot_plan_causal",
                   "qsfi_batch_prefill_execute",
                   tokens,
                   1,
                   pages,
                   kAttentionPageSize,
                   kAttentionQHidden,
                   kAttentionQHeads,
                   kAttentionHeadDim,
                   0,
                   0 };
    const bool ok = run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_batch_prefill_execute(state.qsfi, plan, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
    qsfi_batch_prefill_plan_destroy(plan);
    return ok;
}

bool bench_gdn_causal_conv(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> weight;
    DeviceBuffer<uint16_t> bias;
    DeviceBuffer<uint16_t> conv_state;
    DeviceBuffer<int32_t> seq_indptr;
    DeviceBuffer<int32_t> slot0_indices;
    DeviceBuffer<int32_t> slot1_indices;
    DeviceBuffer<uint16_t> out;
    const std::vector<int32_t> h_seq_indptr { 0, static_cast<int32_t>(tokens) };
    const std::vector<int32_t> slot0 { 0 };
    const std::vector<int32_t> slot1 { 1 };
    if (!x.alloc(static_cast<size_t>(tokens) * kGdnPackedDim, "gdn_conv x")
        || !weight.alloc(static_cast<size_t>(kGdnPackedDim) * kGdnConvWidth, "gdn_conv weight")
        || !bias.alloc(kGdnPackedDim, "gdn_conv bias")
        || !conv_state
                .alloc(static_cast<size_t>(2) * kGdnPackedDim * kGdnConvState, "gdn_conv state")
        || !copy_i32(&seq_indptr, h_seq_indptr, state.stream, "gdn_conv seq_indptr")
        || !copy_i32(&slot0_indices, slot0, state.stream, "gdn_conv state slot0")
        || !copy_i32(&slot1_indices, slot1, state.stream, "gdn_conv state slot1")
        || !out.alloc(static_cast<size_t>(tokens) * kGdnPackedDim, "gdn_conv out")
        || !x.zero(state.stream, "gdn_conv x") || !weight.zero(state.stream, "gdn_conv weight")
        || !bias.zero(state.stream, "gdn_conv bias")
        || !conv_state.zero(state.stream, "gdn_conv state")
        || !out.zero(state.stream, "gdn_conv out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_qwen36_gdn_causal_conv1d_desc live_to_staged {};
    live_to_staged.x = tensor2(x.ptr, QSFI_DTYPE_BF16, tokens, kGdnPackedDim);
    live_to_staged.weight = tensor2(weight.ptr, QSFI_DTYPE_BF16, kGdnPackedDim, kGdnConvWidth);
    live_to_staged.bias = tensor1(bias.ptr, QSFI_DTYPE_BF16, kGdnPackedDim);
    live_to_staged.state
        = tensor3(conv_state.ptr, QSFI_DTYPE_BF16, 2, kGdnPackedDim, kGdnConvState);
    live_to_staged.state_read_indices = tensor1(slot0_indices.ptr, QSFI_DTYPE_I32, 1);
    live_to_staged.state_write_indices = tensor1(slot1_indices.ptr, QSFI_DTYPE_I32, 1);
    live_to_staged.seq_indptr = seq_indptr.ptr;
    live_to_staged.out = tensor2(out.ptr, QSFI_DTYPE_BF16, tokens, kGdnPackedDim);
    live_to_staged.num_tokens = tokens;
    live_to_staged.batch_size = 1;
    live_to_staged.activation = QSCU_ACTIVATION_SILU;
    live_to_staged.update_state = 1;
    qscu_qwen36_gdn_causal_conv1d_desc staged_to_live = live_to_staged;
    staged_to_live.state_read_indices = tensor1(slot1_indices.ptr, QSFI_DTYPE_I32, 1);
    staged_to_live.state_write_indices = tensor1(slot0_indices.ptr, QSFI_DTYPE_I32, 1);

    BenchRow row { "gdn_causal_conv1d_packed8192_live_staged",
                   "qscu_qwen36_gdn_causal_conv1d_bf16",
                   tokens,
                   tokens,
                   kGdnPackedDim,
                   kGdnConvWidth,
                   kGdnPackedDim,
                   0,
                   0,
                   0,
                   0 };
    uint32_t phase = 0;
    return run_timed(
        state,
        options,
        row,
        [&]() {
            qscu_qwen36_gdn_causal_conv1d_desc& active
                = (phase++ & 1u) == 0 ? live_to_staged : staged_to_live;
            return qscu_qwen36_gdn_causal_conv1d_bf16(&active, state.stream);
        },
        []() {}
    );
}

bool bench_gdn_post_conv(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> conv_out;
    DeviceBuffer<uint16_t> a;
    DeviceBuffer<uint16_t> b;
    DeviceBuffer<uint16_t> a_log;
    DeviceBuffer<uint16_t> dt_bias;
    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> k;
    DeviceBuffer<uint16_t> v;
    DeviceBuffer<float> g_out;
    DeviceBuffer<float> beta_out;
    if (!conv_out.alloc(static_cast<size_t>(tokens) * kGdnPackedDim, "gdn_post conv_out")
        || !a.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_post a")
        || !b.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_post b")
        || !a_log.alloc(kGdnVHeads, "gdn_post a_log")
        || !dt_bias.alloc(kGdnVHeads, "gdn_post dt_bias")
        || !q.alloc(static_cast<size_t>(tokens) * kGdnQHeads * kGdnKeyDim, "gdn_post q")
        || !k.alloc(static_cast<size_t>(tokens) * kGdnKHeads * kGdnKeyDim, "gdn_post k")
        || !v.alloc(static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim, "gdn_post v")
        || !g_out.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_post g_out")
        || !beta_out.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_post beta_out")
        || !conv_out.zero(state.stream, "gdn_post conv_out") || !a.zero(state.stream, "gdn_post a")
        || !b.zero(state.stream, "gdn_post b") || !a_log.zero(state.stream, "gdn_post a_log")
        || !dt_bias.zero(state.stream, "gdn_post dt_bias") || !q.zero(state.stream, "gdn_post q")
        || !k.zero(state.stream, "gdn_post k") || !v.zero(state.stream, "gdn_post v")
        || !g_out.zero(state.stream, "gdn_post g_out")
        || !beta_out.zero(state.stream, "gdn_post beta_out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_qwen36_gdn_post_conv_prepare_desc desc {};
    desc.conv_out = tensor2(conv_out.ptr, QSFI_DTYPE_BF16, tokens, kGdnPackedDim);
    desc.a = tensor2(a.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads);
    desc.b = tensor2(b.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads);
    desc.a_log = tensor1(a_log.ptr, QSFI_DTYPE_BF16, kGdnVHeads);
    desc.dt_bias = tensor1(dt_bias.ptr, QSFI_DTYPE_BF16, kGdnVHeads);
    desc.q = tensor3(q.ptr, QSFI_DTYPE_BF16, tokens, kGdnQHeads, kGdnKeyDim);
    desc.k = tensor3(k.ptr, QSFI_DTYPE_BF16, tokens, kGdnKHeads, kGdnKeyDim);
    desc.v = tensor3(v.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    desc.g_out = tensor2(g_out.ptr, QSFI_DTYPE_F32, tokens, kGdnVHeads);
    desc.beta_out = tensor2(beta_out.ptr, QSFI_DTYPE_F32, tokens, kGdnVHeads);
    desc.num_tokens = tokens;
    desc.apply_qk_l2norm = 0;
    desc.l2norm_eps = 1.0e-6f;
    desc.forget_gate_output = QSCU_GDN_FORGET_LOG_DECAY;

    BenchRow row { "gdn_post_conv_prepare_raw_log_decay",
                   "qscu_qwen36_gdn_post_conv_prepare_bf16",
                   tokens,
                   tokens,
                   kGdnPackedDim,
                   0,
                   kGdnPackedDim,
                   kGdnVHeads,
                   kGdnValueDim,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, state.stream); },
        []() {}
    );
}

bool bench_gdn_prefill(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> k;
    DeviceBuffer<uint16_t> v;
    DeviceBuffer<uint16_t> a;
    DeviceBuffer<uint16_t> b;
    DeviceBuffer<uint16_t> a_log;
    DeviceBuffer<uint16_t> dt_bias;
    DeviceBuffer<uint16_t> recurrent_state;
    DeviceBuffer<int32_t> seq_indptr;
    DeviceBuffer<int32_t> slot0_indices;
    DeviceBuffer<int32_t> slot1_indices;
    DeviceBuffer<uint16_t> out;
    const std::vector<int32_t> h_seq_indptr { 0, static_cast<int32_t>(tokens) };
    const std::vector<int32_t> slot0 { 0 };
    const std::vector<int32_t> slot1 { 1 };
    if (!q.alloc(static_cast<size_t>(tokens) * kGdnQHeads * kGdnKeyDim, "gdn_prefill q")
        || !k.alloc(static_cast<size_t>(tokens) * kGdnKHeads * kGdnKeyDim, "gdn_prefill k")
        || !v.alloc(static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim, "gdn_prefill v")
        || !a.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_prefill a")
        || !b.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_prefill b")
        || !a_log.alloc(kGdnVHeads, "gdn_prefill a_log")
        || !dt_bias.alloc(kGdnVHeads, "gdn_prefill dt_bias")
        || !recurrent_state.alloc(
            static_cast<size_t>(2) * kGdnVHeads * kGdnValueDim * kGdnKeyDim,
            "gdn_prefill state"
        )
        || !copy_i32(&seq_indptr, h_seq_indptr, state.stream, "gdn_prefill seq_indptr")
        || !copy_i32(&slot0_indices, slot0, state.stream, "gdn_prefill state slot0")
        || !copy_i32(&slot1_indices, slot1, state.stream, "gdn_prefill state slot1")
        || !out.alloc(static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim, "gdn_prefill out")
        || !q.zero(state.stream, "gdn_prefill q") || !k.zero(state.stream, "gdn_prefill k")
        || !v.zero(state.stream, "gdn_prefill v") || !a.zero(state.stream, "gdn_prefill a")
        || !b.zero(state.stream, "gdn_prefill b") || !a_log.zero(state.stream, "gdn_prefill a_log")
        || !dt_bias.zero(state.stream, "gdn_prefill dt_bias")
        || !recurrent_state.zero(state.stream, "gdn_prefill state")
        || !out.zero(state.stream, "gdn_prefill out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_gdn_prefill_desc live_to_staged {};
    live_to_staged.q = tensor3(q.ptr, QSFI_DTYPE_BF16, tokens, kGdnQHeads, kGdnKeyDim);
    live_to_staged.k = tensor3(k.ptr, QSFI_DTYPE_BF16, tokens, kGdnKHeads, kGdnKeyDim);
    live_to_staged.v = tensor3(v.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    live_to_staged.a = tensor2(a.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads);
    live_to_staged.b = tensor2(b.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads);
    live_to_staged.a_log = tensor1(a_log.ptr, QSFI_DTYPE_BF16, kGdnVHeads);
    live_to_staged.dt_bias = tensor1(dt_bias.ptr, QSFI_DTYPE_BF16, kGdnVHeads);
    live_to_staged.state = gdn_state_tensor(recurrent_state.ptr, QSFI_DTYPE_BF16, 2);
    live_to_staged.seq_indptr = seq_indptr.ptr;
    live_to_staged.state_indices = tensor1(slot0_indices.ptr, QSFI_DTYPE_I32, 1);
    live_to_staged.state_out_indices = tensor1(slot1_indices.ptr, QSFI_DTYPE_I32, 1);
    live_to_staged.out = tensor3(out.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    live_to_staged.batch_size = 1;
    live_to_staged.total_tokens = tokens;
    live_to_staged.num_q_heads = kGdnQHeads;
    live_to_staged.num_k_heads = kGdnKHeads;
    live_to_staged.num_v_heads = kGdnVHeads;
    live_to_staged.key_dim = kGdnKeyDim;
    live_to_staged.value_dim = kGdnValueDim;
    live_to_staged.state_layout = QSCU_GDN_STATE_LAYOUT_VK;
    live_to_staged.scale = kGdnRecurrenceScale;
    live_to_staged.use_qk_l2norm = 1;
    qscu_gdn_prefill_desc staged_to_live = live_to_staged;
    staged_to_live.state_indices = tensor1(slot1_indices.ptr, QSFI_DTYPE_I32, 1);
    staged_to_live.state_out_indices = tensor1(slot0_indices.ptr, QSFI_DTYPE_I32, 1);

    BenchRow row { "gdn_prefill_recurrence_scaled_live_staged",
                   "qscu_gdn_prefill",
                   tokens,
                   tokens,
                   0,
                   0,
                   0,
                   kGdnVHeads,
                   kGdnValueDim,
                   0,
                   0 };
    uint32_t phase = 0;
    return run_timed(
        state,
        options,
        row,
        [&]() {
            qscu_gdn_prefill_desc& active = (phase++ & 1u) == 0 ? live_to_staged : staged_to_live;
            return qscu_gdn_prefill(state.qsfi, &active);
        },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_gdn_decode(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> q;
    DeviceBuffer<uint16_t> k;
    DeviceBuffer<uint16_t> v;
    DeviceBuffer<uint16_t> a;
    DeviceBuffer<uint16_t> b;
    DeviceBuffer<uint16_t> a_log;
    DeviceBuffer<uint16_t> dt_bias;
    DeviceBuffer<uint16_t> recurrent_state;
    DeviceBuffer<int32_t> slot0_indices;
    DeviceBuffer<int32_t> slot1_indices;
    DeviceBuffer<uint16_t> out;
    std::vector<int32_t> h_slot0_indices = make_ping_pong_slots(tokens, 0);
    std::vector<int32_t> h_slot1_indices = make_ping_pong_slots(tokens, 1);

    if (!q.alloc(static_cast<size_t>(tokens) * kGdnQHeads * kGdnKeyDim, "gdn_decode q")
        || !k.alloc(static_cast<size_t>(tokens) * kGdnKHeads * kGdnKeyDim, "gdn_decode k")
        || !v.alloc(static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim, "gdn_decode v")
        || !a.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_decode a")
        || !b.alloc(static_cast<size_t>(tokens) * kGdnVHeads, "gdn_decode b")
        || !a_log.alloc(kGdnVHeads, "gdn_decode a_log")
        || !dt_bias.alloc(kGdnVHeads, "gdn_decode dt_bias")
        || !recurrent_state.alloc(
            static_cast<size_t>(tokens) * 2 * kGdnVHeads * kGdnValueDim * kGdnKeyDim,
            "gdn_decode state"
        )
        || !copy_i32(&slot0_indices, h_slot0_indices, state.stream, "gdn_decode state slot0")
        || !copy_i32(&slot1_indices, h_slot1_indices, state.stream, "gdn_decode state slot1")
        || !out.alloc(static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim, "gdn_decode out")
        || !q.zero(state.stream, "gdn_decode q") || !k.zero(state.stream, "gdn_decode k")
        || !v.zero(state.stream, "gdn_decode v") || !a.zero(state.stream, "gdn_decode a")
        || !b.zero(state.stream, "gdn_decode b") || !a_log.zero(state.stream, "gdn_decode a_log")
        || !dt_bias.zero(state.stream, "gdn_decode dt_bias")
        || !recurrent_state.zero(state.stream, "gdn_decode state")
        || !out.zero(state.stream, "gdn_decode out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_gdn_decode_desc live_to_staged {};
    live_to_staged.q = tensor3(q.ptr, QSFI_DTYPE_BF16, tokens, kGdnQHeads, kGdnKeyDim);
    live_to_staged.k = tensor3(k.ptr, QSFI_DTYPE_BF16, tokens, kGdnKHeads, kGdnKeyDim);
    live_to_staged.v = tensor3(v.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    live_to_staged.a = tensor2(a.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads);
    live_to_staged.b = tensor2(b.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads);
    live_to_staged.a_log = tensor1(a_log.ptr, QSFI_DTYPE_BF16, kGdnVHeads);
    live_to_staged.dt_bias = tensor1(dt_bias.ptr, QSFI_DTYPE_BF16, kGdnVHeads);
    live_to_staged.state
        = gdn_state_tensor(recurrent_state.ptr, QSFI_DTYPE_BF16, static_cast<int64_t>(tokens) * 2);
    live_to_staged.state_indices = tensor1(slot0_indices.ptr, QSFI_DTYPE_I32, tokens);
    live_to_staged.state_out_indices = tensor1(slot1_indices.ptr, QSFI_DTYPE_I32, tokens);
    live_to_staged.out = tensor3(out.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    live_to_staged.num_tokens = tokens;
    live_to_staged.num_q_heads = kGdnQHeads;
    live_to_staged.num_k_heads = kGdnKHeads;
    live_to_staged.num_v_heads = kGdnVHeads;
    live_to_staged.key_dim = kGdnKeyDim;
    live_to_staged.value_dim = kGdnValueDim;
    live_to_staged.state_layout = QSCU_GDN_STATE_LAYOUT_VK;
    live_to_staged.scale = kGdnRecurrenceScale;
    live_to_staged.use_qk_l2norm = 1;
    qscu_gdn_decode_desc staged_to_live = live_to_staged;
    staged_to_live.state_indices = tensor1(slot1_indices.ptr, QSFI_DTYPE_I32, tokens);
    staged_to_live.state_out_indices = tensor1(slot0_indices.ptr, QSFI_DTYPE_I32, tokens);

    BenchRow row { "gdn_decode_recurrence_scaled_live_staged",
                   "qscu_gdn_decode",
                   tokens,
                   tokens,
                   0,
                   0,
                   0,
                   kGdnVHeads,
                   kGdnValueDim,
                   0,
                   0 };
    uint32_t phase = 0;
    return run_timed(
        state,
        options,
        row,
        [&]() {
            qscu_gdn_decode_desc& active = (phase++ & 1u) == 0 ? live_to_staged : staged_to_live;
            return qscu_gdn_decode(state.qsfi, &active);
        },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_gdn_rmsnorm_gated(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<uint16_t> x;
    DeviceBuffer<uint16_t> gate;
    DeviceBuffer<uint16_t> weight;
    DeviceBuffer<uint16_t> out;
    const size_t elems = static_cast<size_t>(tokens) * kGdnVHeads * kGdnValueDim;
    if (!x.alloc(elems, "gdn_rmsnorm_gated x") || !gate.alloc(elems, "gdn_rmsnorm_gated gate")
        || !weight.alloc(kGdnValueDim, "gdn_rmsnorm_gated weight")
        || !out.alloc(elems, "gdn_rmsnorm_gated out")
        || !x.zero(state.stream, "gdn_rmsnorm_gated x")
        || !gate.zero(state.stream, "gdn_rmsnorm_gated gate")
        || !weight.zero(state.stream, "gdn_rmsnorm_gated weight")
        || !out.zero(state.stream, "gdn_rmsnorm_gated out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_qwen36_gdn_rmsnorm_gated_desc desc {};
    desc.x = tensor3(x.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    desc.gate = tensor3(gate.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    desc.weight = tensor1(weight.ptr, QSFI_DTYPE_BF16, kGdnValueDim);
    desc.out = tensor3(out.ptr, QSFI_DTYPE_BF16, tokens, kGdnVHeads, kGdnValueDim);
    desc.num_tokens = tokens;
    desc.eps = 1.0e-6f;
    desc.gate_activation = QSCU_ACTIVATION_SILU;

    BenchRow row { "gdn_rmsnorm_gated",
                   "qscu_qwen36_gdn_rmsnorm_gated_bf16",
                   tokens,
                   tokens,
                   0,
                   0,
                   0,
                   kGdnVHeads,
                   kGdnValueDim,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscu_qwen36_gdn_rmsnorm_gated_bf16(&desc, state.stream); },
        []() {}
    );
}

bool bench_router_topk(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<float> logits;
    DeviceBuffer<int32_t> topk_ids;
    DeviceBuffer<float> topk_weights;
    if (!logits.alloc(static_cast<size_t>(tokens) * kMoeExperts, "router logits")
        || !topk_ids.alloc(static_cast<size_t>(tokens) * kMoeTopK, "router topk ids")
        || !topk_weights.alloc(static_cast<size_t>(tokens) * kMoeTopK, "router topk weights")
        || !logits.zero(state.stream, "router logits")
        || !topk_ids.zero(state.stream, "router topk ids")
        || !topk_weights.zero(state.stream, "router topk weights") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_router_topk_desc desc {};
    desc.logits = tensor2(logits.ptr, QSFI_DTYPE_F32, tokens, kMoeExperts);
    desc.topk_ids = tensor2(topk_ids.ptr, QSFI_DTYPE_I32, tokens, kMoeTopK);
    desc.topk_weights = tensor2(topk_weights.ptr, QSFI_DTYPE_F32, tokens, kMoeTopK);
    desc.num_tokens = tokens;
    desc.num_experts = kMoeExperts;
    desc.top_k = kMoeTopK;
    desc.score = QSCU_ROUTER_SCORE_SOFTMAX;
    desc.renormalize = 1;
    desc.routed_scaling_factor = 1.0f;

    BenchRow row { "router_topk_e256_k8",
                   "qscu_router_topk",
                   tokens,
                   tokens,
                   0,
                   0,
                   0,
                   0,
                   0,
                   kMoeExperts,
                   kMoeTopK };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscu_router_topk(&desc, state.stream); },
        []() {}
    );
}

bool bench_moe_execute_bf16(BenchState& state, const Options& options, uint32_t tokens)
{
    if (tokens > UINT32_MAX / kMoeTopK) {
        std::fprintf(stderr, "moe tokens exceed route count limits\n");
        return false;
    }
    const uint32_t routes = tokens * kMoeTopK;
    std::vector<int32_t> h_topk_ids(routes);
    std::vector<float> h_topk_weights(routes, 1.0f / static_cast<float>(kMoeTopK));
    for (uint32_t route = 0; route < routes; ++route)
        h_topk_ids[route] = static_cast<int32_t>(route % kMoeExperts);

    qsfi_moe_plan_desc plan_desc {};
    plan_desc.backend = QSFI_MOE_BACKEND_FLASHINFER_STAGED_BF16;
    plan_desc.route_mode = QSFI_MOE_ROUTE_PRECOMPUTED_TOPK;
    plan_desc.max_num_tokens = tokens;
    plan_desc.hidden_size = kHidden;
    plan_desc.intermediate_size = kMoeIntermediate;
    plan_desc.num_experts = kMoeExperts;
    plan_desc.top_k = kMoeTopK;
    plan_desc.local_num_experts = kMoeExperts;
    plan_desc.activation_dtype = QSFI_DTYPE_BF16;
    plan_desc.weight_dtype = QSFI_DTYPE_BF16;
    plan_desc.output_dtype = QSFI_DTYPE_BF16;

    MoePlanGuard plan;
    qsfi_status status = qsfi_moe_plan_create(state.qsfi, &plan_desc, &plan.ptr);
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "moe bf16 plan failed with %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        report_qsfi_error(state.qsfi);
        return false;
    }

    size_t workspace_bytes = 0;
    status = qsfi_moe_workspace_size(state.qsfi, plan.ptr, tokens, &workspace_bytes);
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "moe bf16 workspace size failed with %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        report_qsfi_error(state.qsfi);
        return false;
    }

    DeviceBuffer<uint16_t> hidden;
    DeviceBuffer<int32_t> topk_ids;
    DeviceBuffer<float> topk_weights;
    DeviceBuffer<uint16_t> gate_up_weight;
    DeviceBuffer<uint16_t> down_weight;
    DeviceBuffer<uint16_t> out;
    DeviceBuffer<uint8_t> workspace;
    if (!hidden.alloc(static_cast<size_t>(tokens) * kHidden, "moe hidden")
        || !copy_i32(&topk_ids, h_topk_ids, state.stream, "moe topk ids")
        || !copy_f32(&topk_weights, h_topk_weights, state.stream, "moe topk weights")
        || !gate_up_weight.alloc(
            static_cast<size_t>(kMoeExperts) * 2 * kMoeIntermediate * kHidden,
            "moe gate_up_weight"
        )
        || !down_weight.alloc(
            static_cast<size_t>(kMoeExperts) * kHidden * kMoeIntermediate,
            "moe down_weight"
        )
        || !out.alloc(static_cast<size_t>(tokens) * kHidden, "moe out")
        || !workspace.alloc(workspace_bytes, "moe workspace")
        || !hidden.zero(state.stream, "moe hidden")
        || !gate_up_weight.zero(state.stream, "moe gate_up_weight")
        || !down_weight.zero(state.stream, "moe down_weight") || !out.zero(state.stream, "moe out")
        || !sync_setup(state.stream)) {
        return false;
    }

    qsfi_moe_bf16_execute_desc desc {};
    desc.hidden = tensor2(hidden.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.topk_ids = tensor2(topk_ids.ptr, QSFI_DTYPE_I32, tokens, kMoeTopK);
    desc.topk_weights = tensor2(topk_weights.ptr, QSFI_DTYPE_F32, tokens, kMoeTopK);
    desc.gate_up_weight
        = tensor3(gate_up_weight.ptr, QSFI_DTYPE_BF16, kMoeExperts, 2 * kMoeIntermediate, kHidden);
    desc.down_weight
        = tensor3(down_weight.ptr, QSFI_DTYPE_BF16, kMoeExperts, kHidden, kMoeIntermediate);
    desc.out = tensor2(out.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.workspace = tensor1(workspace.ptr, QSFI_DTYPE_U8, static_cast<int64_t>(workspace_bytes));
    desc.num_tokens = tokens;

    BenchRow row { "moe_execute_bf16_h2048_e256_k8_i512_hot_plan",
                   "qsfi_moe_execute_bf16",
                   tokens,
                   routes,
                   2 * kMoeIntermediate,
                   kHidden,
                   kHidden,
                   0,
                   kMoeIntermediate,
                   kMoeExperts,
                   kMoeTopK };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qsfi_moe_execute_bf16(state.qsfi, plan.ptr, &desc); },
        [&]() { report_qsfi_error(state.qsfi); }
    );
}

bool bench_shared_expert_gate_add(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<float> gate_logits;
    DeviceBuffer<uint16_t> shared;
    DeviceBuffer<uint16_t> out;
    if (!gate_logits.alloc(tokens, "shared_gate gate_logits")
        || !shared.alloc(static_cast<size_t>(tokens) * kHidden, "shared_gate shared")
        || !out.alloc(static_cast<size_t>(tokens) * kHidden, "shared_gate out")
        || !gate_logits.zero(state.stream, "shared_gate gate_logits")
        || !shared.zero(state.stream, "shared_gate shared")
        || !out.zero(state.stream, "shared_gate out") || !sync_setup(state.stream)) {
        return false;
    }

    qscu_qwen36_shared_expert_gate_add_desc desc {};
    desc.gate_logits = tensor2(gate_logits.ptr, QSFI_DTYPE_F32, tokens, 1);
    desc.shared = tensor2(shared.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.out = tensor2(out.ptr, QSFI_DTYPE_BF16, tokens, kHidden);
    desc.num_tokens = tokens;
    desc.hidden_size = kHidden;

    BenchRow row { "shared_expert_gate_add_hidden2048",
                   "qscu_qwen36_shared_expert_gate_add_bf16",
                   tokens,
                   tokens,
                   0,
                   0,
                   kHidden,
                   0,
                   0,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscu_qwen36_shared_expert_gate_add_bf16(&desc, state.stream); },
        []() {}
    );
}

bool bench_logits_soft_cap(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<float> logits;
    if (!logits.alloc(static_cast<size_t>(tokens) * kLogitsSmokeVocab, "logits_soft_cap logits")
        || !logits.zero(state.stream, "logits_soft_cap logits") || !sync_setup(state.stream)) {
        return false;
    }

    qsfi_tensor2 logits_tensor = tensor2(logits.ptr, QSFI_DTYPE_F32, tokens, kLogitsSmokeVocab);
    BenchRow row { "logits_soft_cap_f32_small_vocab16",
                   "qscu_logits_soft_cap_f32",
                   tokens,
                   tokens,
                   kLogitsSmokeVocab,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() {
            return qscu_logits_soft_cap_f32(
                &logits_tensor,
                tokens,
                kLogitsSmokeVocab,
                kLogitsSoftCap,
                state.stream
            );
        },
        []() {}
    );
}

bool bench_greedy_argmax(BenchState& state, const Options& options, uint32_t tokens)
{
    DeviceBuffer<float> logits;
    DeviceBuffer<int32_t> next_token_ids;
    if (!logits.alloc(static_cast<size_t>(tokens) * kLogitsSmokeVocab, "argmax logits")
        || !next_token_ids.alloc(tokens, "argmax next_token_ids")
        || !logits.zero(state.stream, "argmax logits")
        || !next_token_ids.zero(state.stream, "argmax next_token_ids")
        || !sync_setup(state.stream)) {
        return false;
    }

    qscu_sampling_desc desc {};
    desc.logits = tensor2(logits.ptr, QSFI_DTYPE_F32, tokens, kLogitsSmokeVocab);
    desc.next_token_ids = tensor1(next_token_ids.ptr, QSFI_DTYPE_I32, tokens);
    desc.batch_size = tokens;
    desc.vocab_size = kLogitsSmokeVocab;
    desc.top_k = 0;
    desc.top_p = 0.0f;
    desc.min_p = 0.0f;
    desc.temperature = 0.0f;

    BenchRow row { "greedy_argmax_f32_small_vocab16",
                   "qscu_greedy_argmax_f32",
                   tokens,
                   tokens,
                   kLogitsSmokeVocab,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0 };
    return run_timed(
        state,
        options,
        row,
        [&]() { return qscu_greedy_argmax_f32(&desc, state.stream); },
        []() {}
    );
}

bool create_state(BenchState* state)
{
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")
        || !check_cuda(cudaStreamCreate(&state->stream), "cudaStreamCreate")
        || !check_cuda(cudaEventCreate(&state->start), "cudaEventCreate start")
        || !check_cuda(cudaEventCreate(&state->stop), "cudaEventCreate stop")) {
        return false;
    }

    qsfi_context_desc qsfi_desc {};
    qsfi_desc.device_ordinal = 0;
    qsfi_desc.stream = state->stream;
    qsfi_status status = qsfi_context_create(&qsfi_desc, &state->qsfi);
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "qsfi_context_create failed: %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        return false;
    }

    qscb_context_desc qscb_desc {};
    qscb_desc.device_ordinal = 0;
    qscb_desc.stream = state->stream;
    status = qscb_context_create(&qscb_desc, &state->qscb);
    if (status != QSFI_STATUS_OK) {
        std::fprintf(
            stderr,
            "qscb_context_create failed: %s (%d)\n",
            status_name(status),
            static_cast<int>(status)
        );
        return false;
    }
    if (!reserve_attention_workspace(*state))
        return false;
    return true;
}

void destroy_state(BenchState* state)
{
    if (state->qscb != nullptr)
        qscb_context_destroy(state->qscb);
    if (state->qsfi != nullptr)
        qsfi_context_destroy(state->qsfi);
    if (state->start != nullptr)
        cudaEventDestroy(state->start);
    if (state->stop != nullptr)
        cudaEventDestroy(state->stop);
    if (state->stream != nullptr)
        cudaStreamDestroy(state->stream);
}

bool run_all(BenchState& state, const Options& options)
{
    std::printf(
        "case\tapi\ttokens\tm\tn\tk\thidden\theads\tdim\texperts\ttop_k\twarmups\titers\tavg_us\n"
    );
    for (uint32_t tokens : options.tokens) {
        if (!bench_linear_bf16(
                state,
                options,
                "linear_q_proj_gate_n8192_k2048",
                tokens,
                8192,
                2048
            )
            || !bench_linear_bf16(
                state,
                options,
                "linear_kv_proj_n512_k2048",
                tokens,
                512,
                2048
            )
            || !bench_linear_bf16(
                state,
                options,
                "linear_o_proj_n2048_k4096",
                tokens,
                2048,
                4096
            )
            || !bench_linear_f32(
                state,
                options,
                "router_logits_linear_f32_n256_k2048",
                tokens,
                kMoeExperts,
                kHidden,
                kHidden,
                kMoeExperts
            )
            || !bench_linear_f32(
                state,
                options,
                "shared_expert_gate_logits_linear_f32_n1_k2048",
                tokens,
                1,
                kHidden,
                kHidden,
                0
            )
            || !bench_linear_f32(
                state,
                options,
                "lm_head_logits_linear_f32_small_vocab16_k2048",
                tokens,
                kLogitsSmokeVocab,
                kHidden,
                kHidden,
                0
            )
            || !bench_rmsnorm(state, options, tokens)
            || !bench_fused_add_rmsnorm(state, options, tokens)
            || !bench_full_attention_q_gate_split(state, options, tokens)
            || !bench_full_attention_head_rmsnorm(
                state,
                options,
                "full_attention_q_head_rmsnorm_h16_d256",
                tokens,
                kAttentionQHeads
            )
            || !bench_full_attention_head_rmsnorm(
                state,
                options,
                "full_attention_k_head_rmsnorm_h2_d256",
                tokens,
                kAttentionKvHeads
            )
            || !bench_full_attention_rope(state, options, tokens)
            || !bench_full_attention_gate(state, options, tokens)
            || !bench_append_paged_kv_decode(state, options, tokens)
            || !bench_append_paged_kv_prefill(state, options, tokens)
            || !bench_batch_decode_execute(state, options, tokens)
            || !bench_batch_prefill_execute(state, options, tokens)
            || !bench_gdn_causal_conv(state, options, tokens)
            || !bench_gdn_post_conv(state, options, tokens)
            || !bench_gdn_prefill(state, options, tokens)
            || !bench_gdn_decode(state, options, tokens)
            || !bench_gdn_rmsnorm_gated(state, options, tokens)
            || !bench_router_topk(state, options, tokens)
            || !bench_moe_execute_bf16(state, options, tokens)
            || !bench_shared_expert_gate_add(state, options, tokens)
            || !bench_logits_soft_cap(state, options, tokens)
            || !bench_greedy_argmax(state, options, tokens)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Options options {};
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return 2;
    }

    BenchState state {};
    if (!create_state(&state)) {
        destroy_state(&state);
        return 1;
    }

    const bool ok = run_all(state, options);
    destroy_state(&state);
    return ok ? 0 : 1;
}
