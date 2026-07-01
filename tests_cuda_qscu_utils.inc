#ifndef TESTS_CUDA_QSCU_UTILS_INC
#define TESTS_CUDA_QSCU_UTILS_INC

#include "qscu.h"

constexpr uint16_t kQscuUtilsBf16Sentinel = 0xA5A5u;

static uint16_t qscu_utils_f32_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static qsfi_tensor1 qscu_utils_tensor1_i32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_I32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

static qsfi_tensor1 qscu_utils_tensor1_u32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_U32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

template <typename T> static void qscu_utils_free_device(T*& ptr)
{
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

static void qscu_utils_check_bf16_close(
    const std::vector<uint16_t>& got,
    const std::vector<float>& want,
    float abs_tol,
    const char* what
)
{
    for (size_t i = 0; i < got.size(); ++i) {
        const float got_f32 = bf16_to_f32(got[i]);
        if (std::fabs(got_f32 - want[i]) > abs_tol) {
            std::fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.8f, want %.8f +/- %.8f\n",
                what,
                i,
                got_f32,
                want[i],
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

static void qscu_utils_check_f32_close(
    const std::vector<float>& got, const std::vector<float>& want, float abs_tol, const char* what
)
{
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > abs_tol) {
            std::fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.8f, want %.8f +/- %.8f\n",
                what,
                i,
                got[i],
                want[i],
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

static void test_qscu_utils_silu_and_mul_bf16()
{
    constexpr int tokens = 2;
    constexpr int intermediate = 3;
    constexpr size_t elems = static_cast<size_t>(tokens) * intermediate;
    const float gate_f32[] = { -2.0f, -0.5f, 0.0f, 1.0f, 2.0f, 4.0f };
    const float up_f32[] = { 3.0f, -1.5f, 2.5f, -2.0f, 0.25f, 1.25f };

    std::vector<uint16_t> h_gate(elems);
    std::vector<uint16_t> h_up(elems);
    std::vector<uint16_t> h_out(elems, kQscuUtilsBf16Sentinel);
    std::vector<float> want(elems);
    for (size_t i = 0; i < elems; ++i) {
        h_gate[i] = qscu_utils_f32_to_bf16(gate_f32[i]);
        h_up[i] = qscu_utils_f32_to_bf16(up_f32[i]);
        const float gate = bf16_to_f32(h_gate[i]);
        const float up = bf16_to_f32(h_up[i]);
        want[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }

    uint16_t* d_gate = nullptr;
    uint16_t* d_up = nullptr;
    uint16_t* d_out = nullptr;
    if (!copy_to_device(&d_gate, h_gate.data(), h_gate.size(), "copy qscu silu gate")
        || !copy_to_device(&d_up, h_up.data(), h_up.size(), "copy qscu silu up")
        || !copy_to_device(&d_out, h_out.data(), h_out.size(), "copy qscu silu out")) {
        return;
    }

    qscu_silu_and_mul_desc desc {};
    desc.gate = tensor2_bf16(d_gate, tokens, intermediate);
    desc.up = tensor2_bf16(d_up, tokens, intermediate);
    desc.out = tensor2_bf16(d_out, tokens, intermediate);
    desc.num_tokens = tokens;
    desc.intermediate_size = intermediate;

    check_status(qscu_silu_and_mul_bf16(&desc, nullptr), QSFI_STATUS_OK, "qscu silu bf16");
    check_cuda(cudaDeviceSynchronize(), "sync qscu silu bf16");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qscu silu out back"
    );
    qscu_utils_check_bf16_close(h_out, want, 0.025f, "qscu silu out");

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
}

static void test_qscu_utils_qwen36_shared_expert_gate_add_bf16()
{
    constexpr int rows = 3;
    constexpr int hidden = 4;
    constexpr size_t elems = static_cast<size_t>(rows) * hidden;
    const float gate_f32[] = { -2.0f, 0.0f, 1.5f };
    const float shared_f32[] = {
        1.0f, -2.0f, 0.25f, 3.0f, -1.5f, 0.5f, 2.0f, -0.75f, 4.0f, -0.25f, 1.25f, 0.0f,
    };
    const float out_f32[] = {
        0.5f, 1.0f, -1.5f, 2.0f, -0.25f, 0.75f, 1.5f, -2.0f, 3.0f, -1.0f, 0.25f, 1.25f,
    };

    std::vector<float> h_gate_f32(gate_f32, gate_f32 + rows);
    std::vector<uint16_t> h_gate_bf16(rows);
    std::vector<uint16_t> h_shared(elems);
    std::vector<uint16_t> h_out_initial(elems);
    std::vector<uint16_t> h_out(elems);
    std::vector<float> want(elems);
    for (int row = 0; row < rows; ++row)
        h_gate_bf16[row] = qscu_utils_f32_to_bf16(gate_f32[row]);
    for (size_t i = 0; i < elems; ++i) {
        h_shared[i] = qscu_utils_f32_to_bf16(shared_f32[i]);
        h_out_initial[i] = qscu_utils_f32_to_bf16(out_f32[i]);
    }

    float* d_gate_f32 = nullptr;
    uint16_t* d_gate_bf16 = nullptr;
    uint16_t* d_shared = nullptr;
    uint16_t* d_out = nullptr;
    if (!copy_to_device(
            &d_gate_f32,
            h_gate_f32.data(),
            h_gate_f32.size(),
            "copy qscu shared f32 gate"
        )
        || !copy_to_device(
            &d_gate_bf16,
            h_gate_bf16.data(),
            h_gate_bf16.size(),
            "copy qscu shared bf16 gate"
        )
        || !copy_to_device(&d_shared, h_shared.data(), h_shared.size(), "copy qscu shared down")
        || !copy_to_device(
            &d_out,
            h_out_initial.data(),
            h_out_initial.size(),
            "copy qscu shared out"
        )) {
        qscu_utils_free_device(d_gate_f32);
        qscu_utils_free_device(d_gate_bf16);
        qscu_utils_free_device(d_shared);
        qscu_utils_free_device(d_out);
        return;
    }

    qscu_qwen36_shared_expert_gate_add_desc desc {};
    desc.gate_logits = tensor2_f32(d_gate_f32, rows, 1);
    desc.shared = tensor2_bf16(d_shared, rows, hidden);
    desc.out = tensor2_bf16(d_out, rows, hidden);
    desc.num_tokens = rows;
    desc.hidden_size = hidden;

    for (int row = 0; row < rows; ++row) {
        const float gate = 1.0f / (1.0f + std::exp(-gate_f32[row]));
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            want[offset]
                = bf16_to_f32(h_out_initial[offset]) + gate * bf16_to_f32(h_shared[offset]);
        }
    }
    check_status(
        qscu_qwen36_shared_expert_gate_add_bf16(&desc, nullptr),
        QSFI_STATUS_OK,
        "qscu qwen36 shared expert f32 gate add"
    );
    check_cuda(cudaDeviceSynchronize(), "sync qscu qwen36 shared expert f32 gate add");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qscu qwen36 shared expert f32 gate out back"
    );
    qscu_utils_check_bf16_close(h_out, want, 0.025f, "qscu shared expert f32 gate out");

    check_cuda(
        cudaMemcpy(
            d_out,
            h_out_initial.data(),
            h_out_initial.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice
        ),
        "reset qscu qwen36 shared expert out"
    );
    desc.gate_logits = tensor2_bf16(d_gate_bf16, rows, 1);
    for (int row = 0; row < rows; ++row) {
        const float gate_logit = bf16_to_f32(h_gate_bf16[row]);
        const float gate = 1.0f / (1.0f + std::exp(-gate_logit));
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            want[offset]
                = bf16_to_f32(h_out_initial[offset]) + gate * bf16_to_f32(h_shared[offset]);
        }
    }
    check_status(
        qscu_qwen36_shared_expert_gate_add_bf16(&desc, nullptr),
        QSFI_STATUS_OK,
        "qscu qwen36 shared expert bf16 gate add"
    );
    check_cuda(cudaDeviceSynchronize(), "sync qscu qwen36 shared expert bf16 gate add");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qscu qwen36 shared expert bf16 gate out back"
    );
    qscu_utils_check_bf16_close(h_out, want, 0.025f, "qscu shared expert bf16 gate out");

    qscu_utils_free_device(d_gate_f32);
    qscu_utils_free_device(d_gate_bf16);
    qscu_utils_free_device(d_shared);
    qscu_utils_free_device(d_out);
}

static void test_qscu_utils_qwen36_full_attention_output_gate_bf16()
{
    constexpr int tokens = 2;
    constexpr int q_hidden = 4096;
    constexpr size_t elems = static_cast<size_t>(tokens) * q_hidden;

    std::vector<uint16_t> h_gate(elems);
    std::vector<uint16_t> h_out_initial(elems);
    std::vector<uint16_t> h_out(elems);
    std::vector<float> want(elems);
    // q_proj extraction, q_norm, and RoPE are upstream; this helper sees only
    // the already-extracted contiguous full-attention gate tensor.
    for (size_t i = 0; i < elems; ++i) {
        const int pattern = static_cast<int>(i % 8);
        const float gate_f32 = pattern == 0 ? -4.0f
            : pattern == 1                  ? -1.0f
            : pattern == 2                  ? -0.25f
            : pattern == 3                  ? 0.0f
            : pattern == 4                  ? 0.5f
            : pattern == 5                  ? 1.0f
            : pattern == 6                  ? 2.5f
                                            : 4.0f;
        const float out_f32 = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
        h_gate[i] = qscu_utils_f32_to_bf16(gate_f32);
        h_out_initial[i] = qscu_utils_f32_to_bf16(out_f32);
        const float gate = bf16_to_f32(h_gate[i]);
        const float out = bf16_to_f32(h_out_initial[i]);
        want[i] = out * (1.0f / (1.0f + std::exp(-gate)));
    }

    uint16_t* d_gate = nullptr;
    uint16_t* d_out = nullptr;
    if (!copy_to_device(&d_gate, h_gate.data(), h_gate.size(), "copy qscu full-attn gate")
        || !copy_to_device(
            &d_out,
            h_out_initial.data(),
            h_out_initial.size(),
            "copy qscu full-attn gated out"
        )) {
        qscu_utils_free_device(d_gate);
        qscu_utils_free_device(d_out);
        return;
    }

    qscu_qwen36_full_attention_output_gate_desc desc {};
    desc.gate = tensor2_bf16(d_gate, tokens, q_hidden);
    desc.out = tensor2_bf16(d_out, tokens, q_hidden);
    desc.num_tokens = tokens;
    desc.q_hidden = q_hidden;

    check_status(
        qscu_qwen36_full_attention_output_gate_bf16(&desc, nullptr),
        QSFI_STATUS_OK,
        "qscu qwen36 extracted full-attn output gate"
    );
    check_cuda(cudaDeviceSynchronize(), "sync qscu qwen36 extracted full-attn output gate");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qscu qwen36 extracted full-attn output gate out back"
    );
    qscu_utils_check_bf16_close(h_out, want, 0.025f, "qscu extracted full-attn gate out");

    qscu_utils_free_device(d_gate);
    qscu_utils_free_device(d_out);
}

static void test_qscu_utils_embedding_gather_bf16()
{
    constexpr int vocab = 4;
    constexpr int hidden = 5;
    constexpr int tokens = 4;
    constexpr int32_t padding_id = 7;
    constexpr size_t embedding_elems = static_cast<size_t>(vocab) * hidden;
    constexpr size_t out_elems = static_cast<size_t>(tokens) * hidden;

    const int32_t token_ids[] = { 2, padding_id, 0, 3 };
    std::vector<uint16_t> h_embedding(embedding_elems);
    std::vector<uint16_t> h_out(out_elems, kQscuUtilsBf16Sentinel);
    for (int row = 0; row < vocab; ++row) {
        for (int col = 0; col < hidden; ++col) {
            h_embedding[static_cast<size_t>(row) * hidden + col]
                = qscu_utils_f32_to_bf16(static_cast<float>(row * 10 + col) + 0.25f);
        }
    }

    int32_t* d_token_ids = nullptr;
    uint16_t* d_embedding = nullptr;
    uint16_t* d_out = nullptr;
    if (!copy_to_device(&d_token_ids, token_ids, tokens, "copy qscu embedding token ids")
        || !copy_to_device(
            &d_embedding,
            h_embedding.data(),
            h_embedding.size(),
            "copy qscu embedding table"
        )
        || !copy_to_device(&d_out, h_out.data(), h_out.size(), "copy qscu embedding out")) {
        qscu_utils_free_device(d_token_ids);
        qscu_utils_free_device(d_embedding);
        qscu_utils_free_device(d_out);
        return;
    }

    qscu_embedding_gather_desc desc {};
    desc.token_ids = qscu_utils_tensor1_i32(d_token_ids, tokens);
    desc.embedding = tensor2_bf16(d_embedding, vocab, hidden);
    desc.out = tensor2_bf16(d_out, tokens, hidden);
    desc.padding_token_id = padding_id;
    desc.validate_token_ids = 1;

    check_status(
        qscu_embedding_gather_bf16(&desc, nullptr),
        QSFI_STATUS_OK,
        "qscu embedding gather bf16"
    );
    check_cuda(cudaDeviceSynchronize(), "sync qscu embedding gather bf16");
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qscu embedding out back"
    );

    for (int token = 0; token < tokens; ++token) {
        for (int col = 0; col < hidden; ++col) {
            const size_t out_offset = static_cast<size_t>(token) * hidden + col;
            const uint16_t want = token_ids[token] == padding_id
                ? kBf16Zero
                : h_embedding[static_cast<size_t>(token_ids[token]) * hidden + col];
            if (h_out[out_offset] != want) {
                std::fprintf(
                    stderr,
                    "FAIL: qscu embedding out[%zu]: got 0x%04x, want 0x%04x\n",
                    out_offset,
                    h_out[out_offset],
                    want
                );
                failures += 1;
                token = tokens;
                break;
            }
        }
    }

#if QSFI_ENABLE_CHECKED_VALIDATION
    const int32_t invalid_token_ids[] = { 2, padding_id, vocab, 3 };
    if (check_cuda(
            cudaMemcpy(
                d_token_ids,
                invalid_token_ids,
                sizeof(invalid_token_ids),
                cudaMemcpyHostToDevice
            ),
            "copy qscu embedding invalid token ids"
        )) {
        check_status(
            qscu_embedding_gather_bf16(&desc, nullptr),
            QSFI_STATUS_INVALID_ARGUMENT,
            "qscu embedding rejects invalid token id"
        );
    }
#endif

    qscu_utils_free_device(d_token_ids);
    qscu_utils_free_device(d_embedding);
    qscu_utils_free_device(d_out);
}

static void test_qscu_utils_logits_soft_cap_f32()
{
    constexpr int rows = 2;
    constexpr int vocab = 5;
    constexpr float soft_cap = 2.0f;
    std::vector<float> h_logits
        = { -8.0f, -2.0f, 0.0f, 2.0f, 8.0f, 0.5f, -0.5f, 10.0f, -10.0f, 1.0f };
    std::vector<float> want(h_logits.size());
    for (size_t i = 0; i < h_logits.size(); ++i)
        want[i] = soft_cap * std::tanh(h_logits[i] / soft_cap);

    float* d_logits = nullptr;
    if (!copy_to_device(&d_logits, h_logits.data(), h_logits.size(), "copy qscu soft cap logits"))
        return;

    qsfi_tensor2 logits = tensor2_f32(d_logits, rows, vocab);
    check_status(
        qscu_logits_soft_cap_f32(&logits, rows, vocab, soft_cap, nullptr),
        QSFI_STATUS_OK,
        "qscu logits soft cap"
    );
    check_cuda(cudaDeviceSynchronize(), "sync qscu logits soft cap");
    check_cuda(
        cudaMemcpy(
            h_logits.data(),
            d_logits,
            h_logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy qscu soft cap logits back"
    );
    qscu_utils_check_f32_close(h_logits, want, 0.00001f, "qscu soft cap logits");

    cudaFree(d_logits);
}

static void test_qscu_utils_greedy_argmax_f32()
{
    constexpr int batch = 4;
    constexpr int vocab = 6;
    std::vector<float> h_logits = {
        0.1f,  0.2f,  3.0f,  2.9f,  -1.0f, 0.0f,  -1.0f,  4.5f,   0.0f,  4.5f,   2.0f,  4.5f,
        -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, -0.5f, -0.25f, -0.25f, -0.5f, -0.25f, -1.0f, -0.25f,
    };
    const int32_t want[] = { 2, 1, 5, 0 };
    std::vector<int32_t> h_next(batch, -1);

    float* d_logits = nullptr;
    int32_t* d_next = nullptr;
    if (!copy_to_device(&d_logits, h_logits.data(), h_logits.size(), "copy qscu argmax logits")
        || !copy_to_device(&d_next, h_next.data(), h_next.size(), "copy qscu argmax out")) {
        return;
    }

    qscu_sampling_desc desc {};
    desc.logits = tensor2_f32(d_logits, batch, vocab);
    desc.next_token_ids = qscu_utils_tensor1_i32(d_next, batch);
    desc.batch_size = batch;
    desc.vocab_size = vocab;
    desc.top_k = 0;
    desc.top_p = 0.0f;
    desc.min_p = 0.0f;
    desc.temperature = 0.0f;

    check_status(qscu_greedy_argmax_f32(&desc, nullptr), QSFI_STATUS_OK, "qscu greedy argmax");
    check_cuda(cudaDeviceSynchronize(), "sync qscu greedy argmax");
    check_cuda(
        cudaMemcpy(h_next.data(), d_next, h_next.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "copy qscu argmax out back"
    );

    for (int row = 0; row < batch; ++row) {
        if (h_next[row] != want[row]) {
            std::fprintf(
                stderr,
                "FAIL: qscu greedy argmax[%d]: got %d, want %d\n",
                row,
                h_next[row],
                want[row]
            );
            failures += 1;
            break;
        }
    }

    cudaFree(d_logits);
    cudaFree(d_next);
}

static void test_qscu_utils_greedy_argmax_nonfinite()
{
    constexpr int batch = 1;
    constexpr int vocab = 4;
    const float h_logits[] = {
        std::numeric_limits<float>::quiet_NaN(),
        1.0f,
        std::numeric_limits<float>::infinity(),
        -2.0f,
    };
    int32_t h_next[] = { -1 };

    float* d_logits = nullptr;
    int32_t* d_next = nullptr;
    if (!copy_to_device(&d_logits, h_logits, vocab, "copy qscu argmax nonfinite logits")
        || !copy_to_device(&d_next, h_next, batch, "copy qscu argmax nonfinite out")) {
        qscu_utils_free_device(d_logits);
        qscu_utils_free_device(d_next);
        return;
    }

    qscu_sampling_desc desc {};
    desc.logits = tensor2_f32(d_logits, batch, vocab);
    desc.next_token_ids = qscu_utils_tensor1_i32(d_next, batch);
    desc.batch_size = batch;
    desc.vocab_size = vocab;
    desc.top_k = 0;
    desc.top_p = 0.0f;
    desc.min_p = 0.0f;
    desc.temperature = 0.0f;

#if QSFI_ENABLE_CHECKED_VALIDATION
    check_status(
        qscu_greedy_argmax_f32(&desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "checked qscu greedy rejects non-finite logits"
    );
#else
    if (check_status(
            qscu_greedy_argmax_f32(&desc, nullptr),
            QSFI_STATUS_OK,
            "release qscu greedy allows non-finite logits"
        )) {
        check_cuda(cudaDeviceSynchronize(), "sync qscu argmax nonfinite");
        if (check_cuda(
                cudaMemcpy(h_next, d_next, sizeof(h_next), cudaMemcpyDeviceToHost),
                "copy qscu argmax nonfinite out"
            )
            && h_next[0] != 2) {
            std::fprintf(
                stderr,
                "FAIL: qscu release greedy nonfinite result: got %d, want 2\n",
                h_next[0]
            );
            failures += 1;
        }
    }
#endif

    qscu_utils_free_device(d_logits);
    qscu_utils_free_device(d_next);
}

static void test_qscu_utils_greedy_argmax_all_nan()
{
    constexpr int batch = 1;
    constexpr int vocab = 4;
    const float h_logits[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    int32_t h_next_i32[] = { 123 };
    uint32_t h_next_u32[] = { 123 };

    float* d_logits = nullptr;
    int32_t* d_next_i32 = nullptr;
    uint32_t* d_next_u32 = nullptr;
    if (!copy_to_device(&d_logits, h_logits, vocab, "copy qscu argmax all-nan logits")
        || !copy_to_device(&d_next_i32, h_next_i32, batch, "copy qscu argmax all-nan i32 out")
        || !copy_to_device(&d_next_u32, h_next_u32, batch, "copy qscu argmax all-nan u32 out")) {
        qscu_utils_free_device(d_logits);
        qscu_utils_free_device(d_next_i32);
        qscu_utils_free_device(d_next_u32);
        return;
    }

    qscu_sampling_desc desc {};
    desc.logits = tensor2_f32(d_logits, batch, vocab);
    desc.batch_size = batch;
    desc.vocab_size = vocab;
    desc.top_k = 0;
    desc.top_p = 0.0f;
    desc.min_p = 0.0f;
    desc.temperature = 0.0f;

#if QSFI_ENABLE_CHECKED_VALIDATION
    desc.next_token_ids = qscu_utils_tensor1_i32(d_next_i32, batch);
    check_status(
        qscu_greedy_argmax_f32(&desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "checked qscu greedy rejects all-NaN i32 logits"
    );
    desc.next_token_ids = qscu_utils_tensor1_u32(d_next_u32, batch);
    check_status(
        qscu_greedy_argmax_f32(&desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "checked qscu greedy rejects all-NaN u32 logits"
    );
#else
    desc.next_token_ids = qscu_utils_tensor1_i32(d_next_i32, batch);
    if (check_status(
            qscu_greedy_argmax_f32(&desc, nullptr),
            QSFI_STATUS_OK,
            "release qscu greedy allows all-NaN i32 logits"
        )) {
        check_cuda(cudaDeviceSynchronize(), "sync qscu argmax all-nan i32");
        if (check_cuda(
                cudaMemcpy(h_next_i32, d_next_i32, sizeof(h_next_i32), cudaMemcpyDeviceToHost),
                "copy qscu argmax all-nan i32 out"
            )
            && h_next_i32[0] != -1) {
            std::fprintf(
                stderr,
                "FAIL: qscu release greedy all-NaN i32 result: got %d, want -1\n",
                h_next_i32[0]
            );
            failures += 1;
        }
    }

    desc.next_token_ids = qscu_utils_tensor1_u32(d_next_u32, batch);
    if (check_status(
            qscu_greedy_argmax_f32(&desc, nullptr),
            QSFI_STATUS_OK,
            "release qscu greedy allows all-NaN u32 logits"
        )) {
        check_cuda(cudaDeviceSynchronize(), "sync qscu argmax all-nan u32");
        if (check_cuda(
                cudaMemcpy(h_next_u32, d_next_u32, sizeof(h_next_u32), cudaMemcpyDeviceToHost),
                "copy qscu argmax all-nan u32 out"
            )
            && h_next_u32[0] != std::numeric_limits<uint32_t>::max()) {
            std::fprintf(
                stderr,
                "FAIL: qscu release greedy all-NaN u32 result: got %u, want %u\n",
                h_next_u32[0],
                std::numeric_limits<uint32_t>::max()
            );
            failures += 1;
        }
    }
#endif

    qscu_utils_free_device(d_logits);
    qscu_utils_free_device(d_next_i32);
    qscu_utils_free_device(d_next_u32);
}

static void test_qscu_utils_router_nonfinite()
{
    constexpr int tokens = 1;
    constexpr int experts = 4;
    constexpr int top_k = 2;
    const float h_logits[] = {
        std::numeric_limits<float>::quiet_NaN(),
        -1.0f,
        std::numeric_limits<float>::infinity(),
        0.0f,
    };
    int32_t h_ids[] = { -1, -1 };
    float h_weights[] = { -1.0f, -1.0f };

    float* d_logits = nullptr;
    int32_t* d_ids = nullptr;
    float* d_weights = nullptr;
    if (!copy_to_device(&d_logits, h_logits, experts, "copy qscu router nonfinite logits")
        || !copy_to_device(&d_ids, h_ids, top_k, "copy qscu router nonfinite ids")
        || !copy_to_device(&d_weights, h_weights, top_k, "copy qscu router nonfinite weights")) {
        qscu_utils_free_device(d_logits);
        qscu_utils_free_device(d_ids);
        qscu_utils_free_device(d_weights);
        return;
    }

    qscu_router_topk_desc desc {};
    desc.logits = tensor2_f32(d_logits, tokens, experts);
    desc.topk_ids = tensor2_i32(d_ids, tokens, top_k);
    desc.topk_weights = tensor2_f32(d_weights, tokens, top_k);
    desc.num_tokens = tokens;
    desc.num_experts = experts;
    desc.top_k = top_k;
    desc.score = QSCU_ROUTER_SCORE_SIGMOID;
    desc.renormalize = 0;
    desc.routed_scaling_factor = 1.0f;

#if QSFI_ENABLE_CHECKED_VALIDATION
    check_status(
        qscu_router_topk(&desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "checked qscu router rejects non-finite logits"
    );
#else
    if (check_status(
            qscu_router_topk(&desc, nullptr),
            QSFI_STATUS_OK,
            "release qscu router allows non-finite logits"
        )) {
        check_cuda(cudaDeviceSynchronize(), "sync qscu router nonfinite");
        const bool copied
            = check_cuda(
                  cudaMemcpy(h_ids, d_ids, sizeof(h_ids), cudaMemcpyDeviceToHost),
                  "copy qscu router nonfinite ids"
              )
            && check_cuda(
                  cudaMemcpy(h_weights, d_weights, sizeof(h_weights), cudaMemcpyDeviceToHost),
                  "copy qscu router nonfinite weights"
            );
        if (copied) {
            if (h_ids[0] != 2 || h_ids[1] != 3) {
                std::fprintf(
                    stderr,
                    "FAIL: qscu release router nonfinite ids: got [%d, %d], want [2, 3]\n",
                    h_ids[0],
                    h_ids[1]
                );
                failures += 1;
            }
            if (!std::isfinite(h_weights[0]) || std::fabs(h_weights[0] - 1.0f) > 0.00001f
                || !std::isfinite(h_weights[1]) || std::fabs(h_weights[1] - 0.5f) > 0.00001f) {
                std::fprintf(
                    stderr,
                    "FAIL: qscu release router nonfinite weights: got [%.8f, %.8f], "
                    "want [1.00000000, 0.50000000]\n",
                    h_weights[0],
                    h_weights[1]
                );
                failures += 1;
            }
        }
    }
#endif

    qscu_utils_free_device(d_logits);
    qscu_utils_free_device(d_ids);
    qscu_utils_free_device(d_weights);
}

static void test_qscu_utils_negative_validation()
{
    test_qscu_utils_greedy_argmax_nonfinite();
    test_qscu_utils_greedy_argmax_all_nan();
    test_qscu_utils_router_nonfinite();

    const uint16_t h_bf16[] = { kBf16One };
    uint16_t* d_gate = nullptr;
    uint16_t* d_up = nullptr;
    uint16_t* d_out = nullptr;
    if (!copy_to_device(&d_gate, h_bf16, 1, "copy qscu negative silu gate")
        || !copy_to_device(&d_up, h_bf16, 1, "copy qscu negative silu up")
        || !copy_to_device(&d_out, h_bf16, 1, "copy qscu negative silu out")) {
        qscu_utils_free_device(d_gate);
        qscu_utils_free_device(d_up);
        qscu_utils_free_device(d_out);
        return;
    }

    qscu_silu_and_mul_desc silu_desc {};
    silu_desc.gate = tensor2_bf16(d_gate, 1, 1);
    silu_desc.up = tensor2_bf16(d_up, 1, 1);
    silu_desc.out = tensor2_bf16(d_out, 1, 1);
    silu_desc.num_tokens = 1;
    silu_desc.intermediate_size = 1;
    silu_desc.intermediate_size = 0;
    check_status(
        qscu_silu_and_mul_bf16(&silu_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu silu rejects zero intermediate size"
    );

    const float h_logits[] = { 0.0f };
    const int32_t h_next[] = { -1 };
    float* d_logits = nullptr;
    int32_t* d_next = nullptr;
    if (!copy_to_device(&d_logits, h_logits, 1, "copy qscu negative sampling logits")
        || !copy_to_device(&d_next, h_next, 1, "copy qscu negative sampling out")) {
        qscu_utils_free_device(d_gate);
        qscu_utils_free_device(d_up);
        qscu_utils_free_device(d_out);
        qscu_utils_free_device(d_logits);
        qscu_utils_free_device(d_next);
        return;
    }

    qscu_sampling_desc sampling_desc {};
    sampling_desc.logits = tensor2_f32(d_logits, 1, 1);
    sampling_desc.next_token_ids = qscu_utils_tensor1_i32(d_next, 1);
    sampling_desc.batch_size = 1;
    sampling_desc.vocab_size = 1;
    sampling_desc.temperature = 1.0f;
    check_status(
        qscu_greedy_argmax_f32(&sampling_desc, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu greedy unsupported temperature"
    );

    qscu_qwen36_shared_expert_gate_add_desc shared_desc {};
    shared_desc.gate_logits = tensor2_f32(d_logits, 1, 1);
    shared_desc.shared = tensor2_bf16(d_up, 1, 1);
    shared_desc.out = tensor2_bf16(d_out, 1, 1);
    shared_desc.num_tokens = 1;
    shared_desc.hidden_size = 0;
    check_status(
        qscu_qwen36_shared_expert_gate_add_bf16(&shared_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu shared expert rejects zero hidden size"
    );
    shared_desc.hidden_size = 1;
    shared_desc.gate_logits = tensor2_f32(d_logits, 1, 2);
    check_status(
        qscu_qwen36_shared_expert_gate_add_bf16(&shared_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu shared expert rejects nonscalar gate"
    );

    qscu_qwen36_full_attention_output_gate_desc full_attn_gate_desc {};
    full_attn_gate_desc.gate = tensor2_bf16(d_gate, 1, 1);
    full_attn_gate_desc.out = tensor2_bf16(d_out, 1, 1);
    full_attn_gate_desc.num_tokens = 1;
    full_attn_gate_desc.q_hidden = 4096;
    check_status(
        qscu_qwen36_full_attention_output_gate_bf16(&full_attn_gate_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu extracted full-attn output gate rejects short gate shape"
    );
    full_attn_gate_desc.q_hidden = 1;
    check_status(
        qscu_qwen36_full_attention_output_gate_bf16(&full_attn_gate_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu extracted full-attn output gate rejects non-qwen q_hidden"
    );
    full_attn_gate_desc.gate = tensor2_bf16(d_gate, 1, 4096);
    full_attn_gate_desc.out = tensor2_bf16(d_out, 1, 4096);
    full_attn_gate_desc.q_hidden = 4096;
    full_attn_gate_desc.gate.dtype = QSFI_DTYPE_F32;
    check_status(
        qscu_qwen36_full_attention_output_gate_bf16(&full_attn_gate_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu extracted full-attn output gate rejects f32 gate"
    );
    full_attn_gate_desc.gate = tensor2_bf16(d_gate, 1, 4096);
    full_attn_gate_desc.gate.stride[0] = 4097;
    check_status(
        qscu_qwen36_full_attention_output_gate_bf16(&full_attn_gate_desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu extracted full-attn output gate rejects padded gate"
    );

    qscu_utils_free_device(d_gate);
    qscu_utils_free_device(d_up);
    qscu_utils_free_device(d_out);
    qscu_utils_free_device(d_logits);
    qscu_utils_free_device(d_next);
}

#endif
