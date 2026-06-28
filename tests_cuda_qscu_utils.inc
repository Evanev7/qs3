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
    desc.clamp_limit = 0.0f;

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

static void test_qscu_utils_embedding_gather_bf16()
{
    constexpr int vocab = 4;
    constexpr int hidden = 5;
    constexpr int tokens = 4;
    constexpr int32_t padding_id = 7;
    constexpr size_t embedding_elems = static_cast<size_t>(vocab) * hidden;
    constexpr size_t out_elems = static_cast<size_t>(tokens) * hidden;

    const int32_t token_ids[] = { 2, padding_id, 0, 3 };
    const int32_t invalid_token_ids[] = { 2, padding_id, vocab, 3 };
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

static void test_qscu_utils_negative_validation()
{
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
    silu_desc.clamp_limit = 1.0f;
    check_status(
        qscu_silu_and_mul_bf16(&silu_desc, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu silu unsupported clamp"
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

    qscu_utils_free_device(d_gate);
    qscu_utils_free_device(d_up);
    qscu_utils_free_device(d_out);
    qscu_utils_free_device(d_logits);
    qscu_utils_free_device(d_next);
}

#endif
