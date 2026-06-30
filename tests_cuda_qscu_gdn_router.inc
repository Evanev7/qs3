#ifndef QS3_TESTS_CUDA_QSCU_GDN_ROUTER_INC
#define QS3_TESTS_CUDA_QSCU_GDN_ROUTER_INC

#ifndef QSCU_H
#include "qscu.h"
#endif

constexpr int kQscuTestGdnPackedDim = 2 * kGdnKHeads * kGdnKeyDim + kGdnVHeads * kGdnValueDim;
constexpr int kQscuTestConvWidth = 4;
constexpr int kQscuTestConvState = kQscuTestConvWidth - 1;

struct QscuTestRouterRef {
    int32_t ids[16];
    float weights[16];
};

uint16_t qscu_test_f32_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float qscu_test_silu(float value)
{
    return value / (1.0f + std::exp(-value));
}

float qscu_test_sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

float qscu_test_softplus(float value)
{
    return value <= 20.0f ? std::log1p(std::exp(value)) : value;
}

float qscu_test_activation(float value, qscu_activation activation)
{
    if (activation == QSCU_ACTIVATION_SILU)
        return qscu_test_silu(value);
    if (activation == QSCU_ACTIVATION_SIGMOID)
        return qscu_test_sigmoid(value);
    return value;
}

void qscu_test_store_bf16(std::vector<uint16_t>& data, size_t offset, float value)
{
    data[offset] = qscu_test_f32_to_bf16(value);
}

size_t qscu_test_conv_state_offset(int slot, int dim, int state_pos)
{
    return (static_cast<size_t>(slot) * kQscuTestGdnPackedDim + dim) * kQscuTestConvState
        + state_pos;
}

size_t qscu_test_conv_weight_offset(int dim, int pos)
{
    return static_cast<size_t>(dim) * kQscuTestConvWidth + pos;
}

size_t qscu_test_packed_q_offset(int token, int head, int dim)
{
    return static_cast<size_t>(token) * kQscuTestGdnPackedDim + head * kGdnKeyDim + dim;
}

size_t qscu_test_packed_k_offset(int token, int head, int dim)
{
    return static_cast<size_t>(token) * kQscuTestGdnPackedDim + kGdnKHeads * kGdnKeyDim
        + head * kGdnKeyDim + dim;
}

size_t qscu_test_packed_v_offset(int token, int head, int dim)
{
    return static_cast<size_t>(token) * kQscuTestGdnPackedDim + 2 * kGdnKHeads * kGdnKeyDim
        + head * kGdnValueDim + dim;
}

size_t qscu_test_gdn_gate_offset(int token, int head)
{
    return static_cast<size_t>(token) * kGdnVHeads + head;
}

qsfi_tensor3 qscu_test_tensor3_f32(void* data, int64_t d0, int64_t d1, int64_t d2)
{
    qsfi_tensor3 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F32;
    tensor.shape[0] = d0;
    tensor.shape[1] = d1;
    tensor.shape[2] = d2;
    tensor.stride[0] = d1 * d2;
    tensor.stride[1] = d2;
    tensor.stride[2] = 1;
    return tensor;
}

qsfi_tensor1 qscu_test_tensor1_bf16(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

void* qscu_test_validation_ptr(uintptr_t tag)
{
    return reinterpret_cast<void*>(0x1000u + tag * 0x1000u);
}

qscu_qwen36_gdn_causal_conv1d_desc qscu_test_valid_conv_validation_desc()
{
    qscu_qwen36_gdn_causal_conv1d_desc desc {};
    desc.x = tensor2_bf16(qscu_test_validation_ptr(1), 1, kQscuTestGdnPackedDim);
    desc.weight
        = tensor2_bf16(qscu_test_validation_ptr(2), kQscuTestGdnPackedDim, kQscuTestConvWidth);
    desc.bias = gdn_tensor1_f32(qscu_test_validation_ptr(3), kQscuTestGdnPackedDim);
    desc.state = qscu_test_tensor3_f32(
        qscu_test_validation_ptr(4),
        1,
        kQscuTestGdnPackedDim,
        kQscuTestConvState
    );
    desc.state_read_indices = gdn_tensor1_i32(qscu_test_validation_ptr(5), 1);
    desc.state_write_indices = gdn_tensor1_i32(qscu_test_validation_ptr(6), 1);
    desc.out = tensor2_bf16(qscu_test_validation_ptr(7), 1, kQscuTestGdnPackedDim);
    desc.num_tokens = 1;
    desc.batch_size = 1;
    desc.activation = QSCU_ACTIVATION_SILU;
    desc.update_state = 1;
    return desc;
}

qscu_qwen36_gdn_post_conv_prepare_desc qscu_test_valid_post_validation_desc()
{
    qscu_qwen36_gdn_post_conv_prepare_desc desc {};
    desc.conv_out = tensor2_bf16(qscu_test_validation_ptr(11), 1, kQscuTestGdnPackedDim);
    desc.a = gdn_tensor2_bf16(qscu_test_validation_ptr(12), 1, kGdnVHeads);
    desc.b = gdn_tensor2_bf16(qscu_test_validation_ptr(13), 1, kGdnVHeads);
    desc.a_log = gdn_tensor1_f32(qscu_test_validation_ptr(14), kGdnVHeads);
    desc.dt_bias = gdn_tensor1_f32(qscu_test_validation_ptr(15), kGdnVHeads);
    desc.q = gdn_tensor3_bf16(qscu_test_validation_ptr(16), 1, kGdnQHeads, kGdnKeyDim);
    desc.k = gdn_tensor3_bf16(qscu_test_validation_ptr(17), 1, kGdnKHeads, kGdnKeyDim);
    desc.v = gdn_tensor3_bf16(qscu_test_validation_ptr(18), 1, kGdnVHeads, kGdnValueDim);
    desc.g_out = tensor2_f32(qscu_test_validation_ptr(19), 1, kGdnVHeads);
    desc.beta_out = tensor2_f32(qscu_test_validation_ptr(20), 1, kGdnVHeads);
    desc.num_tokens = 1;
    desc.apply_qk_l2norm = 1;
    desc.l2norm_eps = 1.0e-6f;
    desc.forget_gate_output = QSCU_GDN_FORGET_LINEAR_ALPHA;
    return desc;
}

qscu_qwen36_gdn_rmsnorm_gated_desc qscu_test_valid_rms_validation_desc()
{
    qscu_qwen36_gdn_rmsnorm_gated_desc desc {};
    desc.x = gdn_tensor3_bf16(qscu_test_validation_ptr(21), 1, kGdnVHeads, kGdnValueDim);
    desc.gate = gdn_tensor3_bf16(qscu_test_validation_ptr(22), 1, kGdnVHeads, kGdnValueDim);
    desc.weight = gdn_tensor1_f32(qscu_test_validation_ptr(23), kGdnValueDim);
    desc.out = gdn_tensor3_bf16(qscu_test_validation_ptr(24), 1, kGdnVHeads, kGdnValueDim);
    desc.num_tokens = 1;
    desc.eps = 1.0e-5f;
    desc.gate_activation = QSCU_ACTIVATION_SILU;
    return desc;
}

qscu_router_topk_desc qscu_test_valid_router_validation_desc(int num_experts, int top_k)
{
    qscu_router_topk_desc desc {};
    desc.logits = tensor2_f32(qscu_test_validation_ptr(31), 1, num_experts);
    desc.topk_ids = tensor2_i32(qscu_test_validation_ptr(32), 1, top_k);
    desc.topk_weights = tensor2_f32(qscu_test_validation_ptr(33), 1, top_k);
    desc.num_tokens = 1;
    desc.num_experts = static_cast<uint32_t>(num_experts);
    desc.top_k = static_cast<uint32_t>(top_k);
    desc.score = QSCU_ROUTER_SCORE_SOFTMAX;
    desc.renormalize = 1;
    desc.routed_scaling_factor = 1.0f;
    return desc;
}

void qscu_test_check_f32_near(float got, float want, float abs_tol, const char* what)
{
    if (approx_equal(got, want, abs_tol))
        return;
    std::fprintf(stderr, "FAIL: %s: got %.8f, want %.8f +/- %.8f\n", what, got, want, abs_tol);
    failures += 1;
}

void qscu_test_check_bf16_near(
    const std::vector<uint16_t>& values, size_t offset, float want, float abs_tol, const char* what
)
{
    const float got = bf16_to_f32(values[offset]);
    if (approx_equal(got, want, abs_tol))
        return;
    std::fprintf(
        stderr,
        "FAIL: %s[%zu]: got %.8f, want %.8f +/- %.8f\n",
        what,
        offset,
        got,
        want,
        abs_tol
    );
    failures += 1;
}

void qscu_test_check_i32_eq(int32_t got, int32_t want, const char* what)
{
    if (got == want)
        return;
    std::fprintf(stderr, "FAIL: %s: got %d, want %d\n", what, got, want);
    failures += 1;
}

QscuTestRouterRef qscu_test_router_reference(
    const std::vector<float>& logits,
    int row,
    int num_experts,
    int top_k,
    qscu_router_score score,
    bool renormalize,
    float routed_scaling_factor
)
{
    QscuTestRouterRef ref {};
    for (int pos = 0; pos < 16; ++pos) {
        ref.ids[pos] = 0x7FFFFFFF;
        ref.weights[pos] = -3.4028234663852886e38f;
    }

    float max_logit = -3.4028234663852886e38f;
    if (score == QSCU_ROUTER_SCORE_SOFTMAX) {
        for (int expert = 0; expert < num_experts; ++expert)
            max_logit = std::max(max_logit, logits[row * num_experts + expert]);
    }

    float softmax_denom = 0.0f;
    if (score == QSCU_ROUTER_SCORE_SOFTMAX) {
        for (int expert = 0; expert < num_experts; ++expert)
            softmax_denom += std::exp(logits[row * num_experts + expert] - max_logit);
    }

    float best_scores[16];
    int32_t best_ids[16];
    for (int pos = 0; pos < 16; ++pos) {
        best_scores[pos] = -3.4028234663852886e38f;
        best_ids[pos] = 0x7FFFFFFF;
    }
    for (int expert = 0; expert < num_experts; ++expert) {
        const float logit = logits[row * num_experts + expert];
        const float select_score
            = score == QSCU_ROUTER_SCORE_SOFTMAX ? logit : qscu_test_sigmoid(logit);
        for (int pos = 0; pos < top_k; ++pos) {
            const bool better = select_score > best_scores[pos]
                || (select_score == best_scores[pos] && expert < best_ids[pos]);
            if (!better)
                continue;
            for (int move = top_k - 1; move > pos; --move) {
                best_scores[move] = best_scores[move - 1];
                best_ids[move] = best_ids[move - 1];
            }
            best_scores[pos] = select_score;
            best_ids[pos] = expert;
            break;
        }
    }

    float selected_sum = 0.0f;
    for (int pos = 0; pos < top_k; ++pos) {
        const float logit = logits[row * num_experts + best_ids[pos]];
        float weight = 0.0f;
        if (score == QSCU_ROUTER_SCORE_SOFTMAX)
            weight = std::exp(logit - max_logit) / softmax_denom;
        else
            weight = qscu_test_sigmoid(logit);
        ref.ids[pos] = best_ids[pos];
        ref.weights[pos] = weight;
        selected_sum += weight;
    }
    const float renorm = renormalize ? 1.0f / std::max(selected_sum, 1.0e-20f) : 1.0f;
    for (int pos = 0; pos < top_k; ++pos)
        ref.weights[pos] *= renorm * routed_scaling_factor;
    return ref;
}

void test_qscu_qwen36_gdn_causal_conv1d_bf16_cpu_reference()
{
    constexpr int kTokens = 3;
    constexpr int kBatchSize = 2;
    constexpr int kStateSlots = 2;
    constexpr qscu_activation kActivation = QSCU_ACTIVATION_SILU;
    const int active_dims[] = { 7, 3000 };

    std::vector<uint16_t> h_x(static_cast<size_t>(kTokens) * kQscuTestGdnPackedDim, kBf16Zero);
    std::vector<uint16_t> h_weight(
        static_cast<size_t>(kQscuTestGdnPackedDim) * kQscuTestConvWidth,
        kBf16Zero
    );
    std::vector<float> h_bias(kQscuTestGdnPackedDim, 0.0f);
    std::vector<float> h_state(
        static_cast<size_t>(kStateSlots) * kQscuTestGdnPackedDim * kQscuTestConvState,
        0.0f
    );

    h_state[qscu_test_conv_state_offset(0, active_dims[0], 0)] = 0.25f;
    h_state[qscu_test_conv_state_offset(0, active_dims[0], 1)] = -0.5f;
    h_state[qscu_test_conv_state_offset(0, active_dims[0], 2)] = 1.0f;
    qscu_test_store_bf16(h_weight, qscu_test_conv_weight_offset(active_dims[0], 0), 0.5f);
    qscu_test_store_bf16(h_weight, qscu_test_conv_weight_offset(active_dims[0], 1), -1.0f);
    qscu_test_store_bf16(h_weight, qscu_test_conv_weight_offset(active_dims[0], 2), 0.25f);
    qscu_test_store_bf16(h_weight, qscu_test_conv_weight_offset(active_dims[0], 3), 2.0f);
    h_bias[active_dims[0]] = 0.125f;
    qscu_test_store_bf16(h_x, active_dims[0], 1.5f);
    qscu_test_store_bf16(h_x, kQscuTestGdnPackedDim + active_dims[0], -2.0f);
    qscu_test_store_bf16(h_x, 2 * kQscuTestGdnPackedDim + active_dims[0], 0.75f);

    h_state[qscu_test_conv_state_offset(0, active_dims[1], 0)] = 1.0f;
    h_state[qscu_test_conv_state_offset(0, active_dims[1], 1)] = 0.0f;
    h_state[qscu_test_conv_state_offset(0, active_dims[1], 2)] = -1.0f;
    for (int pos = 0; pos < kQscuTestConvWidth; ++pos)
        qscu_test_store_bf16(h_weight, qscu_test_conv_weight_offset(active_dims[1], pos), 1.0f);
    h_bias[active_dims[1]] = -0.25f;
    qscu_test_store_bf16(h_x, active_dims[1], 0.5f);
    qscu_test_store_bf16(h_x, kQscuTestGdnPackedDim + active_dims[1], 1.0f);
    qscu_test_store_bf16(h_x, 2 * kQscuTestGdnPackedDim + active_dims[1], -0.5f);

    const int32_t h_read_indices[kBatchSize] = { 0, -1 };
    const int32_t h_write_indices[kBatchSize] = { 1, -1 };
    const int32_t h_seq_indptr[kBatchSize + 1] = { 0, 2, 3 };

    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    float* d_bias = nullptr;
    float* d_state = nullptr;
    int32_t* d_read_indices = nullptr;
    int32_t* d_write_indices = nullptr;
    int32_t* d_seq_indptr = nullptr;

    const bool ok = copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qscu conv x")
        && copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy qscu conv weight")
        && copy_to_device(&d_bias, h_bias.data(), h_bias.size(), "copy qscu conv bias")
        && copy_to_device(&d_state, h_state.data(), h_state.size(), "copy qscu conv state")
        && copy_to_device(
                        &d_read_indices,
                        h_read_indices,
                        kBatchSize,
                        "copy qscu conv read indices"
        )
        && copy_to_device(
                        &d_write_indices,
                        h_write_indices,
                        kBatchSize,
                        "copy qscu conv write indices"
        )
        && copy_to_device(&d_seq_indptr, h_seq_indptr, kBatchSize + 1, "copy qscu conv seq indptr")
        && alloc_device(&d_out,
                        static_cast<size_t>(kTokens) * kQscuTestGdnPackedDim,
                        "alloc qscu conv out");

    if (ok) {
        qscu_qwen36_gdn_causal_conv1d_desc desc {};
        desc.x = tensor2_bf16(d_x, kTokens, kQscuTestGdnPackedDim);
        desc.weight = tensor2_bf16(d_weight, kQscuTestGdnPackedDim, kQscuTestConvWidth);
        desc.bias = gdn_tensor1_f32(d_bias, kQscuTestGdnPackedDim);
        desc.state = qscu_test_tensor3_f32(
            d_state,
            kStateSlots,
            kQscuTestGdnPackedDim,
            kQscuTestConvState
        );
        desc.state_read_indices = gdn_tensor1_i32(d_read_indices, kBatchSize);
        desc.state_write_indices = gdn_tensor1_i32(d_write_indices, kBatchSize);
        desc.seq_indptr = d_seq_indptr;
        desc.out = tensor2_bf16(d_out, kTokens, kQscuTestGdnPackedDim);
        desc.num_tokens = kTokens;
        desc.batch_size = kBatchSize;
        desc.activation = kActivation;
        desc.update_state = 1;

        check_status(
            qscu_qwen36_gdn_causal_conv1d_bf16(&desc, nullptr),
            QSFI_STATUS_OK,
            "qscu conv1d"
        );
        check_cuda(cudaDeviceSynchronize(), "sync qscu conv1d");

        std::vector<uint16_t> h_out(
            static_cast<size_t>(kTokens) * kQscuTestGdnPackedDim,
            kBf16Zero
        );
        std::vector<float> h_state_after(h_state.size(), 0.0f);
        if (check_cuda(
                cudaMemcpy(
                    h_out.data(),
                    d_out,
                    h_out.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu conv out"
            )
            && check_cuda(
                cudaMemcpy(
                    h_state_after.data(),
                    d_state,
                    h_state_after.size() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu conv state"
            )) {
            for (int active_idx = 0; active_idx < 2; ++active_idx) {
                const int dim = active_dims[active_idx];
                float h0 = h_state[qscu_test_conv_state_offset(0, dim, 0)];
                float h1 = h_state[qscu_test_conv_state_offset(0, dim, 1)];
                float h2 = h_state[qscu_test_conv_state_offset(0, dim, 2)];
                for (int token = 0; token < 2; ++token) {
                    const float x_value = bf16_to_f32(
                        h_x[static_cast<size_t>(token) * kQscuTestGdnPackedDim + dim]
                    );
                    const float w0 = bf16_to_f32(h_weight[qscu_test_conv_weight_offset(dim, 0)]);
                    const float w1 = bf16_to_f32(h_weight[qscu_test_conv_weight_offset(dim, 1)]);
                    const float w2 = bf16_to_f32(h_weight[qscu_test_conv_weight_offset(dim, 2)]);
                    const float w3 = bf16_to_f32(h_weight[qscu_test_conv_weight_offset(dim, 3)]);
                    const float want = qscu_test_activation(
                        h0 * w0 + h1 * w1 + h2 * w2 + x_value * w3 + h_bias[dim],
                        kActivation
                    );
                    qscu_test_check_bf16_near(
                        h_out,
                        static_cast<size_t>(token) * kQscuTestGdnPackedDim + dim,
                        want,
                        0.04f,
                        "qscu conv out"
                    );
                    h0 = h1;
                    h1 = h2;
                    h2 = x_value;
                }
                qscu_test_check_f32_near(
                    h_state_after[qscu_test_conv_state_offset(1, dim, 0)],
                    h0,
                    1.0e-6f,
                    "qscu conv write state h0"
                );
                qscu_test_check_f32_near(
                    h_state_after[qscu_test_conv_state_offset(1, dim, 1)],
                    h1,
                    1.0e-6f,
                    "qscu conv write state h1"
                );
                qscu_test_check_f32_near(
                    h_state_after[qscu_test_conv_state_offset(1, dim, 2)],
                    h2,
                    1.0e-6f,
                    "qscu conv write state h2"
                );

                const float token2_x
                    = bf16_to_f32(h_x[2 * static_cast<size_t>(kQscuTestGdnPackedDim) + dim]);
                const float token2_w3 = bf16_to_f32(h_weight[qscu_test_conv_weight_offset(dim, 3)]);
                const float token2_want
                    = qscu_test_activation(token2_x * token2_w3 + h_bias[dim], kActivation);
                qscu_test_check_bf16_near(
                    h_out,
                    2 * static_cast<size_t>(kQscuTestGdnPackedDim) + dim,
                    token2_want,
                    0.04f,
                    "qscu conv zero-read out"
                );
                qscu_test_check_f32_near(
                    h_state_after[qscu_test_conv_state_offset(0, dim, 0)],
                    h_state[qscu_test_conv_state_offset(0, dim, 0)],
                    1.0e-6f,
                    "qscu conv read state unchanged"
                );
            }
            qscu_test_check_bf16_near(
                h_out,
                kQscuTestGdnPackedDim + 11,
                0.0f,
                0.0f,
                "qscu conv inactive out"
            );
        }
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_state);
    cudaFree(d_read_indices);
    cudaFree(d_write_indices);
    cudaFree(d_seq_indptr);
    cudaFree(d_out);
}

void test_qscu_qwen36_gdn_post_conv_prepare_bf16_cpu_reference()
{
    constexpr int kTokens = 2;
    constexpr int kQHead = 2;
    constexpr int kVHead = 5;
    constexpr int kDim0 = 3;
    constexpr int kDim1 = 4;
    constexpr int kVDim = 9;
    constexpr float kL2Eps = 1.0e-6f;

    std::vector<uint16_t> h_conv(static_cast<size_t>(kTokens) * kQscuTestGdnPackedDim, kBf16Zero);
    std::vector<uint16_t> h_a(static_cast<size_t>(kTokens) * kGdnVHeads, kBf16Zero);
    std::vector<uint16_t> h_b(static_cast<size_t>(kTokens) * kGdnVHeads, kBf16Zero);
    std::vector<float> h_a_log(kGdnVHeads, 0.0f);
    std::vector<float> h_dt_bias(kGdnVHeads, 0.0f);

    qscu_test_store_bf16(h_conv, qscu_test_packed_q_offset(0, kQHead, kDim0), 3.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_q_offset(0, kQHead, kDim1), 4.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_k_offset(0, kQHead, kDim0), -6.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_k_offset(0, kQHead, kDim1), 8.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_v_offset(0, kVHead, kVDim), 1.25f);
    qscu_test_store_bf16(h_a, qscu_test_gdn_gate_offset(0, kVHead), 0.5f);
    qscu_test_store_bf16(h_b, qscu_test_gdn_gate_offset(0, kVHead), -0.5f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_q_offset(1, kQHead, kDim0), -5.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_q_offset(1, kQHead, kDim1), 12.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_k_offset(1, kQHead, kDim0), 9.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_k_offset(1, kQHead, kDim1), -12.0f);
    qscu_test_store_bf16(h_conv, qscu_test_packed_v_offset(1, kVHead, kVDim), -2.5f);
    qscu_test_store_bf16(h_a, qscu_test_gdn_gate_offset(1, kVHead), -1.0f);
    qscu_test_store_bf16(h_b, qscu_test_gdn_gate_offset(1, kVHead), 1.5f);
    h_a_log[kVHead] = std::log(2.0f);
    h_dt_bias[kVHead] = 0.25f;

    uint16_t* d_conv = nullptr;
    uint16_t* d_a = nullptr;
    uint16_t* d_b = nullptr;
    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    float* d_a_log = nullptr;
    float* d_dt_bias = nullptr;
    float* d_g = nullptr;
    float* d_beta = nullptr;

    const bool ok = copy_to_device(&d_conv, h_conv.data(), h_conv.size(), "copy qscu post conv")
        && copy_to_device(&d_a, h_a.data(), h_a.size(), "copy qscu post a")
        && copy_to_device(&d_b, h_b.data(), h_b.size(), "copy qscu post b")
        && copy_to_device(&d_a_log, h_a_log.data(), h_a_log.size(), "copy qscu post a_log")
        && copy_to_device(&d_dt_bias, h_dt_bias.data(), h_dt_bias.size(), "copy qscu post dt_bias")
        && alloc_device(&d_q,
                        static_cast<size_t>(kTokens) * kGdnQHeads * kGdnKeyDim,
                        "alloc qscu post q")
        && alloc_device(&d_k,
                        static_cast<size_t>(kTokens) * kGdnKHeads * kGdnKeyDim,
                        "alloc qscu post k")
        && alloc_device(&d_v,
                        static_cast<size_t>(kTokens) * kGdnVHeads * kGdnValueDim,
                        "alloc qscu post v")
        && alloc_device(&d_g, static_cast<size_t>(kTokens) * kGdnVHeads, "alloc qscu post g")
        && alloc_device(&d_beta, static_cast<size_t>(kTokens) * kGdnVHeads, "alloc qscu post beta");

    if (ok) {
        qscu_qwen36_gdn_post_conv_prepare_desc desc {};
        desc.conv_out = tensor2_bf16(d_conv, kTokens, kQscuTestGdnPackedDim);
        desc.a = gdn_tensor2_bf16(d_a, kTokens, kGdnVHeads);
        desc.b = gdn_tensor2_bf16(d_b, kTokens, kGdnVHeads);
        desc.a_log = gdn_tensor1_f32(d_a_log, kGdnVHeads);
        desc.dt_bias = gdn_tensor1_f32(d_dt_bias, kGdnVHeads);
        desc.q = gdn_tensor3_bf16(d_q, kTokens, kGdnQHeads, kGdnKeyDim);
        desc.k = gdn_tensor3_bf16(d_k, kTokens, kGdnKHeads, kGdnKeyDim);
        desc.v = gdn_tensor3_bf16(d_v, kTokens, kGdnVHeads, kGdnValueDim);
        desc.g_out = tensor2_f32(d_g, kTokens, kGdnVHeads);
        desc.beta_out = tensor2_f32(d_beta, kTokens, kGdnVHeads);
        desc.num_tokens = kTokens;
        desc.apply_qk_l2norm = 1;
        desc.l2norm_eps = kL2Eps;
        desc.forget_gate_output = QSCU_GDN_FORGET_LINEAR_ALPHA;

        check_status(
            qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, nullptr),
            QSFI_STATUS_OK,
            "qscu post conv prepare"
        );
        check_cuda(cudaDeviceSynchronize(), "sync qscu post conv prepare");

        std::vector<uint16_t> h_q(static_cast<size_t>(kTokens) * kGdnQHeads * kGdnKeyDim);
        std::vector<uint16_t> h_k(static_cast<size_t>(kTokens) * kGdnKHeads * kGdnKeyDim);
        std::vector<uint16_t> h_v(static_cast<size_t>(kTokens) * kGdnVHeads * kGdnValueDim);
        std::vector<float> h_g(static_cast<size_t>(kTokens) * kGdnVHeads);
        std::vector<float> h_beta(static_cast<size_t>(kTokens) * kGdnVHeads);
        if (check_cuda(
                cudaMemcpy(h_q.data(), d_q, h_q.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                "copy qscu post q"
            )
            && check_cuda(
                cudaMemcpy(h_k.data(), d_k, h_k.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                "copy qscu post k"
            )
            && check_cuda(
                cudaMemcpy(h_v.data(), d_v, h_v.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                "copy qscu post v"
            )
            && check_cuda(
                cudaMemcpy(h_g.data(), d_g, h_g.size() * sizeof(float), cudaMemcpyDeviceToHost),
                "copy qscu post g"
            )
            && check_cuda(
                cudaMemcpy(
                    h_beta.data(),
                    d_beta,
                    h_beta.size() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu post beta"
            )) {
            auto check_qkv = [&](int token, float v_want) {
                const float q0
                    = bf16_to_f32(h_conv[qscu_test_packed_q_offset(token, kQHead, kDim0)]);
                const float q1
                    = bf16_to_f32(h_conv[qscu_test_packed_q_offset(token, kQHead, kDim1)]);
                const float k0
                    = bf16_to_f32(h_conv[qscu_test_packed_k_offset(token, kQHead, kDim0)]);
                const float k1
                    = bf16_to_f32(h_conv[qscu_test_packed_k_offset(token, kQHead, kDim1)]);
                const float q_rstd = 1.0f / std::sqrt(q0 * q0 + q1 * q1 + kL2Eps);
                const float k_rstd = 1.0f / std::sqrt(k0 * k0 + k1 * k1 + kL2Eps);
                qscu_test_check_bf16_near(
                    h_q,
                    gdn_qk_offset(token, kQHead, kDim0),
                    q0 * q_rstd,
                    0.01f,
                    "qscu post q"
                );
                qscu_test_check_bf16_near(
                    h_q,
                    gdn_qk_offset(token, kQHead, kDim1),
                    q1 * q_rstd,
                    0.01f,
                    "qscu post q"
                );
                qscu_test_check_bf16_near(
                    h_k,
                    gdn_qk_offset(token, kQHead, kDim0),
                    k0 * k_rstd,
                    0.01f,
                    "qscu post k"
                );
                qscu_test_check_bf16_near(
                    h_k,
                    gdn_qk_offset(token, kQHead, kDim1),
                    k1 * k_rstd,
                    0.01f,
                    "qscu post k"
                );
                qscu_test_check_bf16_near(
                    h_v,
                    gdn_v_offset(token, kVHead, kVDim),
                    v_want,
                    0.0f,
                    "qscu post v"
                );
            };
            check_qkv(0, 1.25f);
            check_qkv(1, -2.5f);
            qscu_test_check_bf16_near(
                h_q,
                gdn_qk_offset(0, kQHead, 12),
                0.0f,
                0.0f,
                "qscu post inactive q"
            );

            auto expected_log_decay = [&](int token) {
                const size_t offset = qscu_test_gdn_gate_offset(token, kVHead);
                const float a_value = bf16_to_f32(h_a[offset]);
                return -std::exp(h_a_log[kVHead]) * qscu_test_softplus(a_value + h_dt_bias[kVHead]);
            };
            auto check_gates = [&](int token) {
                const size_t offset = qscu_test_gdn_gate_offset(token, kVHead);
                const float b_value = bf16_to_f32(h_b[offset]);
                qscu_test_check_f32_near(
                    h_g[offset],
                    std::exp(expected_log_decay(token)),
                    1.0e-5f,
                    "qscu post g linear alpha"
                );
                qscu_test_check_f32_near(
                    h_beta[offset],
                    qscu_test_sigmoid(b_value),
                    1.0e-5f,
                    "qscu post beta"
                );
            };
            check_gates(0);
            check_gates(1);

            desc.forget_gate_output = QSCU_GDN_FORGET_LOG_DECAY;
            if (check_status(
                    qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, nullptr),
                    QSFI_STATUS_OK,
                    "qscu post conv prepare log decay"
                )) {
                check_cuda(cudaDeviceSynchronize(), "sync qscu post conv prepare log decay");
                if (check_cuda(
                        cudaMemcpy(
                            h_g.data(),
                            d_g,
                            h_g.size() * sizeof(float),
                            cudaMemcpyDeviceToHost
                        ),
                        "copy qscu post g log decay"
                    )) {
                    qscu_test_check_f32_near(
                        h_g[qscu_test_gdn_gate_offset(0, kVHead)],
                        expected_log_decay(0),
                        1.0e-5f,
                        "qscu post g log decay"
                    );
                    qscu_test_check_f32_near(
                        h_g[qscu_test_gdn_gate_offset(1, kVHead)],
                        expected_log_decay(1),
                        1.0e-5f,
                        "qscu post g log decay"
                    );
                }
            }

            desc.apply_qk_l2norm = 0;
            desc.g_out = qsfi_tensor2 {};
            desc.beta_out = qsfi_tensor2 {};
            desc.forget_gate_output = QSCU_GDN_FORGET_LINEAR_ALPHA;
            if (check_status(
                    qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, nullptr),
                    QSFI_STATUS_OK,
                    "qscu post conv prepare no l2 optional gates"
                )) {
                check_cuda(cudaDeviceSynchronize(), "sync qscu post conv prepare no l2");
                if (check_cuda(
                        cudaMemcpy(
                            h_q.data(),
                            d_q,
                            h_q.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost
                        ),
                        "copy qscu post q no l2"
                    )
                    && check_cuda(
                        cudaMemcpy(
                            h_k.data(),
                            d_k,
                            h_k.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost
                        ),
                        "copy qscu post k no l2"
                    )
                    && check_cuda(
                        cudaMemcpy(
                            h_v.data(),
                            d_v,
                            h_v.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost
                        ),
                        "copy qscu post v no l2"
                    )) {
                    auto check_raw_qkv = [&](int token, float v_want) {
                        qscu_test_check_bf16_near(
                            h_q,
                            gdn_qk_offset(token, kQHead, kDim0),
                            bf16_to_f32(h_conv[qscu_test_packed_q_offset(token, kQHead, kDim0)]),
                            0.0f,
                            "qscu post no l2 q"
                        );
                        qscu_test_check_bf16_near(
                            h_q,
                            gdn_qk_offset(token, kQHead, kDim1),
                            bf16_to_f32(h_conv[qscu_test_packed_q_offset(token, kQHead, kDim1)]),
                            0.0f,
                            "qscu post no l2 q"
                        );
                        qscu_test_check_bf16_near(
                            h_k,
                            gdn_qk_offset(token, kQHead, kDim0),
                            bf16_to_f32(h_conv[qscu_test_packed_k_offset(token, kQHead, kDim0)]),
                            0.0f,
                            "qscu post no l2 k"
                        );
                        qscu_test_check_bf16_near(
                            h_k,
                            gdn_qk_offset(token, kQHead, kDim1),
                            bf16_to_f32(h_conv[qscu_test_packed_k_offset(token, kQHead, kDim1)]),
                            0.0f,
                            "qscu post no l2 k"
                        );
                        qscu_test_check_bf16_near(
                            h_v,
                            gdn_v_offset(token, kVHead, kVDim),
                            v_want,
                            0.0f,
                            "qscu post no l2 v"
                        );
                    };
                    check_raw_qkv(0, 1.25f);
                    check_raw_qkv(1, -2.5f);
                }
            }
        }
    }

    cudaFree(d_conv);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_log);
    cudaFree(d_dt_bias);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_beta);
}

void test_qscu_qwen36_gdn_rmsnorm_gated_bf16_cpu_reference()
{
    constexpr int kTokens = 1;
    constexpr int kHead = 7;
    constexpr float kEps = 1.0e-5f;
    const int dims[] = { 0, 1, 2, 3 };
    const float x_values[] = { 1.0f, 2.0f, -1.5f, 0.5f };
    const float gate_values[] = { 1.0f, -0.5f, 0.25f, 2.0f };
    const float weight_values[] = { 1.0f, 1.25f, 0.75f, 1.5f };

    std::vector<uint16_t> h_x(static_cast<size_t>(kTokens) * kGdnVHeads * kGdnValueDim, kBf16Zero);
    std::vector<uint16_t> h_gate(h_x.size(), kBf16Zero);
    std::vector<float> h_weight(kGdnValueDim, 1.0f);
    std::vector<uint16_t> h_weight_bf16(kGdnValueDim, kBf16One);

    float sum_squares = 0.0f;
    for (int i = 0; i < 4; ++i) {
        qscu_test_store_bf16(h_x, gdn_v_offset(0, kHead, dims[i]), x_values[i]);
        qscu_test_store_bf16(h_gate, gdn_v_offset(0, kHead, dims[i]), gate_values[i]);
        h_weight[dims[i]] = weight_values[i];
        h_weight_bf16[dims[i]] = qscu_test_f32_to_bf16(weight_values[i]);
        const float x_bf16 = bf16_to_f32(h_x[gdn_v_offset(0, kHead, dims[i])]);
        sum_squares += x_bf16 * x_bf16;
    }

    uint16_t* d_x = nullptr;
    uint16_t* d_gate = nullptr;
    uint16_t* d_out = nullptr;
    uint16_t* d_weight_bf16 = nullptr;
    float* d_weight = nullptr;

    const bool ok = copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qscu rms x")
        && copy_to_device(&d_gate, h_gate.data(), h_gate.size(), "copy qscu rms gate")
        && copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy qscu rms weight")
        && copy_to_device(
                        &d_weight_bf16,
                        h_weight_bf16.data(),
                        h_weight_bf16.size(),
                        "copy qscu rms bf16 weight"
        )
        && alloc_device(&d_out, h_x.size(), "alloc qscu rms out");

    if (ok) {
        qscu_qwen36_gdn_rmsnorm_gated_desc desc {};
        desc.x = gdn_tensor3_bf16(d_x, kTokens, kGdnVHeads, kGdnValueDim);
        desc.gate = gdn_tensor3_bf16(d_gate, kTokens, kGdnVHeads, kGdnValueDim);
        desc.weight = gdn_tensor1_f32(d_weight, kGdnValueDim);
        desc.out = gdn_tensor3_bf16(d_out, kTokens, kGdnVHeads, kGdnValueDim);
        desc.num_tokens = kTokens;
        desc.eps = kEps;
        desc.gate_activation = QSCU_ACTIVATION_SILU;

        check_status(
            qscu_qwen36_gdn_rmsnorm_gated_bf16(&desc, nullptr),
            QSFI_STATUS_OK,
            "qscu rmsnorm gated"
        );
        check_cuda(cudaDeviceSynchronize(), "sync qscu rmsnorm gated");

        std::vector<uint16_t> h_out(h_x.size());
        if (check_cuda(
                cudaMemcpy(
                    h_out.data(),
                    d_out,
                    h_out.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu rms out"
            )) {
            const float rstd
                = 1.0f / std::sqrt(sum_squares / static_cast<float>(kGdnValueDim) + kEps);
            for (int i = 0; i < 4; ++i) {
                const size_t offset = gdn_v_offset(0, kHead, dims[i]);
                const float x = bf16_to_f32(h_x[offset]);
                const float gate = bf16_to_f32(h_gate[offset]);
                qscu_test_check_bf16_near(
                    h_out,
                    offset,
                    x * rstd * h_weight[dims[i]] * qscu_test_silu(gate),
                    0.04f,
                    "qscu rms out"
                );
            }
            qscu_test_check_bf16_near(
                h_out,
                gdn_v_offset(0, kHead, 12),
                0.0f,
                0.0f,
                "qscu rms inactive out"
            );
        }

        desc.weight = qscu_test_tensor1_bf16(d_weight_bf16, kGdnValueDim);
        desc.gate_activation = QSCU_ACTIVATION_SIGMOID;
        check_status(
            qscu_qwen36_gdn_rmsnorm_gated_bf16(&desc, nullptr),
            QSFI_STATUS_OK,
            "qscu rmsnorm gated sigmoid bf16 weight"
        );
        check_cuda(cudaDeviceSynchronize(), "sync qscu rmsnorm gated sigmoid");
        if (check_cuda(
                cudaMemcpy(
                    h_out.data(),
                    d_out,
                    h_out.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu rms sigmoid out"
            )) {
            const float rstd
                = 1.0f / std::sqrt(sum_squares / static_cast<float>(kGdnValueDim) + kEps);
            for (int i = 0; i < 4; ++i) {
                const size_t offset = gdn_v_offset(0, kHead, dims[i]);
                const float x = bf16_to_f32(h_x[offset]);
                const float gate = bf16_to_f32(h_gate[offset]);
                const float weight = bf16_to_f32(h_weight_bf16[dims[i]]);
                qscu_test_check_bf16_near(
                    h_out,
                    offset,
                    x * rstd * weight * qscu_test_sigmoid(gate),
                    0.04f,
                    "qscu rms sigmoid out"
                );
            }
        }
    }

    cudaFree(d_x);
    cudaFree(d_gate);
    cudaFree(d_weight);
    cudaFree(d_weight_bf16);
    cudaFree(d_out);
}

void test_qscu_router_topk_cpu_reference()
{
    constexpr int kTokens = 2;
    constexpr int kExperts = 6;
    constexpr int kTopK = 3;
    constexpr float kScale = 1.25f;

    const std::vector<float> h_logits {
        0.1f, 2.0f, 2.0f, -1.0f, 3.0f, 0.5f, -2.0f, 1.0f, 1.0f, 1.0f, 0.0f, -3.0f,
    };

    float* d_logits = nullptr;
    int32_t* d_ids = nullptr;
    float* d_weights = nullptr;

    const bool ok = copy_to_device(&d_logits, h_logits.data(), h_logits.size(), "copy qscu router")
        && alloc_device(&d_ids, static_cast<size_t>(kTokens) * kTopK, "alloc qscu router ids")
        && alloc_device(&d_weights,
                        static_cast<size_t>(kTokens) * kTopK,
                        "alloc qscu router weights");

    if (ok) {
        qscu_router_topk_desc desc {};
        desc.logits = tensor2_f32(d_logits, kTokens, kExperts);
        desc.topk_ids = tensor2_i32(d_ids, kTokens, kTopK);
        desc.topk_weights = tensor2_f32(d_weights, kTokens, kTopK);
        desc.num_tokens = kTokens;
        desc.num_experts = kExperts;
        desc.top_k = kTopK;
        desc.score = QSCU_ROUTER_SCORE_SOFTMAX;
        desc.renormalize = 1;
        desc.routed_scaling_factor = kScale;

        check_status(qscu_router_topk(&desc, nullptr), QSFI_STATUS_OK, "qscu router softmax");
        check_cuda(cudaDeviceSynchronize(), "sync qscu router softmax");

        std::vector<int32_t> h_ids(static_cast<size_t>(kTokens) * kTopK);
        std::vector<float> h_weights(static_cast<size_t>(kTokens) * kTopK);
        if (check_cuda(
                cudaMemcpy(
                    h_ids.data(),
                    d_ids,
                    h_ids.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu router ids"
            )
            && check_cuda(
                cudaMemcpy(
                    h_weights.data(),
                    d_weights,
                    h_weights.size() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "copy qscu router weights"
            )) {
            for (int row = 0; row < kTokens; ++row) {
                const QscuTestRouterRef ref = qscu_test_router_reference(
                    h_logits,
                    row,
                    kExperts,
                    kTopK,
                    QSCU_ROUTER_SCORE_SOFTMAX,
                    true,
                    kScale
                );
                for (int pos = 0; pos < kTopK; ++pos) {
                    qscu_test_check_i32_eq(
                        h_ids[row * kTopK + pos],
                        ref.ids[pos],
                        "qscu router softmax id"
                    );
                    qscu_test_check_f32_near(
                        h_weights[row * kTopK + pos],
                        ref.weights[pos],
                        1.0e-5f,
                        "qscu router softmax weight"
                    );
                }
            }
        }
    }

    cudaFree(d_logits);
    cudaFree(d_ids);
    cudaFree(d_weights);

    constexpr int kSigmoidExperts = 5;
    constexpr int kSigmoidTopK = 2;
    constexpr float kSigmoidScale = 0.75f;
    std::vector<uint16_t> h_sigmoid_logits_bf16(kSigmoidExperts);
    const float h_sigmoid_values[kSigmoidExperts] = { 0.0f, 1.0f, 1.0f, -2.0f, 0.5f };
    std::vector<float> h_sigmoid_logits(kSigmoidExperts);
    for (int expert = 0; expert < kSigmoidExperts; ++expert) {
        h_sigmoid_logits_bf16[expert] = qscu_test_f32_to_bf16(h_sigmoid_values[expert]);
        h_sigmoid_logits[expert] = bf16_to_f32(h_sigmoid_logits_bf16[expert]);
    }

    uint16_t* d_sigmoid_logits = nullptr;
    int32_t* d_sigmoid_ids = nullptr;
    float* d_sigmoid_weights = nullptr;
    const bool sigmoid_ok = copy_to_device(
                                &d_sigmoid_logits,
                                h_sigmoid_logits_bf16.data(),
                                h_sigmoid_logits_bf16.size(),
                                "copy qscu router sigmoid"
                            )
        && alloc_device(&d_sigmoid_ids, kSigmoidTopK, "alloc qscu router sigmoid ids")
        && alloc_device(&d_sigmoid_weights, kSigmoidTopK, "alloc qscu router sigmoid weights");

    if (sigmoid_ok) {
        qscu_router_topk_desc desc {};
        desc.logits = tensor2_bf16(d_sigmoid_logits, 1, kSigmoidExperts);
        desc.topk_ids = tensor2_i32(d_sigmoid_ids, 1, kSigmoidTopK);
        desc.topk_weights = tensor2_f32(d_sigmoid_weights, 1, kSigmoidTopK);
        desc.num_tokens = 1;
        desc.num_experts = kSigmoidExperts;
        desc.top_k = kSigmoidTopK;
        desc.score = QSCU_ROUTER_SCORE_SIGMOID;
        desc.renormalize = 0;
        desc.routed_scaling_factor = kSigmoidScale;

        check_status(qscu_router_topk(&desc, nullptr), QSFI_STATUS_OK, "qscu router sigmoid");
        check_cuda(cudaDeviceSynchronize(), "sync qscu router sigmoid");

        int32_t h_ids[kSigmoidTopK] {};
        float h_weights[kSigmoidTopK] {};
        if (check_cuda(
                cudaMemcpy(h_ids, d_sigmoid_ids, sizeof(h_ids), cudaMemcpyDeviceToHost),
                "copy qscu router sigmoid ids"
            )
            && check_cuda(
                cudaMemcpy(h_weights, d_sigmoid_weights, sizeof(h_weights), cudaMemcpyDeviceToHost),
                "copy qscu router sigmoid weights"
            )) {
            const QscuTestRouterRef ref = qscu_test_router_reference(
                h_sigmoid_logits,
                0,
                kSigmoidExperts,
                kSigmoidTopK,
                QSCU_ROUTER_SCORE_SIGMOID,
                false,
                kSigmoidScale
            );
            for (int pos = 0; pos < kSigmoidTopK; ++pos) {
                qscu_test_check_i32_eq(h_ids[pos], ref.ids[pos], "qscu router sigmoid id");
                qscu_test_check_f32_near(
                    h_weights[pos],
                    ref.weights[pos],
                    1.0e-5f,
                    "qscu router sigmoid weight"
                );
            }
        }
    }

    cudaFree(d_sigmoid_logits);
    cudaFree(d_sigmoid_ids);
    cudaFree(d_sigmoid_weights);
}

#if QSFI_ENABLE_CHECKED_VALIDATION

void test_qscu_qwen36_gdn_causal_conv1d_bf16_checked_validation()
{
    constexpr int kTokens = 2;
    constexpr int kBatchSize = 2;
    constexpr int kStateSlots = 1;

    std::vector<uint16_t> h_x(static_cast<size_t>(kTokens) * kQscuTestGdnPackedDim, kBf16Zero);
    std::vector<uint16_t> h_weight(
        static_cast<size_t>(kQscuTestGdnPackedDim) * kQscuTestConvWidth,
        kBf16Zero
    );
    std::vector<float> h_state(
        static_cast<size_t>(kStateSlots) * kQscuTestGdnPackedDim * kQscuTestConvState,
        0.0f
    );
    const int32_t h_seq_indptr[kBatchSize + 1] = { 0, 1, 2 };
    const int32_t h_bad_seq_indptr[kBatchSize + 1] = { 0, 2, 1 };
    const int32_t h_indices[kBatchSize] = { 0, 0 };
    const int32_t h_bad_read_indices[kBatchSize] = { kStateSlots, 0 };

    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    float* d_state = nullptr;
    int32_t* d_read_indices = nullptr;
    int32_t* d_write_indices = nullptr;
    int32_t* d_seq_indptr = nullptr;

    const bool ok = copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qscu conv checked x")
        && copy_to_device(
                        &d_weight,
                        h_weight.data(),
                        h_weight.size(),
                        "copy qscu conv checked weight"
        )
        && copy_to_device(
                        &d_state,
                        h_state.data(),
                        h_state.size(),
                        "copy qscu conv checked state"
        )
        && copy_to_device(
                        &d_read_indices,
                        h_indices,
                        kBatchSize,
                        "copy qscu conv checked read indices"
        )
        && copy_to_device(
                        &d_write_indices,
                        h_indices,
                        kBatchSize,
                        "copy qscu conv checked write indices"
        )
        && copy_to_device(
                        &d_seq_indptr,
                        h_seq_indptr,
                        kBatchSize + 1,
                        "copy qscu conv checked seq indptr"
        )
        && alloc_device(
                        &d_out,
                        static_cast<size_t>(kTokens) * kQscuTestGdnPackedDim,
                        "alloc qscu conv checked out"
        );

    if (ok) {
        qscu_qwen36_gdn_causal_conv1d_desc desc {};
        desc.x = tensor2_bf16(d_x, kTokens, kQscuTestGdnPackedDim);
        desc.weight = tensor2_bf16(d_weight, kQscuTestGdnPackedDim, kQscuTestConvWidth);
        desc.state = qscu_test_tensor3_f32(
            d_state,
            kStateSlots,
            kQscuTestGdnPackedDim,
            kQscuTestConvState
        );
        desc.state_read_indices = gdn_tensor1_i32(d_read_indices, kBatchSize);
        desc.state_write_indices = gdn_tensor1_i32(d_write_indices, kBatchSize);
        desc.seq_indptr = d_seq_indptr;
        desc.out = tensor2_bf16(d_out, kTokens, kQscuTestGdnPackedDim);
        desc.num_tokens = kTokens;
        desc.batch_size = kBatchSize;
        desc.activation = QSCU_ACTIVATION_SILU;
        desc.update_state = 1;

        if (check_cuda(
                cudaMemcpy(
                    d_read_indices,
                    h_bad_read_indices,
                    sizeof(h_bad_read_indices),
                    cudaMemcpyHostToDevice
                ),
                "copy qscu conv checked bad read index"
            )) {
            check_status(
                qscu_qwen36_gdn_causal_conv1d_bf16(&desc, nullptr),
                QSFI_STATUS_INVALID_ARGUMENT,
                "qscu conv rejects invalid state index"
            );
        }

        if (check_cuda(
                cudaMemcpy(d_read_indices, h_indices, sizeof(h_indices), cudaMemcpyHostToDevice),
                "restore qscu conv checked read indices"
            )
            && check_cuda(
                cudaMemcpy(
                    d_seq_indptr,
                    h_bad_seq_indptr,
                    sizeof(h_bad_seq_indptr),
                    cudaMemcpyHostToDevice
                ),
                "copy qscu conv checked bad seq indptr"
            )) {
            check_status(
                qscu_qwen36_gdn_causal_conv1d_bf16(&desc, nullptr),
                QSFI_STATUS_INVALID_ARGUMENT,
                "qscu conv rejects invalid seq indptr"
            );
        }
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_state);
    cudaFree(d_read_indices);
    cudaFree(d_write_indices);
    cudaFree(d_seq_indptr);
    cudaFree(d_out);
}

#endif

void test_qscu_gdn_router_validation_errors()
{
    check_status(
        qscu_qwen36_gdn_causal_conv1d_bf16(nullptr, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu conv null desc"
    );
    check_status(
        qscu_qwen36_gdn_post_conv_prepare_bf16(nullptr, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu post null desc"
    );
    check_status(
        qscu_qwen36_gdn_rmsnorm_gated_bf16(nullptr, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu rms null desc"
    );
    check_status(
        qscu_router_topk(nullptr, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu router null"
    );

    qscu_qwen36_gdn_causal_conv1d_desc conv = qscu_test_valid_conv_validation_desc();
    conv.activation = QSCU_ACTIVATION_SIGMOID;
    check_status(
        qscu_qwen36_gdn_causal_conv1d_bf16(&conv, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu conv unsupported activation"
    );

    qscu_qwen36_gdn_post_conv_prepare_desc post = qscu_test_valid_post_validation_desc();
    post.forget_gate_output = static_cast<qscu_gdn_forget_gate_output>(99);
    check_status(
        qscu_qwen36_gdn_post_conv_prepare_bf16(&post, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu post unsupported forget output"
    );

    qscu_qwen36_gdn_rmsnorm_gated_desc rms = qscu_test_valid_rms_validation_desc();
    rms.eps = 0.0f;
    check_status(
        qscu_qwen36_gdn_rmsnorm_gated_bf16(&rms, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu rms invalid eps"
    );

    rms = qscu_test_valid_rms_validation_desc();
    rms.gate_activation = QSCU_ACTIVATION_NONE;
    check_status(
        qscu_qwen36_gdn_rmsnorm_gated_bf16(&rms, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu rms unsupported activation"
    );

    qscu_router_topk_desc router = qscu_test_valid_router_validation_desc(1, 2);
    check_status(
        qscu_router_topk(&router, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu router topk exceeds experts"
    );

    router = qscu_test_valid_router_validation_desc(2, 1);
    router.score = static_cast<qscu_router_score>(99);
    check_status(
        qscu_router_topk(&router, nullptr),
        QSFI_STATUS_UNSUPPORTED,
        "qscu router unsupported score"
    );

    router = qscu_test_valid_router_validation_desc(2, 1);
    router.routed_scaling_factor = 0.0f;
    check_status(
        qscu_router_topk(&router, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscu router invalid scale"
    );
}

void test_qscu_gdn_router_helpers()
{
    test_qscu_qwen36_gdn_causal_conv1d_bf16_cpu_reference();
    test_qscu_qwen36_gdn_post_conv_prepare_bf16_cpu_reference();
    test_qscu_qwen36_gdn_rmsnorm_gated_bf16_cpu_reference();
    test_qscu_router_topk_cpu_reference();
#if QSFI_ENABLE_CHECKED_VALIDATION
    test_qscu_qwen36_gdn_causal_conv1d_bf16_checked_validation();
#endif
    test_qscu_gdn_router_validation_errors();
}

#endif
