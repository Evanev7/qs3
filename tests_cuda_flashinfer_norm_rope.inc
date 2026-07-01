static uint16_t norm_rope_f32_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

static qsfi_tensor1 norm_rope_tensor1_bf16(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

static qsfi_tensor1 norm_rope_tensor1_f32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

static qsfi_tensor1 norm_rope_tensor1_i32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_I32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

static qsfi_tensor1 norm_rope_tensor1_u32(void* data, int64_t n)
{
    qsfi_tensor1 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_U32;
    tensor.shape[0] = n;
    tensor.stride[0] = 1;
    return tensor;
}

static qsfi_tensor3 norm_rope_tensor3_f32(void* data, int64_t d0, int64_t d1, int64_t d2)
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

static std::vector<uint16_t> norm_rope_bf16_from_f32(const std::vector<float>& values)
{
    std::vector<uint16_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        out[i] = norm_rope_f32_to_bf16(values[i]);
    return out;
}

static void norm_rope_check_bf16_close(
    const std::vector<uint16_t>& got,
    const std::vector<float>& want,
    float abs_tol,
    const char* what
)
{
    if (got.size() != want.size()) {
        std::fprintf(
            stderr,
            "FAIL: %s size mismatch: got %zu, want %zu\n",
            what,
            got.size(),
            want.size()
        );
        failures += 1;
        return;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const float got_f32 = bf16_to_f32(got[i]);
        if (std::fabs(got_f32 - want[i]) > abs_tol) {
            std::fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.8f (0x%04x), want %.8f +/- %.8f\n",
                what,
                i,
                got_f32,
                static_cast<unsigned>(got[i]),
                want[i],
                abs_tol
            );
            failures += 1;
            return;
        }
    }
}

static void norm_rope_check_f32_close(
    const std::vector<float>& got, const std::vector<float>& want, float abs_tol, const char* what
)
{
    if (got.size() != want.size()) {
        std::fprintf(
            stderr,
            "FAIL: %s size mismatch: got %zu, want %zu\n",
            what,
            got.size(),
            want.size()
        );
        failures += 1;
        return;
    }
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

static std::vector<float> norm_rope_rmsnorm_reference(
    const std::vector<uint16_t>& x,
    const std::vector<uint16_t>& weight,
    int rows,
    int hidden,
    float eps,
    float weight_bias = 0.0f
)
{
    std::vector<float> out(x.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < hidden; ++col) {
            const float value = bf16_to_f32(x[static_cast<size_t>(row) * hidden + col]);
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            out[offset] = bf16_to_f32(x[offset]) * inv_rms * (weight_bias + bf16_to_f32(weight[col]));
        }
    }
    return out;
}

static std::vector<float> norm_rope_rmsnorm_reference(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    int rows,
    int hidden,
    float eps,
    float weight_bias = 0.0f
)
{
    std::vector<float> out(x.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < hidden; ++col) {
            const float value = x[static_cast<size_t>(row) * hidden + col];
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            out[offset] = x[offset] * inv_rms * (weight_bias + weight[col]);
        }
    }
    return out;
}

static void norm_rope_fused_add_rmsnorm_reference(
    const std::vector<uint16_t>& x,
    const std::vector<uint16_t>& residual,
    const std::vector<uint16_t>& weight,
    int rows,
    int hidden,
    float eps,
    std::vector<float>& norm_out,
    std::vector<float>& residual_out
)
{
    norm_out.assign(x.size(), 0.0f);
    residual_out.assign(x.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            const float value = bf16_to_f32(x[offset]) + bf16_to_f32(residual[offset]);
            residual_out[offset] = value;
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            norm_out[offset] = residual_out[offset] * inv_rms * bf16_to_f32(weight[col]);
        }
    }
}

static void norm_rope_fused_add_rmsnorm_reference(
    const std::vector<float>& x,
    const std::vector<float>& residual,
    const std::vector<float>& weight,
    int rows,
    int hidden,
    float eps,
    std::vector<float>& norm_out,
    std::vector<float>& residual_out
)
{
    norm_out.assign(x.size(), 0.0f);
    residual_out.assign(x.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            const float value = x[offset] + residual[offset];
            residual_out[offset] = value;
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
        for (int col = 0; col < hidden; ++col) {
            const size_t offset = static_cast<size_t>(row) * hidden + col;
            norm_out[offset] = residual_out[offset] * inv_rms * weight[col];
        }
    }
}

static size_t norm_rope_q_offset(int token, int head, int dim, int num_heads, int head_dim)
{
    return (static_cast<size_t>(token) * num_heads + head) * head_dim + dim;
}

static void norm_rope_check_bf16_tail_unchanged(
    const std::vector<uint16_t>& input,
    const std::vector<uint16_t>& got,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    const char* what
)
{
    for (int token = 0; token < num_tokens; ++token) {
        for (int head = 0; head < num_heads; ++head) {
            for (int dim = rotary_dim; dim < head_dim; ++dim) {
                const size_t offset = norm_rope_q_offset(token, head, dim, num_heads, head_dim);
                if (got[offset] != input[offset]) {
                    std::fprintf(
                        stderr,
                        "FAIL: %s token=%d head=%d dim=%d: got 0x%04x, want input 0x%04x\n",
                        what,
                        token,
                        head,
                        dim,
                        static_cast<unsigned>(got[offset]),
                        static_cast<unsigned>(input[offset])
                    );
                    failures += 1;
                    return;
                }
            }
        }
    }
}

static void norm_rope_reference_tensor(
    const std::vector<uint16_t>& input,
    const std::vector<int32_t>& positions,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    float rope_scale,
    float rope_theta,
    std::vector<float>& out
)
{
    out.assign(input.size(), 0.0f);
    const int half = rotary_dim / 2;
    const float scale = rope_scale == 0.0f ? 1.0f : rope_scale;
    const float theta = rope_theta == 0.0f ? 10000.0f : rope_theta;
    for (int token = 0; token < num_tokens; ++token) {
        for (int head = 0; head < num_heads; ++head) {
            for (int dim = 0; dim < half; ++dim) {
                const float freq
                    = std::pow(theta, -2.0f * static_cast<float>(dim) / rotary_dim) / scale;
                const float angle = static_cast<float>(positions[token]) * freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const size_t lo = norm_rope_q_offset(token, head, dim, num_heads, head_dim);
                const size_t hi = norm_rope_q_offset(token, head, dim + half, num_heads, head_dim);
                const float x_lo = bf16_to_f32(input[lo]);
                const float x_hi = bf16_to_f32(input[hi]);
                out[lo] = x_lo * c - x_hi * s;
                out[hi] = x_hi * c + x_lo * s;
            }
            for (int dim = rotary_dim; dim < head_dim; ++dim) {
                const size_t offset = norm_rope_q_offset(token, head, dim, num_heads, head_dim);
                out[offset] = bf16_to_f32(input[offset]);
            }
        }
    }
}

static void norm_rope_reference_tensor(
    const std::vector<float>& input,
    const std::vector<int32_t>& positions,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    float rope_scale,
    float rope_theta,
    std::vector<float>& out
)
{
    out.assign(input.size(), 0.0f);
    const int half = rotary_dim / 2;
    const float scale = rope_scale == 0.0f ? 1.0f : rope_scale;
    const float theta = rope_theta == 0.0f ? 10000.0f : rope_theta;
    for (int token = 0; token < num_tokens; ++token) {
        for (int head = 0; head < num_heads; ++head) {
            for (int dim = 0; dim < half; ++dim) {
                const float freq
                    = std::pow(theta, -2.0f * static_cast<float>(dim) / rotary_dim) / scale;
                const float angle = static_cast<float>(positions[token]) * freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const size_t lo = norm_rope_q_offset(token, head, dim, num_heads, head_dim);
                const size_t hi = norm_rope_q_offset(token, head, dim + half, num_heads, head_dim);
                const float x_lo = input[lo];
                const float x_hi = input[hi];
                out[lo] = x_lo * c - x_hi * s;
                out[hi] = x_hi * c + x_lo * s;
            }
            for (int dim = rotary_dim; dim < head_dim; ++dim) {
                const size_t offset = norm_rope_q_offset(token, head, dim, num_heads, head_dim);
                out[offset] = input[offset];
            }
        }
    }
}

static void test_qsfi_rmsnorm_f32_matches_cpu()
{
    constexpr int rows = 2;
    constexpr int hidden = 8;
    constexpr float eps = 1.0e-6f;

    const std::vector<float> h_x = {
        -1.0f, 0.5f,  1.5f,  -2.5f,  3.0f,  -0.25f, 0.75f, 2.25f,
        2.0f,  -1.5f, 0.25f, -0.75f, 1.25f, -3.0f,  0.5f,  1.75f,
    };
    const std::vector<float> h_weight = { 0.5f, -1.25f, 1.5f, 0.75f, -0.5f, 2.0f, -1.0f, 1.25f };
    const std::vector<float> want = norm_rope_rmsnorm_reference(h_x, h_weight, rows, hidden, eps);

    qsfi_context* ctx = nullptr;
    float* d_x = nullptr;
    float* d_weight = nullptr;
    float* d_out = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy rmsnorm f32 x")
        || !copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy rmsnorm f32 weight")
        || !alloc_device(&d_out, h_x.size(), "alloc rmsnorm f32 out")) {
        cudaFree(d_x);
        cudaFree(d_weight);
        cudaFree(d_out);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_rmsnorm_desc desc {};
    desc.x = tensor2_f32(d_x, rows, hidden);
    desc.weight = norm_rope_tensor1_f32(d_weight, hidden);
    desc.out = tensor2_f32(d_out, rows, hidden);
    desc.hidden_size = hidden;
    desc.eps = eps;
    check_status(qsfi_rmsnorm(ctx, &desc), QSFI_STATUS_OK, "rmsnorm f32");
    check_cuda(cudaDeviceSynchronize(), "sync rmsnorm f32");

    std::vector<float> h_out(h_x.size(), 0.0f);
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy rmsnorm f32 out"
    );
    norm_rope_check_f32_close(h_out, want, 0.00002f, "rmsnorm f32 out");

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    qsfi_context_destroy(ctx);
}

static void test_qsfi_rmsnorm_bf16_qwen_qk_norm_flattened_per_head_matches_cpu()
{
    constexpr int rows = 2;
    constexpr int heads = 3;
    constexpr int head_dim = 256;
    constexpr int flat_rows = rows * heads;
    constexpr float eps = 1.0e-6f;

    std::vector<float> x_f32(flat_rows * head_dim);
    for (size_t i = 0; i < x_f32.size(); ++i) {
        const int centered = static_cast<int>(i % 37) - 18;
        const int head = static_cast<int>((i / head_dim) % heads);
        x_f32[i] = centered * 0.03125f + head * 0.0078125f;
    }
    const std::vector<uint16_t> h_x = norm_rope_bf16_from_f32(x_f32);
    const std::vector<uint16_t> h_weight(head_dim, norm_rope_f32_to_bf16(0.0f));
    const std::vector<float> want
        = norm_rope_rmsnorm_reference(h_x, h_weight, flat_rows, head_dim, eps, 1.0f);

    qsfi_context* ctx = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    uint16_t* d_inplace = nullptr;
    if (!make_context(&ctx)
        || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qk rmsnorm x")
        || !copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy qk rmsnorm weight")
        || !alloc_device(&d_out, h_x.size(), "alloc qk rmsnorm out")
        || !copy_to_device(&d_inplace, h_x.data(), h_x.size(), "copy qk rmsnorm inplace")) {
        cudaFree(d_x);
        cudaFree(d_weight);
        cudaFree(d_out);
        cudaFree(d_inplace);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_rmsnorm_desc desc {};
    desc.x = tensor2_bf16(d_x, flat_rows, head_dim);
    desc.weight = norm_rope_tensor1_bf16(d_weight, head_dim);
    desc.out = tensor2_bf16(d_out, flat_rows, head_dim);
    desc.hidden_size = head_dim;
    desc.weight_bias = 1.0f;
    desc.eps = eps;
    check_status(qsfi_rmsnorm(ctx, &desc), QSFI_STATUS_OK, "qwen qk rmsnorm flattened");
    check_cuda(cudaDeviceSynchronize(), "sync qwen qk rmsnorm flattened");

    std::vector<uint16_t> h_out(h_x.size(), 0);
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qwen qk rmsnorm out"
    );
    norm_rope_check_bf16_close(h_out, want, 0.025f, "qwen qk rmsnorm flattened out");

    desc.x = tensor2_bf16(d_inplace, flat_rows, head_dim);
    desc.out = desc.x;
    check_status(qsfi_rmsnorm(ctx, &desc), QSFI_STATUS_OK, "qwen qk rmsnorm inplace");
    check_cuda(cudaDeviceSynchronize(), "sync qwen qk rmsnorm inplace");

    std::vector<uint16_t> h_inplace(h_x.size(), 0);
    check_cuda(
        cudaMemcpy(
            h_inplace.data(),
            d_inplace,
            h_inplace.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy qwen qk rmsnorm inplace"
    );
    norm_rope_check_bf16_close(h_inplace, want, 0.025f, "qwen qk rmsnorm inplace");

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    cudaFree(d_inplace);
    qsfi_context_destroy(ctx);
}

static void norm_rope_test_rmsnorm_rejects_invalid_desc_shape_dtype()
{
    constexpr int rows = 1;
    constexpr int hidden = 8;
    const std::vector<uint16_t> h_x
        = norm_rope_bf16_from_f32({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
    const std::vector<uint16_t> h_weight
        = norm_rope_bf16_from_f32({ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

    qsfi_context* ctx = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy rms invalid x")
        || !copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy rms invalid weight")
        || !alloc_device(&d_out, h_x.size(), "alloc rms invalid out")) {
        cudaFree(d_x);
        cudaFree(d_weight);
        cudaFree(d_out);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_rmsnorm_desc desc {};
    desc.x = tensor2_bf16(d_x, rows, hidden);
    desc.weight = norm_rope_tensor1_bf16(d_weight, hidden);
    desc.out = tensor2_bf16(d_out, rows, hidden);
    desc.hidden_size = hidden;
    desc.eps = 1.0e-6f;

    check_status(qsfi_rmsnorm(ctx, nullptr), QSFI_STATUS_INVALID_ARGUMENT, "rmsnorm rejects null");

    qsfi_rmsnorm_desc bad = desc;
    bad.hidden_size = 0;
    check_status(
        qsfi_rmsnorm(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rmsnorm rejects zero hidden"
    );

    bad = desc;
    bad.weight.shape[0] = hidden - 1;
    check_status(
        qsfi_rmsnorm(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rmsnorm rejects weight shape"
    );

    bad = desc;
    bad.x.dtype = QSFI_DTYPE_F16;
    check_status(
        qsfi_rmsnorm(ctx, &bad),
        QSFI_STATUS_UNSUPPORTED,
        "rmsnorm rejects unsupported dtype"
    );

    bad = desc;
    bad.eps = 0.0f;
    check_status(
        qsfi_rmsnorm(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rmsnorm rejects non-positive eps"
    );

    bad = desc;
    bad.weight_bias = 0.5f;
    check_status(
        qsfi_rmsnorm(ctx, &bad),
        QSFI_STATUS_UNSUPPORTED,
        "rmsnorm rejects unsupported weight_bias"
    );

    bad = desc;
    bad.weight_bias = std::numeric_limits<float>::quiet_NaN();
    check_status(
        qsfi_rmsnorm(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rmsnorm rejects non-finite weight_bias"
    );

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    qsfi_context_destroy(ctx);
}

static void test_qsfi_fused_add_rmsnorm_f32_inplace_updates_residual()
{
    constexpr int rows = 2;
    constexpr int hidden = 8;
    constexpr float eps = 1.0e-6f;

    const std::vector<float> h_x = {
        0.25f, -0.75f, 1.25f, -1.75f, 2.25f,  -2.75f, 0.5f, 1.5f,
        -1.0f, 2.0f,   -0.5f, 0.75f,  -1.25f, 1.75f,  2.5f, -2.0f,
    };
    const std::vector<float> h_residual = {
        1.0f, 0.5f,  -0.25f, 0.75f,  -1.0f, 1.25f, -1.5f, 0.25f,
        0.5f, -1.5f, 1.0f,   -0.75f, 1.5f,  -0.5f, -1.0f, 0.75f,
    };
    const std::vector<float> h_weight = { 1.25f, -0.5f, 0.75f, 1.5f, -1.0f, 0.5f, 2.0f, -1.25f };

    std::vector<float> want_norm;
    std::vector<float> want_residual;
    norm_rope_fused_add_rmsnorm_reference(
        h_x,
        h_residual,
        h_weight,
        rows,
        hidden,
        eps,
        want_norm,
        want_residual
    );

    qsfi_context* ctx = nullptr;
    float* d_x = nullptr;
    float* d_residual = nullptr;
    float* d_weight = nullptr;
    if (!make_context(&ctx)
        || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy fused add rmsnorm f32 x")
        || !copy_to_device(
            &d_residual,
            h_residual.data(),
            h_residual.size(),
            "copy fused add rmsnorm f32 residual"
        )
        || !copy_to_device(
            &d_weight,
            h_weight.data(),
            h_weight.size(),
            "copy fused add rmsnorm f32 weight"
        )) {
        cudaFree(d_x);
        cudaFree(d_residual);
        cudaFree(d_weight);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_fused_add_rmsnorm_desc desc {};
    desc.x = tensor2_f32(d_x, rows, hidden);
    desc.residual_inout = tensor2_f32(d_residual, rows, hidden);
    desc.weight = norm_rope_tensor1_f32(d_weight, hidden);
    desc.out = desc.x;
    desc.hidden_size = hidden;
    desc.eps = eps;
    check_status(qsfi_fused_add_rmsnorm(ctx, &desc), QSFI_STATUS_OK, "fused add rmsnorm f32");
    check_cuda(cudaDeviceSynchronize(), "sync fused add rmsnorm f32");

    std::vector<float> h_out(h_x.size(), 0.0f);
    std::vector<float> h_residual_out(h_residual.size(), 0.0f);
    check_cuda(
        cudaMemcpy(h_out.data(), d_x, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy fused add rmsnorm f32 x out"
    );
    check_cuda(
        cudaMemcpy(
            h_residual_out.data(),
            d_residual,
            h_residual_out.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy fused add rmsnorm f32 residual out"
    );
    norm_rope_check_f32_close(h_out, want_norm, 0.00002f, "fused add rmsnorm f32 out");
    norm_rope_check_f32_close(
        h_residual_out,
        want_residual,
        0.00001f,
        "fused add rmsnorm f32 residual"
    );

    cudaFree(d_x);
    cudaFree(d_residual);
    cudaFree(d_weight);
    qsfi_context_destroy(ctx);
}

static void test_qsfi_fused_add_rmsnorm_rejects_invalid_desc_shape_dtype()
{
    constexpr int rows = 1;
    constexpr int hidden = 8;
    const std::vector<uint16_t> h_x
        = norm_rope_bf16_from_f32({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
    const std::vector<uint16_t> h_residual
        = norm_rope_bf16_from_f32({ 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f, -2.0f });
    const std::vector<uint16_t> h_weight
        = norm_rope_bf16_from_f32({ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

    qsfi_context* ctx = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_residual = nullptr;
    uint16_t* d_weight = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy fused invalid x")
        || !copy_to_device(
            &d_residual,
            h_residual.data(),
            h_residual.size(),
            "copy fused invalid residual"
        )
        || !copy_to_device(
            &d_weight,
            h_weight.data(),
            h_weight.size(),
            "copy fused invalid weight"
        )) {
        cudaFree(d_x);
        cudaFree(d_residual);
        cudaFree(d_weight);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_fused_add_rmsnorm_desc desc {};
    desc.x = tensor2_bf16(d_x, rows, hidden);
    desc.residual_inout = tensor2_bf16(d_residual, rows, hidden);
    desc.weight = norm_rope_tensor1_bf16(d_weight, hidden);
    desc.out = desc.x;
    desc.hidden_size = hidden;
    desc.eps = 1.0e-6f;

    check_status(
        qsfi_fused_add_rmsnorm(ctx, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "fused add rmsnorm rejects null"
    );

    qsfi_fused_add_rmsnorm_desc bad = desc;
    bad.residual_inout.shape[0] = rows + 1;
    check_status(
        qsfi_fused_add_rmsnorm(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "fused add rmsnorm rejects residual shape"
    );

    bad = desc;
    bad.x.dtype = QSFI_DTYPE_F16;
    check_status(
        qsfi_fused_add_rmsnorm(ctx, &bad),
        QSFI_STATUS_UNSUPPORTED,
        "fused add rmsnorm rejects unsupported dtype"
    );

    bad = desc;
    bad.eps = 0.0f;
    check_status(
        qsfi_fused_add_rmsnorm(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "fused add rmsnorm rejects non-positive eps"
    );

    cudaFree(d_x);
    cudaFree(d_residual);
    cudaFree(d_weight);
    qsfi_context_destroy(ctx);
}

static void test_qsfi_rope_apply_f32_u32_positions_matches_cpu()
{
    constexpr int tokens = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr float rope_scale = 1.0f;
    constexpr float rope_theta = 10000.0f;

    std::vector<float> h_q(tokens * q_heads * head_dim);
    std::vector<float> h_k(tokens * kv_heads * head_dim);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = (static_cast<int>(i % 17) - 8) * 0.0625f;
    for (size_t i = 0; i < h_k.size(); ++i)
        h_k[i] = (static_cast<int>(i % 13) - 6) * 0.09375f;

    const std::vector<int32_t> ref_positions = { 1, 5 };
    const std::vector<uint32_t> h_positions = { 1, 5 };

    std::vector<float> want_q;
    std::vector<float> want_k;
    norm_rope_reference_tensor(
        h_q,
        ref_positions,
        tokens,
        q_heads,
        head_dim,
        head_dim,
        rope_scale,
        rope_theta,
        want_q
    );
    norm_rope_reference_tensor(
        h_k,
        ref_positions,
        tokens,
        kv_heads,
        head_dim,
        head_dim,
        rope_scale,
        rope_theta,
        want_k
    );

    qsfi_context* ctx = nullptr;
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_q_out = nullptr;
    float* d_k_out = nullptr;
    uint32_t* d_positions = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_q, h_q.data(), h_q.size(), "copy rope f32 q")
        || !copy_to_device(&d_k, h_k.data(), h_k.size(), "copy rope f32 k")
        || !copy_to_device(
            &d_positions,
            h_positions.data(),
            h_positions.size(),
            "copy rope f32 positions"
        )
        || !alloc_device(&d_q_out, h_q.size(), "alloc rope f32 q out")
        || !alloc_device(&d_k_out, h_k.size(), "alloc rope f32 k out")) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_q_out);
        cudaFree(d_k_out);
        cudaFree(d_positions);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_rope_apply_desc desc {};
    desc.q = norm_rope_tensor3_f32(d_q, tokens, q_heads, head_dim);
    desc.k = norm_rope_tensor3_f32(d_k, tokens, kv_heads, head_dim);
    desc.q_out = norm_rope_tensor3_f32(d_q_out, tokens, q_heads, head_dim);
    desc.k_out = norm_rope_tensor3_f32(d_k_out, tokens, kv_heads, head_dim);
    desc.positions = norm_rope_tensor1_u32(d_positions, tokens);
    desc.num_qo_heads = q_heads;
    desc.num_kv_heads = kv_heads;
    desc.head_dim = head_dim;
    desc.rotary_dim = head_dim;
    desc.rope_scale = rope_scale;
    desc.rope_theta = rope_theta;
    desc.interleave = 0;
    check_status(qsfi_rope_apply(ctx, &desc), QSFI_STATUS_OK, "rope apply f32 u32 positions");
    check_cuda(cudaDeviceSynchronize(), "sync rope apply f32 u32 positions");

    std::vector<float> h_q_out(h_q.size(), 0.0f);
    std::vector<float> h_k_out(h_k.size(), 0.0f);
    check_cuda(
        cudaMemcpy(h_q_out.data(), d_q_out, h_q_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy rope f32 q out"
    );
    check_cuda(
        cudaMemcpy(h_k_out.data(), d_k_out, h_k_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy rope f32 k out"
    );
    norm_rope_check_f32_close(h_q_out, want_q, 0.00005f, "rope f32 q out");
    norm_rope_check_f32_close(h_k_out, want_k, 0.00005f, "rope f32 k out");

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_q_out);
    cudaFree(d_k_out);
    cudaFree(d_positions);
    qsfi_context_destroy(ctx);
}

static void test_qsfi_rope_apply_rejects_invalid_desc_shape_dtype()
{
    constexpr int tokens = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    std::vector<uint16_t> h_q(tokens * q_heads * head_dim, kBf16One);
    std::vector<uint16_t> h_k(tokens * kv_heads * head_dim, kBf16One);
    const std::vector<int32_t> h_positions = { 0, 1 };

    qsfi_context* ctx = nullptr;
    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_q_out = nullptr;
    uint16_t* d_k_out = nullptr;
    int32_t* d_positions = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_q, h_q.data(), h_q.size(), "copy rope invalid q")
        || !copy_to_device(&d_k, h_k.data(), h_k.size(), "copy rope invalid k")
        || !copy_to_device(
            &d_positions,
            h_positions.data(),
            h_positions.size(),
            "copy rope invalid positions"
        )
        || !alloc_device(&d_q_out, h_q.size(), "alloc rope invalid q out")
        || !alloc_device(&d_k_out, h_k.size(), "alloc rope invalid k out")) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_q_out);
        cudaFree(d_k_out);
        cudaFree(d_positions);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_rope_apply_desc desc {};
    desc.q = tensor3_bf16(d_q, tokens, q_heads, head_dim);
    desc.k = tensor3_bf16(d_k, tokens, kv_heads, head_dim);
    desc.q_out = tensor3_bf16(d_q_out, tokens, q_heads, head_dim);
    desc.k_out = tensor3_bf16(d_k_out, tokens, kv_heads, head_dim);
    desc.positions = norm_rope_tensor1_i32(d_positions, tokens);
    desc.num_qo_heads = q_heads;
    desc.num_kv_heads = kv_heads;
    desc.head_dim = head_dim;
    desc.rotary_dim = head_dim;
    desc.rope_scale = 1.0f;
    desc.rope_theta = 10000.0f;
    desc.interleave = 0;

    check_status(
        qsfi_rope_apply(ctx, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects null"
    );

    qsfi_rope_apply_desc bad = desc;
    bad.head_dim = 63;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects odd head_dim"
    );

    bad = desc;
    bad.head_dim = 96;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_UNSUPPORTED,
        "rope apply rejects unsupported head_dim"
    );

    bad = desc;
    bad.rotary_dim = 0;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects zero rotary_dim"
    );

    bad = desc;
    bad.rotary_dim = 63;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects odd rotary_dim"
    );

    bad = desc;
    bad.rotary_dim = 66;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects rotary_dim past head_dim"
    );

    bad = desc;
    bad.q = tensor3_bf16(d_q, tokens, q_heads, 256);
    bad.k = tensor3_bf16(d_k, tokens, kv_heads, 256);
    bad.q_out = tensor3_bf16(d_q_out, tokens, q_heads, 256);
    bad.k_out = tensor3_bf16(d_k_out, tokens, kv_heads, 256);
    bad.head_dim = 256;
    bad.rotary_dim = 128;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_UNSUPPORTED,
        "rope apply rejects qwen36 head_dim 256 with non-64 rotary_dim"
    );

    bad = desc;
    bad.positions.dtype = QSFI_DTYPE_F32;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects positions dtype"
    );

    bad = desc;
    bad.positions.shape[0] = tokens + 1;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects positions shape"
    );

    bad = desc;
    bad.k.shape[0] = tokens + 1;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_INVALID_ARGUMENT,
        "rope apply rejects q/k token mismatch"
    );

    bad = desc;
    bad.q.dtype = QSFI_DTYPE_F16;
    check_status(
        qsfi_rope_apply(ctx, &bad),
        QSFI_STATUS_UNSUPPORTED,
        "rope apply rejects unsupported q dtype"
    );

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_q_out);
    cudaFree(d_k_out);
    cudaFree(d_positions);
    qsfi_context_destroy(ctx);
}

void test_qsfi_rmsnorm_bf16_matches_cpu()
{
    constexpr int rows = 2;
    constexpr int hidden = 8;
    constexpr float eps = 1.0e-6f;

    const std::vector<uint16_t> h_x = norm_rope_bf16_from_f32(
        {
            1.0f,
            -2.0f,
            0.5f,
            3.0f,
            -1.5f,
            2.5f,
            0.25f,
            -0.75f,
            -0.5f,
            1.25f,
            -3.0f,
            0.75f,
            2.0f,
            -1.0f,
            1.5f,
            -2.5f,
        }
    );
    const std::vector<uint16_t> h_weight
        = norm_rope_bf16_from_f32({ 0.75f, -1.0f, 1.25f, 0.5f, -0.25f, 1.5f, 2.0f, -0.75f });
    const std::vector<float> want = norm_rope_rmsnorm_reference(h_x, h_weight, rows, hidden, eps);

    qsfi_context* ctx = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy rmsnorm x")
        || !copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy rmsnorm weight")
        || !alloc_device(&d_out, h_x.size(), "alloc rmsnorm out")) {
        cudaFree(d_x);
        cudaFree(d_weight);
        cudaFree(d_out);
        qsfi_context_destroy(ctx);
        return;
    }
    check_cuda(cudaMemset(d_out, 0xA5, h_x.size() * sizeof(uint16_t)), "clear rmsnorm out");

    qsfi_rmsnorm_desc desc {};
    desc.x = tensor2_bf16(d_x, rows, hidden);
    desc.weight = norm_rope_tensor1_bf16(d_weight, hidden);
    desc.out = tensor2_bf16(d_out, rows, hidden);
    desc.hidden_size = hidden;
    desc.eps = eps;
    check_status(qsfi_rmsnorm(ctx, &desc), QSFI_STATUS_OK, "rmsnorm bf16");
    check_cuda(cudaDeviceSynchronize(), "sync rmsnorm bf16");

    std::vector<uint16_t> h_out(h_x.size(), 0);
    check_cuda(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy rmsnorm out"
    );
    norm_rope_check_bf16_close(h_out, want, 0.018f, "rmsnorm out");

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    qsfi_context_destroy(ctx);

    test_qsfi_rmsnorm_f32_matches_cpu();
    test_qsfi_rmsnorm_bf16_qwen_qk_norm_flattened_per_head_matches_cpu();
    norm_rope_test_rmsnorm_rejects_invalid_desc_shape_dtype();
}

void test_qsfi_fused_add_rmsnorm_bf16_inplace_updates_residual()
{
    constexpr int rows = 2;
    constexpr int hidden = 8;
    constexpr float eps = 1.0e-6f;

    const std::vector<uint16_t> h_x = norm_rope_bf16_from_f32(
        {
            0.5f,
            -1.0f,
            1.5f,
            -2.0f,
            0.25f,
            2.5f,
            -0.75f,
            1.25f,
            -1.5f,
            0.75f,
            2.0f,
            -0.25f,
            1.0f,
            -2.5f,
            0.5f,
            1.75f,
        }
    );
    const std::vector<uint16_t> h_residual = norm_rope_bf16_from_f32(
        {
            1.0f,
            0.5f,
            -0.5f,
            1.0f,
            -1.25f,
            0.75f,
            1.5f,
            -0.25f,
            0.25f,
            -1.25f,
            0.5f,
            2.25f,
            -0.75f,
            1.0f,
            -1.5f,
            0.25f,
        }
    );
    const std::vector<uint16_t> h_weight
        = norm_rope_bf16_from_f32({ 1.0f, 0.5f, -1.5f, 0.75f, 1.25f, -0.5f, 2.0f, -1.0f });

    std::vector<float> want_norm;
    std::vector<float> want_residual;
    norm_rope_fused_add_rmsnorm_reference(
        h_x,
        h_residual,
        h_weight,
        rows,
        hidden,
        eps,
        want_norm,
        want_residual
    );

    qsfi_context* ctx = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_residual = nullptr;
    uint16_t* d_weight = nullptr;
    if (!make_context(&ctx)
        || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy fused add rmsnorm x")
        || !copy_to_device(
            &d_residual,
            h_residual.data(),
            h_residual.size(),
            "copy fused add rmsnorm residual"
        )
        || !copy_to_device(
            &d_weight,
            h_weight.data(),
            h_weight.size(),
            "copy fused add rmsnorm weight"
        )) {
        cudaFree(d_x);
        cudaFree(d_residual);
        cudaFree(d_weight);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_fused_add_rmsnorm_desc desc {};
    desc.x = tensor2_bf16(d_x, rows, hidden);
    desc.residual_inout = tensor2_bf16(d_residual, rows, hidden);
    desc.weight = norm_rope_tensor1_bf16(d_weight, hidden);
    desc.out = desc.x;
    desc.hidden_size = hidden;
    desc.eps = eps;
    check_status(qsfi_fused_add_rmsnorm(ctx, &desc), QSFI_STATUS_OK, "fused add rmsnorm bf16");
    check_cuda(cudaDeviceSynchronize(), "sync fused add rmsnorm bf16");

    std::vector<uint16_t> h_out(h_x.size(), 0);
    std::vector<uint16_t> h_residual_out(h_residual.size(), 0);
    check_cuda(
        cudaMemcpy(h_out.data(), d_x, h_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy fused add rmsnorm x out"
    );
    check_cuda(
        cudaMemcpy(
            h_residual_out.data(),
            d_residual,
            h_residual_out.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy fused add rmsnorm residual out"
    );
    norm_rope_check_bf16_close(h_out, want_norm, 0.018f, "fused add rmsnorm out");
    norm_rope_check_bf16_close(h_residual_out, want_residual, 0.012f, "fused add rmsnorm residual");

    cudaFree(d_x);
    cudaFree(d_residual);
    cudaFree(d_weight);
    qsfi_context_destroy(ctx);

    test_qsfi_fused_add_rmsnorm_f32_inplace_updates_residual();
}

void test_qsfi_fused_add_rmsnorm_rejects_non_alias_out()
{
    constexpr int rows = 1;
    constexpr int hidden = 8;
    const std::vector<uint16_t> h_x
        = norm_rope_bf16_from_f32({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
    const std::vector<uint16_t> h_residual
        = norm_rope_bf16_from_f32({ 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f, -2.0f });
    const std::vector<uint16_t> h_weight
        = norm_rope_bf16_from_f32({ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

    qsfi_context* ctx = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_residual = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_x, h_x.data(), h_x.size(), "copy fused alias x")
        || !copy_to_device(
            &d_residual,
            h_residual.data(),
            h_residual.size(),
            "copy fused alias residual"
        )
        || !copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy fused alias weight")
        || !alloc_device(&d_out, h_x.size(), "alloc fused alias out")) {
        cudaFree(d_x);
        cudaFree(d_residual);
        cudaFree(d_weight);
        cudaFree(d_out);
        qsfi_context_destroy(ctx);
        return;
    }

    qsfi_fused_add_rmsnorm_desc desc {};
    desc.x = tensor2_bf16(d_x, rows, hidden);
    desc.residual_inout = tensor2_bf16(d_residual, rows, hidden);
    desc.weight = norm_rope_tensor1_bf16(d_weight, hidden);
    desc.out = tensor2_bf16(d_out, rows, hidden);
    desc.hidden_size = hidden;
    desc.eps = 1.0e-6f;
    check_status(
        qsfi_fused_add_rmsnorm(ctx, &desc),
        QSFI_STATUS_INVALID_ARGUMENT,
        "fused add rmsnorm rejects separate out"
    );

    desc.out = desc.x;
    desc.out.stride[0] = hidden + 1;
    check_status(
        qsfi_fused_add_rmsnorm(ctx, &desc),
        QSFI_STATUS_INVALID_ARGUMENT,
        "fused add rmsnorm rejects non-exact out alias"
    );

    cudaFree(d_x);
    cudaFree(d_residual);
    cudaFree(d_weight);
    cudaFree(d_out);
    qsfi_context_destroy(ctx);

    test_qsfi_fused_add_rmsnorm_rejects_invalid_desc_shape_dtype();
}

static void test_qsfi_rope_apply_bf16_qwen36_partial_rotary_copies_tail()
{
    constexpr int tokens = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 256;
    constexpr int rotary_dim = 64;
    constexpr float rope_scale = 1.0f;
    constexpr float rope_theta = 10000.0f;

    std::vector<float> q_f32(tokens * q_heads * head_dim);
    std::vector<float> k_f32(tokens * kv_heads * head_dim);
    for (size_t i = 0; i < q_f32.size(); ++i)
        q_f32[i] = (static_cast<int>(i % 29) - 14) * 0.046875f;
    for (size_t i = 0; i < k_f32.size(); ++i)
        k_f32[i] = (static_cast<int>(i % 31) - 15) * 0.0390625f;

    const std::vector<uint16_t> h_q = norm_rope_bf16_from_f32(q_f32);
    const std::vector<uint16_t> h_k = norm_rope_bf16_from_f32(k_f32);
    const std::vector<int32_t> h_positions = { 2, 7 };

    std::vector<float> want_q;
    std::vector<float> want_k;
    norm_rope_reference_tensor(
        h_q,
        h_positions,
        tokens,
        q_heads,
        head_dim,
        rotary_dim,
        rope_scale,
        rope_theta,
        want_q
    );
    norm_rope_reference_tensor(
        h_k,
        h_positions,
        tokens,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_scale,
        rope_theta,
        want_k
    );

    qsfi_context* ctx = nullptr;
    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_q_out = nullptr;
    uint16_t* d_k_out = nullptr;
    int32_t* d_positions = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_q, h_q.data(), h_q.size(), "copy partial rope q")
        || !copy_to_device(&d_k, h_k.data(), h_k.size(), "copy partial rope k")
        || !copy_to_device(
            &d_positions,
            h_positions.data(),
            h_positions.size(),
            "copy partial rope positions"
        )
        || !alloc_device(&d_q_out, h_q.size(), "alloc partial rope q out")
        || !alloc_device(&d_k_out, h_k.size(), "alloc partial rope k out")) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_q_out);
        cudaFree(d_k_out);
        cudaFree(d_positions);
        qsfi_context_destroy(ctx);
        return;
    }
    check_cuda(cudaMemset(d_q_out, 0xA5, h_q.size() * sizeof(uint16_t)), "clear partial rope q out");
    check_cuda(cudaMemset(d_k_out, 0xA5, h_k.size() * sizeof(uint16_t)), "clear partial rope k out");

    qsfi_rope_apply_desc desc {};
    desc.q = tensor3_bf16(d_q, tokens, q_heads, head_dim);
    desc.k = tensor3_bf16(d_k, tokens, kv_heads, head_dim);
    desc.q_out = tensor3_bf16(d_q_out, tokens, q_heads, head_dim);
    desc.k_out = tensor3_bf16(d_k_out, tokens, kv_heads, head_dim);
    desc.positions = norm_rope_tensor1_i32(d_positions, tokens);
    desc.num_qo_heads = q_heads;
    desc.num_kv_heads = kv_heads;
    desc.head_dim = head_dim;
    desc.rotary_dim = rotary_dim;
    desc.rope_scale = rope_scale;
    desc.rope_theta = rope_theta;
    desc.interleave = 0;
    check_status(qsfi_rope_apply(ctx, &desc), QSFI_STATUS_OK, "rope apply bf16 qwen36 partial");
    check_cuda(cudaDeviceSynchronize(), "sync rope apply bf16 qwen36 partial");

    std::vector<uint16_t> h_q_out(h_q.size(), 0);
    std::vector<uint16_t> h_k_out(h_k.size(), 0);
    check_cuda(
        cudaMemcpy(
            h_q_out.data(),
            d_q_out,
            h_q_out.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy partial rope q out"
    );
    check_cuda(
        cudaMemcpy(
            h_k_out.data(),
            d_k_out,
            h_k_out.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy partial rope k out"
    );
    norm_rope_check_bf16_close(h_q_out, want_q, 0.025f, "partial rope q out");
    norm_rope_check_bf16_close(h_k_out, want_k, 0.025f, "partial rope k out");
    norm_rope_check_bf16_tail_unchanged(
        h_q,
        h_q_out,
        tokens,
        q_heads,
        head_dim,
        rotary_dim,
        "partial rope q tail"
    );
    norm_rope_check_bf16_tail_unchanged(
        h_k,
        h_k_out,
        tokens,
        kv_heads,
        head_dim,
        rotary_dim,
        "partial rope k tail"
    );

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_q_out);
    cudaFree(d_k_out);
    cudaFree(d_positions);
    qsfi_context_destroy(ctx);
}

void test_qsfi_rope_apply_bf16_neox_full_head_matches_cpu()
{
    constexpr int tokens = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr float rope_scale = 1.0f;
    constexpr float rope_theta = 10000.0f;

    std::vector<float> q_f32(tokens * q_heads * head_dim);
    std::vector<float> k_f32(tokens * kv_heads * head_dim);
    for (size_t i = 0; i < q_f32.size(); ++i)
        q_f32[i] = (static_cast<int>(i % 23) - 11) * 0.125f;
    for (size_t i = 0; i < k_f32.size(); ++i)
        k_f32[i] = (static_cast<int>(i % 19) - 9) * 0.15625f;

    const std::vector<uint16_t> h_q = norm_rope_bf16_from_f32(q_f32);
    const std::vector<uint16_t> h_k = norm_rope_bf16_from_f32(k_f32);
    const std::vector<int32_t> h_positions = { 0, 3 };

    std::vector<float> want_q;
    std::vector<float> want_k;
    norm_rope_reference_tensor(
        h_q,
        h_positions,
        tokens,
        q_heads,
        head_dim,
        head_dim,
        rope_scale,
        rope_theta,
        want_q
    );
    norm_rope_reference_tensor(
        h_k,
        h_positions,
        tokens,
        kv_heads,
        head_dim,
        head_dim,
        rope_scale,
        rope_theta,
        want_k
    );

    qsfi_context* ctx = nullptr;
    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_q_out = nullptr;
    uint16_t* d_k_out = nullptr;
    int32_t* d_positions = nullptr;
    if (!make_context(&ctx) || !copy_to_device(&d_q, h_q.data(), h_q.size(), "copy rope q")
        || !copy_to_device(&d_k, h_k.data(), h_k.size(), "copy rope k")
        || !copy_to_device(
            &d_positions,
            h_positions.data(),
            h_positions.size(),
            "copy rope positions"
        )
        || !alloc_device(&d_q_out, h_q.size(), "alloc rope q out")
        || !alloc_device(&d_k_out, h_k.size(), "alloc rope k out")) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_q_out);
        cudaFree(d_k_out);
        cudaFree(d_positions);
        qsfi_context_destroy(ctx);
        return;
    }
    check_cuda(cudaMemset(d_q_out, 0xA5, h_q.size() * sizeof(uint16_t)), "clear rope q out");
    check_cuda(cudaMemset(d_k_out, 0xA5, h_k.size() * sizeof(uint16_t)), "clear rope k out");

    qsfi_rope_apply_desc desc {};
    desc.q = tensor3_bf16(d_q, tokens, q_heads, head_dim);
    desc.k = tensor3_bf16(d_k, tokens, kv_heads, head_dim);
    desc.q_out = tensor3_bf16(d_q_out, tokens, q_heads, head_dim);
    desc.k_out = tensor3_bf16(d_k_out, tokens, kv_heads, head_dim);
    desc.positions = norm_rope_tensor1_i32(d_positions, tokens);
    desc.num_qo_heads = q_heads;
    desc.num_kv_heads = kv_heads;
    desc.head_dim = head_dim;
    desc.rotary_dim = head_dim;
    desc.rope_scale = rope_scale;
    desc.rope_theta = rope_theta;
    desc.interleave = 0;
    check_status(qsfi_rope_apply(ctx, &desc), QSFI_STATUS_OK, "rope apply bf16 neox");
    check_cuda(cudaDeviceSynchronize(), "sync rope apply bf16 neox");

    std::vector<uint16_t> h_q_out(h_q.size(), 0);
    std::vector<uint16_t> h_k_out(h_k.size(), 0);
    check_cuda(
        cudaMemcpy(
            h_q_out.data(),
            d_q_out,
            h_q_out.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy rope q out"
    );
    check_cuda(
        cudaMemcpy(
            h_k_out.data(),
            d_k_out,
            h_k_out.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost
        ),
        "copy rope k out"
    );
    norm_rope_check_bf16_close(h_q_out, want_q, 0.025f, "rope q out");
    norm_rope_check_bf16_close(h_k_out, want_k, 0.025f, "rope k out");

    desc.interleave = 1;
    check_status(
        qsfi_rope_apply(ctx, &desc),
        QSFI_STATUS_UNSUPPORTED,
        "rope apply rejects interleave"
    );

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_q_out);
    cudaFree(d_k_out);
    cudaFree(d_positions);
    qsfi_context_destroy(ctx);

    test_qsfi_rope_apply_bf16_qwen36_partial_rotary_copies_tail();
    test_qsfi_rope_apply_f32_u32_positions_matches_cpu();
    test_qsfi_rope_apply_rejects_invalid_desc_shape_dtype();
}
