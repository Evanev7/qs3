#ifndef QSCB_H
#error "Include qscb.h before including tests_cuda_qscb_gemm.inc inside test.cu."
#endif

static uint16_t qscb_test_f32_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

static qsfi_tensor2 qscb_test_tensor2_bf16(void* data, int64_t rows, int64_t cols, int64_t stride0)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_BF16;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.stride[0] = stride0;
    tensor.stride[1] = 1;
    return tensor;
}

static qsfi_tensor2 qscb_test_tensor2_f32(void* data, int64_t rows, int64_t cols, int64_t stride0)
{
    qsfi_tensor2 tensor {};
    tensor.data = data;
    tensor.dtype = QSFI_DTYPE_F32;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.stride[0] = stride0;
    tensor.stride[1] = 1;
    return tensor;
}

static void
qscb_test_store_bf16(std::vector<uint16_t>& values, int row, int col, int stride, float value)
{
    values[static_cast<size_t>(row) * stride + col] = qscb_test_f32_to_bf16(value);
}

static float qscb_test_bf16_dot(
    const std::vector<uint16_t>& x,
    int x_stride,
    const std::vector<uint16_t>& weight,
    int weight_stride,
    int row,
    int out_col,
    int in_features
)
{
    float sum = 0.0f;
    for (int in_col = 0; in_col < in_features; ++in_col) {
        const float x_value = bf16_to_f32(x[static_cast<size_t>(row) * x_stride + in_col]);
        const float w_value
            = bf16_to_f32(weight[static_cast<size_t>(out_col) * weight_stride + in_col]);
        sum += x_value * w_value;
    }
    return sum;
}

static void qscb_test_check_bf16_output(
    const std::vector<uint16_t>& out,
    int out_stride,
    const std::vector<uint16_t>& x,
    int x_stride,
    const std::vector<uint16_t>& weight,
    int weight_stride,
    int rows,
    int in_features,
    int out_features,
    float abs_tol,
    const char* what
)
{
    for (int row = 0; row < rows; ++row) {
        for (int out_col = 0; out_col < out_features; ++out_col) {
            const float got = bf16_to_f32(out[static_cast<size_t>(row) * out_stride + out_col]);
            const float want
                = qscb_test_bf16_dot(x, x_stride, weight, weight_stride, row, out_col, in_features);
            if (!approx_equal(got, want, abs_tol)) {
                std::fprintf(
                    stderr,
                    "FAIL: %s[%d,%d]: got %.8f, want %.8f +/- %.8f\n",
                    what,
                    row,
                    out_col,
                    got,
                    want,
                    abs_tol
                );
                failures += 1;
                return;
            }
        }
    }
}

static void qscb_test_check_f32_output(
    const std::vector<float>& out,
    int out_stride,
    const std::vector<float>& out_before,
    const std::vector<uint16_t>& x,
    int x_stride,
    const std::vector<uint16_t>& weight,
    int weight_stride,
    int rows,
    int in_features,
    int out_features,
    float beta,
    float abs_tol,
    const char* what
)
{
    for (int row = 0; row < rows; ++row) {
        for (int out_col = 0; out_col < out_features; ++out_col) {
            const size_t offset = static_cast<size_t>(row) * out_stride + out_col;
            const float got = out[offset];
            const float want
                = qscb_test_bf16_dot(x, x_stride, weight, weight_stride, row, out_col, in_features)
                + beta * out_before[offset];
            if (!approx_equal(got, want, abs_tol)) {
                std::fprintf(
                    stderr,
                    "FAIL: %s[%d,%d]: got %.8f, want %.8f +/- %.8f\n",
                    what,
                    row,
                    out_col,
                    got,
                    want,
                    abs_tol
                );
                failures += 1;
                return;
            }
        }
    }
}

static void qscb_test_check_bf16_padding(
    const std::vector<uint16_t>& out,
    int rows,
    int out_features,
    int out_stride,
    uint16_t sentinel,
    const char* what
)
{
    for (int row = 0; row < rows; ++row) {
        for (int col = out_features; col < out_stride; ++col) {
            const size_t offset = static_cast<size_t>(row) * out_stride + col;
            if (out[offset] != sentinel) {
                std::fprintf(
                    stderr,
                    "FAIL: %s padding[%d,%d]: got 0x%04x, want 0x%04x\n",
                    what,
                    row,
                    col,
                    static_cast<unsigned>(out[offset]),
                    static_cast<unsigned>(sentinel)
                );
                failures += 1;
                return;
            }
        }
    }
}

static void qscb_test_check_f32_padding(
    const std::vector<float>& out,
    int rows,
    int out_features,
    int out_stride,
    float sentinel,
    const char* what
)
{
    for (int row = 0; row < rows; ++row) {
        for (int col = out_features; col < out_stride; ++col) {
            const size_t offset = static_cast<size_t>(row) * out_stride + col;
            if (!approx_equal(out[offset], sentinel, 0.0f)) {
                std::fprintf(
                    stderr,
                    "FAIL: %s padding[%d,%d]: got %.8f, want %.8f\n",
                    what,
                    row,
                    col,
                    out[offset],
                    sentinel
                );
                failures += 1;
                return;
            }
        }
    }
}

static bool qscb_test_make_context(qscb_context** ctx, const char* what)
{
    qscb_context_desc desc {};
    desc.device_ordinal = 0;
    desc.stream = nullptr;
    if (!check_status(qscb_context_create(&desc, ctx), QSFI_STATUS_OK, what))
        return false;
    if (*ctx != nullptr)
        return true;
    std::fprintf(stderr, "FAIL: %s: context is null after successful create\n", what);
    failures += 1;
    return false;
}

static void test_qscb_context_lifecycle()
{
    qscb_context_desc desc {};
    desc.device_ordinal = 0;
    desc.stream = nullptr;
    qscb_context* created_context = nullptr;
    check_status(
        qscb_context_create(&desc, &created_context),
        QSFI_STATUS_OK,
        "qscb context create"
    );
    qscb_context_destroy(created_context);

    check_status(
        qscb_context_create(&desc, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscb create rejects null output pointer"
    );

    check_status(
        qscb_gemm_bf16(nullptr, nullptr),
        QSFI_STATUS_INVALID_ARGUMENT,
        "qscb gemm rejects null context"
    );

    qscb_context* ctx = nullptr;
    if (qscb_test_make_context(&ctx, "qscb context create helper"))
        qscb_context_destroy(ctx);
    qscb_context_destroy(nullptr);
}

static void test_qscb_gemm_bf16_hidden_output_strided()
{
    qscb_context* ctx = nullptr;
    if (!qscb_test_make_context(&ctx, "qscb hidden context create"))
        return;

    constexpr int rows = 3;
    constexpr int in_features = 5;
    constexpr int out_features = 4;
    constexpr int x_stride = 7;
    constexpr int weight_stride = 6;
    constexpr int out_stride = 6;
    const uint16_t bf16_sentinel = qscb_test_f32_to_bf16(-17.0f);

    const float x_values[rows][in_features] = {
        { 1.0f, -0.5f, 0.25f, 1.5f, -1.25f },
        { -0.75f, 0.5f, 2.0f, -0.25f, 1.25f },
        { 0.125f, -1.5f, 0.75f, 0.5f, -0.5f },
    };
    const float weight_values[out_features][in_features] = {
        { 0.5f, -1.0f, 0.25f, 0.75f, -0.5f },
        { -0.25f, 0.5f, 1.0f, -1.5f, 0.25f },
        { 1.25f, -0.75f, 0.5f, 0.25f, 1.0f },
        { -1.0f, 0.25f, -0.5f, 0.5f, 1.5f },
    };

    std::vector<uint16_t> h_x(static_cast<size_t>(rows) * x_stride, bf16_sentinel);
    std::vector<uint16_t> h_weight(
        static_cast<size_t>(out_features) * weight_stride,
        bf16_sentinel
    );
    std::vector<uint16_t> h_out(static_cast<size_t>(rows) * out_stride, bf16_sentinel);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < in_features; ++col)
            qscb_test_store_bf16(h_x, row, col, x_stride, x_values[row][col]);
    }
    for (int row = 0; row < out_features; ++row) {
        for (int col = 0; col < in_features; ++col)
            qscb_test_store_bf16(h_weight, row, col, weight_stride, weight_values[row][col]);
    }

    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    uint16_t* d_out = nullptr;
    const bool copied = copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qscb hidden x")
        && copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy qscb hidden weight")
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy qscb hidden out");

    if (copied) {
        qscb_bf16_gemm_desc desc {};
        desc.x = qscb_test_tensor2_bf16(d_x, rows, in_features, x_stride);
        desc.weight = qscb_test_tensor2_bf16(d_weight, out_features, in_features, weight_stride);
        desc.out = qscb_test_tensor2_bf16(d_out, rows, out_features, out_stride);
        desc.rows = rows;
        desc.in_features = in_features;
        desc.out_features = out_features;

        const bool ran
            = check_status(qscb_gemm_bf16(ctx, &desc), QSFI_STATUS_OK, "qscb hidden bf16 gemm")
            && check_cuda(cudaDeviceSynchronize(), "sync qscb hidden bf16 gemm")
            && check_cuda(
                  cudaMemcpy(
                      h_out.data(),
                      d_out,
                      h_out.size() * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost
                  ),
                  "copy qscb hidden out back"
            );
        if (ran) {
            qscb_test_check_bf16_output(
                h_out,
                out_stride,
                h_x,
                x_stride,
                h_weight,
                weight_stride,
                rows,
                in_features,
                out_features,
                0.04f,
                "qscb hidden bf16 out"
            );
            qscb_test_check_bf16_padding(
                h_out,
                rows,
                out_features,
                out_stride,
                bf16_sentinel,
                "qscb hidden bf16 out"
            );
        }
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    qscb_context_destroy(ctx);
}

static void qscb_test_run_f32_accumulation_sensitive(qscb_context* ctx)
{
    constexpr int rows = 1;
    constexpr int in_features = 4096;
    constexpr int out_features = 3;
    constexpr int x_stride = in_features + 7;
    constexpr int weight_stride = in_features + 9;
    constexpr int out_stride = 5;
    constexpr float f32_sentinel = -71.0f;
    const uint16_t bf16_sentinel = qscb_test_f32_to_bf16(-29.0f);

    std::vector<uint16_t> h_x(static_cast<size_t>(rows) * x_stride, bf16_sentinel);
    std::vector<uint16_t> h_weight(
        static_cast<size_t>(out_features) * weight_stride,
        bf16_sentinel
    );
    std::vector<float> h_out(static_cast<size_t>(rows) * out_stride, f32_sentinel);
    for (int col = 0; col < in_features; ++col) {
        const float x_value = 1.0f + static_cast<float>((col % 17) - 8) * 0.00390625f;
        qscb_test_store_bf16(h_x, 0, col, x_stride, x_value);
    }
    for (int col = 0; col < in_features; ++col) {
        const float weight0 = 1.0f + static_cast<float>(((col + 3) % 19) - 9) * 0.005859375f;
        const float weight1 = 0.75f + static_cast<float>(((col * 5 + 1) % 23) - 11) * 0.00390625f;
        const float weight2 = 1.25f + static_cast<float>(((col * 7 + 4) % 29) - 14) * 0.00390625f;
        qscb_test_store_bf16(h_weight, 0, col, weight_stride, weight0);
        qscb_test_store_bf16(h_weight, 1, col, weight_stride, weight1);
        qscb_test_store_bf16(h_weight, 2, col, weight_stride, weight2);
    }
    const std::vector<float> h_out_before = h_out;

    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    float* d_out = nullptr;
    const bool copied = copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qscb sensitive x")
        && copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy qscb sensitive weight")
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy qscb sensitive out");

    if (copied) {
        qscb_bf16_gemm_desc desc {};
        desc.x = qscb_test_tensor2_bf16(d_x, rows, in_features, x_stride);
        desc.weight = qscb_test_tensor2_bf16(d_weight, out_features, in_features, weight_stride);
        desc.out = qscb_test_tensor2_f32(d_out, rows, out_features, out_stride);
        desc.rows = rows;
        desc.in_features = in_features;
        desc.out_features = out_features;

        const bool ran = check_status(
                             qscb_gemm_bf16(ctx, &desc),
                             QSFI_STATUS_OK,
                             "qscb sensitive f32 accumulation gemm"
                         )
            && check_cuda(cudaDeviceSynchronize(), "sync qscb sensitive f32 accumulation gemm")
            && check_cuda(cudaMemcpy(
                              h_out.data(),
                              d_out,
                              h_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost
                          ),
                          "copy qscb sensitive out back");
        if (ran) {
            qscb_test_check_f32_output(
                h_out,
                out_stride,
                h_out_before,
                h_x,
                x_stride,
                h_weight,
                weight_stride,
                rows,
                in_features,
                out_features,
                0.0f,
                0.10f,
                "qscb sensitive f32 accumulation out"
            );
            qscb_test_check_f32_padding(
                h_out,
                rows,
                out_features,
                out_stride,
                f32_sentinel,
                "qscb sensitive f32 accumulation out"
            );
        }
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
}

static void test_qscb_gemm_bf16_logits_f32_output_beta()
{
    qscb_context* ctx = nullptr;
    if (!qscb_test_make_context(&ctx, "qscb logits context create"))
        return;

    constexpr int rows = 2;
    constexpr int in_features = 5;
    constexpr int out_features = 3;
    constexpr int x_stride = 6;
    constexpr int weight_stride = 7;
    constexpr int out_stride = 5;
    constexpr float beta = 0.25f;
    constexpr float f32_sentinel = -31.0f;
    const uint16_t bf16_sentinel = qscb_test_f32_to_bf16(19.0f);

    const float x_values[rows][in_features] = {
        { 1.0f, 0.5f, -1.0f, 0.25f, 1.5f },
        { -0.5f, 1.25f, 0.75f, -1.5f, 0.5f },
    };
    const float weight_values[out_features][in_features] = {
        { 0.25f, -0.75f, 1.0f, 0.5f, -1.25f },
        { -1.5f, 0.5f, 0.25f, 1.0f, 0.75f },
        { 1.25f, -0.25f, -0.5f, 0.75f, 0.5f },
    };
    const float out_before_values[rows][out_features] = {
        { 0.5f, -1.0f, 2.0f },
        { -0.25f, 1.5f, -2.0f },
    };

    std::vector<uint16_t> h_x(static_cast<size_t>(rows) * x_stride, bf16_sentinel);
    std::vector<uint16_t> h_weight(
        static_cast<size_t>(out_features) * weight_stride,
        bf16_sentinel
    );
    std::vector<float> h_out(static_cast<size_t>(rows) * out_stride, f32_sentinel);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < in_features; ++col)
            qscb_test_store_bf16(h_x, row, col, x_stride, x_values[row][col]);
    }
    for (int row = 0; row < out_features; ++row) {
        for (int col = 0; col < in_features; ++col)
            qscb_test_store_bf16(h_weight, row, col, weight_stride, weight_values[row][col]);
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < out_features; ++col)
            h_out[static_cast<size_t>(row) * out_stride + col] = out_before_values[row][col];
    }
    const std::vector<float> h_out_before = h_out;

    uint16_t* d_x = nullptr;
    uint16_t* d_weight = nullptr;
    float* d_out = nullptr;
    const bool copied = copy_to_device(&d_x, h_x.data(), h_x.size(), "copy qscb logits x")
        && copy_to_device(&d_weight, h_weight.data(), h_weight.size(), "copy qscb logits weight")
        && copy_to_device(&d_out, h_out.data(), h_out.size(), "copy qscb logits out");

    if (copied) {
        qscb_bf16_gemm_desc desc {};
        desc.x = qscb_test_tensor2_bf16(d_x, rows, in_features, x_stride);
        desc.weight = qscb_test_tensor2_bf16(d_weight, out_features, in_features, weight_stride);
        desc.out = qscb_test_tensor2_f32(d_out, rows, out_features, out_stride);
        desc.rows = rows;
        desc.in_features = in_features;
        desc.out_features = out_features;
        desc.beta = beta;

        const bool ran
            = check_status(qscb_gemm_bf16(ctx, &desc), QSFI_STATUS_OK, "qscb logits f32 gemm beta")
            && check_cuda(cudaDeviceSynchronize(), "sync qscb logits f32 gemm")
            && check_cuda(
                  cudaMemcpy(
                      h_out.data(),
                      d_out,
                      h_out.size() * sizeof(float),
                      cudaMemcpyDeviceToHost
                  ),
                  "copy qscb logits out back"
            );
        if (ran) {
            qscb_test_check_f32_output(
                h_out,
                out_stride,
                h_out_before,
                h_x,
                x_stride,
                h_weight,
                weight_stride,
                rows,
                in_features,
                out_features,
                beta,
                0.02f,
                "qscb logits f32 out"
            );
            qscb_test_check_f32_padding(
                h_out,
                rows,
                out_features,
                out_stride,
                f32_sentinel,
                "qscb logits f32 out"
            );
        }
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    qscb_test_run_f32_accumulation_sensitive(ctx);
    qscb_context_destroy(ctx);
}
