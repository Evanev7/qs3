// Independent host decoding references for the native quantized boundaries.
// Every context uses a non-default stream; all scratch and scales are caller-owned.
template <typename T> struct QuantBuffer {
    T* data = nullptr;
    size_t count;
    explicit QuantBuffer(size_t n)
        : count(n)
    {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data), std::max<size_t>(n, 1) * sizeof(T)),
            "quantized allocation"
        );
    }
    ~QuantBuffer()
    {
        if (data)
            cudaFree(data);
    }
    QuantBuffer(const QuantBuffer&) = delete;
    void upload(const std::vector<T>& v)
    {
        check_cuda(
            cudaMemcpy(data, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice),
            "quantized upload"
        );
        // Pageable H2D copies may return after staging, before device completion.
        // These tests launch on nonblocking streams, which do not wait on the
        // default stream used by cudaMemcpy. Complete the fixture upload first.
        check_cuda(cudaStreamSynchronize(nullptr), "quantized upload completion");
    }
    std::vector<T> download(cudaStream_t stream) const
    {
        check_cuda(cudaStreamSynchronize(stream), "quantized completion");
        std::vector<T> v(count);
        check_cuda(
            cudaMemcpy(v.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost),
            "quantized download"
        );
        return v;
    }
};

static qsfi_tensor1 quant_vector(void* p, qsfi_dtype dtype, size_t n)
{
    return { p, dtype, { static_cast<int64_t>(n) }, { 1 } };
}
static qsfi_tensor2 quant_matrix(void* p, qsfi_dtype dtype, int rows, int cols, int stride = 0)
{
    return { p, dtype, { rows, cols }, { stride ? stride : cols, 1 } };
}
static void quant_check_error(double error, double reference, const char* name)
{
    const double relative = std::sqrt(error / std::max(reference, 1e-30));
    if (!std::isfinite(relative) || relative > 0.005) {
        std::fprintf(stderr, "FAIL: %s relative L2 %.8g\n", name, relative);
        ++failures;
    }
}
static float quant_fp8_value(uint8_t bits)
{
    // Independent positive/negative E4M3FN decode, including subnormals.
    const int exponent = (bits >> 3) & 15, mantissa = bits & 7;
    const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -9)
                                          : std::ldexp(1.0f + float(mantissa) / 8, exponent - 7);
    return bits & 128 ? -magnitude : magnitude;
}
static float quant_fp4_value(uint8_t bits)
{
    const float magnitudes[] = { 0, .5f, 1, 1.5f, 2, 3, 4, 6 };
    return (bits & 8 ? -1 : 1) * magnitudes[bits & 7];
}
static size_t quant_sf_size(int rows, int k)
{
    return size_t((rows + 127) / 128) * 128 * (k / 16);
}
static size_t quant_sf_offset(int row, int block, int k)
{
    // Derive position from the six-dimensional scale layout.
    const int a = row / 128, b = (row % 128) / 32, c = row % 32;
    const int d = block / 4, e = block % 4;
    return (((size_t(a) * (k / 64) + d) * 32 + c) * 4 + b) * 4 + e;
}

static void test_qscb_fp8()
{
    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "FP8 stream"))
        return;
    qscb_context* ctx = nullptr;
    qscb_context_desc cd { 0, stream };
    if (!check_status(qscb_context_create(&cd, &ctx), QSFI_STATUS_OK, "FP8 context"))
        return;
    QuantBuffer<uint8_t> workspace(64ull << 20);
    QuantBuffer<float> xs(1), ws(1), xs2(1), ws2(1);
    xs.upload({ .25f });
    ws.upload({ .125f });
    xs2.upload({ .5f });
    ws2.upload({ .25f });
    for (int m : { 1, 2, 4, 5, 16, 128, 512 })
        for (bool f32 : { false, true }) {
            const int n = 128, k = 128, stride = k + 16, ostride = n + 16;
            std::vector<uint16_t> hx(size_t(m) * stride, 0), hw(size_t(n) * stride, 0);
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < k; ++c)
                    hx[size_t(r) * stride + c]
                        = qscb_test_f32_to_bf16(float((r * 19 + c * 7) % 113 - 56) / 17);
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < k; ++c)
                    hw[size_t(r) * stride + c]
                        = qscb_test_f32_to_bf16(float((r * 7 + c * 13) % 71 - 35) / 19);
            hx[0] = qscb_test_f32_to_bf16(512); // finite saturation in the quantizer
            QuantBuffer<uint16_t> x(hx.size()), w(hw.size());
            x.upload(hx);
            w.upload(hw);
            QuantBuffer<uint8_t> xq(hx.size()), wq(hw.size());
            QuantBuffer<uint8_t> y(size_t(m) * ostride * (f32 ? 4 : 2));
            qscb_fp8_quantize_desc qx { quant_matrix(x.data, QSFI_DTYPE_BF16, m, k, stride),
                                        quant_matrix(xq.data, QSFI_DTYPE_FP8_E4M3, m, k, stride),
                                        quant_vector(xs.data, QSFI_DTYPE_F32, 1) };
            auto qw = qx;
            qw.x = quant_matrix(w.data, QSFI_DTYPE_BF16, n, k, stride);
            qw.out = quant_matrix(wq.data, QSFI_DTYPE_FP8_E4M3, n, k, stride);
            qw.scale = quant_vector(ws.data, QSFI_DTYPE_F32, 1);
            if (!check_status(qscb_fp8_quantize(ctx, &qx), QSFI_STATUS_OK, "FP8 quantize x")
                || !check_status(qscb_fp8_quantize(ctx, &qw), QSFI_STATUS_OK, "FP8 quantize w"))
                break;
            auto qhost = xq.download(stream);
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < k; ++c) {
                    __nv_fp8_e4m3 ref(bf16_to_f32(hx[size_t(r) * stride + c]) / .25f);
                    if (qhost[size_t(r) * stride + c] != ref.__x) {
                        ++failures;
                        std::fprintf(stderr, "FAIL: FP8 quantizer bytes\n");
                        goto fp8_bytes_done;
                    }
                }
        fp8_bytes_done:
            qscb_fp8_linear_desc d {};
            d.linear.x = qx.out;
            d.linear.weight = qw.out;
            d.linear.out
                = quant_matrix(y.data, f32 ? QSFI_DTYPE_F32 : QSFI_DTYPE_BF16, m, n, ostride);
            d.linear.rows = m;
            d.linear.in_features = k;
            d.linear.out_features = n;
            d.linear.alpha = .7f;
            d.linear.workspace = workspace.data;
            d.linear.workspace_bytes = workspace.count;
            d.x_scale = qx.scale;
            d.weight_scale = qw.scale;
            qscb_linear_plan* plan = nullptr;
            if (!check_status(
                    qscb_fp8_linear_plan_create(ctx, &d, &plan),
                    QSFI_STATUS_OK,
                    "FP8 plan"
                )) {
                qsfi_error_info err {};
                qscb_context_get_last_error(ctx, &err);
                std::fprintf(stderr, "%s\n", err.message);
                break;
            }
            auto verify = [&](float sx, float sw, const std::vector<uint8_t>* previous = nullptr) {
                auto ax = xq.download(stream), bw = wq.download(stream),
                     output = y.download(stream);
                double error = 0, norm = 0;
                for (int r = 0; r < m; ++r)
                    for (int col = 0; col < n; ++col) {
                        double ref = 0;
                        for (int c = 0; c < k; ++c)
                            ref += double(quant_fp8_value(ax[size_t(r) * stride + c]))
                                * quant_fp8_value(bw[size_t(col) * stride + c]);
                        ref *= double(sx) * sw * .7f;
                        if (previous) {
                            float old;
                            if (f32)
                                std::memcpy(
                                    &old,
                                    previous->data() + (size_t(r) * ostride + col) * 4,
                                    4
                                );
                            else {
                                uint16_t bits;
                                std::memcpy(
                                    &bits,
                                    previous->data() + (size_t(r) * ostride + col) * 2,
                                    2
                                );
                                old = bf16_to_f32(bits);
                            }
                            ref += .25f * old;
                        }
                        float got;
                        if (f32)
                            std::memcpy(&got, output.data() + (size_t(r) * ostride + col) * 4, 4);
                        else {
                            uint16_t bits;
                            std::memcpy(&bits, output.data() + (size_t(r) * ostride + col) * 2, 2);
                            got = bf16_to_f32(bits);
                        }
                        error += (got - ref) * (got - ref);
                        norm += ref * ref;
                    }
                char label[128];
                std::snprintf(
                    label,
                    sizeof(label),
                    "FP8 GEMM M=%d dtype=%s sx=%g sw=%g beta=%g",
                    m,
                    f32 ? "F32" : "BF16",
                    sx,
                    sw,
                    previous ? .25 : 0.
                );
                quant_check_error(error, norm, label);
            };
            check_status(qscb_fp8_linear_execute(ctx, plan, &d), QSFI_STATUS_OK, "FP8 execute");
            verify(.25f, .125f);
            // Same plan, new scale addresses and values. Operand bytes stay fixed.
            d.x_scale.data = xs2.data;
            d.weight_scale.data = ws2.data;
            check_status(
                qscb_fp8_linear_execute(ctx, plan, &d),
                QSFI_STATUS_OK,
                "FP8 rebind scales"
            );
            verify(.5f, .25f);
            const auto previous = y.download(stream);
            d.linear.beta = .25f;
            check_status(qscb_fp8_linear_execute(ctx, plan, &d), QSFI_STATUS_OK, "FP8 beta reuse");
            verify(.5f, .25f, &previous);
            d.linear.beta = 0;
            auto bad = d;
            bad.x_scale.shape[0] = 2;
            check_status(
                qscb_fp8_linear_execute(ctx, plan, &bad),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP8 scale extent rejection"
            );
            bad = d;
            bad.linear.x.stride[0] += 16;
            check_status(
                qscb_fp8_linear_execute(ctx, plan, &bad),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP8 plan key rejection"
            );
            bad = d;
            bad.linear.x.dtype = QSFI_DTYPE_BF16;
            check_status(
                qscb_fp8_linear_execute(ctx, plan, &bad),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP8 dtype rejection"
            );
            check_status(
                qscb_linear_execute(ctx, plan, &d.linear),
                QSFI_STATUS_INVALID_ARGUMENT,
                "BF16 API rejects FP8"
            );
            if (m == 5) {
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t executable = nullptr;
                check_cuda(
                    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                    "FP8 capture begin"
                );
                check_status(
                    qscb_fp8_quantize(ctx, &qx),
                    QSFI_STATUS_OK,
                    "FP8 captured quantization"
                );
                check_status(
                    qscb_fp8_linear_execute(ctx, plan, &d),
                    QSFI_STATUS_OK,
                    "FP8 capture execute"
                );
                check_cuda(cudaStreamEndCapture(stream, &graph), "FP8 capture end");
                check_cuda(
                    cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                    "FP8 graph instantiate"
                );
                for (int replay = 0; replay < 3; ++replay) {
                    xs2.upload({ .5f / (replay + 1) });
                    check_cuda(cudaGraphLaunch(executable, stream), "FP8 graph replay");
                    verify(.5f / (replay + 1), .25f);
                }
                xs2.upload({ .5f });
                cudaGraphExecDestroy(executable);
                cudaGraphDestroy(graph);
            }
            qscb_linear_plan_destroy(plan);
        }
    auto badq = qscb_fp8_quantize_desc {};
    check_status(
        qscb_fp8_quantize(ctx, &badq),
        QSFI_STATUS_INVALID_ARGUMENT,
        "FP8 invalid quantizer"
    );
    qscb_context_destroy(ctx);
    cudaStreamDestroy(stream);
}

#define QSFI_TEST_NVFP4_TACTIC(ID, N, STREAM_K, LAUNCHER) ID,
static constexpr qsfi_nvfp4_tactic kNvfp4Tactics[] = { QSFI_NVFP4_TACTICS(QSFI_TEST_NVFP4_TACTIC) };
#undef QSFI_TEST_NVFP4_TACTIC

static void test_qsfi_nvfp4()
{
    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "FP4 stream"))
        return;
    qsfi_context* ctx = nullptr;
    qsfi_context_desc cd { 0, stream };
    if (!check_status(qsfi_context_create(&cd, &ctx), QSFI_STATUS_OK, "FP4 context"))
        return;
    QuantBuffer<float> multiplier(1), alpha(1), alpha2(1);
    multiplier.upload({ 2.f });
    alpha.upload({ .0685f });
    alpha2.upload({ .137f });
    const int n = 136, k = 256;
    std::vector<uint8_t> hw(size_t(n) * k / 2), hs(size_t(n) * k / 16);
    for (size_t i = 0; i < hw.size(); ++i)
        hw[i] = uint8_t((i * 17 + i / 7) % 256);
    for (size_t i = 0; i < hs.size(); ++i)
        hs[i] = i % 19 == 0 ? 0 : uint8_t(40 + (i * 7) % 24);
    QuantBuffer<uint8_t> w(hw.size()), scales(hs.size()), swizzled(quant_sf_size(n, k));
    w.upload(hw);
    scales.upload(hs);
    auto source = quant_matrix(scales.data, QSFI_DTYPE_FP8_E4M3, n, k / 16);
    auto target = quant_vector(swizzled.data, QSFI_DTYPE_FP8_E4M3, swizzled.count);
    check_status(qsfi_nvfp4_swizzle_scales(ctx, &source, &target), QSFI_STATUS_OK, "FP4 swizzle");
    auto sw = swizzled.download(stream);
    for (int r = 0; r < 256; ++r)
        for (int c = 0; c < k / 16; ++c) {
            if (sw[quant_sf_offset(r, c, k)] != (r < n ? hs[size_t(r) * (k / 16) + c] : 0)) {
                std::fprintf(stderr, "FAIL: FP4 swizzle/padding\n");
                ++failures;
                goto swizzle_done;
            }
        }
swizzle_done:
    for (int m : { 1, 2, 4, 5, 16, 128, 512 }) {
        std::vector<uint16_t> hx(size_t(m) * k);
        for (size_t i = 0; i < hx.size(); ++i)
            hx[i] = qscb_test_f32_to_bf16(
                i / 16 % 11 == 0 ? 0.f : float(int((i * 13 + i / 17) % 97) - 48) / 16
            );
        QuantBuffer<uint16_t> x(hx.size()), y(size_t(m) * n);
        x.upload(hx);
        QuantBuffer<uint8_t> xq(hx.size() / 2), xsc(quant_sf_size(m, k));
        qsfi_nvfp4_quantize_desc q { quant_matrix(x.data, QSFI_DTYPE_BF16, m, k),
                                     quant_matrix(xq.data, QSFI_DTYPE_NVFP4_E2M1, m, k),
                                     quant_vector(xsc.data, QSFI_DTYPE_FP8_E4M3, xsc.count),
                                     quant_vector(multiplier.data, QSFI_DTYPE_F32, 1) };
        if (!check_status(qsfi_nvfp4_quantize(ctx, &q), QSFI_STATUS_OK, "FP4 quantization"))
            break;
        auto aq = xq.download(stream), as = xsc.download(stream);
        // Check quantization independently: block max/6, rounded E4M3 scale,
        // nearest E2M1 with ties to even. Test zero blocks and padding too.
        for (int r = 0; r < ((m + 127) / 128) * 128; ++r)
            for (int block = 0; block < k / 16; ++block) {
                float maxabs = 0;
                if (r < m)
                    for (int c = 0; c < 16; ++c)
                        maxabs = std::max(
                            maxabs,
                            std::abs(bf16_to_f32(hx[size_t(r) * k + block * 16 + c]))
                        );
                __nv_fp8_e4m3 expected(maxabs * 2.f / 6.f);
                const uint8_t actual = as[quant_sf_offset(r, block, k)];
                if (actual != expected.__x) {
                    ++failures;
                    std::fprintf(stderr, "FAIL: FP4 quantized block scale\n");
                    goto quantize_done;
                }
                if (r >= m)
                    continue;
                const float scale = quant_fp8_value(actual) / 2.f;
                for (int c = 0; c < 16; ++c) {
                    const size_t index = size_t(r) * k + block * 16 + c;
                    const float value = bf16_to_f32(hx[index]);
                    int best = 0;
                    float distance = INFINITY;
                    for (int code = 0; code < 8; ++code) {
                        float error = std::abs(std::abs(value) - quant_fp4_value(code) * scale);
                        if (error < distance || (error == distance && !(code & 1))) {
                            best = code;
                            distance = error;
                        }
                    }
                    const int got = (aq[index / 2] >> ((index % 2) * 4)) & 15;
                    if (scale == 0 ? quant_fp4_value(got) != 0
                                   : got != (best | (std::signbit(value) ? 8 : 0))) {
                        ++failures;
                        std::fprintf(
                            stderr,
                            "FAIL: FP4 quantized nibble got %d want %d\n",
                            got,
                            best
                        );
                        goto quantize_done;
                    }
                }
            }
    quantize_done:
        std::vector<double> reference(size_t(m) * n, 0);
        for (int r = 0; r < m; ++r)
            for (int col = 0; col < n; ++col) {
                double sum = 0;
                for (int c = 0; c < k; ++c) {
                    const size_t ai = size_t(r) * k + c, wi = size_t(col) * k + c;
                    const float av = quant_fp4_value((aq[ai / 2] >> ((ai % 2) * 4)) & 15)
                        * quant_fp8_value(as[quant_sf_offset(r, c / 16, k)]);
                    const float wv = quant_fp4_value((hw[wi / 2] >> ((wi % 2) * 4)) & 15)
                        * quant_fp8_value(hs[size_t(col) * (k / 16) + c / 16]);
                    sum += double(av) * wv;
                }
                reference[size_t(r) * n + col] = sum;
            }
        for (auto tactic : kNvfp4Tactics) {
            qsfi_nvfp4_plan_desc pd { uint32_t(m), k, n, tactic };
            qsfi_nvfp4_plan* plan = nullptr;
            size_t bytes = 0;
            if (!check_status(
                    qsfi_nvfp4_plan_create(ctx, &pd, &plan, &bytes),
                    QSFI_STATUS_OK,
                    "FP4 plan"
                ))
                break;
            QuantBuffer<uint8_t> workspace(bytes);
            qsfi_nvfp4_execute_desc d { q.out,
                                        quant_matrix(w.data, QSFI_DTYPE_NVFP4_E2M1, n, k),
                                        q.scales,
                                        target,
                                        quant_vector(alpha.data, QSFI_DTYPE_F32, 1),
                                        quant_matrix(y.data, QSFI_DTYPE_BF16, m, n),
                                        workspace.data,
                                        bytes };
            auto verify = [&](float global) {
                auto result = y.download(stream);
                double error = 0, norm = 0;
                for (size_t i = 0; i < result.size(); ++i) {
                    const double ref = reference[i] * global, diff = bf16_to_f32(result[i]) - ref;
                    error += diff * diff;
                    norm += ref * ref;
                }
                quant_check_error(error, norm, "FP4 GEMM");
            };
            if (check_status(qsfi_nvfp4_execute(ctx, plan, &d), QSFI_STATUS_OK, "FP4 execute"))
                verify(.0685f);
            else {
                qsfi_error_info err {};
                qsfi_context_get_last_error(ctx, &err);
                std::fprintf(stderr, "%s\n", err.message);
            }
            d.alpha.data = alpha2.data;
            check_status(qsfi_nvfp4_execute(ctx, plan, &d), QSFI_STATUS_OK, "FP4 rebind alpha");
            verify(.137f);
            auto bad = d;
            bad.x_scales.shape[0]--;
            check_status(
                qsfi_nvfp4_execute(ctx, plan, &bad),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP4 scale capacity rejection"
            );
            bad = d;
            bad.weight.stride[0] += 128;
            check_status(
                qsfi_nvfp4_execute(ctx, plan, &bad),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP4 stride rejection"
            );
            bad = d;
            bad.x.data = xq.data + 1;
            check_status(
                qsfi_nvfp4_execute(ctx, plan, &bad),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP4 alignment rejection"
            );
            if (bytes) {
                bad = d;
                bad.workspace_bytes--;
                check_status(
                    qsfi_nvfp4_execute(ctx, plan, &bad),
                    QSFI_STATUS_INVALID_ARGUMENT,
                    "FP4 workspace rejection"
                );
            }
            if (m == 5) {
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t executable = nullptr;
                check_cuda(
                    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                    "FP4 capture begin"
                );
                check_status(
                    qsfi_nvfp4_quantize(ctx, &q),
                    QSFI_STATUS_OK,
                    "FP4 captured quantization"
                );
                check_status(
                    qsfi_nvfp4_execute(ctx, plan, &d),
                    QSFI_STATUS_OK,
                    "FP4 captured GEMM"
                );
                check_cuda(cudaStreamEndCapture(stream, &graph), "FP4 capture end");
                check_cuda(
                    cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                    "FP4 graph instantiate"
                );
                for (int replay = 0; replay < 3; ++replay) {
                    const float global = .137f / (replay + 1);
                    alpha2.upload({ global });
                    check_cuda(cudaGraphLaunch(executable, stream), "FP4 graph replay");
                    verify(global);
                }
                alpha2.upload({ .137f });
                cudaGraphExecDestroy(executable);
                cudaGraphDestroy(graph);
            }
            qsfi_nvfp4_plan_destroy(plan);
        }
    }
    qsfi_nvfp4_plan* invalid = nullptr;
    size_t bytes = 123;
    qsfi_nvfp4_plan_desc bad { 1, 129, n, QSFI_NVFP4_TILE128X32_DP };
    check_status(
        qsfi_nvfp4_plan_create(ctx, &bad, &invalid, &bytes),
        QSFI_STATUS_INVALID_ARGUMENT,
        "FP4 K rejection"
    );
    bad.in_features = k;
    bad.tactic = static_cast<qsfi_nvfp4_tactic>(9);
    check_status(
        qsfi_nvfp4_plan_create(ctx, &bad, &invalid, &bytes),
        QSFI_STATUS_INVALID_ARGUMENT,
        "FP4 tactic rejection"
    );
    for (auto tactic : { QSFI_NVFP4_TILE128X32_DP,
                         QSFI_NVFP4_TILE128X32_STREAM_K,
                         QSFI_NVFP4_TILE128X64_DP,
                         QSFI_NVFP4_TILE128X64_STREAM_K }) {
        bool enabled = false;
        for (auto compiled : kNvfp4Tactics)
            enabled |= tactic == compiled;
        if (!enabled) {
            bad.tactic = tactic;
            check_status(
                qsfi_nvfp4_plan_create(ctx, &bad, &invalid, &bytes),
                QSFI_STATUS_INVALID_ARGUMENT,
                "FP4 disabled tactic rejection"
            );
        }
    }
    auto badtarget = target;
    badtarget.shape[0]--;
    check_status(
        qsfi_nvfp4_swizzle_scales(ctx, &source, &badtarget),
        QSFI_STATUS_INVALID_ARGUMENT,
        "FP4 swizzle extent rejection"
    );
    qsfi_context_destroy(ctx);
    cudaStreamDestroy(stream);
}

// Full Qwen dimensions with a separable reference, including the padded LM head.
// Different row/block scales expose swizzle mistakes without a CPU-sized GEMM.
__global__ void quant_fill_large_fp4(uint8_t* weights, uint8_t* scales, int n, int k)
{
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(n) * k / 2;
         i += size_t(blockDim.x) * gridDim.x) {
        const int c = (i % (k / 2)) * 2;
        weights[i] = uint8_t((c % 8) | (((c + 1) % 8) << 4));
    }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(n) * k / 16;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = i / (k / 16), block = i % (k / 16);
        scales[i] = uint8_t(48 + (row % 3) * 8 + (block % 2) * 8); // powers of two
    }
}

static void test_qsfi_nvfp4_qwen_shapes()
{
    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "FP4 Qwen stream"))
        return;
    qsfi_context* ctx = nullptr;
    qsfi_context_desc cd { 0, stream };
    if (!check_status(qsfi_context_create(&cd, &ctx), QSFI_STATUS_OK, "FP4 Qwen context"))
        return;
    QuantBuffer<float> multiplier(1), alpha(1);
    multiplier.upload({ 2.f });
    alpha.upload({ .0685f });
    const int shapes[][3] = { { 1, 17408, 5120 },
                              { 16, 5120, 17408 },
                              { 128, 17408, 5120 },
                              { 1, 248320, 5120 },
                              { 5, 248320, 5120 } };
    for (const auto& shape : shapes) {
        const int m = shape[0], n = shape[1], k = shape[2];
        QuantBuffer<uint8_t> w(size_t(n) * k / 2), ws(size_t(n) * k / 16), sw(quant_sf_size(n, k));
        quant_fill_large_fp4<<<1024, 256, 0, stream>>>(w.data, ws.data, n, k);
        check_cuda(cudaGetLastError(), "FP4 Qwen fixture");
        auto source = quant_matrix(ws.data, QSFI_DTYPE_FP8_E4M3, n, k / 16);
        auto target = quant_vector(sw.data, QSFI_DTYPE_FP8_E4M3, sw.count);
        check_status(
            qsfi_nvfp4_swizzle_scales(ctx, &source, &target),
            QSFI_STATUS_OK,
            "FP4 Qwen swizzle"
        );
        std::vector<uint16_t> host(size_t(m) * k);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = qscb_test_f32_to_bf16(float((i % 17) + 1) / 16);
        QuantBuffer<uint16_t> x(host.size()), y(size_t(m) * n);
        x.upload(host);
        QuantBuffer<uint8_t> aq(host.size() / 2), as(quant_sf_size(m, k));
        qsfi_nvfp4_quantize_desc q { quant_matrix(x.data, QSFI_DTYPE_BF16, m, k),
                                     quant_matrix(aq.data, QSFI_DTYPE_NVFP4_E2M1, m, k),
                                     quant_vector(as.data, QSFI_DTYPE_FP8_E4M3, as.count),
                                     quant_vector(multiplier.data, QSFI_DTYPE_F32, 1) };
        check_status(qsfi_nvfp4_quantize(ctx, &q), QSFI_STATUS_OK, "FP4 Qwen quantization");
        auto packed = aq.download(stream), scales = as.download(stream);
        std::vector<double> sums(m, 0);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < k; ++c) {
                const size_t index = size_t(r) * k + c;
                sums[r] += double(quant_fp4_value((packed[index / 2] >> (index % 2 * 4)) & 15))
                    * quant_fp8_value(scales[quant_sf_offset(r, c / 16, k)])
                    * quant_fp4_value(c % 8) * quant_fp8_value(48 + (c / 16 % 2) * 8);
            }
        for (auto tactic : kNvfp4Tactics) {
            qsfi_nvfp4_plan_desc pd { uint32_t(m), uint32_t(k), uint32_t(n), tactic };
            qsfi_nvfp4_plan* plan = nullptr;
            size_t bytes = 0;
            if (!check_status(
                    qsfi_nvfp4_plan_create(ctx, &pd, &plan, &bytes),
                    QSFI_STATUS_OK,
                    "FP4 Qwen plan"
                ))
                continue;
            QuantBuffer<uint8_t> work(bytes);
            qsfi_nvfp4_execute_desc d { q.out,
                                        quant_matrix(w.data, QSFI_DTYPE_NVFP4_E2M1, n, k),
                                        q.scales,
                                        target,
                                        quant_vector(alpha.data, QSFI_DTYPE_F32, 1),
                                        quant_matrix(y.data, QSFI_DTYPE_BF16, m, n),
                                        work.data,
                                        bytes };
            if (check_status(qsfi_nvfp4_execute(ctx, plan, &d), QSFI_STATUS_OK, "FP4 Qwen GEMM")) {
                const auto output = y.download(stream);
                double error = 0, norm = 0;
                for (int r = 0; r < m; ++r)
                    for (int col = 0; col < n; ++col) {
                        const double expected = sums[r] * (1 << (col % 3)) * .0685f;
                        const double diff = bf16_to_f32(output[size_t(r) * n + col]) - expected;
                        error += diff * diff;
                        norm += expected * expected;
                    }
                quant_check_error(error, norm, "FP4 full Qwen shape");
            }
            qsfi_nvfp4_plan_destroy(plan);
        }
    }
    qsfi_context_destroy(ctx);
    cudaStreamDestroy(stream);
}
