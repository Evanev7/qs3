#include "../../qscu.cu"
#include "kernels.inc"
#include <cstdio>
#include <cstdlib>
#include <vector>

void ck(cudaError_t s)
{
    if (s != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(s));
        exit(1);
    }
}
template <class T> T* upload(const std::vector<T>& v)
{
    T* p;
    ck(cudaMalloc(&p, v.size() * sizeof(T)));
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> std::vector<T> download(T* p, size_t n)
{
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
template <class F> float bench(F f)
{
    for (int i = 0; i < 5; ++i)
        f();
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a));
    ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a));
    for (int i = 0; i < 50; ++i)
        f();
    ck(cudaEventRecord(b));
    ck(cudaEventSynchronize(b));
    float ms;
    ck(cudaEventElapsedTime(&ms, a, b));
    ck(cudaEventDestroy(a));
    ck(cudaEventDestroy(b));
    return ms / 50;
}

template <class S> void run(unsigned tokens, bool same_slot, unsigned batch, bool timing)
{
    const unsigned dim = 8192, slots = 4;
    std::vector<__nv_bfloat16> x(size_t(tokens) * dim), w(size_t(dim) * 4), z(x.size());
    std::vector<S> initial(size_t(slots) * dim * 3);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2bfloat16(float(int(i % 37) - 18) * 0.03125f);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2bfloat16(float(int(i % 13) - 6) * 0.0625f);
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = S(float(int(i % 31) - 15) * 0.021f);
    std::vector<int32_t> indptr = batch == 1
        ? std::vector<int32_t> { 0, int(tokens) }
        : std::vector<int32_t> { 0, int(tokens / 3), int(tokens) };
    std::vector<int32_t> reads
        = batch == 1 ? std::vector<int32_t> { 0 } : std::vector<int32_t> { 0, -1 };
    std::vector<int32_t> writes = batch == 1 ? std::vector<int32_t> { same_slot ? 0 : 2 }
                                             : std::vector<int32_t> { same_slot ? 0 : 2, 3 };
    auto* dx = upload(x);
    auto* dw = upload(w);
    auto* a = upload(z);
    auto* b = upload(z);
    auto* sa = upload(initial);
    auto* sb = upload(initial);
    auto* di = upload(indptr);
    auto* dr = upload(reads);
    auto* ds = upload(writes);
    conv1d_params p {};
    p.x = dx;
    p.x_stride0 = dim;
    p.x_stride1 = 1;
    p.weight = dw;
    p.weight_stride0 = 4;
    p.weight_stride1 = 1;
    p.state_stride0 = dim * 3;
    p.state_stride1 = 3;
    p.state_stride2 = 1;
    p.read_indices = dr;
    p.write_indices = ds;
    p.seq_indptr = di;
    p.out = a;
    p.out_stride0 = dim;
    p.out_stride1 = 1;
    p.conv_dim = dim;
    p.activation = QSCU_ACTIVATION_SILU;
    p.update_state = 1;
    qwen36_gdn_causal_conv1d_kernel<<<batch, 256>>>(p, sa);
    p.out = b;
    launch_parallel(p, sb, tokens, batch);
    ck(cudaDeviceSynchronize());
    auto ha = download(a, x.size()), hb = download(b, x.size());
    auto hsa = download(sa, initial.size()), hsb = download(sb, initial.size());
    if (memcmp(ha.data(), hb.data(), ha.size() * sizeof(ha[0]))
        || memcmp(hsa.data(), hsb.data(), hsa.size() * sizeof(S))) {
        fprintf(
            stderr,
            "mismatch rows=%u batch=%u inplace=%d state_bytes=%zu\n",
            tokens,
            batch,
            same_slot,
            sizeof(S)
        );
        exit(1);
    }
    if (timing) {
        p.update_state = 1;
        p.out = a;
        float old = bench([&] { qwen36_gdn_causal_conv1d_kernel<<<batch, 256>>>(p, sa); });
        p.out = b;
        float next = bench([&] { launch_parallel(p, sb, tokens, batch); });
        printf("%u\t%u\t%zu\t%.6f\t%.6f\n", tokens, batch, sizeof(S), old, next);
    }
    for (void* ptr : { (void*)dx,
                       (void*)dw,
                       (void*)a,
                       (void*)b,
                       (void*)sa,
                       (void*)sb,
                       (void*)di,
                       (void*)dr,
                       (void*)ds })
        ck(cudaFree(ptr));
}
int main()
{
    puts("tokens\tbatch\tstate_bytes\tserial_ms\tparallel_ms");
    for (unsigned n : { 1, 2, 3, 4, 5, 31, 32, 33, 1024 })
        for (bool same : { false, true })
            for (unsigned batch : { 1, 2 }) {
                run<__nv_bfloat16>(n, same, batch, n == 1024 && !same && batch == 1);
                run<float>(n, same, batch, false);
            }
    puts("All outputs and final states match bitwise");
}
