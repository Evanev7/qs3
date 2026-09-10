#include "../../qscu_gdn.cu"
#include "kernels.inc"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
void ck(cudaError_t e)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(e));
        exit(1);
    }
}
template <class T> T* up(const std::vector<T>& v)
{
    T* p;
    ck(cudaMalloc(&p, v.size() * sizeof(T)));
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> std::vector<T> down(T* p, size_t n)
{
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
template <class T> T* data(size_t n, unsigned seed)
{
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = T(float(int((i * 17 + seed) % 101) - 50) / 256.f);
    return up(v);
}
template <class F> float bench(F f)
{
    for (int i = 0; i < 3; ++i)
        f();
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a));
    ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a));
    for (int i = 0; i < 20; ++i)
        f();
    ck(cudaEventRecord(b));
    ck(cudaEventSynchronize(b));
    float ms;
    ck(cudaEventElapsedTime(&ms, a, b));
    ck(cudaEventDestroy(a));
    ck(cudaEventDestroy(b));
    return ms / 20;
}
template <class S> void run(unsigned tokens, bool norm)
{
    const unsigned qh = 16, vh = 32, d = 128;
    const size_t sn = size_t(2) * vh * d * d, on = size_t(tokens) * vh * d;
    auto* q = data<__nv_bfloat16>(size_t(tokens) * qh * d, 1);
    auto* k = data<__nv_bfloat16>(size_t(tokens) * qh * d, 7);
    auto* v = data<__nv_bfloat16>(on, 13);
    auto* a = data<__nv_bfloat16>(size_t(tokens) * vh, 5);
    auto* b = data<__nv_bfloat16>(size_t(tokens) * vh, 9);
    auto* al = data<__nv_bfloat16>(vh, 11);
    auto* dt = data<__nv_bfloat16>(vh, 17);
    auto* old = data<S>(sn, 29);
    auto* next = data<S>(sn, 29);
    auto* out = data<__nv_bfloat16>(on, 2);
    auto* out2 = data<__nv_bfloat16>(on, 2);
    auto* ind = up(std::vector<int> { 0, int(tokens) });
    auto* rd = up(std::vector<int> { 0 });
    auto* wr = up(std::vector<int> { 1 });
    gdn_kernel_params p {};
    p.q = { q, qh * d, d, 1 };
    p.k = { k, qh * d, d, 1 };
    p.v = { v, vh * d, d, 1 };
    p.a = { a, vh, 1 };
    p.b = { b, vh, 1 };
    p.a_log = { al, 1 };
    p.dt_bias = { dt, 1 };
    p.state = { vh * d * d, d * d, d, 1 };
    p.seq_indptr = ind;
    p.state_indices = rd;
    p.state_out_indices = wr;
    p.out = { out, vh * d, d, 1 };
    p.outer_count = 1;
    p.num_q_heads = qh;
    p.num_k_heads = qh;
    p.num_v_heads = vh;
    p.key_dim = d;
    p.value_dim = d;
    p.scale = 1.f / sqrtf(float(d));
    p.use_qk_l2norm = norm;
    gdn_prefill_kernel<<<vh * d, 128>>>(p, old);
    p.out.data = out2;
    gdn_warp_kernel<S, true><<<(vh * d + 3) / 4, 128>>>(p, next);
    ck(cudaDeviceSynchronize());
    auto ho = down(out, on), hn = down(out2, on);
    auto so = down(old, sn), snew = down(next, sn);
    size_t badout = 0, badstate = 0;
    for (size_t i = 0; i < on; ++i)
        badout += memcmp(&ho[i], &hn[i], sizeof(ho[i])) != 0;
    for (size_t i = 0; i < sn; ++i)
        badstate += memcmp(&so[i], &snew[i], sizeof(S)) != 0;
    if (badout || badstate) {
        fprintf(
            stderr,
            "mismatch tokens=%u norm=%d state=%zu outputs=%zu state_values=%zu\n",
            tokens,
            norm,
            sizeof(S),
            badout,
            badstate
        );
        exit(1);
    }
    if (tokens == 1024 || tokens == 1) {
        p.out.data = out;
        float t0 = bench([&] { gdn_prefill_kernel<<<vh * d, 128>>>(p, old); });
        p.out.data = out2;
        float t1 = bench([&] { gdn_warp_kernel<S, true><<<(vh * d + 3) / 4, 128>>>(p, next); });
        printf("%u\t%d\t%zu\t%.6f\t%.6f\n", tokens, norm, sizeof(S), t0, t1);
        fflush(stdout);
    }
    for (void* p : { (void*)q,
                     (void*)k,
                     (void*)v,
                     (void*)a,
                     (void*)b,
                     (void*)al,
                     (void*)dt,
                     (void*)old,
                     (void*)next,
                     (void*)out,
                     (void*)out2,
                     (void*)ind,
                     (void*)rd,
                     (void*)wr })
        ck(cudaFree(p));
}
int main()
{
    puts("tokens\tnorm\tstate_bytes\tblock_ms\twarp_ms");
    for (unsigned n : { 1, 2, 4, 31, 128, 1024 })
        for (bool norm : { false, true }) {
            run<float>(n, norm);
            run<__nv_bfloat16>(n, norm);
        }
    puts("All output and state bits match");
}
