#define main qs3_original_benchmark_main
#include "../../bench_native.cu"
#undef main
int main() {
    Options options {};
    options.warmups = 100;
    options.iters = 1000;
    BenchState state {};
    if (!create_state(&state)) { destroy_state(&state); return 1; }
    for (uint32_t tokens : {1u, 16u, 1024u}) {
        if (!bench_router_topk(state, options, tokens)) { destroy_state(&state); return 1; }
    }
    destroy_state(&state);
    return 0;
}
