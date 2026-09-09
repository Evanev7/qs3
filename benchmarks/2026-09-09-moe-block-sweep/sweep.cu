#define main qs3_original_benchmark_main
#include "../../bench_native.cu"
#undef main
int qs3_moe_threadblocks = 4;
int main() {
    Options options {};
    options.warmups = 10;
    options.iters = 100;
    BenchState state {};
    if (!create_state(&state)) { destroy_state(&state); return 1; }
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int blocks : {4, 12, 24, 48, 96}) {
            qs3_moe_threadblocks = blocks;
            for (uint32_t tokens : {1u, 16u, 102u}) {
                std::printf("repeat=%d blocks=%d ", repeat, blocks);
                if (!bench_moe_execute_bf16(state, options, tokens)) {
                    destroy_state(&state); return 1;
                }
                std::fflush(stdout);
            }
        }
    }
    destroy_state(&state);
    return 0;
}
