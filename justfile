cuda_root := env_var_or_default("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib:" + cuda_root + "/lib/stubs:"

fmt:
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
        cargo fmt
        just build_tools/fmt
check:
        just build_tools/check
build: ninja
        ninja -C build
        cargo build --lib

test:
        just build_tools/python-test triton-test
        just cargo-test cuda-test

cuda-test: ninja
        ninja -C build tests
        build/qsfi_test_checked
        build/qsfi_test_release
        build/qsfi_test_qwen27_checked
        build/qsfi_test_qwen27_release
cargo-test: build _generate-vectors
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test

_generate-vectors:
        just build_tools/generate-vectors

norm-test: build _generate-vectors
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --test vector_harness qwen36_norm_concurrent_widths -- --nocapture

[positional-arguments]
model-test repetitions="20" *args: build _generate-vectors
        #!/usr/bin/env bash
        set -euo pipefail
        export LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}"
        repetitions=$1
        shift
        for ((iteration=1; iteration<=repetitions; iteration++)); do
            echo "Model test repetition $iteration"
            cargo test --test model -- "$@"
        done

norm-validation: test (model-test "20") real-model-test

[positional-arguments]
scores-test run: build _generate-vectors
        QS3_SCORE_INPUT="$HOME/qs3-scores/$1/input.json" QS3_SCORE_OUTPUT="$HOME/qs3-scores/$1/qs3" LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --lib real_qwen36_same_prefix_scores -- --ignored --nocapture

weight-loader-compare: build _generate-vectors
        #!/usr/bin/env bash
        set -euo pipefail
        export LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}"
        for backend in managed_uma pinned_upload managed_uma; do
            echo "Weight-load backend: $backend"
            date -u
            grep -E 'MemAvailable:|^Cached:|SwapFree:' /proc/meminfo
            cargo test --lib "bench_real_qwen36_bf16_${backend}_load" -- --ignored --nocapture --test-threads=1
        done

[positional-arguments]
weight-loader-cold-bench snapshot: build _generate-vectors
        #!/usr/bin/env bash
        set -euo pipefail
        export LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}"
        export QS3_QWEN36_MODEL_DIR="$1"
        for backend in managed_uma pinned_upload; do
            echo "Cold weight-load backend: $backend"
            date -u
            python3 benchmarks/2026-09-09-weight-load-comparison/cold_files.py "$1"
            grep -E 'MemAvailable:|^Cached:|SwapFree:' /proc/meminfo
            cargo test --lib "bench_real_qwen36_bf16_${backend}_load" -- --ignored --nocapture --test-threads=1
        done

bench *args: ninja
        ninja -C build bench
        build/qsfi_bench_native {{args}}

model-tps: ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo run --release --bin qs3-bench

benchmark: (_benchmark "102" "32") (_benchmark "1024" "256")

_benchmark context_tokens decode_samples:
        #!/usr/bin/env bash
        set -euo pipefail
        driver_libs=$(mktemp -d)
        trap 'rm -rf "$driver_libs"' EXIT
        ln -s /lib/aarch64-linux-gnu/libcuda.so* /lib/aarch64-linux-gnu/libnvidia-*.so* "$driver_libs/"
        # Keep host CUDA and system libraries out of the Nix runtime's search path.
        QS3_BENCH_CONTEXT_TOKENS={{context_tokens}} QS3_BENCH_DECODE_SAMPLES={{decode_samples}} \
        QS3_BENCH_MOE_KERNEL=decode_gemv64 QS3_BENCH_GDN_STATE=f32 \
        LD_LIBRARY_PATH="$driver_libs" nix run --impure .#benchmark

weight-loader-bench: ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --release bench_real_qwen36_bf16_ -- --ignored --nocapture --test-threads=1

real-model-test: ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --release loader::tests::real_qwen36_bf16_generates_reference_tokens -- --ignored --exact --nocapture --test-threads=1

ninja:
        mkdir -p build/triton
        nix eval --offline --raw --file build_tools/nixsrc/ninja.nix > build/triton/kernels.ninja
        cp build_tools/build.ninja build/build.ninja
        cp build_tools/cuda.ninja build/cuda.ninja

render-benchmark-history:
        python3 benchmarks/render_history.py
        firefox benchmarks/performance.html
