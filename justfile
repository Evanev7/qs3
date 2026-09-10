cuda_root := env_var_or_default("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib:" + cuda_root + "/lib/stubs:"

fmt:
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
        cargo fmt
        just build_tools/fmt
check:
        just build_tools/check
build: copy-ninja
        ninja -C build
        cargo build --lib

test: build cargo-test cuda-test
cuda-test: copy-ninja
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

bench *args: copy-ninja
        ninja -C build bench
        build/qsfi_bench_native {{args}}

model-tps: copy-ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo run --release --bin qs3-bench

weight-loader-bench: copy-ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --release bench_real_qwen36_bf16_ -- --ignored --nocapture --test-threads=1

real-model-test: copy-ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --release loader::tests::real_qwen36_bf16_generates_reference_tokens -- --ignored --exact --nocapture --test-threads=1

copy-ninja:
        mkdir -p build
        cp build_tools/build.ninja build/build.ninja

render-benchmark-history:
        python3 benchmarks/render_history.py
        firefox benchmarks/performance.html
