cuda_root := env_var_or_default("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib:" + cuda_root + "/lib/stubs:"
qwen36_vectors := "build_tools/.venv/bin/qwen36-vectors"

fmt:
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
        cargo fmt
build: copy-ninja
        ninja -C build
        cargo build --lib

test: build cargo-test cuda-test
cuda-test: copy-ninja
        ninja -C build tests
        build/qsfi_test_checked
        build/qsfi_test_release
cargo-test: build generate-vectors
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test

generate-vectors: uv-sync copy-ninja
        ninja -C build vectors/qwen36_semantics/.oracle-ok

uv-sync:
        uv sync --locked --project build_tools

refresh-vector-oracles: uv-sync
        {{qwen36_vectors}} generate-all --output-root build/vectors/qwen36_semantics --clean
        {{qwen36_vectors}} write-oracles --input-root build/vectors/qwen36_semantics --oracle-root build_tools/qwen36-vectors/oracle_hashes

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
