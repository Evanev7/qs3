cuda_root := env_var_or_default("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib:" + cuda_root + "/lib/stubs:"

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

generate-vectors: copy-ninja
        ninja -C build vectors/qwen36_semantics/.oracle-ok

refresh-vector-oracles:
        python3 build_tools/qwen36-vectors/src/qwen36_vectors/__main__.py generate-all --output-root build/vectors/qwen36_semantics --clean
        python3 build_tools/qwen36-vectors/src/qwen36_vectors/__main__.py write-oracles --input-root build/vectors/qwen36_semantics

bench *args: copy-ninja
        ninja -C build bench
        build/qsfi_bench_native {{args}}

model-bench *args: copy-ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo run --release --bin qs3_model_bench -- {{args}}

weight-loader-bench: copy-ninja
        ninja -C build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --release bench_real_qwen36_bf16_ -- --ignored --nocapture --test-threads=1

copy-ninja:
        mkdir -p build
        cp build_tools/build.ninja build/build.ninja
