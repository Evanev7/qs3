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
cargo-test: build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test

copy-ninja:
        mkdir -p build
        cp build_tools/build.ninja build/build.ninja
