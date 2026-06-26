cuda_root := env_var_or_default("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib:"

fmt:
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
build: copy-ninja
        ninja -C build
        cargo build --lib

test: append-test cargo-test
append-test: copy-ninja
        ninja -C build qsfi_test
        build/qsfi_test
cargo-test: build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test

copy-ninja:
        cp build_tools/build.ninja build/build.ninja
