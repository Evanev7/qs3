cuda_root := env_var_or_default("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib"

build:
        ninja
        cargo build --lib
test: append-test engine-test
append-test:
        ninja build/qsfi_test
        build/qsfi_test
engine-test: build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --test engine
fmt:
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
