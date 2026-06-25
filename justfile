cuda_root := env_var_or_default("CUDA_HOME", `dirname $(dirname $(readlink -f $(which nvcc)))`)
cuda_lib_path := cuda_root + "/lib64:" + cuda_root + "/lib"
nvccflags := "-I3pty/flashinfer/3rdparty/cccl/libcudacxx/include \
-I3pty/flashinfer/3rdparty/cccl/cub \
-I3pty/flashinfer/3rdparty/cccl/thrust \
-I3pty/flashinfer/3rdparty/cccl/cudax/include \
-I3pty/flashinfer/include \
-Ibuild
"

build: always
        mkdir -p build
        nvcc -std=c++17 -arch=sm_121 -c qsfi.cu -o build/qsfi.o {{nvccflags}}
        cargo build --lib
test: append-test session-test
append-test: build
        nvcc -std=c++17 -arch=sm_121 test.cu build/qsfi.o -o build/qsfi_test {{nvccflags}}
        build/qsfi_test
session-test: build
        LIBRARY_PATH="{{cuda_lib_path}}:${LIBRARY_PATH:-}" cargo test --test session
fmt: always
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
always:
        build_tools/generate_macros.c > build/qsfi_macros.h
