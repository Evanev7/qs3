build: always
        mkdir -p build
        nvcc -std=c++17 -arch=sm_121 -c qsfi.cu -o build/qsfi.o \
          -I3pty/flashinfer/3rdparty/cccl/libcudacxx/include \
          -I3pty/flashinfer/3rdparty/cccl/cub \
          -I3pty/flashinfer/3rdparty/cccl/thrust \
          -I3pty/flashinfer/3rdparty/cccl/cudax/include \
          -I3pty/flashinfer/include \
          -Ibuild
test: append-test
append-test: build
        nvcc -std=c++17 -arch=sm_121 test.cu build/qsfi.o -o build/qsfi_test \
          -I3pty/flashinfer/3rdparty/cccl/libcudacxx/include \
          -I3pty/flashinfer/3rdparty/cccl/cub \
          -I3pty/flashinfer/3rdparty/cccl/thrust \
          -I3pty/flashinfer/3rdparty/cccl/cudax/include \
          -I3pty/flashinfer/include \
          -Ibuild
        build/qsfi_test
fmt: always
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
always:
        build_tools/generate_macros.c > build/qsfi_macros.h
