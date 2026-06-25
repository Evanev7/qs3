cuda_root := `dirname $(dirname $(readlink -f $(which nvcc)))`
cflags := "-std=c11 -Wall -Wextra -Werror"
debug_cflags := cflags + " -Og -g3 -fno-omit-frame-pointer"
san_cflags := debug_cflags + " -fsanitize=address,undefined"
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
        gcc {{cflags}} -I{{cuda_root}}/include -c session.c -o build/session.o
test: append-test session-test
append-test: build
        nvcc -std=c++17 -arch=sm_121 test.cu build/qsfi.o -o build/qsfi_test {{nvccflags}}
        build/qsfi_test
session-test-bin: build
        gcc {{cflags}} -I{{cuda_root}}/include -c session_test.c -o build/session_test.o
        nvcc -std=c++17 -arch=sm_121 build/session_test.o build/session.o build/qsfi.o -o build/session_test {{nvccflags}}
session-test: session-test-bin
        build/session_test
session-test-debug: build
        gcc {{debug_cflags}} -I{{cuda_root}}/include -c session.c -o build/session_debug.o
        gcc {{debug_cflags}} -I{{cuda_root}}/include -c session_test.c -o build/session_test_debug.o
        nvcc -std=c++17 -arch=sm_121 build/session_test_debug.o build/session_debug.o build/qsfi.o -o build/session_test_debug {{nvccflags}}
        build/session_test_debug
session-test-asan: build
        gcc {{san_cflags}} -I{{cuda_root}}/include -c session.c -o build/session_asan.o
        gcc {{san_cflags}} -I{{cuda_root}}/include -c session_test.c -o build/session_test_asan.o
        nvcc -std=c++17 -arch=sm_121 build/session_test_asan.o build/session_asan.o build/qsfi.o -o build/session_test_asan {{nvccflags}}
        ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 build/session_test_asan
session-test-memcheck: session-test-bin
        compute-sanitizer --tool memcheck --leak-check full build/session_test
fmt: always
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
always:
        build_tools/generate_macros.c > build/qsfi_macros.h
