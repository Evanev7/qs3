maybe-build:
        #!/usr/bin/env bash
        if [ ! -e build/qsfi.o ] \
        || [ flashinfer.cu -nt build/qsfi.o ] \
        || [ flashinfer.h -nt build/qsfi.o ];
        then just build;
        fi
build:
        mkdir -p build
        build_tools/generate_macros.c > flashinfer_macros.h
        nvcc -std=c++17 -I3pty/flashinfer/include -c flashinfer.cu -o build/qsfi.o
test: maybe-build
    nvcc -std=c++17 -I3pty/flashinfer/include test.cu build/qsfi.o -o build/qsfi_test
    build/qsfi_test
fmt:
    rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
