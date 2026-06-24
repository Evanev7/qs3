maybe-build:
        #!/usr/bin/env bash
        if [ ! -e build/qsfi.o ] \
        || [ qsfi.cu -nt build/qsfi.o ] \
        || [ qsfi.h -nt build/qsfi.o ];
        then just build;
        fi
build: always
        mkdir -p build
        nvcc -std=c++17 -Ibuild -c qsfi.cu -o build/qsfi.o
test: maybe-build
        nvcc -std=c++17 test.cu build/qsfi.o -o build/qsfi_test
        build/qsfi_test
fmt: always
        rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
always:
        build_tools/generate_macros.c > build/qsfi_macros.h
