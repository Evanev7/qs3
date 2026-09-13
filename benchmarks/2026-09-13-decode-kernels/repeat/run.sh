#!/usr/bin/env bash
set -euo pipefail
# Invoked through run_cuda_test.sh after the full profiled core run completes.
repeat_dir=".prototypes/out/decode-repeat-$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "$repeat_dir"
git rev-parse HEAD > "$repeat_dir/commit.txt"
git diff HEAD > "$repeat_dir/source.diff"
driver_libs=$(mktemp -d)
trap 'rm -rf "$driver_libs"' EXIT
ln -s /lib/aarch64-linux-gnu/libcuda.so* /lib/aarch64-linux-gnu/libnvidia-*.so* "$driver_libs/"
for kernel in tile32_blocks96 decode_gemv64; do
    echo "Measuring $kernel in $repeat_dir" >&2
    QS3_BENCH_CONTEXT_TOKENS=102 QS3_BENCH_DECODE_SAMPLES=32 \
    QS3_BENCH_MOE_KERNEL="$kernel" QS3_BENCH_GDN_STATE=f32 \
    LD_LIBRARY_PATH="$driver_libs" nix run --impure .#benchmark -- --measure-pass \
        > "$repeat_dir/$kernel.json" 2> "$repeat_dir/$kernel.stderr.log"
done
printf '%s\n' "$repeat_dir"
