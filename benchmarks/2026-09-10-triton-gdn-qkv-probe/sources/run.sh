#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
probe=.prototypes/gdn_qkv_aot
mkdir -p "$probe/out"
uv sync --locked --project build_tools
build_tools/.venv/bin/python "$probe/compile.py" --out "$probe/out"
python3 "$probe/extract.py"
nvcc -std=c++17 -O3 -arch=sm_121 -I. -I"$probe/out" -DQSFI_ENABLE_CHECKED_VALIDATION=1 \
    "$probe/bench.cu" qscb.cu -lcuda -lcublasLt -o "$probe/out/probe"
{
    date -u
    git rev-parse HEAD
    nvcc --version
    nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv
    ldd "$probe/out/probe"
    sha256sum qscb.cu "$probe/bench.cu" build_tools/uv.lock
} > "$probe/out/host.txt"
for layer in 0 18 38; do
    for mode in device managed_upload managed_cpu; do
        result="$probe/out/layer${layer}_${mode}"
        "$probe/out/probe" "$probe/out" "$mode" "$probe/out/layer${layer}.bf16" > "$result.csv" 2> "$result.log"
        cat "$result.log" "$result.csv"
    done
done
