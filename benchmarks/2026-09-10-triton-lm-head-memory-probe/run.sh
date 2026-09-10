#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
probe=build/lm-head-memory-probe
mkdir -p "$probe"
just copy-ninja
ninja -C build triton/lm_head.rs
python3 - <<'PY'
import json
from pathlib import Path
source = Path('build/triton')
output = Path('build/lm-head-memory-probe')
m = json.loads((source / 'lm_head.json').read_text())
meta = m['metadata']
(output / 'lm_head.cubin').write_bytes((source / 'lm_head.cubin').read_bytes())
(output / 'lm_head.json').write_text(json.dumps(m, indent=2) + '\n')
(output / 'kernels.h').write_text('''struct KernelSpec { const char *name, *file, *symbol;
int n, k, rows, warps, shared; bool fp32; };
static const KernelSpec kernels[] = {
{"row", "lm_head.cubin", "%s", %d, %d, 1, %d, %d, true}};
''' % (meta['name'], m['grid'][0], m['constexprs']['K'], meta['num_warps'], meta['shared']))
PY
nvcc -std=c++17 -O3 -arch=sm_121 -I. -I"$probe" -DQSFI_ENABLE_CHECKED_VALIDATION=1 \
    benchmarks/2026-09-10-triton-lm-head-memory-probe/bench.cu qscb.cu \
    -lcuda -lcublasLt -o "$probe/probe"
{
    date -u
    nvcc --version
    nvidia-smi --query-gpu=name,driver_version --format=csv
    ldd "$probe/probe"
    sha256sum "$probe/lm_head.cubin" qscb.cu
} > "$probe/host.txt"
for mode in device managed_gpu managed_cpu; do
    "$probe/probe" "$probe" "$mode" > "$probe/$mode.csv" 2> "$probe/$mode.log"
    cat "$probe/$mode.log" "$probe/$mode.csv"
done
