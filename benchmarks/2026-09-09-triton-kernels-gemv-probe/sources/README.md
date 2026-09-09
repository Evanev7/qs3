# Upstream triton_kernels LM-head comparison

Run `bash .prototypes/triton_kernels_gemv/run.sh` from the repository root.
This uses the disposable sp10 checkout, so run sequentially with the normal
test and core benchmark scripts.

The runner extracts the library from the vendored repo’s `v3.8.0` tag
(`c01b6774b1865984607d89d89d3a10833de92037`) into `upstream/`. The exporter imports the unchanged
`upstream/triton_kernels/matmul_details/_matmul.py`.
The newer vendored HEAD imports `triton._compile_warmup_state`, which is absent
from the pinned compiler wheel. The matching release avoids that dependency.
It specializes dense BF16 inputs/weights and FP32 output for M=1, N=248320,
K=2048, with the existing row-major [N,K] weight storage. No weight transpose
or repacking is performed. Optional scaling, routing, bias and epilogues are
disabled. The public Python matmul wrapper is not timed or invoked.

The regular matmul path is appropriate for the upstream FP32-output heuristic.
Four explicitly selected tile configurations are compiled: BLOCK_M=16 and
(BLOCK_N, BLOCK_K, warps, stages) = (256,64,2,2), (128,64,2,4), (64,64,2,4),
(32,128,4,4). The first matches the regular NVIDIA heuristic's tile dimensions
and warp count. This is a bounded tile sweep, not upstream autotuning or an
exhaustive search. Persistent/TMA and BF16-output paths are not tested here.
The copied b12x row/eight-warps kernel provides an in-run control.

The isolated `uv run` compilation environment contains Triton 3.8.0 and CPU
Torch 2.10.0 (needed by upstream imports); it does not modify the build-tools
workspace environment. Compilation uses ASTSource and an explicit SM121 target.
The generated header, metadata exporter and native harness are qs3 prototype
code. The matmul implementation and its imported helpers are upstream code.

The exporter supplies 16-byte pointer-alignment hints, matching the normal
launcher’s specialization for these cudaMalloc buffers. All variants use
`enable_fp_fusion=False`, as in the earlier probes.

The native harness loads cubins and launches via the CUDA driver. The matmul
ABI is output/input/weight pointers, followed by Triton's scratch pointers;
b12x uses input/weight/output instead. All dimensions, strides and optional
arguments have been specialized at build time. No Python or JIT runs during
measurement.

The fixtures, full-output cuBLASLt comparison, sampled CPU double reference,
CUDA-event timing, cache-eviction procedure and alternating provider order
are copied from the b12x probe. Results are synthetic eager LM-head timings,
not real-model correctness or end-to-end TPS. Raw outputs land in `sp10/`.
