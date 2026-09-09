# triton_kernels vocabulary matmul on GB10

The unchanged upstream regular matmul kernel passes the LM-head probe, but the
best of four tested tiles remains about 5% slower than the copied b12x GEMV.
Keep b12x row/eight-warps as the next real-model candidate.

BF16 inputs and weights, FP32 output, M=1, N=248320, K=2048. Median of three
50-sample eager CUDA-event p50 measurements:

| Kernel | BLOCK_M/N/K | Warps / stages | Repeated weights | After eviction | Matched cuBLASLt after eviction |
| --- | --- | --- | ---: | ---: | ---: |
| b12x row | one output row | 8 / 1 | **4.076 ms** | **4.123 ms** | 5.717 ms |
| triton_kernels | 16/256/64 | 2 / 2 | 4.447 ms | 4.503 ms | 5.691 ms |
| triton_kernels | 16/128/64 | 2 / 4 | 4.414 ms | 4.467 ms | 5.685 ms |
| triton_kernels | 16/64/64 | 2 / 4 | 4.453 ms | 4.507 ms | 5.696 ms |
| triton_kernels | 16/32/128 | 4 / 4 | **4.293 ms** | **4.345 ms** | 5.693 ms |

The best tested upstream matmul reduces eviction-case latency by 23.7% against
its paired cuBLASLt control. The b12x kernel saves a further 0.221 ms versus
that matmul. These results cover this synthetic single-token LM-head shape;
they do not establish a general ranking for GEMM, prefill or MoE.

All five specializations pass comparison with cuBLASLt across all 248320
outputs and CPU double accumulation at 65 rows, including both boundaries.
Maximum absolute differences against cuBLASLt are 2.15e-5 for each matmul tile
and 1.07e-6 for b12x. Gates are unchanged: 0.0002 absolute plus 0.0002 relative.

## Source and AOT build

The library is from Triton's `v3.8.0` tag, commit
`c01b6774b1865984607d89d89d3a10833de92037`, matching the pinned Triton 3.8.0
wheel. The newer vendored HEAD (`2074a1b8a5cd283ce584f8ebca825c4f0b215fc1`)
cannot be imported with that wheel: its scale-layout code imports the absent
internal `triton._compile_warmup_state` module. No compiler shim or library
source modification is used; the runner extracts the matching release.

`matmul_details/_matmul.py` and its helpers are upstream code. The Python
exporter, generated header and C++ fixture/launcher are qs3 prototype code.
The exporter specializes away optional bias, scales, routing, activation and
epilogue features, and supplies fixed dimensions and strides. The runtime
matmul ABI is Y/X/W plus two unused Triton scratch pointers. Weight storage
stays row-major [N,K], with no repacking or transpose pass.

The first tile matches the regular NVIDIA heuristic's dimensions, warp count
and stage count for this device. The other three tiles are explicit choices.
The FP32-output heuristic selects regular matmul; persistent/TMA and BF16
output are outside this probe. The public Python wrapper and its dispatch
are not invoked, and this is not an exhaustive tuning sweep.

Compilation uses an isolated uv environment with Triton 3.8.0 and CPU Torch
2.10.0, needed by upstream imports. The build-tools workspace dependencies
are unchanged by this experiment. Compiler target is SM121, with
`enable_fp_fusion=False` for every candidate, matching the previous probes.
The exporter supplies the 16-byte pointer-alignment hints that the normal
launcher specializes on for these cudaMalloc buffers. An initial build
omitting those hints caused the 256-column tile to use 4920 bytes of local
memory and take roughly 60 ms; the corrected build uses zero local bytes and
takes 4.50 ms. Preserve alignment information in a production AOT exporter.
Final resource attributes are recorded in `validation.log`.

The native executable loads cubins through the CUDA driver and launches them
without Python or JIT. Host provenance records CUDA toolkit 13.0.88 and driver
580.142; the ARM Triton wheel's actual ptxas 13.3.33 path/version/hash and all
cubin/source hashes are in `manifest.json`.

## Reproduction and limits

Restore `sources/` into `.prototypes/triton_kernels_gemv/` and run
`bash .prototypes/triton_kernels_gemv/run.sh` from the repository root. It
extracts the upstream release from the vendored repo. A matching source archive
is also retained here; its root is the `triton_kernels` package contents.
The runner uses the disposable sp10 checkout and must run sequentially with
the normal test/core benchmark scripts. Full cubins, PTX and executable remain
in `.prototypes/triton_kernels_gemv/sp10/` and the remote prototype directory.

Signed synthetic BF16 fixtures use cudaMalloc. Ten warmups precede each
50-sample batch; three repetitions alternate provider order. Eviction touches
at least 128 MiB / four times L2 before each sample, outside the timed interval.
Clocks are not locked, and cache residency is not verified by counters.
No real-model score, managed-weight or end-to-end TPS claim follows from this
microbenchmark.
