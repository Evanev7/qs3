# Copied b12x vocabulary GEMV on GB10

Copied b12x's `_row_kernel` and `_row_loop_kernel` without changing their
bodies, compiled them AOT with the pinned Triton 3.8.0 wheel, and benchmarked
against prepared qs3 cuBLASLt and the earlier custom row kernel. All eight
specializations pass full-output cuBLASLt comparisons and sampled CPU double
references. The copied row kernel at eight warps is the fastest tested choice.

FP32 output, median of three 50-sample CUDA-event p50 measurements:

| Kernel | Warps | Repeated weights | After cache eviction | Matched cuBLASLt after eviction |
| --- | ---: | ---: | ---: | ---: |
| Earlier qs3 row | 4 | 4.277 ms | 4.326 ms | 5.730 ms |
| b12x row | 4 | 4.264 ms | 4.310 ms | 5.701 ms |
| b12x row | 8 | **4.139 ms** | **4.186 ms** | **5.706 ms** |
| b12x loop, BLOCK_K=512 | 4 | 4.225 ms | 4.278 ms | 5.700 ms |

The copied row kernel's eight-warps setting matches b12x's default vocabulary
launch policy. It reduces eviction-case latency by 26.6% versus its matched
cuBLASLt control, saving 1.52 ms per synthetic projection. It is about 3.2%
faster than the earlier qs3 row/four-warps candidate in this run; kernel source
and warp count both differ, so that comparison does not isolate either change.

With BF16 output, the eight-warp b12x row kernel measures 4.136 ms repeated and
4.185 ms after eviction, versus 5.699/5.703 ms for cuBLASLt. Thus the speedup
does not require changing qs3's FP32 vocabulary output. For the eight-warp
kernel, maximum absolute differences from cuBLASLt are 1.19e-6 (FP32) and
0.0078125 (BF16). Every variant passes the probe's existing gates.

## Source and build

Upstream source:
`b12x/gemm/bf16_vocab_projection/_kernel.py`, revision
`75ffee6375b0577ce2c8d6931ffacefda3ecbdd6`.
The copy omits only the Torch import and Python host wrappers. Both function
ASTs were checked against the vendored source. Apache-2.0 license and
attribution are retained in `sources/`.

The upstream wrapper allocates BF16 output. This experiment instantiates the
unchanged kernel bodies with either BF16 or FP32 output-pointer signatures.
It does not invoke b12x's Python wrapper, dispatch fallback or tuning system.
The loop variant is one supported configuration, not an exhaustive sweep.

All candidates use `num_stages=1` and `enable_fp_fusion=False`, matching the
earlier probe. Native invocations use CUDA driver module loading and kernel
launches, with no runtime Python or JIT. The target is SM121 on sp10, driver
580.142 and CUDA toolkit 13.0.88. The Triton ARM wheel supplies ptxas 13.3.33;
its actual path/version/hash and cubin hashes are recorded in manifest.json.

## Measurement limits and reproduction

One token, 248320 outputs, 2048 inputs, BF16 inputs/weights. Synthetic nonzero
signed weights use cudaMalloc. Ten warmups precede 50 individual eager samples;
three repetitions alternate candidate/baseline order. Before eviction samples,
a kernel touches at least 128 MiB / four times L2 outside the timed interval.
Clocks are not locked, and cache residency is not verified with counters.

Each candidate is checked against cuBLASLt for every output and against CPU
double accumulation for 65 rows. Probe tolerances remain 0.0002 absolute plus
0.0002 relative for FP32, and 0.01 absolute plus 0.01 relative for BF16.
These synthetic checks and timings do not establish real-model score agreement,
managed-weight behavior or end-to-end TPS.

Raw results, summary medians, validation output, host provenance and exact
sources are included here. Restore `sources/` into `.prototypes/b12x_gemv/`
and run `bash .prototypes/b12x_gemv/run.sh` from the repository root. The runner
uses the disposable sp10 checkout and must run sequentially with the existing
test/core benchmark scripts. Full cubins/PTX/native binary remain in
`.prototypes/b12x_gemv/sp10/` and the matching remote prototype directory.

The next real-model trial should use the copied b12x row kernel at eight warps
with FP32 output. The custom row kernel is retained here only as a comparison.
