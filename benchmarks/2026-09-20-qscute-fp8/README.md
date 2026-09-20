# CuTe FP8 QKV runtime integration

Qwen3.8-27B-NVFP4, NVIDIA snapshot
`dbb8f445b3145f8a4c18ddc769f032d57d32867c`, GB10 SM121. Replaces only
single-token GDN QKV projection at M1/N10240/K5120. Prefill and other FP8
projections retain cuBLASLt. The implementation is adapted from b12x
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`: 32x64x128 tile, four TMA stages,
four MMA warps plus one load warp, two FP32 split-K outputs, then a Triton FP32
sum and BF16 store. Rust retains quantization, scratch ownership and scheduling.

## Validation

- Full `./remote.sh test` passes: [log](required-tests.log). Includes compiler
  tests, direct CuTe launcher tests, Rust backend/model tests, and all four native
  checked/release suites. The unused SAXPY source is removed; dedicated qscute
  tests now use the real FP8 kernel.
- GPU-free Nix venv build and real AOT export pass. The pinned wheel's optional
  profiler needs `libcuda.so.1` only on a driver-equipped host; Nix permits that
  missing dependency during packaging.
- A literal Python M=1 caused CuTe to emit a faulting 1D TMA load on SM121.
  Internal `cutlass.Int32(1)` preserves the working 2D descriptor without an
  external M argument. The required suite passed with this fix.

## Numerical comparison

[fp8_model_parity.py](fp8_model_parity.py) loads one model and retains one runner's weights,
scratch and existing plans. For prefixes of 4, 102 and 1024 tokens, it runs
cuBLASLt then CuTe, resetting state and forcing the same 32 baseline-greedy tokens.
The CuTe module owner is retained while its dispatch is disabled for the baseline.
Diagnostic copies/synchronization invalidate timings. The runner uses the standard
Nix benchmark's actual cuBLASLt 13.1.1.3 and CUDA runtime 13.0.96:
[library resolution](libraries.txt).

[Results](model-comparison.json): all three prefill rows are exact; 91/96 decode
rows have the same greedy winner (94/99 including prefills). This is a numerical
change, not exact parity. Downstream logit drift can be substantial despite small
individual QKV differences: maximum logit absolute error 13.640625 and relative
L2 0.38317. This numerical difference is accepted for the replacement.

[fp8_intermediates.py](fp8_intermediates.py) traces one forced decode after the
four-token prompt, comparing CuTe and cuBLASLt QKV on identical inputs and feeding
the cuBLASLt result onward. Across 48 GDN layers, 0–4 of 10240 BF16 values differ
per projection; maximum relative L2 is 0.0001873. Feeding the reference QKV onward
restores exact final logits for this step. [Per-layer results](qkv-intermediates.log).

[fp8_cpu_rounding.py](fp8_cpu_rounding.py) checks captured FP8 inputs and weights
for the first three GDN layers against FP64 products/accumulation and several
FP32 scale orders. Both implementations match the CPU BF16 result in the first
two layers. The first disagreement is layer 2, column 7180: CuTe `0xbe26` matches
the CPU result; cuBLASLt produces `0xbe27`. All checked scale orders select
`0xbe26`. This establishes that the first difference is not evidence of a CuTe
indexing error; it does not establish which implementation is more accurate for
every subsequent layer. [CPU results](cpu-rounding.log).

Diagnostic scripts build disposable source overlays under `.prototypes/out` and
do not add downloads, synchronization, or provider switches to production.
Original GPU captures remain in `.prototypes/out/kr03-runtime-parity-el3z30x2`;
same-runner model comparison is `kr03-runtime-parity-u1ts9tbu`.

## Performance

Measured revision: `c9138b3`.

| Prompt / measured decode tokens | Prior `f284ac6` decode median | CuTe decode median | CuTe throughput | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| 102 / 32 | 96.511 ms | 88.594 ms | 11.278 tok/s | +8.9% |
| 1024 / 256 | 94.598 ms | 89.306 ms | 11.195 tok/s | +5.9% |

Prefill medians were 130.491 and 416.284 ms, versus 129.014 and 413.405 ms in
the prior measurements. Prefill does not use this replacement. These are single
standard runs; the prior cuBLASLt path has documented process-dependent timing
variation, so the percentages are comparisons to those recorded runs.

Raw standard results include unprofiled measurements and Nsight captures:
[102 / 32](../2026-09-20T095517.621315286Z-c9138b3.json),
[1024 / 256](../2026-09-20T095555.794403482Z-c9138b3.json).
