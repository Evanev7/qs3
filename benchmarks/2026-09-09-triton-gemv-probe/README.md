# Triton 3.8 AOT GEMV on GB10

The standalone prototype builds and runs all 12 SM121 specializations with
native CUDA launches. Every candidate passes comparison against prepared qs3
cuBLASLt for every output and CPU double accumulation on 65 selected rows.
The FP32 vocabulary projection is the clearest performance candidate.

Final run, median of three per-provider CUDA-event p50 measurements:

| LM head, 248320 outputs x 2048 inputs | cuBLASLt | Triton, 1 row / 4 warps | Latency reduction |
| --- | ---: | ---: | ---: |
| Repeated weight matrix | 5.731 ms | 4.283 ms | 25.3% |
| Cache eviction before each sample | 5.738 ms | 4.332 ms | 24.5% |

The first complete run gives 5.737/4.286 ms repeated and 5.745/4.333 ms evicted
for that same configuration. FP32 LM-head maximum absolute difference from
cuBLASLt is 1.43e-6. Inputs and weights are BF16; output stays FP32. The 4-row
and 8-row variants are slightly slower here.

Smaller BF16 projections are mixed. QKV/packed-Q-gate (8192 x 2048) improves
modestly in the eviction measurements. Z (4096 x 2048) and output projection
(2048 x 4096) improve with repeated weights but are slower after eviction.
Their absolute timings and some configuration rankings vary between complete
runs, so use matched pairs in the raw CSV rather than treating a cross-run
minimum as a stable result. These results do not justify replacing all GEMVs.

## Conditions and scope

- sp10 NVIDIA GB10, driver 580.142, native CUDA toolkit 13.0.88.
- Pinned Triton 3.8.0 ARM wheel; its actual bundled Blackwell assembler is
  13.3.33. Explicit `cuda:121:32` compilation emits `sm_121a` cubins. No assembler
  override is required in this experiment.
- Three fixed configurations (1 row / 4 warps, 4 / 4, 8 / 8) for each of four
  real 35B projection shapes. No runtime compilation or autotuning.
- Synthetic nonzero signed weights allocated with cudaMalloc. The real model's
  managed allocations, weight distributions and full execution schedule are
  not measured here.
- Ten warmups, 50 samples, three repetitions; provider order alternates between
  repetitions. Each sample uses CUDA events around one eager launch. The eviction
  kernel touches at least 128 MiB / four times L2 before the timed interval;
  this is an eviction attempt, not verified hardware cache residency.
- Setup, allocation, module loading, copies and validation are outside timing.
  Clocks are not locked. The host was idle before the probe.
- BF16 comparison gate is `0.01 + 0.01*abs(reference)` and FP32 gate is
  `0.0002 + 0.0002*abs(reference)`. Actual maximum errors are in validation.log;
  these synthetic gates do not replace real-model logit/token checks.

The generated PTX confirms the five native pointer arguments: X, W, Y, global
scratch and profiling scratch. The last two are null; the exporter rejects
nonzero compiler scratch requirements. Dynamic shared-memory configuration is
set once at module preparation, outside timed launches.

## Evidence and reproduction

- `results.csv`, `initial-results.csv`: raw final and initial measurements.
- `summary.json`: final medians grouped by shape/configuration/cache mode.
- `validation.log`, `host.txt`, `manifest.json`: numerical checks, host details,
  compiler/assembler provenance and cubin hashes.
- `sources/`: exact final probe sources. `qscb.cu` SHA256 is in host.txt; repository
  base revision is `2f4ed1d3c1bf3f6ff6bb03a77eef8648700427e9`, with the pinned
  Triton dependency addition. The two runs produce identical cubin hashes.

Restore the source copies into `.prototypes/gemv_aot/` if needed, then run
`bash .prototypes/gemv_aot/run.sh` from the repository root. It uses the disposable
sp10 checkout; run sequentially with the standard test/core benchmark scripts.
Cubins, PTX and the native executable remain in `.prototypes/gemv_aot/sp10/`
and `sp10@sp10:qs3/.prototypes/gemv_aot/out/`.

Next gate: try the 1-row/4-warp LM-head kernel against real managed weights,
compare identical-prefix logits, and measure eager full-model decode. The
approximately 1.4 ms kernel saving is not an established end-to-end TPS gain.
