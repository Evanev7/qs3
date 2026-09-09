# Parallel Qwen router probe

sp10 GB10, 2026-09-09, baseline router source from `bb36781`. The AOT prototype
replaces the one-thread selection kernel with one warp per token. Score/exp
calculation and ordered top-k comparison run cooperatively; lane zero retains
expert-order denominator addition and top-k-order weight normalization. It keeps
lower-ID ties, BF16/F32 input paths and softmax/sigmoid semantics.

The existing native benchmark supplies zero logits for 256 experts, top-k eight,
softmax, normalization enabled, scale one. CUDA events time 1000 calls after 100
warmups. Inputs/outputs and provider setup are outside the measured interval.
These probes establish timing, not numerical correctness.

| Tokens | Serial (µs) | Warp with old 4096-expert bounds (µs) | Qwen 256-expert warp (µs) |
| --- | ---: | ---: | ---: |
| 1 | 143.479 | 22.542 | 8.188 |
| 16 | 143.291 | 22.539 | 8.197 |
| 1024 | 261.899 | 164.258 | 10.259 |

The 4096-expert prototype reserved two 4096-float shared arrays and iterated the
larger fixed bound, despite the actual model having 256 experts. Specializing to
Qwen's 256 experts/top eight cuts both scratch and loop work. The one-token probe
improves about 17.5x from the original; this is not a model-throughput claim.

`kernel.inc` is the prototype body; `bench.cu` reuses the native benchmark fixture.
To reconstruct under `.prototypes/router_parallel/` on the baseline checkout,
copy `qscu.cu`, replace its router kernel body with `kernel.inc`, change the launch
from one to 32 threads, and set `kRouterMaxExperts=256`, `kRouterMaxTopK=8` for the
Qwen run. `build.py` links separate serial and warp binaries against the existing
native build objects. It assumes those objects still contain the baseline serial
router. The original source and executables remain in the ignored prototype
folder on sp10; production integration has separate checked/release and real
model validation.
