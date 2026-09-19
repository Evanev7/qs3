# Decode slowdown following b143912

Target: the existing committed Qwen3.8-27B NVFP4 binaries on sp10/GB10,
102-token prompt and 32 measured greedy decode steps, four decode warmups.
No production code changed in this investigation. The exact cause is still open.

## What is established

The historical short-workload median rises from 93.245 ms at `df6f07e` to
96.313 ms at `b143912`, then 96.395/96.511 ms in the next two benchmarks.
The long workload rises less: 93.226 to 94.147/94.229/94.598 ms.
`df6f07e` is the last measured predecessor, not the immediate parent:
`5a62fc7` changes convolution rounding between it and `b143912`.

[Historical trace comparison](historical-comparison.json) localizes a captured
slow run (`a6cb1ba`, short workload) to the cuBLASLt FP8
`nvjet_sm121_qqtst_mma_64x128x128_4_16x128x128_tmaAB_bz_TNNN` kernel group:
16.411 → 19.374 ms/token versus the earlier fast `b143912` capture, +2.963 ms.
There are 48 calls/token with the same grid/block/shared-memory dimensions.
The native call log identifies the M1/N10240/K5120 projection, matching GDN QKV.
GDN recurrence is 0.946 → 0.950 ms; idle gaps 2.147 → 2.109 ms.
The captured slowdown therefore is not extra recurrence work or host gaps.

Unprofiled and profiled measurements are separate processes. The short
`b143912` benchmark is 96.313 ms unprofiled but 93.902 ms profiled; the latest
is 96.511 versus 94.410 ms. A fast captured run cannot explain the corresponding
slow unprofiled run by subtracting its kernel totals.

## Alternating committed binaries

Run [repeat.py](repeat.py.txt) with [these binaries](binaries.json). This calls
the existing Nix executables directly, reproducing the standard measurement
settings and driver-library setup. Trials reverse the revision order every
other iteration. No recompilation, alternate CUDA library, or source overlay.
[Raw measurements and summary](repeat/summary.json):

| Revision | Trial 1 ms/token | Trial 2 | Trial 3 |
| --- | ---: | ---: | ---: |
| df6f07e | 93.284 | 93.780 | 93.616 |
| b143912 | 96.278 | 96.410 | 96.653 |
| f284ac6 | 96.350 | 96.388 | 93.817 |

Each revision generates identical tokens across its own three repeats.
The current executable can recover most of the missing time without changing
code. This is evidence of distinct process-dependent timings, not proof of a
specific allocator, cache, thermal, or algorithm-selection mechanism.

## Workspace hypotheses: negative / inconclusive

[interpose.cpp](interpose.cpp.txt) wraps `cublasLtMatmul` in an isolated preload
library. It records shapes, pointers and algorithm bytes; the last version also
records public algorithm attributes. It optionally substitutes one fresh
`cudaMalloc` workspace of the same 64 MiB capacity. The production allocator
uses `cudaMallocAsync`; this intentionally tests a different placement and
must not be treated as a neutral production change.

[Separate-process sweep](workspace/summary.json): logging-only controls ran at
93.653 ms (`df6f07e`) and 93.784 ms (`b143912`). Fresh allocations with offsets
0, 256, 65536 and 2097152 bytes all ran at 96.67–96.72 ms. Tokens remain exact
among the `b143912` variants. This does not establish causality: other allocation
and process differences remain. These first runs logged opaque algorithm bytes
only; those contain process-specific data, so whole-byte comparisons are not
algorithm-ID comparisons.

The stronger [within-process check](alternate-analysis.json) alternates only
the GDN QKV projection's workspace binding each decode step, using 48 calls per
step and four warmups. Other allocations, capacity, model, and selected plan
remain shared. Original versus replacement medians:

- Trial 1: 93.738 versus 93.727 ms.
- Trial 2: 93.819 versus 93.749 ms.

This did not reproduce the slowdown or implicate the workspace pointer alone.
The interposer is diagnostic only; its host-side descriptor queries affect
launch overhead. These are not standard benchmark/adoption results.

## Next decisive work

Capture a slow and fast case with public Lt algorithm attributes plus bindings,
then hold the selected plan and weights fixed while varying activation/output
allocation in an isolated QKV probe. Distinguish virtual-address effects from
allocation lifetime/physical placement; do not patch allocation order based on
correlation. `b143912` adds prefill buffers before existing scratch/workspace
allocations, but leaves the decode recurrence branch and kernels in place.
That is a plausible trigger, not an established cause.

For decode improvement, KR03's native b12x FP32 split-K QKV candidate targets
this same projection (~350 → 284 µs in the existing fixture). Qualify it against
the production cuBLASLt library and real-model logits, including M<=16, before
adoption. KR06's fused activation/quantizer saves much less single-token time.
