# Parallel causal-convolution prefill probe

sp10 GB10, CUDA 13, `qscu.cu` from source commit `8a8f910` (the convolution
implementation is unchanged from `bf86169`). Standalone AOT probe; no Python/JIT
at execution. Reproduce by compiling `probe.cu` with the same flags and native
objects as the repository CUDA test target, excluding `qscu.o` because the probe
includes that source directly.

At 1024 tokens and 8192 channels, BF16 convolution state, one sequence, separate
read/write slots, the CUDA-event mean of 50 calls after five warmups is:

| Path | Mean |
| --- | ---: |
| Serial per-sequence convolution | 14.367512 ms |
| Parallel outputs plus final-state kernel | 0.190377 ms |

Timing includes final-state writeback for both paths. Numerical checks compare
all BF16 output bits and every state byte against the existing serial kernel for
1/2/3/4/5/31/32/33/1024 tokens, one/two sequences, separate/same-slot updates,
BF16/FP32 state, and negative read slots. Two-sequence cases include empty
segments for very short inputs. All 72 cases pass. Input and output buffers are
disjoint. State writeback follows output computation on the same stream, so
same-slot history cannot be overwritten before output blocks finish reading it.

These are standalone kernel measurements alongside the asset download, not
isolated core throughput. Integration uses a flat, bounded grid for final-state
writeback, avoiding a grid-y batch-size limit. Full-runner validation and core
prefill comparisons are recorded separately in FINDINGS.md.
