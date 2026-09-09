# Findings and direction

This is an evidence log for Qwen3.6 on Spark. TODO.md remains the completion
checklist; competitive performance against vLLM has **not** been established.

## Measurement contract

Use the gitignored `run_core_benchmark.sh` to transfer the current commit to the
disposable `sp10@sp10:qs3` checkout and record release JSON in `benchmarks/`.
Do not run it concurrently with the test script: both replace that checkout.
Tests run through the gitignored `run_cuda_test.sh`.

The current core workload uses the pinned Qwen3.6-35B-A3B BF16 snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, managed weights, eager execution,
greedy sampling, a 102-token prompt, 4 decode warmups and 32 measured decode
steps (contexts 106 through 138). Decode wall time includes the public runner
call and sampled-token delivery. Tokenizer decode validation occurs outside the
timed interval. This is a short-context regression measurement, not a sustained
or matched-vLLM comparison. Prefill samples follow resets of the same runner.

| Revision | Decode tok/s | Decode p50 ms | Prefill p50 ms | Evidence |
| --- | ---: | ---: | ---: | --- |
| `ba4ad6e` | 7.498 | 133.401 | 1022.074 | [JSON](benchmarks/2026-09-09T010751.748761081Z-ba4ad6e.json) |
| `f32f0e5` | 7.550 | 132.793 | 1023.724 | [JSON](benchmarks/2026-09-09T011848.800087754Z-f32f0e5.json) |

Both runs report GB10, driver 580.142 and CUDA runtime 13.0. Model, precision,
context, software and kernel choices must remain explicit in comparisons.

## Prepared dense projections

Inspection found descriptor creation/destruction and
`cublasLtMatmulAlgoGetHeuristic` in every dense projection, including decode.
The first optimization gives Rust ownership of reusable native linear plans.
Keys include dimensions, row strides, output dtype, input/weight/output pointer
alignment (capped at 256 bytes), and workspace capacity. They belong to one
execution context. Alpha, beta and addresses can change while that contract
holds. Native execution checks the contract before addressing device tensors.

Alignment must be an actual heuristic preference, not just a cache key:
cuBLASLt defaults to 256-byte matrix alignment, while subviews may provide less.
Workspace requires 256-byte alignment. See the [cuBLASLt reference](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul).

This removes repeated preparation; it is not yet the final static schedule.
The first call for each key still selects an algorithm, and Rust still looks
up a plan on execution. Prefix rebuilding currently creates a fresh Engine,
which also discards provider handles and prepared plans. That lifetime coupling
is a separate TODO.

## Architecture to test next

The target is Rust-owned, near-static execution of a small set of exact Qwen
model schedules with AOT CUDA kernels. Build-time Python for artifacts is
acceptable; inference must not need Python, NVRTC, or Python-driven JIT. The
current cuBLASLt setup query is an intermediate preparation mechanism. Ultimately
kernel choices should be explicit prepared schedule entries; graph replay must
not select algorithms, allocate storage or compile kernels.

1. Preserve execution resources across prefix transactions. Candidate attention
   and recurrent states may be replaced without replacing the device/stream and
   provider handles. Keep failure rollback separate from reusable execution
   resources.
2. Reuse attention planning storage while retaining exact metadata validation.
   Page IDs and last-page lengths cannot be dropped from the key. An update path
   must account for pinned staging still being consumed by asynchronous copies.
3. Preallocate decode storage and represent changing page metadata, sequence
   positions, sampled input IDs and GDN state slots in persistent device buffers.
   Capture only after preparation; page boundaries and prefix resets are
   correctness gates. Alternating recurrence slots need explicit replay parity.
4. Measure prepared eager CPU submission and GPU timelines before choosing
   fusion. Then compare graph replay. A short-context wall-time improvement
   alone does not establish a GPU kernel bottleneck or vLLM parity.
5. Establish exact 27B BF16 support and matched-vLLM baselines before claiming
   model coverage. Quantization needs verified NVFP4 packing/scales and SM121
   kernels; an emulation path is not evidence of competitive quantized execution.

These are hypotheses and implementation gates, not measured speedup claims.

## Validation log

2026-09-09, prepared linear plans: the prescribed script passed 115 library
tests (3 ignored), 1 benchmark timestamp test, 3 engine integration tests,
16 model integration tests, 13 vector tests, and both native checked/release
suites. Native cases cover repeated execution with changed alpha and rejection
of changed strides, dtype, alignment and workspace. The separately requested
real BF16 regression passed its prefill/first-decode logit tolerances, greedy
IDs `[5, 6, 24218, 10]` and reset/replay.
