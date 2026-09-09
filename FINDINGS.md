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
| `6a3d39d` (prepared linear) | 7.638 | 131.361 | 1028.085 | [JSON](benchmarks/2026-09-09T013243.445741248Z-6a3d39d.json) |
| `6bf75db` (norm fix) | 7.515 | 133.235 | 1026.774 | [JSON](benchmarks/2026-09-09T014957.161677547Z-6bf75db.json) |

The first prepared-linear run is 1.17% higher in decode throughput than the fresh
baseline, with prefill 0.43% slower. This is one pair of runs and does not resolve
small effects from run-to-run variation. It does show that removing this host
preparation alone leaves the overall decode cost close to 131–133 ms/token.
The norm-fix measurement returned to 7.515 tok/s, reinforcing that these small
wall-time differences do not yet establish a performance gain.

All runs report GB10, driver 580.142 and CUDA runtime 13.0. Model, precision,
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

## Norm launch race: reproduced and fixed

FlashInfer revision `b3baedbbef2686df91b6dc43818ee56fe26ceba2` calls
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, requested_bytes)` immediately
before each Gemma RMSNorm/fused-add launch. Different hidden widths instantiate
the same vector-width-8 BF16 kernel. That CUDA function attribute is shared
across host threads using the device primary context; a smaller-width call can
lower the limit between another call's setter and launch.

On 2026-09-09, the focused `qwen36_norm_concurrent_widths_keep_launch_configuration_independent`
regression reproduced the exact `invalid argument` at `norm.cuh:679`: width 2048
failed on iteration 35 and width 5120 on iteration 102. It creates four independent
contexts with widths 8, 256, 2048 and 5120. A temporary mutex around only the native
launch calls made the same test pass. Removing the per-call attribute mutation
also passed, without serialization. This is direct evidence for the launch race,
not a prefix-rebuild or stream-lifetime hypothesis.

`qsfi_norm_rope.cu` now launches the same AOT FlashInfer kernels with a zero-initialized
local CUDA launch configuration, Gemma weight bias 1, and PDL disabled. All Qwen
shapes fit the default shared-memory allowance; oversized fused norms fail before
launch. No mutex or stream synchronization was added. CUDA requires explicit
opt-in for dynamic shared memory above 48 KiB; see the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html).

Validation through `QS3_TEST_MODE=norm_validation ./run_cuda_test.sh` passed:
115 library tests (3 ignored), 1 benchmark test, 3 engine tests, 16 model tests,
14 vector tests including 16,000 concurrent norm launches, both native suites,
and a build of `qsfi_bench_native`. Twenty additional normal-parallelism model
runs passed all 320 executions. The separate real 35B BF16 regression also passed
its logits, greedy IDs and reset/replay checks.

## First steady-decode GPU timeline

The [Nsight capture and summaries](benchmarks/2026-09-09T015556Z-0a7eef9-decode/README.md)
for `0a7eef9` isolate the 32 warmed decode samples. Two costs now have evidence:

- Grouped MoE GEMM: 59.3% of aggregate GPU kernel time, 1.576 seconds across
  2,560 launches (49.25 ms/token). The pinned FlashInfer grouped-GEMM wrapper
  hardcodes four thread blocks; the trace confirms `[4,1,1]` for all 2,560
  launches. Benchmark a wider launch before replacing its arithmetic or adding
  fusion.
- Attention replanning: 32 pinned allocations cost 816.15 ms and 32 pinned frees
  cost 547.49 ms (42.61 ms/token combined). These correspond to the 64 MiB host
  planning workspace being recreated as each token changes metadata. Reuse this
  storage with explicit asynchronous-copy lifetime management; weakening the
  metadata key or adding a release-mode stream synchronization is unnecessary.

The SQLite timeline has 1,539.70 ms without recorded GPU work within a
4,200.81 ms first-to-last-kernel span. The pinned allocation/free calls have
zero overlap with recorded GPU work in this capture. Their total also includes
calls preceding the first kernel, so these interval endpoints differ.

Host and GPU timings in general can overlap and must not be summed into a speedup prediction.
In particular, blocking host copies include waits for queued GPU work. Confirm
improvements with the unprofiled core benchmark after each change.

A matched-vLLM baseline can use the existing local container
`vllm-node@sha256:d966c1831d5da55c0cc52c6bd40f7d02cfc3d83404c3bd599139b055232d3970`.
Read-only package inspection found vLLM 0.21.0, PyTorch 2.11.0+cu130, and
Transformers 5.8.1. No matched inference run has been recorded yet. The expected
27B BF16 cache path is absent; a third-party 27B NVFP4/MTP cache does not satisfy
the pinned 27B BF16 correctness requirement.

## Reusing attention planning storage

The prepare API replaces the one-shot native create entrypoints for both paged
prefill and decode. Rust retains the plan handle and its full metadata key.
Candidate key vectors are allocated before reprepare; the key is installed only
on success. Validation/allocation failure preserves a native plan. A planner or
upload failure invalidates it until successful preparation, so stale schedule
metadata cannot execute.

The native plan keeps its device integer workspace and a pool of pinned host
buffers. Device updates are ordered with attention execution on the same stream.
Each host buffer records an event after its planning upload and can be overwritten
only when that event reports completion. If all slots remain busy, preparation
adds a slot instead of waiting. An event-record failure quarantines that slot.
Normal synchronous token delivery should need one slot; enqueue-only callers can
retain more. Teardown releases the buffers and events. No stream synchronization
was added to preparation or execution.

The queued native regression alternates query offsets, retains every output,
and compares all results against CPU attention. A short GPU delay makes uploads
remain in flight in the release suite. It also verifies execution after a rejected
update. Existing cache tests still reject matches when page IDs or last-page
lengths change. The device-workspace address now persists across reprepare;
graph capture still needs persistent batch buffers and explicit metadata updates.

Validation on sp10 (2026-09-09): the prescribed full test script passed 115
library tests (3 ignored), 1 benchmark test, 3 engine tests, 16 model tests,
14 vector tests, both native suites, and the native benchmark build. The real
35B BF16 regression passed separately, including logits, greedy IDs and reset
replay. Performance and allocation behavior are measured in subsequent runs.
