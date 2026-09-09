# Findings and direction

This is an evidence log for Qwen3.6 on Spark. TODO.md remains the completion
checklist; competitive performance against vLLM has **not** been established.

## Measurement contract

Use the gitignored `run_core_benchmark.sh` to transfer the current commit to the
disposable `sp10@sp10:qs3` checkout and record release JSON in `benchmarks/`.
Do not run it concurrently with the test script: both replace that checkout.
Tests run through the gitignored `run_cuda_test.sh`.

`QS3_BENCH_CONTEXT_TOKENS` and `QS3_BENCH_DECODE_SAMPLES` select longer core
workloads (defaults 102 and 32). Longer prompts repeat the base prompt's token
IDs. JSON includes the full prompt fingerprint, generated IDs including warmups,
and explicit GDN state precision. The default workload remains unchanged.

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
| `90b52c4` (attention workspace reuse) | 11.905 | 83.701 | 1015.461 | [JSON](benchmarks/2026-09-09T021833.133622539Z-90b52c4.json) |
| `95f87e3` (96-block MoE) | 20.724 | 48.252 | 406.284 | [JSON](benchmarks/2026-09-09T025651.359461497Z-95f87e3.json) |
| `95f87e3` (4-block control) | 11.876 | 83.834 | 1023.041 | [JSON](benchmarks/2026-09-09T025757.341711005Z-95f87e3.json) |

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
Transformers 5.8.1. An initial inference comparison is recorded below; recurrent precision is not
yet matched. The expected
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
replay.

The unprofiled core run now measures **11.905 tok/s**, up 58.4% from the norm-fix
run, with decode p50 falling from 133.235 to 83.701 ms. Prefill p50 is 1015.461 ms.
The [repeat Nsight capture](benchmarks/2026-09-09T022039Z-90b52c4-decode/README.md)
contains no pinned allocation/free calls. Across 32 decode steps, event records
and queries cost 0.591 ms combined. Time without recorded GPU work within the
first-to-last-kernel span falls from 1539.70 to 97.53 ms; recorded GPU work falls
only from 2661.11 to 2608.21 ms. This supports host planning allocation removal
as the main cause of the wall-time improvement. Small differences in GPU time
remain subject to run variation.

Grouped MoE still takes 58.9% of GPU kernel time with all 2560 launches using four
blocks. Benchmark wider launches next. Persistent batch metadata and explicit
GDN parity remain prerequisites for graph replay; this result alone establishes
neither graph readiness nor competitive vLLM performance.

## First vLLM inference comparison

The [pinned-container run](benchmarks/2026-09-09T023628Z-vllm-bf16-core/README.md)
measures 30.826 tok/s and 32.355 ms decode p50 for the same 35B BF16 weights,
102-token prompt, greedy sampling and 32 warmed decode forwards. Full decode
CUDA graphs are enabled. Selected providers include FlashInfer CUTLASS MoE,
FlashAttention 2, and Triton/FLA GDN prefill. qs3's current eager result is
11.905 tok/s and 83.701 ms p50. Prefill is 181.369 ms in vLLM, including sampling
and delivery of its first token, versus qs3's 1015.461 ms to logits.

The effective vLLM GDN recurrent state is **FP32**, while qs3 stores BF16; its
convolution state is BF16. Thus this is a useful performance target but not the
matched-precision completion gate. The container rejects explicit BF16 Mamba
cache arguments and resolves `auto` recurrence to FP32. Record resolved cache
types, not just requested weight precision. Output tokens are preserved in the
artifact; output equivalence, longer contexts and sustained quality remain open.

The first vLLM prefill warmup triggers additional JIT and costs 16.592 seconds;
the second takes 183.219 ms. Neither is included in steady samples. The harness
includes empty engine steps in token-delivery intervals, preventing asynchronous
submission from being mistaken for completed inference latency.

## Wider grouped MoE launch probe

A [two-repetition sweep](benchmarks/2026-09-09-moe-block-sweep/README.md) holds
CUTLASS arithmetic fixed and tests 4, 12, 24, 48 and 96 blocks. Complete staged
MoE time for one token falls from 1285–1286 µs to 315–317 µs at 96 blocks. The
16- and 102-token cases improve about 5x. This supports increasing the launch
grid before replacing arithmetic or adding fusion. The probe uses zero weights
and device allocations; numerical tests and managed real-model timing are separate
gates. 96 is the best tested grid, not an established global optimum.

The subsequent output-recording core run `3bb273e` reproduces 11.908 tok/s and
83.724 ms decode p50 with the original four-block launch. All 36 generated IDs
(including warmups) match the first 36 vLLM IDs on the fixed prompt. This supports
short-run greedy agreement; recurrent precision and sustained quality remain open.

## Explicit AOT MoE launch selection

`QwenConfig::moe_bf16_kernel` selects `CutlassBlocks4` or `CutlassBlocks96`.
Both lower to the same local two-stage SM80 CUTLASS grouped GEMM, compiled for
SM121. The default is 96 blocks. `QS3_BENCH_MOE_BLOCKS=4` selects the comparison
launch in the core benchmark; JSON reports the kernel tile, stages and block
count. Native and Rust validation reject other grid settings before execution.
The local launch replaces the vendor wrapper's fixed four-block call without
changing routing, arithmetic, activation, or reduction. No vendor files change.

The full prescribed suite passed on sp10: 115 library tests (3 ignored), one
benchmark test, three engine tests, 16 model tests, 14 vector tests, both native
suites and the native benchmark build. Native numerical cases now run at both
four and 96 blocks, including repeated routes and weighted top-2 accumulation.
The separate pinned 35B BF16 regression also passed, including prefill/first-decode
logit tolerances, greedy IDs `[5, 6, 24218, 10]`, and reset/replay. End-to-end
measurements follow.


The unprofiled 96-block core run measures **20.724 tok/s**, **48.252 ms decode
p50**, and **406.284 ms prefill p50**. The four-block control from the same commit
returns to 11.876 tok/s, 83.834 ms decode p50 and 1023.041 ms prefill p50. Thus
changing the grid improves throughput by 74.5% in this pair and reduces prefill
p50 by 60.3%. All 36 generated IDs match between the runs and match the first
36 IDs from vLLM. This establishes a real-model gain from wider MoE scheduling,
beyond the zero-weight microbenchmark. The measured eager result remains below
vLLM's 30.826 tok/s, and FP32-vs-BF16 recurrence is still a comparison limitation.

The preceding GPU trace also identifies serial router selection as a next target:
`router_topk_kernel` totals 178.34 ms over 32 decode steps (5.57 ms/token).
`qscu_router_topk` launches one thread per token; that thread scans all 256 experts
and maintains the top eight. Test a parallel warp/block selection while preserving
lower-ID tie-breaking, softmax/sigmoid behavior and route-weight normalization.
Reprofile after the MoE grid change before assigning its new share of runtime.


The [post-change timeline](benchmarks/2026-09-09T030025Z-95f87e3-decode/README.md)
confirms all 2560 MoE launches use 96 blocks. Aggregate MoE time falls from
1533.57 to 393.82 ms; total kernel time falls from 2603.22 to 1465.65 ms with the
same 38,948 launches. Time without recorded GPU work stays near 98 ms across
32 steps, supporting reduced GPU execution time as the cause of this gain.
Dense cuBLAS GEMV now dominates; serial router top-k remains 178.05 ms (12.1%).
Prioritize those measured kernels alongside persistent metadata and graph work.

## Longer prompt and sustained decode

The [1024-token / 256-step comparison](benchmarks/2026-09-09T030615Z-vllm-bf16-core/README.md)
uses repeated base prompt IDs and contexts 1028 through 1284. qs3 holds at
20.730 tok/s (48.216 ms decode p50, 1941.522 ms prefill p50); vLLM measures
30.769 tok/s (32.481 ms decode p50, 367.629 ms prefill p50). Thus the short-context
qs3 throughput persists across this longer run. This is a synthetic performance
probe, not a natural-language quality benchmark.

The first 56 generated IDs agree; they diverge at index 56 (qs3 32956, vLLM 33027).
Subsequent token contexts therefore differ. Recurrent state remains BF16 in qs3
and FP32 in vLLM, while kernel arithmetic/prefill paths also differ. Do not assign
a cause from this alone. Native `qscu_gdn_prefill/decode` already support FP32
state, and `GdnRecurrentState` can describe it; the runner currently allocates
`DeviceBuffer<u16>`. Expose explicit FP32 runner storage and compare again to
address the precision gate. Sustained quality and a third context remain open.

## Explicit recurrent-state precision

`QwenConfig::gdn_recurrent_precision` now selects BF16 or FP32 storage while
keeping BF16 activations and convolution history. Owned typed buffers determine
the native descriptor dtype; reset and prefix reconstruction allocate/clear the
selected type. The native AOT kernels already implement both forms, so no new
runtime kernel selection framework, compilation or synchronization is needed.
Loaded real models now default to FP32; narrow fixtures retain their BF16 state.
`QS3_BENCH_GDN_STATE=f32 ./run_core_benchmark.sh` selects FP32, and JSON reports
the effective state dtype.

The prescribed full suite passed (115 library, 1 benchmark, 3 engine, 16 model,
14 vector tests and both native suites). The pinned 35B BF16 real-model test now
runs both recurrent precisions sequentially; both passed the existing prefill
and first-decode logit checks, greedy IDs `[5, 6, 24218, 10]`, and reset/replay.
The modes must remain explicit in subsequent throughput and output comparisons.


The [FP32 short run](benchmarks/2026-09-09T032157.505542859Z-bb36781.json)
measures 20.653 tok/s and 48.343 ms decode p50, with all 36 generated IDs matching
vLLM. The [1024-token/256-step run](benchmarks/2026-09-09T032256.107210192Z-bb36781.json)
measures 20.658 tok/s, 48.380 ms decode p50 and 1934.079 ms prefill p50. These
throughputs are within 0.4% of the corresponding BF16-state measurements; this
small difference is not resolved beyond run variation.

FP32 recurrence extends the sustained matching prefix from 56 to 162 generated
IDs. At index 162, qs3 emits 8340 while vLLM emits 79091. Thus matching recurrent
storage helps this trajectory but does not establish complete output equivalence.
Different projection/output rounding and prefill algorithms still need attention;
compare scores under identical token prefixes before assigning a remaining cause.
Loaded model configurations now default to FP32 recurrence. BF16 remains an
explicit comparison choice, and benchmark metadata records the effective mode.
