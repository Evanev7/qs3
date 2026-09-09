# Steady-decode profile

Commit `0a7eef9e8ad2591f0dd6aceb8da2c5c243470c10`, sp10, GB10,
Nsight Systems 2025.3.2.474. The benchmark uses the pinned 35B BF16 snapshot,
eager execution, greedy sampling, and managed weights. Capture covers exactly
the 32 measured decode steps, after four warmups (context 106 through 138).
See the runner JSON and native kernel names in the CSV summaries.

Invocation after building the release benchmark with Nix:

```sh
QS3_PROFILE_DECODE=1 nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=true --output=decode /absolute/path/to/qs3-bench
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv --output summary decode.nsys-rep
```

The full trace and SQLite database remain on sp10 at
`/home/sp10/qs3-profiles/2026-09-09T015556Z-0a7eef9-decode/`.
The local transfer targets `.prototypes/profiles/` under the same run name.
These profiled timings must not be mixed with the unprofiled core results.

The grouped MoE GEMM accounts for 59.3% of aggregate GPU kernel duration:
1.576 seconds in 2,560 launches, or 49.25 ms per decoded token. Its vendored
FlashInfer wrapper hardcodes four thread blocks (`group_gemm.cuh:87`).
The aggregate GPU percentage is not a percentage of wall-clock decode time.

On the CPU, 32 `cudaHostAlloc` calls total 816.15 ms and 32 `cudaFreeHost`
calls total 547.49 ms: 42.61 ms per token combined. Source inspection connects
these to the 64 MiB pinned integer planning workspace destroyed/recreated on
each attention metadata change. This makes persistent attention planning
storage the next host-side target. Device allocation/free also remains visible.

Do not add host API duration to GPU kernel duration: the two can overlap, and
blocking copies include waits for preceding GPU work. The 1.967 seconds inside
`cudaMemcpyAsync` is not evidence of 1.967 seconds of actual memory transfer.

`timing_summary.json` is derived from the exported SQLite activity tables. It
merges the kernel, memcpy and memset intervals, then intersects their union
with the host allocation/free intervals (`cudaHostAlloc_v3020` and
`cudaFreeHost_v3020` in `StringIds`). The span from the first kernel start to the
last kernel end is 4,200.81 ms, with 2,661.11 ms covered by recorded GPU work and
1,539.70 ms without it. The 1,363.65 ms of pinned-host API calls has zero overlap
with recorded GPU work. The host API total includes calls before the first
kernel, so its interval is not identical to the kernel span.

Grouping `CUPTI_ACTIVITY_KIND_KERNEL` rows whose demangled name contains
`GemmGrouped` by `gridX/gridY/gridZ` confirms all 2,560 launches used `[4,1,1]`.
The profiled runner took 4,253.26 ms for its 32 timed calls; range boundaries,
inter-call work and profiled activity intervals have different endpoints.
