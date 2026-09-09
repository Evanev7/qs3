# Steady-decode profile after attention workspace reuse

Commit `90b52c455d3a92abaa586daa8b856cc51cd1db00`, sp10, GB10,
Nsight Systems 2025.3.2.474. This repeats the [previous capture](../2026-09-09T015556Z-0a7eef9-decode/README.md)
with the same pinned 35B BF16 snapshot, managed weights, eager execution,
102-token prompt, four warmups and 32 measured decode steps (contexts 106–138).
Invocation and interval analysis follow that capture; runner metadata and native
kernel names are included here. Full trace and SQLite files remain at
`sp10@sp10:/home/sp10/qs3-profiles/2026-09-09T022039Z-90b52c4-decode/`
and locally under `.prototypes/profiles/` with the same run name.

| Timeline measure | Before | After |
| --- | ---: | ---: |
| First-to-last kernel span (ms) | 4200.81 | 2705.74 |
| Recorded GPU work within span (ms) | 2661.11 | 2608.21 |
| No recorded GPU work within span (ms) | 1539.70 | 97.53 |
| Pinned allocation/free API duration (ms) | 1363.65 | 0 |

There are no `cudaHostAlloc` or `cudaFreeHost` calls in this capture. The 32
`cudaEventRecord` and 32 `cudaEventQuery` calls total 0.591 ms. Eight device
allocations and frees remain, costing 0.131 ms combined. The aggregate kernel
duration is 2603.22 ms in 38,948 launches. Grouped MoE GEMM accounts for 58.9%
of that duration: 1533.57 ms across 2,560 launches, all still using `[4,1,1]`.
This makes the four-block MoE launch the next substantial GPU target.

These are profiled activity intervals, whose endpoints differ from timed runner
calls. The separate [unprofiled core run](../2026-09-09T021833.133622539Z-90b52c4.json)
measured 11.905 tok/s and 83.701 ms decode p50, versus 7.515 tok/s and 133.235 ms
for the preceding norm-fix core run. Host and GPU durations can overlap and must
not be added together.
