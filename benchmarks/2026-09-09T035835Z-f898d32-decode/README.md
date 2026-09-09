# Integrated warp-router decode trace

sp10 GB10, commit `f898d321e6fb9c893ee2b044966b65d70642859b`, Nsight Systems
2025.3.2. Capture covers the same 32 warmed decode forwards after a 102-token
prompt and four decode warmups. Managed BF16 weights, FP32 GDN recurrence,
96-block MoE, eager Rust/AOT execution. The 27B download was paused for capture.

| Captured quantity | Value |
| --- | ---: |
| Kernel launches | 38,948 |
| Kernel duration sum | 1300.622 ms |
| First-to-last kernel span | 1399.623 ms |
| GPU work union inside that span | 1305.862 ms |
| No GPU work inside that span | 93.761 ms |
| Router: 1280 calls | 8.662 ms |
| Dense GEMV kernels | 774.393 ms |
| Grouped MoE GEMMs: 2560 calls, all grid 96 | 395.691 ms |

The previous `95f87e3` trace has 178.054 ms of scalar routing, compared with
8.662 ms here: about 20.6 times less router kernel time. The kernel sum falls
from 1465.653 to 1300.622 ms, while dense GEMV and MoE costs stay close. That
older trace used BF16 recurrent state; the current one uses FP32, so this is
not a strictly single-variable whole-trace comparison. The unprofiled matched
FP32 core runs separately measure the router's 12.3–12.4% throughput improvement.
Dense projections remain the largest GPU cost. Roughly 94 ms without GPU work
across 32 steps leaves host preparation/capture worth pursuing as well.

No pinned-host allocation/free calls appear. Raw summaries and interval totals
are recorded here; full Nsight/SQLite artifacts are in the gitignored local
`.prototypes/profiles/2026-09-09T035835Z-f898d32-decode/` and remote
`~/qs3-profiles/2026-09-09T035835Z-f898d32-decode/`. Collection uses
`.prototypes/collect_decode_profile.py`. Profiled elapsed time includes profiler
overhead; use the adjacent unprofiled core JSON files for throughput.
