# One warmed 1024-token prefill

sp10 GB10, commit `bf86169`, managed BF16 weights, FP32 recurrence, 96-block MoE,
eager Rust/AOT execution, final-row vocabulary projection. `QS3_PROFILE=prefill`
captures the first measured prefill after two warmups. Reset and its completion
sync precede capture. Profiler start/stop remain outside the sample timer.
This is a kernel-cost diagnostic: the 27B asset download was active, so do not
use this run as an isolated end-to-end throughput comparison.

| Captured quantity | Value |
| --- | ---: |
| Kernel launches | 1174 |
| Kernel duration sum | 1854.543 ms |
| GDN FP32 recurrence, 30 calls | 962.632 ms (51.9%) |
| GDN causal convolution, 30 calls | 497.233 ms (26.8%) |
| Grouped MoE GEMMs, 80 calls | 314.039 ms (16.9%) |
| First-to-last kernel span | 1866.715 ms |
| No GPU work inside that span | 2.399 ms |
| Pinned-host API union | 40.320 ms, no overlap with GPU work |

The local convolution uses one block per sequence and walks tokens serially.
Each output only needs four input values, so prefill tokens can be processed in
parallel. Preserve initial state reads and final state writes, including same-slot
updates; do not introduce a cross-block race. GDN recurrence remains the larger
cost and needs a separate chunked-prefill investigation. Dense vocabulary
projection is not the main prefill bottleneck.

The pinned allocation cost occurs before the main kernel span, consistent with
prefix rebuild still discarding provider resources. Full capture artifacts are
in `.prototypes/profiles/2026-09-09T041450Z-bf86169-prefill/` locally and
`~/qs3-profiles/2026-09-09T041450Z-bf86169-prefill/` on sp10. Raw summaries,
runner metadata and interval analysis are included here. Reproduce with the
gitignored `.prototypes/run_prefill_profile.sh` and
`.prototypes/collect_prefill_profile.py`.
