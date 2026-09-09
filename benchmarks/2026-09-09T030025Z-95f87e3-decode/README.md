# Steady decode after widening MoE to 96 blocks

Commit `95f87e3d3d084b41a9603bb99443ac7f8cdedbec`, sp10, GB10,
Nsight Systems 2025.3.2.474. Same pinned 35B BF16 weights, managed storage, eager
execution, 102-token prompt and 32 measured decode steps as the
[attention-reuse capture](../2026-09-09T022039Z-90b52c4-decode/README.md).
Profiler invocation and interval-union analysis follow the earlier captures.
The full trace and SQLite database are retained under
`sp10@sp10:/home/sp10/qs3-profiles/2026-09-09T030025Z-95f87e3-decode/`.
Summaries were computed on sp10 and copied here independently of the bulk trace
transfer to `.prototypes/profiles/`.

All 2,560 grouped MoE launches now use `[96,1,1]`, confirmed from SQLite grid
fields. Their aggregate time falls from 1533.57 to **393.82 ms**, or 12.31 ms per
token, reducing MoE's share of kernel time from 58.9% to 26.9%. Kernel count is
unchanged at 38,948. Total kernel time falls from 2603.22 to 1465.65 ms.

The first-to-last-kernel span is 1568.56 ms; its recorded GPU work union is
1470.81 ms, leaving 97.75 ms without recorded GPU work. The preceding capture
had 97.53 ms without GPU work. Pinned allocations/frees remain absent; eight
device allocations/frees remain. This change reduces GPU execution time rather
than the host allocation gaps removed by attention workspace reuse.

Dense cuBLAS GEMV kernels now total **772.90 ms** across the capture and dominate
GPU time. Serial router top-k remains **178.05 ms** (5.56 ms/token, 12.1% of
kernel duration). GDN convolution and recurrence total about 70.25 ms together.
Test parallel router selection and examine the large GEMV shapes. Persistent
metadata and graph replay remain architectural work, but idle-span duration
alone does not predict their performance gain.

The [unprofiled 96-block run](../2026-09-09T025651.359461497Z-95f87e3.json)
measures 20.724 tok/s and 48.252 ms p50. The
[same-revision four-block control](../2026-09-09T025757.341711005Z-95f87e3.json)
measures 11.876 tok/s and 83.834 ms. All 36 generated IDs match. Profiled activity
intervals and unprofiled runner calls have different endpoints and must remain
separate. The earlier vLLM comparison still has an FP32/BF16 recurrent-state
precision difference.
