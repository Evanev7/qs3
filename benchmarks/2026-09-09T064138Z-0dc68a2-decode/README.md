# Decode trace after tiled convolution and 32-row MoE

Source: `0dc68a28994e05a6adee2a2c0c2975d965b3637a`, sp10 GB10, CUDA 13.0,
Nsight 2025.3.2.474. Pinned 35B BF16 weights, FP32 GDN recurrence, eager AOT,
102-token core prompt, four warmups, 32 captured decode forwards. Timing under
profiling is diagnostic; use the separate core JSON for throughput.

The capture has 38,948 kernels, including 2,560 grouped GEMMs, 960 GDN recurrences
and 960 decode convolutions. Every grouped GEMM uses the compiled 32x128x64 tile
and 96-block grid. Kernel sum is 1,215.089 ms over a 1,315.389 ms span; GPU work
union is 1,220.507 ms, leaving 94.882 ms without GPU work. No pinned-host allocation
or release occurs in the range. Eight device allocations/frees remain.

Against the aligned earlier qs3 trace `2026-09-09T045948Z-262b4f2-decode`:

| Kernel family | Earlier ms | Current ms |
| --- | ---: | ---: |
| Grouped MoE GEMMs | 392.597 | 358.915 |
| Decode convolution | 37.418 | 2.764 |
| GDN recurrence | 13.351 | 13.297 |
| cuBLAS GEMV | 775.139 | 779.276 |
| No GPU work within kernel span | 95.782 | 94.882 |

This comparison includes both the convolution tiling and MoE tile changes.
It confirms their kernel savings and persistent host gaps; it does not isolate
the tile's end-to-end effect. The same-commit sustained core A/B isolates that.
CUDA API duration includes waiting in token-delivery copies and launch calls;
it must not be added to GPU time or labeled pure CPU preparation.

Generated using `.prototypes/run_decode_profile.sh` and
`.prototypes/collect_decode_profile.py`. Full report/SQLite are retained in
`.prototypes/profiles/2026-09-09T064138Z-0dc68a2-decode/` and
`sp10@sp10:~/qs3-profiles/2026-09-09T064138Z-0dc68a2-decode/`.
