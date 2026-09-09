# Grouped MoE launch sweep

sp10, GB10, 2026-09-09. Source baseline `9a3fd5e`, with the same MoE arithmetic
as `90b52c4`. Pinned FlashInfer revision
`b3baedbbef2686df91b6dc43818ee56fe26ceba2`. The prototype changes only the grouped
GEMM argument `threadblock_count` from 4 to a host-selected value. The existing
SM80 BF16 kernel, 128x128x32 CTA tile, 64x64x32 warp tile, and two stages remain.
The vendor checkout is unchanged. Sources and executable remain in
`sp10@sp10:qs3/.prototypes/moe_sweep/`; the driver and build script are included.
To reconstruct, copy `qsfi.cu` and `qsfi_moe.cu` into that directory, redirect the
latter's grouped-GEMM include to a local copy of the pinned `group_gemm.cuh`, and
replace its literal four blocks with `extern int qs3_moe_threadblocks`. Resolve
other relative includes against the original repository and FlashInfer paths.

The driver reuses `bench_native.cu`'s complete staged MoE benchmark: hidden 2048,
256 experts, top-k 8, intermediate 512, sequential round-robin routes and uniform
route scales. Inputs and weights are zero, so these are performance probes,
not correctness evidence. Each row averages 100 CUDA-event timed calls after 10
warmups. Weights are device allocations and each case starts fresh. There are two
complete repetitions in the same block-count order. Times include route prep,
both GEMMs, activation and final reduction, excluding allocation/setup.

| Blocks | 1 token (µs, repetitions) | 16 tokens (µs) | 102 tokens (µs) |
| --- | ---: | ---: | ---: |
| 4 | 1284.945 / 1286.153 | 19775.496 / 19834.724 | 39592.498 / 39775.059 |
| 12 | 565.257 / 559.645 | 7595.026 / 7602.900 | 15031.580 / 15100.536 |
| 24 | 401.386 / 402.594 | 5175.815 / 5201.896 | 10285.596 / 10302.502 |
| 48 | 350.462 / 355.265 | 4208.156 / 4218.918 | 8348.116 / 8356.841 |
| 96 | 317.142 / 315.063 | 3951.450 / 3949.678 | 7878.549 / 7886.304 |

96 blocks is the best tested setting, not a claim of a global optimum. The
one-token case improves about 4.1x and larger cases about 5x. Integrating this
requires nonzero numerical checks, real-model regression and an unprofiled core
measurement; do not equate this microbenchmark speedup with model throughput.
