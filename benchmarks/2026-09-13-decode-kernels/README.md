# Single-token kernel sweeps, 2026-09-13

Control source: `8451569`, sp10 GB10, CUDA 13.0, SM121, native AOT kernels.
No graphs, runtime Python, weight repacking, or runner changes. Synthetic inputs
use device allocations. These isolated timings are not end-to-end speedups.

## Routed MoE

`build.py` and `kernels.cuh` reproduce the initial sweep from the control checkout:
copy them to `.prototypes/moe_decode/`, run a normal native build, then run
`python3 .prototypes/moe_decode/build.py` from the repository root. The script
renames only the prototype's MoE API symbols and links the existing native
objects. `moe.log` is the complete output.

One token, hidden 2048, intermediate 512, 256 experts, eight selected IDs spread
across the expert range. BF16 activations/weights and all intermediate rounding
stages match the staged implementation; scalar FP32 reductions differ from
CUTLASS's tensor-core accumulation order. Each row loads eight BF16 values per
thread at a time. Existing workspace holds the two projected outputs and SiLU
activation. Four launches replace grouped routing preparation and GEMMs.

Three repetitions share allocations, input values, weights, routes, and plans;
each uses five warmups and 100 full-stage launches timed with CUDA events.
Every output is compared against CUTLASS. Median means, microseconds:

| Kernel | us |
| --- | ---: |
| CUTLASS tile32/96 control | 276.674 |
| GEMV 32 threads/row | 227.889 |
| GEMV 64 threads/row | 221.519 |
| GEMV 128 threads/row | 218.837 |
| GEMV 256 threads/row | 224.684 |

All GEMV variants produce the same output: 386/2048 BF16 values differ from
CUTLASS, maximum absolute error 0.000488281, relative RMS error 0.00155179.
The 128-thread variant spans 218.560–225.687 us; the 64-thread variant spans
221.130–221.520 us. This fixture is a screening test. Real-model scores and
continuation remain the integration gate.

## Greedy argmax

`argmax.cu` contains the copied `8451569` control and the candidate kernels.
Build from the repository root:

```
nvcc -std=c++17 -arch=sm_121 -I. -Ibuild -DQSFI_ENABLE_CHECKED_VALIDATION=0 benchmarks/2026-09-13-decode-kernels/argmax.cu -lcuda -o /tmp/qs3-argmax-probe
/tmp/qs3-argmax-probe
```

248320 FP32 logits, one row, device allocation, repeated warm data. All kernels
select ID 200001 over an equal maximum at 240000. Three repetitions, 200 launches
per event measurement; `argmax.log` contains all raw results. Median means:

| Kernel | us |
| --- | ---: |
| Existing shared-memory tree, 256 threads | 160.919 |
| Warp reduction, 256 threads | 34.811 |
| Warp reduction, 512 threads | 20.485 |
| Warp reduction, 1024 threads | 16.384 |
| Four adjacent elements/thread, 256 threads | 84.051 |
| Four adjacent elements/thread, 512 threads | 45.128 |
| Four adjacent elements/thread, 1024 threads | 28.341 |

Keep scalar coalesced loads, warp shuffles, and one shared-memory exchange of
warp winners. The production candidate uses 1024 threads for vocabularies at
least 65536 wide and 256 for smaller fixtures. Existing finite-value validation,
lowest-ID tie-breaking and signed/unsigned result semantics stay unchanged.
The real-model trace's ~0.42 ms argmax differs from this repeated warm microprobe;
measure its integrated result separately.
