# LM-head allocation comparison

The integrated Triton row kernel reaches its earlier isolated speed only with
device allocations. Managed allocations remove the advantage over cuBLASLt.
This accounts for the nearly flat real-model benchmark without establishing
the underlying memory-system mechanism (no translation/cache counters collected).

Median of three 50-sample CUDA-event p50 measurements, after cache eviction:

| Weight allocation / initialization | cuBLASLt | Triton |
| --- | ---: | ---: |
| `cudaMalloc`, GPU initialized | 5.698 ms | **4.167 ms** |
| `cudaMallocManaged`, GPU initialized | 6.089 ms | 6.110 ms |
| `cudaMallocManaged`, CPU initialized | 6.094 ms | 6.105 ms |

Repeated-weight results show the same distinction; see `summary.json` and raw
CSVs. All three allocation cases pass comparisons against cuBLASLt over every
output and a CPU double reference over 65 rows. Maximum FP32 difference from
cuBLASLt is 1.19209e-6. Initialization and allocation are outside timing, ten
warmups precede each measurement, and candidate/control order alternates across
the three repetitions. The eviction pass touches at least 128 MiB / four times
L2 outside the measured interval. Clocks are not locked.

The probe uses the exact cubin emitted by the integrated `qstriton` builder:
248320 x 2048 BF16 weights, BF16 activation, FP32 output, eight warps, one stage,
FP32 accumulation, FP fusion disabled. Only the weight allocation/initialization
changes; activation, output, workspace and eviction allocations use `cudaMalloc`.
No managed-memory advice or prefetch is applied. Synthetic weights occupy about
0.95 GiB, so this does not measure full-model allocation pressure or load cost.

`bench.cu` is the earlier b12x probe harness with weight allocation and
initialization selectable. It consumes the current generated manifest/cubin
instead of recompiling the archived kernel. Run from the repository root:

```sh
./run_cuda_test.sh bash benchmarks/2026-09-10-triton-lm-head-memory-probe/run.sh
```

Outputs are under `build/lm-head-memory-probe` on sp10. `host.txt` records the
system-toolkit linkage and hashes; `lm_head.json` records compiler inputs. This
probe uses the installed CUDA toolkit, while the core benchmark uses its Nix
closure. Allocation comparisons within this probe share identical libraries.

Next experiment: device-back just the real LM-head weight using the existing
loader transfer machinery and measure both loading cost and steady decode.
The approximately 1.94 ms Triton allocation difference here is an isolated
measurement, not a promised end-to-end saving. Further Triton GEMV candidates
must be tested with the actual runtime allocation path.
