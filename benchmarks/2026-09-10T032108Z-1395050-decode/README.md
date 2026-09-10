# Integrated Triton LM-head decode trace

Source: `1395050ccaf82ca53b0aa779a0c1a0b5e5e6d5bb`. GB10, driver 580.142,
Nix CUDA 13.0, pinned 35B BF16 snapshot, managed weights, FP32 GDN state,
32-row/96-block MoE, eager execution. The 102-token core prompt is followed by
four warmups and 32 captured decode forwards. Use the unprofiled core JSON for
throughput; profiler timing is diagnostic.

| Work | GPU time per decode forward |
| --- | ---: |
| Other cuBLAS GEMVs | 18.218 ms |
| Grouped MoE GEMMs | 11.214 ms |
| Triton LM head | 6.043 ms |
| GDN recurrence | 0.406 ms |
| GPU idle time within kernel span | 2.974 ms |

The capture contains 38,948 kernels, including exactly 32 `row_kernel` launches.
Total kernel duration is 1,211.088 ms; kernel span is 1,311.697 ms. GPU-work union
is 1,216.519 ms, leaving 95.178 ms without GPU work. No pinned-host allocation or
release occurs in the captured interval. The prior September 9 decode trace had
1,215.089 ms of kernels and 94.882 ms without GPU work: no large gain appears.

The [matching unprofiled run](../2026-09-10T031951.967883789Z-1395050.json)
measures 24.791 tok/s versus the user's immediately preceding
[cuBLASLt baseline](../2026-09-10T030755.492864205Z-77fbfaf.json) at 24.716 tok/s.
Median decode latency is 40.303 versus 40.421 ms; prefill median is 239.421 versus
241.164 ms. All 36 generated IDs (warmup plus measured) match. A single 0.3%
throughput change is too small to establish a meaningful improvement.

The [allocation probe](../2026-09-10-triton-lm-head-memory-probe/README.md)
reproduces the earlier 4.17 ms Triton result with `cudaMalloc`, but measures
6.10 ms with managed weights, matching cuBLASLt and this runtime trace.

Validation: 153 Rust tests and two standalone launcher tests passed. The real
35B reference test passed with both BF16 and FP32 GDN recurrence, including
prefill/first-decode comparisons of all 248,320 logits against cuBLASLt on the
same actual model activations. Maximum absolute difference was 2.861023e-6.
Nix and Ninja produce identical SASS; source-path debug metadata differs.

For further Triton kernels, first revisit GDN's 8192x2048 QKV projection on real
managed/device weights. The existing evicted-cache prototype saves only about
6–7% there and regresses on smaller gate/output shapes, so do not replace all
GEMVs together. Routed-expert GEMV is a larger opportunity in the 11.2 ms MoE
budget, but needs an independent prototype preserving BF16 intermediates,
top-k weighting and reduction, with the tested FlashInfer path as control.
Q/K normalization plus partial-RoPE preparation is a smaller fusion candidate;
keep KV append separate. The 2.97 ms idle budget remains visible but is not a
GPU kernel-duration saving and is outside this traditional-kernel experiment.

Produced with `.prototypes/run_decode_profile.sh` and
`.prototypes/collect_decode_profile.py`. Full report/SQLite are retained at
`.prototypes/profiles/2026-09-10T032108Z-1395050-decode/` locally and
`sp10@sp10:~/qs3-profiles/2026-09-10T032108Z-1395050-decode/`.
