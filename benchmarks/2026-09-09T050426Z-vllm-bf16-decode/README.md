# Aligned vLLM 35B BF16 decode trace

The pinned image and model are recorded in `image.txt` and `result.json`.
Capture uses Nsight Systems 2025.3.2, graph node tracing, BF16 model weights,
BF16 convolution history, FP32 recurrence, default graph execution, 102 prompt
IDs and four warmup decode deliveries. Capture starts one delivery earlier than
the first attempt: the engine queues model work ahead of delivery. Counts now
confirm 32 forwards: 1,280 calls per MoE projection and 960 GDN calls.
`result.json` records 37 generated IDs: one prefill output and 36 decode outputs.
The corresponding qs3 run returns the same first 36 IDs.

| Captured quantity | qs3 | vLLM |
| --- | ---: | ---: |
| Kernels | 38,948 | 31,840 |
| Summed kernel duration, ms | 1278.544 | 1172.791 |
| First-to-last-kernel span, ms | 1379.419 | 1049.259 |
| GPU work union within span, ms | 1283.637 | 1034.467 |
| No GPU work within span, ms | 95.782 | 14.792 |
| Decode convolution, 960 calls, ms | 37.418 | 2.683 |
| GDN recurrence, 960 calls, ms | 13.351 | 13.022 |
| MoE projection kernels, 2560 calls, ms | 392.597 | 324.712 |

Kernel sums include overlapping work and must not be treated as elapsed latency.
The vLLM graph trace has overlapping kernels, so its sum exceeds its span. The
implementations differ in graph mode, projection packing, fusion, and some
activation/output precision. These are diagnostic comparisons, not isolated
kernel A/B benchmarks. Use unprofiled core artifacts for throughput conclusions.

vLLM uses packed QKV/Z projections and two distinct MoE kernels: a fused SM80
kernel (222.815 ms) and CUTLASS MoeFCGemm (101.896 ms). qs3 uses separate
projections and its 96-block grouped GEMM. Decode convolution and projection
packing are concrete candidates for AOT probes; preserving device addresses and
moving sampling/output completion out of enqueueing are prerequisites for graphs.

The presumed LM-head GEMV (grid 62,080 = 248,320 output rows / 4) has 32 calls,
164.707 ms, and BF16 matrix/input/output. qs3's matching kernel signature has FP32 input/output
and 197.355 ms. Its caller supplies BF16 activations, as confirmed by the Rust
scratch type and native cuBLASLt matrix layouts; FP32 input here belongs to the
internal library path. Output precision is the confirmed model-level difference.
Shape and signatures identify the projection; a same-prefix score
comparison and controlled precision probe are still needed before changing it.
Do not attribute the sustained greedy divergence to this difference without that
comparison.

Launch and collection scripts are included. Full traces remain under
`.prototypes/vllm-runs/` and `sp10@sp10:~/qs3-vllm-bench/` with this capture name.
The 27B asset download was running, so this is not an isolated core timing run.
