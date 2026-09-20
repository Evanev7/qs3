# Fused SiLU×up and NVFP4 quantization

Qwen3.8-27B NVIDIA NVFP4 on GB10. Preserves BF16 rounding and zero-filled
scale padding. Compared with the committed CuTe baseline `1b298a0`.

| Workload | Baseline prefill | Fused prefill | Baseline decode | Fused decode |
| --- | ---: | ---: | ---: | ---: |
| 102/32 | 129.689 ms | 127.893 ms | 85.262 ms/token | 85.027 ms/token |
| 1024/256 | 417.179 ms | 399.731 ms | 85.984 ms/token | 85.626 ms/token |

Pooled medians from two runs per provider, ordered baseline/fused/fused/baseline.
Core benchmark protocol, identical Nix CUDA libraries, eager execution, no MTP.
The prefill gain is clear (~4.2% at 1024 tokens); decode savings are small relative
to drift. Long-run decode medians ranged 85.671–86.062 ms for the baseline and
85.563–85.668 ms for the fused path. All generated IDs match.

The production operation at K17408 measures 4.67→3.67 µs at M1 and 654→371 µs
at M1024, including scale padding. Kernel timings use 128 captured invocations,
three warmup replays, and the median of 15 replays, repeated in reverse order.

Full required tests and real-checkpoint reset/replay pass. All 81 native cases
match packed activations/scales exactly, covering row tails, output guards,
invalid bindings, and graph replay.

[results.json](results.json) contains every raw model sample and generated ID,
kernel measurements, checkpoint/source hashes, and environment metadata.
The candidate was measured from the working tree; a standard committed
benchmark/Nsight pass remains after review/commit.
