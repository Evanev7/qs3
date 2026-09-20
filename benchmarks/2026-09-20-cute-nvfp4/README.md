# CuTe NVFP4 GEMM, Qwen3.8-27B on GB10

Selected recipe: **32x64x512, no split-K, BF16 output**, using b12x
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. Three AOT exports share the kernel:
gate/up (N17408/K5120), down (N5120/K17408), head (N248320/K5120).
M=1/2/4/8/16 is qualified. Existing packed weights, scales and quantization remain.

The integrated working tree passes `./remote.sh test` and the explicit real
checkpoint prefill/decode/reset test; logs are in `validation/`.

| Integrated runtime | Previous committed decode | CuTe decode | CuTe tok/s | CuTe prefill |
| --- | ---: | ---: | ---: | ---: |
| 102/32 | 88.668 ms | 85.126 ms | 11.731 | 127.496 ms |
| 1024/256 | 89.207 ms | 85.558 ms | 11.685 | 414.671 ms |

The sustained gain is **3.648 ms/token (4.1% lower latency)** against `e1befe1`,
and 3.859 ms against the paired prototype CUTLASS run below. `runtime/` contains
the final measurements, source snapshot/hashes, reproduction script, and exact
Nix dynamic libraries. These are unprofiled working-tree measurements using
the core benchmark protocol; the committed benchmark/Nsight pass remains to run
after review. Stock vLLM without MTP measures 83.582 ms/token on the long
workload, with full decode graphs and an independent continuation.
Both integrated continuations match their paired CUTLASS runs exactly: 36/36
and 260/260 generated IDs, including warmups (`runtime/comparison.json`).

| M=1, weights evicted | CUTLASS N32 | CuTe | Speedup |
| --- | ---: | ---: | ---: |
| Gate/up | 303.84 µs | 283.36 µs | 1.072x |
| Down | 318.69 µs | 289.46 µs | 1.101x |
| LM head | 3352.38 µs | 3080.35 µs | 1.088x |

`wide/results.jsonl` contains paired samples; `wide/summary.json` includes all
batch sizes. GPU event measurements alternate providers for 30 samples, with
warm and 128 MiB eviction conditions. Host enqueue times are recorded separately.
There are 120 passing correctness cases in this extension, all bit-exact;
the initial 24-candidate sweep adds 240 passing cases (split-two has small
rounding differences). Tests cover varied block scales, alpha 1/0.137, output
canaries and FP64 spot checks. The N32 CuTe tile trial failed during compilation;
`n32-compile-failure.log` records that unsupported experiment.

The selected recipe matches **1,544/1,544 real intermediates exactly**, across
prefill, forced decode and reset/replay, with CUTLASS results fed onward:
`sjpp7_4p/real-results.jsonl`. The same probe for the initial tile16/K128 recipe
is in `asriptyg/`.

The sustained experimental runtime replacement measures **85.481 ms/token,
11.694 tok/s**, versus matched CUTLASS **89.418 ms/token, 11.177 tok/s**
(1024 context, 256 samples). See `v1wz5e3n/model-benchmark.json` and
`89wvrv9k/model-benchmark.json`. These use the core benchmark protocol and
its Nix CUDA libraries, with isolated source overlays. This is prototype
evidence, separate from the final integrated runtime validation. All 260
generated IDs (four warmups plus 256 samples) match the paired CUTLASS run.

Initial tile16/K128 results: 88.550→87.966 ms at 102/32 and 89.418→88.686 ms at
1024/256, with identical generated IDs. Those four runs are in `34s2z0ty/`,
`469y4dup/`, `89wvrv9k/`, and `6pql67xs/` respectively.

Frozen experiment sources are in `probe/`; run scripts assume the pre-integration
source tree recorded in their metadata. Current implementation is
`cute_kernels/nvfp4_gemm.py`, with three exports owned by `Nvfp4Linears`.
