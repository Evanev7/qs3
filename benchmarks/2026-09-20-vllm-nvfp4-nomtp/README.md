# Qwen3.8-27B NVFP4: stock vLLM without MTP

GB10, NVIDIA checkpoint `dbb8f445b3145f8a4c18ddc769f032d57d32867c`.
vLLM 0.29.0, Torch 2.13.0+cu130, FlashInfer 0.6.18, CuTe DSL 4.6.2.
Exact image digest, checkpoint configuration and driver are in `metadata.json`,
`image.json`, `checkpoint-config.json` and `gpu.csv`.

| Workload | qs3 prefill p50 | vLLM prefill p50 | qs3 decode p50 | vLLM decode p50 | qs3 tok/s | vLLM tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 102/32 | 129.010 ms | 120.533 ms | 88.668 ms | 82.707 ms | 11.264 | 12.093 |
| 1024/256 | 414.538 ms | 386.701 ms | 89.207 ms | 83.582 ms | 11.205 | 11.951 |

qs3: committed `e1befe1` core benchmark artifacts linked in `comparison.json`.
vLLM: stock image/kernel selection, default graphs (full decode), v1 runner;
`speculative_config=None` asserted after initialization. Each delivery appends
one token. No MTP, prefix caching, chunked prefill, detokenization, or HTTP server.
Batch one, greedy, BF16 KV/conv state and FP32 recurrent state match qs3.
Selected linear providers: FlashInfer CUTLASS NVFP4 and FlashInfer FP8.

Exact prompt fingerprints and decode context intervals match. Two prefill
warmups precede five samples; four decode warmups precede 32/256 samples.
Setup/loading/compilation/autotuning/capture are excluded. Throughput uses total
sample time, not reciprocal p50. Raw timing samples and generated IDs are in
`102-32/vllm.json` and `1024-256/vllm.json`; logs retain resolved configuration
and kernel choices.

Comparison limits: qs3 is eager while vLLM uses graphs. Continuations are
independent: first generated-token differences occur at zero-based indices
18 and 4. Matching later token positions does not establish matching state.
vLLM prefill includes sampling/delivering token one; qs3 prefill requests zero
new tokens. Both decode measurements use wall time.

Reproduce with `.prototypes/vllm_nvfp4_baseline/run.py` on the GPU checkout,
sequentially with other GPU work. Frozen scripts and exact Docker commands are
included here. `SUCCESS.json` records both completed workloads.
