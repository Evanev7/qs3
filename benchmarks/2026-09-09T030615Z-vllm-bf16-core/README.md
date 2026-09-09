# Longer-context vLLM comparison

Same sp10 GB10 and pinned vLLM container/model snapshot as the
[short-context comparison](../2026-09-09T023628Z-vllm-bf16-core/README.md).
Invocation from `.prototypes/`:

```sh
VLLM_BENCH_CONTEXT_TOKENS=1024 VLLM_BENCH_DECODE_SAMPLES=256 .prototypes/run_vllm_core_benchmark.sh
```

The prompt repeats the same base 102 token IDs and truncates to 1024 tokens.
The base fingerprint is `6ca602a4cc15238d`; the qs3 run reports the full repeated
prompt fingerprint `c8cd97958d34675c`. This synthetic repetition controls compute
work; it is not a natural-language quality benchmark. Both use four decode
warmups and 256 measured decode forwards, contexts 1028 through 1284, greedy
sampling, and the same BF16 weights. vLLM has full decode graphs enabled.

| Runtime | Decode tok/s | Decode p50 ms | Prefill p50 ms |
| --- | ---: | ---: | ---: |
| qs3 `82c18b0`, 96-block MoE, eager | 20.730 | 48.216 | 1941.522 |
| vLLM 0.21.0, graphs | 30.769 | 32.481 | 367.629 |

[qs3 JSON](../2026-09-09T030333.392798072Z-82c18b0.json) records 260 generated
IDs including warmups; vLLM records 261 because it samples the first token at
prefill and the last after its final decode forward. The first 56 generated IDs
agree. At zero-based index 56, qs3 produces 32956 and vLLM 33027; subsequent
contexts then differ. Full generated sequences and per-step times are retained.
This is no longer an identical-token trajectory after that point.

Recurrent-state precision remains unmatched: qs3 BF16 versus vLLM FP32;
convolution state is BF16 in both. Different kernel arithmetic and prefill paths
also remain possible causes of divergence. This observation alone does not
establish an accuracy bug or its cause. The native qs3 GDN kernels already accept
FP32 recurrent state; expose that choice through model storage and rerun the
comparison before interpreting sustained output differences. Broader quality
validation and a third context length remain open.

Prefill timing has the same endpoint difference as the short run: qs3 internally projects and samples every prompt row, returning no generated
token for `max_new_tokens=0`, while vLLM returns its first generated token. The vLLM log includes
its actual provider, graph, cache and compilation choices. Host token-delivery
intervals include empty engine steps.
