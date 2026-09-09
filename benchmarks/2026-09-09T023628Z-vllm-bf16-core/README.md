# First vLLM short-context comparison

sp10 GB10, driver 580.142; vLLM 0.21.0, PyTorch 2.11.0+cu130,
Transformers 5.8.1. Container:
`vllm-node@sha256:d966c1831d5da55c0cc52c6bd40f7d02cfc3d83404c3bd599139b055232d3970`.
The immutable model snapshot is Qwen3.6-35B-A3B
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16 weights/activations.
The harness and launch script are copied here with the raw log and result JSON.
The scripts were run from `.prototypes/`; mounts and output paths assume sp10.

Decode measures **30.826 tok/s**, p50 **32.355 ms**, versus qs3 `90b52c4` at
11.905 tok/s and 83.701 ms. Both use the exact 102-token prompt fingerprint
`6ca602a4cc15238d`, greedy sampling, one request, four decode warmups and 32
measured decode forwards (contexts 106 through 138). vLLM emits the first token
at prefill, so the harness requests 37 outputs to measure 36 decode forwards.
qs3 emits 36 tokens and processes the last one before returning. Generated IDs
are recorded for later comparison; a full output equivalence check is pending.

This is **not yet a fully matched-precision comparison**. vLLM resolves its GDN
convolution state to BF16 and recurrent state to FP32; qs3 stores both in BF16.
Explicit `bfloat16` cache options are rejected by this container's configuration
schema. The harness requests `auto` and records the effective configuration and
resolved state dtypes. The effective recurrent cache setting becomes `float32`.
Do not infer matched recurrent precision from the BF16 weight dtype alone.

vLLM enables full decode CUDA graphs and piecewise prefill graphs. Its log selects
FlashInfer CUTLASS unquantized MoE, FlashAttention 2, and Triton/FLA GDN prefill.
qs3 uses eager AOT execution, the four-block grouped CUTLASS MoE kernel, and local
GDN. Prefix caching and chunked prefill are disabled in vLLM, tokenizer execution
is skipped, detokenization is disabled, and KV cache budget is 512 MiB. The
container runs offline with one engine process and an existing compile cache.

Each sample is wall time between host token deliveries, including engine steps
that produce no output. One empty step was observed per request, before its first
output. This includes scheduling and host output processing; it is not CUDA event
kernel time. Prefill p50 is 181.369 ms, including first-token sampling/delivery;
qs3's 1015.461 ms prefill ends with logits and performs no sampling. Setup takes
272.161 seconds. The first prefill warmup takes 16.592 seconds and logs additional
JIT compilation. The second warmup is 183.219 ms; both are excluded from the five
prefill samples. Setup and warmup are not steady decode latency.

Further work: match recurrent precision, compare generated tokens and sustained
quality, add several contexts and sustained decode, and capture a vLLM GPU trace.
