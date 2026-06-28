this is quasar3, a minimal qwen3.6 runtime built from flashinfer, inspired by
dwarfstar4.

rarely record durable facts here unless they took significant information
gathering.

run tests with the gitignored test script

ground rules:
- do not worry about backward compatibility, abi stability or versioning.
- no cmake; we'll figure out a build system later.
- keep the runtime qwen3.6-specific. validate early, fail loudly, avoid generic
  runtime compatibility work. take the time to cleanup and tidy after refactors.

current state:
- `EngineCore` already owns request ids, sequence lengths, a page allocator,
  page tables, last-page lengths, append positions, and staged batch state.
- `runtime::EngineInner` already owns per-layer paged K/V caches and reuses
  FlashInfer prefill/decode plans when the batch/page-table shape permits.
- the current repo can run FlashInfer attention over externally supplied Q/K/V
  tensors. it is not yet a runnable model runtime.

GDN direction:
- keep qwen3.6-specific GDN prep glue local for now: causal conv, post-conv
  Q/K/V split, decay/beta materialization, gated RMSNorm, and a local recurrence
  fallback.
- FlashInfer GDN looks like the first replacement path to investigate, not
  Triton/vLLM: BF16 decode matches the `[pool, HV, V, K]` state shape, and SM12
  chunked prefill matches the long-prompt path if prep emits linear alpha
  `exp(g)` plus f32 beta.
- if FlashInfer GDN is wired, keep it in one FlashInfer-owned TU. do not spread
  FlashInfer GDN headers/JIT plumbing through local `qscu` files.
- main mismatches to resolve before replacing the local recurrence: FlashInfer
  BF16 decode treats `-1` state indices as a sacrificial slot with undefined
  output, and FlashInfer prefill wants f32 final state while qs3 may want bf16
  live decode state.

DS4 lessons worth preserving:
- keep the public boundary narrow: a loaded model/runtime object plus mutable
  inference timelines. higher layers should not know tensor internals.
- durable session state is more than a token count. keep exact request tokens,
  last logits, paged KV tables, page ownership, and append/frontier positions as
  explicit engine state.
- prefix sync policy should be conservative. if a requested prompt extends the
  live token prefix, append only the suffix. if it rewrites behind the live tail,
  rebuild or restore an older checkpoint; do not patch only the token vector.
- separate paths:
  - long suffix/prompt: chunked prefill.
  - short/live generation: decode.
- cache/snapshot payloads, when added, should be engine-owned: exact tokens,
  logits, page-table/frontier state, and KV contents needed to make the next
  token match an uninterrupted session.

fastest path to actually running the model:
- target one qwen3.6 shape first, BF16 correctness first, then NVFP4 for the
  first optimized path
- add model loading for config + safetensors, map fixed Qwen tensor names, upload
  weights to CUDA.
- add the missing non-attention kernels/wrappers: embedding gather, RMSNorm,
  QKV/output projections, gated SiLU MLP, final norm, LM head, logits, and a
  minimal sampler.
- build a `ModelRunner` above `Engine` that computes layer activations, supplies
  Q/K/V to the existing attention engine, stores logits, samples next tokens,
  and owns exact-prefix sync/rebuild.
- first runnable CLI may accept and print token ids. tokenizer/chat template can
  follow immediately after the model produces tokens.
