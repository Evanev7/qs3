this is quasar3, a minimal qwen3.6 runtime built from flashinfer, inspired by
dwarfstar4.

rarely record durable facts here unless they took significant information
gathering.

run tests with the gitignored test script

ground rules:
- do not maintain backward compatibility, abi stability or versioning.
- if you introduce a replacement api or build target, remove the old entrypoint.
- take the time to cleanup and remove legacy calls after refactors.
- no cmake.
- keep the runtime qwen3.6-specific. validate early, fail loudly, avoid generic
  runtime compatibility work. 

current state:
- `3pty` contains vendored kernels and refernce code
- `target` is the rust build directory
- `EngineCore` already owns request ids, sequence lengths, a page allocator,
  page tables, last-page lengths, append positions, and staged batch state.
- `EngineCore` begin/commit/abort/reset/release paths should stay
  transactional: build candidate state and live views before installing them.
  keep allocator invariant checks debug-only.
- `runtime::EngineInner` already owns per-layer paged K/V caches and reuses
  FlashInfer prefill/decode plans when the batch/page-table metadata permits.
  plan-cache keys include page ids and last-page lengths, not just CSR shape.
- the public `ModelRunner` can run a randomized dense BF16 Qwen-shaped model
  end-to-end: embedding gather, RMSNorm, Q/K/V/O projections, FlashInfer
  attention, gated SiLU MLP, final norm, LM head logits, and greedy sampling.
  it owns exact-prefix sync/rebuild for the current single-runner path.
- the randomized runner is intentionally narrow. current compiled attention
  support is head dim 64 with no GQA; Rust config/runtime validation should
  reject unsupported head dims or grouped-query shapes before CUDA setup.
- normal native objects compile with checked device-content validation disabled
  (`QSFI_ENABLE_CHECKED_VALIDATION=0`). checked CUDA test objects enable it and
  cover negative metadata cases. do not add release-mode stream synchronizes for
  transactionality or validation unless there is a deliberate performance trade.
- validation before device addressing is part of the contract: Rust descriptors
  validate shapes/strides/modes, and checked native paths cover attention page
  ids and append positions, GDN seq/state metadata, embedding ids, and MoE
  routes/weights.
- model loading from config+safetensors, real qwen3.6 tensor mapping, tokenizer,
  CLI, GDN-in-runner, and MoE-in-runner are still future work.

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
- the first randomized dense BF16 path exists. keep it correct and narrow while
  adding real-model features.
- add model loading for config + safetensors, map fixed Qwen tensor names, upload
  weights to CUDA.
- keep `ModelRunner` as the boundary above `Engine`: it computes activations,
  supplies Q/K/V to the attention engine, stores logits, samples next tokens,
  and owns exact-prefix sync/rebuild.
- wire qwen3.6 GDN and MoE into `ModelRunner` only after their state ownership
  is engine/runner-owned enough to survive abort, rebuild, and release.
- after BF16 real-model correctness, add the first optimized NVFP4 path.
- first runnable CLI may accept and print token ids. tokenizer/chat template can
  follow immediately after the model produces tokens.
