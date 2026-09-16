this is quasar3, a minimal qwen3.6/8 runtime inspired by dwarfstar4

rarely record durable facts here unless they took significant information
gathering

run tests with the gitignored test script

ground rules:
- no cmake
- no backward compatibility, abi stability, or versioning work
- if replacing an api/build target, remove the old entrypoint and legacy calls
- keep the runtime qwen3.6-specific: prototype early, fail loudly, avoid generic
  model/runtime compatibility
- avoid release-mode stream synchronizes for transactionality or validation
  unless making a deliberate performance trade
- rust owns model schedules, preparation, memory, and cuda graph orchestration;
  c/c++ is kernel glue. inference must use aot kernels without runtime python or
  jit

no failing tests:
- do not call implementation work complete or verified until the full required
  test workflow passes; passing an earlier stage is not enough
- all failures block completion, including pre-existing failures. fix them;
  do not dismiss them as baseline failures or unrelated to the current change
- do not delete, skip, or weaken tests merely to get a passing run. preserve
  meaningful coverage when updating fixtures or model-specific test selection
- if a required run is blocked, explicitly report the work as unverified and
  state the blocker; a blocked or partial run is not a pass

current architecture:
- `3pty` is vendored kernels/reference code; `target` is the rust build dir
- `.prototypes` for prototype work
- `EngineCore` owns request ids, tokens, sequence lengths, page allocator,
  page tables, last-page lengths, append positions, and staged batch state
  begin/commit/abort/reset/release paths should stay transactional: build
  candidate state and live views before installing them. allocator invariant
  checks stay debug-only
- `engine::attention::AttentionSession` owns stream/provider context state,
  per-layer paged k/v caches, device batch metadata, and flashinfer plan caches.
  plan keys include page ids and last-page lengths, not just csr shape. reprepare
  retains device workspace; pinned planning slots are reused only after their
  upload event completes
- `backend::qscb::Qscb` owns reusable cublaslt linear plans keyed by dimensions,
  strides, dtype, actual pointer alignment, and workspace capacity. prefix
  rebuilds replace only `PrefixState` (core and kv caches) and gdn state; provider
  contexts, linear/attention plans, workspaces and batch metadata buffers survive
- `ModelRunner` is the boundary above `Engine`: it computes activations, supplies
  q/k/v to attention, stores logits, samples, and owns exact-prefix sync/rebuild
- `QwenTokenizer` is a separate host-side asset boundary: it strictly loads the
  nfc + bytelevel bpe `tokenizer.json`, validating supported semantics and
  vocabulary structure, and hands `i32` ids to callers. `ModelRunner` receives
  the loaded tokenizer's addressable token count; it does not own text formatting
  or tokenization. pinned 3.6 tokenizers have 248070 ids and 3.8 has 248077,
  while model logits have 248320 slots. padded ids must fail sampling and decode
- validation before device addressing is contract: rust descriptors validate
  shapes/strides/modes; checked native paths cover attention page ids and append
  positions, gdn metadata, embedding ids, and moe routes/weights
- public randomized/vector bf16 runner paths cover the real hidden/gqa/gdn/moe
  shape work for narrow test shapes. keep them correct while adding real-model
  loading

real qwen3.6-35b-a3b findings:
- `sp10@sp10:qs3` is the disposable gpu checkout used by the gitignored test
  and core benchmark scripts; both replace it, so run them sequentially.
  the pinned 35b bf16 and nvidia nvfp4 snapshots are cached on sp10. model paths
  are resolved by `src/test_assets.rs`; use `QS3_QWEN36_MODEL_DIR` to override
- real text model prefix is `model.language_model.*`; ignore `model.visual.*`
  and `mtp.*` for the first text-only loader
- real bf16 shape differs from the randomized fixture: hidden size 2048, 40
  layers, 3 linear-attention layers then 1 full-attention layer repeated
  full attention uses gqa, head dim 256, q/k norms, and q/out dims that do not
  match the old no-gqa/head-dim-64 fixture assumptions
- bf16 experts use fused `mlp.experts.gate_up_proj` / `down_proj`; nvfp4 uses
  split per-expert tensors plus `input_scale`, `weight_scale`, `weight_scale_2`
  reject nvfp4 until its scale/packing semantics are implemented
- current full attention should keep explicit q/k norm + partial rope before
  `POS_ENCODING_NONE` attention as the correctness baseline. future fusion, if
  profiling justifies it, should be a qwen-specific prep kernel for packed q
  extraction, output gate extraction, q/k norm, and `rotary_dim=64` rope. keep
  paged kv append separate first; only fuse k prep into append if launch/memory
  pass overhead proves worth coupling model prep to cache transaction details
- bf16 snapshot 995ad96eacd98c81ed38be0c5b274b04031597b0 generates
  greedy ids [5, 6, 24218, 10] for prompt ids [1, 2, 3, 4], matching the
  external bf16 reference at prefill and first decode within the existing
  logit tolerances

loader direction:
- `src/loader/transfer.rs` has the backend trait. keep qwen-specific manifest
  parsing/validation above it
- a loaded bf16 plan owns its backend until consuming materialization drains
  the backend's final allocation list into devicebuffers. backend drop frees
  pre-handoff failures and still cleans pinned staging resources; do not forget
  the whole backend
- validate full config + safetensors indexes/headers before cuda allocation:
  duplicate, missing, unexpected, wrong dtype/shape, overlapping, or out-of-range
  tensors must fail before device addressing
- first backend for gb10/uma: `cudaMallocManaged` final weights, `preadv`
  directly into managed pointers, optional advise/prefetch, one load-end sync
  weight-load comparisons must include backend/ring setup and record file-cache
  residency. full cold sp10 results are in findings.md; cached small probes do
  not predict full-snapshot load time or settle first gpu use
- keep pinned staging as the dgpu fallback: `cudaMalloc` final weights,
  `cudaHostAlloc` ring, `preadv`, `cudaMemcpyAsync`, events
- do not retain mmap or instanttensor staging pointers as committed weights
  mmap can be a source view only. cufile/gds stays optional behind a hard probe

norm launch finding:
- flashinfer's per-launch maxdynamicsharedmemorysize setter races between host
  threads using different widths of the same kernel specialization. keep the
  local aot norm launches free of shared function-attribute mutation; see the
  concurrent-width regression and findings.md.

gdn direction:
- keep qwen3.6-specific gdn prep glue local for now: causal conv, post-conv
  q/k/v split, decay/beta materialization, gated rmsnorm, and local recurrence
- if flashinfer gdn is wired, keep it in one flashinfer-owned tu; do not spread
  flashinfer gdn headers/jit plumbing through local `qscu` files

near-term todos:
- `TODO.md` is the completion checklist; `FINDINGS.md` records benchmark evidence
  and architecture direction. current correctness covers 35b bf16; exact 27b,
  graphs, optimized nvfp4, and matched vllm comparisons remain open
