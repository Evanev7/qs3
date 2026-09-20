this is quasar3, a minimal qwen3.6/8 runtime inspired by dwarfstar4

rarely record durable facts here unless they took significant information
gathering

run tests with the gitignored remote.sh script. it holds a flock so is safely
reentrant.
`sp10@sp10:qs3` is a disposable gpu checkout; tests, benchmarks, and prototypes
replace it.

ground rules:
- no cmake
- no backward compatibility, abi stability, or versioning work
- if replacing an api/build target, remove the old entrypoint and legacy calls
- keep the runtime qwen3.6/8-specific: prototype early, fail loudly, avoid generic
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
- `build_tools` for qstriton/qscute compilers and nix based codegen
- `config.nix` contains all new backend kernel selection shared constants
  - it never imports nixpkgs, its strictly pure evaluation.
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
- `src/loader/transfer.rs` has the backend trait. keep qwen-specific manifest
  parsing/validation above it
- the loader enqueues transfers, scale swizzling and scale-parameter uploads on
  the caller's CudaCtx before backend.finish waits and hands off allocation
  owners. materialization attaches final storage to QwenWeights; the runner owns
  activation scratch and plans. retire canonical scales only after completion;
  backend drop still cleans pre-handoff failures and pinned staging resources
- validate full config + safetensors indexes/headers before cuda allocation:
  duplicate, missing, unexpected, wrong dtype/shape, overlapping, or out-of-range
  tensors must fail before device addressing
- do not retain mmap or instanttensor staging pointers as committed weights
  mmap can be a source view only

near-term todos:
- `TODO.md` is the completion checklist; `FINDINGS.md` records benchmark evidence
  and architecture direction.
