this is quasar3, a minimal qwen3.6 runtime built from flashinfer, inspired by
dwarfstar4

rarely record durable facts here unless they took significant information
gathering

run tests with the gitignored test script

ground rules:
- no cmake
- no backward compatibility, ABI stability, or versioning work
- if replacing an api/build target, remove the old entrypoint and legacy calls
- keep the runtime qwen3.6-specific: prototype early, fail loudly, avoid generic
  model/runtime compatibility
- avoid release-mode stream synchronizes for transactionality or validation
  unless making a deliberate performance trade
- Rust owns model schedules, preparation, memory, and CUDA graph orchestration;
  C/C++ is kernel glue. Build-time Python is fine; inference must use AOT kernels
  without runtime Python or JIT

current architecture:
- `3pty` is vendored kernels/reference code; `target` is the rust build dir
- `.prototypes` for prototype work
- `EngineCore` owns request ids, tokens, sequence lengths, page allocator,
  page tables, last-page lengths, append positions, and staged batch state
  begin/commit/abort/reset/release paths should stay transactional: build
  candidate state and live views before installing them. allocator invariant
  checks stay debug-only
- `engine::attention::AttentionSession` owns stream/provider context state,
  per-layer paged K/V caches, device batch metadata, and FlashInfer plan caches.
  Plan keys include page ids and last-page lengths, not just CSR shape. Reprepare
  retains device workspace; pinned planning slots are reused only after their
  upload event completes
- `backend::qscb::Qscb` owns reusable cuBLASLt linear plans keyed by dimensions,
  strides, dtype, actual pointer alignment, and workspace capacity. Prefix
  rebuilds still replace the Engine and discard these execution resources
- `ModelRunner` is the boundary above `Engine`: it computes activations, supplies
  Q/K/V to attention, stores logits, samples, and owns exact-prefix sync/rebuild
- `QwenTokenizer` is a separate host-side asset boundary: it strictly loads the
  pinned NFC + ByteLevel BPE `tokenizer.json` and hands `i32` IDs to callers;
  `ModelRunner` does not own text formatting or tokenization. the tokenizer has
  248070 addressable IDs while the model vocabulary is padded to 248320, so
  padded logit slots must fail decode rather than be treated as tokenizer IDs
- validation before device addressing is contract: Rust descriptors validate
  shapes/strides/modes; checked native paths cover attention page ids and append
  positions, GDN metadata, embedding ids, and MoE routes/weights
- public randomized/vector BF16 runner paths cover the real hidden/GQA/GDN/MoE
  shape work for narrow test shapes. keep them correct while adding real-model
  loading

real qwen3.6-35b-a3b findings:
- `sp10@sp10:qs3` is the disposable GPU checkout used by the gitignored test
  and core benchmark scripts; both replace it, so run them sequentially.
  The pinned 35B BF16 and NVIDIA NVFP4 snapshots are cached on sp10. Model paths
  are resolved by `src/test_assets.rs`; use `QS3_QWEN36_MODEL_DIR` to override
- real text model prefix is `model.language_model.*`; ignore `model.visual.*`
  and `mtp.*` for the first text-only loader
- real BF16 shape differs from the randomized fixture: hidden size 2048, 40
  layers, 3 linear-attention layers then 1 full-attention layer repeated
  full attention uses GQA, head dim 256, q/k norms, and q/out dims that do not
  match the old no-GQA/head-dim-64 fixture assumptions
- BF16 experts use fused `mlp.experts.gate_up_proj` / `down_proj`; NVFP4 uses
  split per-expert tensors plus `input_scale`, `weight_scale`, `weight_scale_2`
  reject NVFP4 until its scale/packing semantics are implemented
- current full attention should keep explicit q/k norm + partial RoPE before
  `POS_ENCODING_NONE` attention as the correctness baseline. future fusion, if
  profiling justifies it, should be a qwen-specific prep kernel for packed Q
  extraction, output gate extraction, q/k norm, and `rotary_dim=64` RoPE. keep
  paged KV append separate first; only fuse K prep into append if launch/memory
  pass overhead proves worth coupling model prep to cache transaction details
- BF16 snapshot 995ad96eacd98c81ed38be0c5b274b04031597b0 generates
  greedy ids [5, 6, 24218, 10] for prompt ids [1, 2, 3, 4], matching the
  external BF16 reference at prefill and first decode within the existing
  logit tolerances

loader direction:
- `src/loader/transfer.rs` has the backend trait. keep qwen-specific manifest
  parsing/validation above it
- a loaded BF16 plan owns its backend until consuming materialization drains
  the backend's final allocation list into DeviceBuffers. backend Drop frees
  pre-handoff failures and still cleans pinned staging resources; do not forget
  the whole backend
- validate full config + safetensors indexes/headers before CUDA allocation:
  duplicate, missing, unexpected, wrong dtype/shape, overlapping, or out-of-range
  tensors must fail before device addressing
- first backend for GB10/UMA: `cudaMallocManaged` final weights, `preadv`
  directly into managed pointers, optional advise/prefetch, one load-end sync
  probes saw ~5 GiB/s direct managed read and no first-touch penalty
- keep pinned staging as the dGPU fallback: `cudaMalloc` final weights,
  `cudaHostAlloc` ring, `preadv`, `cudaMemcpyAsync`, events
- do not retain mmap or InstantTensor staging pointers as committed weights
  mmap can be a source view only. cuFile/GDS stays optional behind a hard probe

norm launch finding:
- FlashInfer's per-launch MaxDynamicSharedMemorySize setter races between host
  threads using different widths of the same kernel specialization. Keep the
  local AOT norm launches free of shared function-attribute mutation; see the
  concurrent-width regression and FINDINGS.md.

GDN direction:
- keep qwen3.6-specific GDN prep glue local for now: causal conv, post-conv
  Q/K/V split, decay/beta materialization, gated RMSNorm, and local recurrence
- if FlashInfer GDN is wired, keep it in one FlashInfer-owned TU; do not spread
  FlashInfer GDN headers/JIT plumbing through local `qscu` files

near-term todos:
- `TODO.md` is the completion checklist; `FINDINGS.md` records benchmark evidence
  and architecture direction. Current correctness covers 35B BF16; exact 27B,
  graphs, optimized NVFP4, and matched vLLM comparisons remain open
- rerun managed versus pinned weight-load tests on sp10, including the
  interrupted `4 x 1 GiB` pinned-ring run
