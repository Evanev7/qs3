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

current architecture:
- `3pty` is vendored kernels/reference code; `target` is the rust build dir
- `.prototypes` for prototype work
- `EngineCore` owns request ids, tokens, sequence lengths, page allocator,
  page tables, last-page lengths, append positions, and staged batch state
  begin/commit/abort/reset/release paths should stay transactional: build
  candidate state and live views before installing them. allocator invariant
  checks stay debug-only
- `runtime::EngineInner` owns CUDA stream/context state, per-layer paged K/V
  caches, device batch metadata, and FlashInfer prefill/decode plan caches
  plan-cache keys include page ids and last-page lengths, not just CSR shape
- `ModelRunner` is the boundary above `Engine`: it computes activations, supplies
  Q/K/V to attention, stores logits, samples, and owns exact-prefix sync/rebuild
- validation before device addressing is contract: Rust descriptors validate
  shapes/strides/modes; checked native paths cover attention page ids and append
  positions, GDN metadata, embedding ids, and MoE routes/weights
- public randomized BF16 runner works end-to-end for current narrow shapes. keep
  it correct while adding real-model support

real qwen3.6-35b-a3b findings:
- materialized BF16 and NVFP4 safetensors exist on `spark-1565`
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

loader direction:
- `src/weight_loader.rs` has the backend trait. keep qwen-specific manifest
  parsing/validation above it
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

GDN direction:
- keep qwen3.6-specific GDN prep glue local for now: causal conv, post-conv
  Q/K/V split, decay/beta materialization, gated RMSNorm, and local recurrence
- if FlashInfer GDN is wired, keep it in one FlashInfer-owned TU; do not spread
  FlashInfer GDN headers/JIT plumbing through local `qscu` files

near-term todos:
- adjust `QwenConfig`, `QwenWeights`, attention runtime, and runner for the real
  BF16 model shapes: hidden 2048, GQA, head dim 256, q/k norms, GDN packed/output
  dims, and fused MoE expert layout
- implement BF16 config+safetensors loading through the qwen-specific manifest
  validator and `WeightLoadBackend`
- then add the first CLI that accepts/prints token ids. tokenizer/chat template
  can follow after model token generation works
- after BF16 real-model correctness, add the first optimized NVFP4 path
