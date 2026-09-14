# Proposed NVFP4 integration

Survey only; no runtime implementation or commits. Decisions below distinguish
checkpoint precision from a kernel's ability to consume packed FP4 weights.

## Order and acceptance gates

1. **Make the mixed checkpoint contract explicit.** Extend the Qwen manifest in
   `src/loader/format.rs` to accept the exact ModelOpt mixed-precision metadata,
   including per-layer W4A16_NVFP4, NVFP4 W4A4, FP8 and BF16 assignments. Validate
   all indexes, tensor headers, packed dimensions and scale shapes before CUDA
   allocation. Preserve the existing fail-before-device-addressing contract.
   The pinned 35B and 3.6-27B use W4A16, while 3.8-27B uses W4A4.
   In vLLM 0.29, the 3.6-27B metadata makes automatic KV selection resolve to
   FP8, but its index supplies no K/V scales (vLLM falls back to unit scales).
   Use explicit BF16 KV for the initial comparison of weight precision; treat
   checkpoint-default FP8 KV as a separate policy and quality check. Pinned 35B
   and 3.8 NVFP4 payloads are now cached and pass eager vLLM load/generation;
   3.6 NVFP4 is downloading. All three BF16 logit references are complete.
   A model name ending in NVFP4 is insufficient to select computation.

2. **Represent quantized projections without expanding the model to BF16.**
   Add a Qwen projection representation in `src/model/weights.rs` with BF16,
   FP8 and NVFP4 cases. NVFP4 owns packed U8 payload, E4M3 block scales and F32
   global multipliers plus its required activation precision. Keep the canonical
   checkpoint interpretation separate from prepared kernel layouts. In
   `src/loader/plan.rs` and `materialize.rs`, reuse the existing transfer backend
   and ownership handoff; add scale swizzling or Marlin repacking during load.
   Do not create a second loader, generic provider registry or inference-time
   conversion of the full weights.

3. **Land the two dense compute operations needed by 3.8-27B.** Use the tested
   native FlashInfer CUTLASS W4A4 operation and native activation quantizer in
   one FlashInfer-owned translation unit. Add FP8 activation quantization and
   cuBLASLt support through `qscb.cu` / `src/backend/qscb.rs`. Plan keys must
   include input/weight/output dtype and scale mode; refresh device scale
   bindings when executing a retained plan. Rust owns output, activation-scale,
   packed-activation and CUTLASS workspaces. Prepare tactics outside execution.
   Keep BF16 activation intermediates and preserve established normalization,
   attention, GDN and residual precision unless checkpoint semantics require
   otherwise.
   FP8 projection weights do not imply FP8 recurrence or KV caches.

4. **Wire dense 27B through the existing runner.** Dispatch projections in
   `src/model/runner/mlp.rs`, `attention.rs`, `gdn.rs` and the LM-head path using
   their prepared weight representation. Preserve schedules and cache state.
   Begin with separate gate/up operations. Fusing projections with different
   global scales requires an explicit policy and numerical reference; a max
   reduction of scales alone does not preserve the packed weights' values.
   Keep padded vocabulary behavior and validate raw logits as well as tokens.

5. **Add checkpoint-faithful W4A16 for 35B dense/shared/LM-head operations.**
   Use Marlin as the initial dense W4A16 candidate: its native decode results
   lead the measured dense shapes. Qualify its prefill and large-vocabulary
   paths before calling it the complete provider. Retain b12x and AOT Triton
   as measured alternatives rather than adding all three to production. Do not
   let a backend default based on M silently
   switch A16 to A4 at M=16. Use a single qualified W4A16 provider first;
   additional kernels should earn their maintenance cost with a measured gain.
   Marlin requires load-time weight/scale layout preparation. b12x's generated
   host launch code already linked and ran on ARM; adopting it still requires
   a reproducible build-time CuTe export in our build pipeline. The Triton path
   already fits qs3's build-time cubin/CUDA-driver boundary.

6. **Then add the routed 35B NVFP4 MoE operation.** b12x is the first candidate
   for this slice, where its complete routed operation is useful. Dense Marlin
   wins do not establish a routed-MoE winner. Refreshed vLLM also runs the
   complete 35B model with Marlin A16 MoE; compare its routed operation before
   committing to b12x as the production provider. Rust retains router/top-k,
   route validation, scratch ownership and schedule selection. Pass explicit
   route IDs/weights into a prepared expert operation. Qualify both the direct
   low-M path and the packed M=16 path, including gate/up, SiLU, down projection
   and routed reduction. Shared expert and router work must be included in the
   eventual layer benchmark. Prepare b12x w13 as `[up, gate]`: the current
   qsfi BF16 operation expects `[gate, up]`, so copying its fused ordering
   directly would change SiLU semantics. Validate each gate/up pair's weight
   globals before fusion. Start with checkpoint A16 semantics; A4 MoE is a
   separate quality/performance experiment. Avoid duplicating the old vLLM
   adapter's lossy global-scale folding merely to fit an upstream signature.

## Validation before calling a model supported

- Negative manifest fixtures: missing/wrong/overlapping payloads, wrong FP8 or
  scale dtypes, malformed block dimensions, unsupported quantization recipes;
  no CUDA allocations before header validation completes. Check global scales
  for finite positive values; block scales may be zero for zero-valued blocks.
  Reject invalid scale encodings before kernels consume them.
- Kernel references reconstruct packed E2M1 × E4M3 × F32 weights explicitly.
  For W4A4, compare against the same quantized activations as a distinct check
  from error against original BF16 activations. Exercise non-unit globals,
  tails/padding, M=1, 2, 4, 5, and 16 and representative prefill shapes.
- Real payload checks at first/middle/last layers, gate/up/down, FP8 projections
  and LM head. The survey's real-payload tests use synthetic inputs, so they do
  not establish model quality or activation calibration coverage.
- Capture reference logits from a vLLM version that actually accepts each
  pinned recipe. Record backend, activation precision, fusion/scale treatment,
  recurrent and KV precision, and exact revisions. Compare identical prefixes at
  prefill, first decode and sustained positions; do not infer correctness from
  plausible generated text.
- Run the existing BF16 suites for 35B, 3.6-27B and 3.8-27B, then quantized model
  tests. Keep reference availability visible rather than silently skipping a
  model. Kernel-only probes do not close the remaining exact-model gates.
- Benchmark full load and first use, then eager 102/32 and 1024/256 workloads.
  Weight bandwidth, GPU time and CPU launch gaps must be reported separately;
  matrix speedups do not predict an end-to-end tokens/sec number.

## Relationship to MTP and build scope

Support M=1 through 16 from the first dense API, with explicit activation
precision and Rust-owned persistent scratch. These are useful verification
batch sizes without coupling the operator to a particular draft model. Draft
model selection and speculative state commit/rollback remain above the kernel
layer. No CUDA graph or scheduler changes are necessary for this integration.

Build with the existing Ninja/Nix/AOT pipeline. FlashInfer/CUTLASS templates,
Marlin CUDA templates and exported CuTe/Triton artifacts are candidate build
inputs; Python/Torch used in this survey are not proposed runtime dependencies.
Each numbered item is a reviewable implementation slice, with loader/weight
representation changes made concrete alongside their first consumer. Do not
land unused generic infrastructure ahead of that consumer.
