# TODO

## Scope

- Support Qwen3.6-35B-A3B and Qwen3.6-27B, initially text-only.
- Competitive single-user decode latency and tokens/sec against vLLM on Spark
  is a completion criterion. Compare matched precision, context lengths, and
  generation settings; record the exact models, kernels, and software revisions.
- Keep Rust as the primary implementation: model schedules, memory ownership,
  preparation, provider selection, and CUDA graph orchestration. C/C++ is kernel
  glue. Python may generate artifacts at build time; no runtime Python or
  Python-driven JIT compilation.
- Use explicit model paths and kernel choices. Share useful storage and launch
  helpers; defer generic provider frameworks, contract crates, operation
  catalogues, graph solvers, and e-graphs.
- Keep kernel choices visible and benchmarkable. Add fused paths where measured
  launch or memory traffic costs justify them.
- Keep the existing no-CMake build direction. Static linking is secondary to
  execution performance and a clear Rust implementation.

## 1. Set up the new Spark and establish a baseline

- [x] Set up a checkout on `sp10@sp10`; it does not currently have the repo.
  Initialize the submodules required for building and validation without
  unnecessarily copying or traversing all of `3pty`.
- [x] Verify CUDA, Rust, Ninja, and the Python vector-generation dependencies.
- [x] Adapt the gitignored `run_cuda_test.sh` to the new host and checkout.
  Its existing reset/clean/sync workflow assumes a disposable remote checkout.
- [x] Provision model snapshots and retarget hardcoded `/home/exo` asset paths
  in tests and benchmarks. Keep the known 35B BF16 snapshot
  `995ad96eacd98c81ed38be0c5b274b04031597b0`; pin a 27B snapshot separately.
- [x] Fix the vector-generation Ninja dependencies to include generator sources,
  so edits cannot silently reuse stale fixtures and the `.oracle-ok` stamp.
- [x] Run the prescribed test script and the real 35B BF16 regression on the
  new Spark.
- [x] Diagnose and fix the intermittent FlashInfer norm launch error. A focused
  concurrent-width regression reproduced `norm.cuh:679`; serializing only the
  launches passed. The host wrapper was racing on the CUDA function's dynamic
  shared-memory limit. Launch the same AOT kernels without changing shared
  function attributes. The full suite, 20 parallel model-test repetitions, and
  real 35B BF16 reference regression passed on sp10 (2026-09-09). See FINDINGS.md.
- [ ] Record 35B BF16 single-request latency/TPS and a GPU timeline against a
  matched vLLM run. Include several context lengths and sustained decode;
  separate setup, prefill, steady decode, and output delivery. Initial short-context
  qs3 measurements and a 32-step decode timeline are recorded in FINDINGS.md;
  102- and 1024-token vLLM comparisons are recorded, the latter with 256 decode
  samples. FP32 recurrence is now available and the loaded-model default; it
  extends the longer matching prefix from 56 to 162 generated IDs with little
  throughput change. Identical-prefix score comparison, sustained quality, a
  third context remain open. Aligned qs3/vLLM 32-forward GPU timelines are now
  recorded, including graph gaps, kernel choices and LM-head precision differences.

## 2. Remove repeated preparation from decode

- [x] First implementation change: retain cuBLASLt linear descriptors and chosen
  algorithms for reuse. Rust now owns prepared plans keyed by dimensions, strides,
  output dtype, actual pointer alignment, and workspace capacity. The prescribed
  full suite and real 35B BF16 reference regression passed on sp10 (2026-09-09).
  The native one-shot entrypoint was removed; benchmarks prepare outside timing.
- [x] Reuse attention planning workspaces instead of destroying and reallocating
  device and pinned-host storage on every metadata change. Introduce a correct
  update/replan path; do not obtain reuse by weakening the existing metadata key.
  Native prepare now retains device workspace and uses event-protected pinned
  staging slots. The full suite, queued native replan regression, and real 35B
  BF16 reference test passed on sp10. The full metadata key is unchanged.
- [x] Separate execution-context/provider-handle lifetime from attention and
  prefix state so rebuilds preserve reusable execution resources. PrefixState
  now owns replaceable core/KV state; AttentionSession keeps providers, plans and
  metadata. Late-failure rollback and real BF16/FP32 continuation regressions pass.
- [ ] Simplify bindings as these paths are changed: typed Rust tensor views into
  one lowering/launch method, removing redundant descriptor wrappers and
  forwarding methods. Preserve validation before device addressing.
- [ ] Re-measure CPU preparation, GPU idle gaps, and end-to-end decode after
  each substantive change.

## 3. Add the exact 27B path

- [ ] Pin and validate the 27B config, tokenizer, and safetensors manifest.
  Revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` and all shard headers are
  recorded under `model_manifests/`. An independent host check validates 851
  text tensor shapes; tokenizer bytes match the pinned 35B asset. All 15 full
  shards are now cached on sp10 and match their Hub SHA256 hashes and pinned
  headers. Production Rust dense-manifest validation remains open.
- [x] Probe native full attention with 24 Q heads, 4 KV heads, head dimension
  256, and GQA ratio 6. The isolated AOT probe passes CPU-reference prefill/decode
  and append cases on sp10; see FINDINGS.md. Production dispatch and Rust shape
  validation still need to adopt the tested 6/8 dispatch.
- [ ] Validate GDN with 16 key heads and 48 value heads, head dimensions 128,
  and convolution width 4. Audit fixed 35B constants in native kernels, Rust
  views, scratch allocation, and recurrent state.
- [ ] Add the 64-layer, hidden-size-5120 dense model path with MLP intermediate
  size 17408. Preserve the three-GDN/one-full-attention schedule and explicit
  support for the 35B shape; reject unsupported model configurations.
- [ ] Implement dense weight materialization and validate public-runner prefill,
  decode, prefix extension, and rebuilds against external reference results.
  Check recurrent-state precision and sustained decode, not just a short prompt.

## 4. Capture steady decode

- [ ] Make decode buffers, workspaces, and device metadata addresses persistent.
  Complete provider initialization and preparation before capture.
- [ ] Handle changing sequence/page metadata and GDN slot alternation explicitly
  across graph replays; test page boundaries and prefix resets/rebuilds.
- [ ] Separate enqueueing from output delivery. Keep the sampled token available
  on-device for the next step and make host completion points explicit.
- [ ] Compare prepared eager execution with graph replay for both models.

## 5. Tune the measured GPU work

- [ ] Report selected kernel implementations, precision, graph mode, and workspace
  sizes in benchmarks. Keep alternatives named and forceable in Rust. Current
  core JSON records provider paths, workspace sizes, GDN state dtype, and output
  IDs. The BF16 MoE grid has explicit four- and 96-block Rust selections.
- [x] Profile attention preparation, dense projections, GDN, and MoE before
  choosing fusion, projection packing, or replacement provider kernels. Recorded
  qs3 prefill/decode and aligned vLLM decode traces identify serial convolution,
  recurrence reductions, MoE tiling, graph gaps and projection precision/packing;
  convolution and recurrence prefill improvements pass bitwise probes and core runs.
- [ ] Implement optimized quantized paths, beginning with NVFP4 after BF16
  correctness. Validate packing/scales and actual SM121 kernel support; compare
  matched quantization with vLLM.
- [ ] Revisit managed versus pinned weight-load performance on the new host,
  including the previously interrupted four-buffer, 1 GiB-per-buffer run.

## Model configuration references

- [Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- [Qwen3.6-27B config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)

These links describe the model shapes; validation runs must use pinned snapshots.
