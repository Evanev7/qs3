# GB10 NVFP4 kernel survey — 2026-09-14

The proposed first implementation is **3.8-27B mixed NVFP4/FP8**, using native
FlashInfer CUTLASS for W4A4 and cuBLASLt for FP8. The 35B checkpoint needs a
separate W4A16 route; Marlin, b12x and AOT Triton were investigated rather than
assuming one NVFP4 kernel family fits both checkpoints. See the
[concrete integration sequence](INTEGRATION.md).

This is an experimental report, not a claim of integrated NVFP4 model support.
No runtime changes, CUDA graphs or commits were made. Only root ran GPU work;
three agents independently inspected upstream code and prepared probes.

## Environment and method

- qs3 `e2d46e799fb1ef300daa33a84cfc84b90557b2d6`, sp10 NVIDIA GB10 SM 12.1,
  driver 580.142. Source revisions are in [source-pins.json](source-pins.json).
- Pinned ARM64 image `vllm-node@sha256:d966c1831d5da55c0cc52c6bd40f7d02cfc3d83404c3bd599139b055232d3970`:
  Torch 2.11.0+cu130, FlashInfer 0.6.8.post1, vLLM 0.21.0, Triton 3.6.0.
  Native Triton artifacts use the host build environment's Triton 3.8.0.
- b12x uses an isolated CUTLASS DSL 4.6.2 overlay. The installed 4.4.2 compiler
  failed dense A16 and M=16 MoE on SM121; correct import path fixed those tests.
  Those probes retain their original image. Package metadata asks for Torch >=2.12;
  these particular probes passed on 2.11, which is the tested scope.
- GPU runs were serialized through `./remote.sh prototype ...`. A model download
  was active during part of the survey, so system-memory/storage traffic was
  not fully controlled. The GPU itself was available. Treat close rankings as
  candidates for a later isolated repeat, not a universal winner.
- Native probes time CUDA work with native C++ event/launch loops, no Python in
  the timed dispatch and no graph replay. They measure warm reuse and a
  128 MiB eviction pass before each operation. The eviction itself is excluded.
  Three trials are retained. Allocation, packing and compilation are excluded
  from steady-state kernel times; activation quantization is reported separately
  and as part of the native FlashInfer/FP8 pipeline.
- Python probes also use CUDA events, but host enqueue gaps contaminate short
  warm measurements. Their evicted measurements queue behind a flush, reducing
  that distortion. They remain a different benchmark protocol: FlashInfer and
  Marlin Python use 64 MiB flush; b12x uses 128 MiB. Do not merge their raw rankings
  with the native table without qualification.
- Native FlashInfer, Triton and Marlin use the same deterministic packed E2M1 /
  E4M3 fixture. W4A16 and W4A4 still compute different operations: A16 keeps BF16
  activations, while A4 consumes their quantized counterpart. BF16 controls use
  the reconstructed weights. FP32-output Triton and BF16-output providers have
  separate output-precision labels. Actual retained qs3 cuBLASLt plans, with
  the production 64 MiB workspace, are included as controls.

## Checkpoint semantics and real payload validation

| NVIDIA checkpoint | Exact snapshot | Requested low-bit computation |
| --- | --- | --- |
| Qwen3.6-35B-A3B-NVFP4 | `6c7f09d4036e97393f82e9f9ecd1a5c35ca5ee92` | 161 W4A16_NVFP4 + 130 FP8 entries in authoritative mixed map |
| Qwen3.6-27B-NVFP4 | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` | 193 W4A16_NVFP4 + 208 FP8 entries |
| Qwen3.8-27B-NVFP4 | `dbb8f445b3145f8a4c18ddc769f032d57d32867c` | 193 NVFP4 W4A4 + 208 FP8 entries |

Those are quantization-map entries, not counts of individual expert tensors.
The 35B snapshot is cached on sp10 and was read directly. Both 27B recipes
were verified from pinned official metadata; their payloads were not tested in
the initial survey. The user subsequently authorized downloading all three.
Exact download revisions, sizes and metadata hashes are in
[download-pins.json](download-pins.json). Exact configs, hashes, dispatch-source
analysis and upstream URLs are in [the checkpoint audit](research-notes/b12x.md).

For 3.6-27B NVFP4, vLLM 0.29 maps automatic KV selection to FP8, while
35B and 3.8 leave it automatic. Neither 27B index includes K/V scales, so
vLLM uses unit scales in that FP8 path. Explicit BF16 KV is the proposed
initial policy for comparing weight precision; record it in the oracle.

The cached 35B payload is split per expert, with U8 packed values, E4M3 block
scales per 16 weights and F32 global multipliers. Reconstruct with
`E2M1 × block_scale × weight_scale_2`; the global is not an inverse. FP4 plus
block scales takes 0.5625 bytes/weight, a 3.56× reduction versus BF16 before padding
and other metadata. W4A16 does not need an activation quantization launch.

Real-payload tests used expert 0/layer 0, expert 7/layer 20 and expert 255/layer 39,
all gate/up/down projections at M=1, 5, and 16: 27 cases per provider. **All 81 passed**
against their appropriate same-operand reference. Maximum relative L2 was
0.001738 for Marlin/b12x and 0.001751 for FlashInfer W4A4 (BF16 output rounding).
Inputs were synthetic BF16, not captured model activations. Switching to A4
caused roughly 9–10% relative L2 against the A16 reference for these inputs;
that is quantization distortion, not an implementation error or model-quality
measurement. The files retain both comparisons.

The audit checked **all 10,240 gate/up expert pairs**: both weight globals and
input globals match within each pair. Fusing these pairs is therefore compatible
with the pinned 35B scale values. This does not establish that other checkpoints
or independently fused Q/K/V projections share scales. Preserve and validate
that assumption instead of silently taking a maximum.

## Native dense results

Tables are populated from the saved CSV/JSON evidence; all times are µs. The
large-projection results establish a bandwidth benefit, not an end-to-end TPS
prediction. M=5 and 16 represent useful speculative-verification row counts.

| N × K | M | qs3 cuBLASLt BF16 | FI W4A4 incl. quant | Triton W4A4* | Triton W4A16 | Marlin W4A16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17408 × 5120 | 1 | 777.1 | 292.5 | 302.4 | 280.5 | 268.0 |
| 17408 × 5120 | 5 | 835.6 | 292.6 | 294.6 | 301.4 | 268.0 |
| 17408 × 5120 | 16 | 841.2 | 294.6 | 295.0 | 304.9 | 271.2 |
| 5120 × 17408 | 1 | 758.9 | 307.9 | 311.4 | 289.8 | 265.0 |
| 5120 × 17408 | 5 | 830.9 | 308.9 | 313.1 | 317.2 | 264.9 |
| 5120 × 17408 | 16 | 835.3 | 311.0 | 319.9 | 319.0 | 267.0 |
| 10240 × 5120 | 1 | 440.9 | 197.3 | 197.3 | 182.0 | 176.9 |
| 10240 × 5120 | 5 | 514.3 | 198.2 | 197.2 | 214.8 | 176.9 |
| 10240 × 5120 | 16 | 517.5 | 198.3 | 196.9 | 216.2 | 177.9 |
| 1024 × 2048 | 1 | 38.8 | 16.1 | 16.3 | 13.3 | 16.1 |
| 1024 × 2048 | 5 | 35.4 | 16.1 | 17.3 | 20.1 | 17.1 |
| 1024 × 2048 | 16 | 35.5 | 16.1 | 18.1 | 21.7 | 16.3 |

*The Triton W4A4 column excludes activation quantization.* Each column uses
its best measured native tactic; no dynamic runtime tuning is proposed.
CSV: [dense-summary.csv](dense-summary.csv).

For prefill, native FlashInfer also passes full-output checks at M=128 and 512.
Quantization+GEMM takes 309/419 µs for N=17408, K=5120 and 337/481 µs for
N=5120, K=17408. Those prefill controls use cuBLAS GEMM, not the retained qs3
cuBLASLt baseline. Tactic choice matters here: the slower StreamK variant
reaches 2 ms at M=512, while the best data-parallel variant is about 400 µs.
Marlin/b12x low-M results do not qualify their complete prefill path.

The actual N=248320, K=5120 vocabulary-sized matrix also passes all 17 Triton
variants at M=1, 5, and 16. Best evicted packed-weight times are 3.23/3.43/3.51 ms;
cuBLASLt BF16 controls are 14.93/10.98/11.06 ms. **These are not comparisons
against qs3's already optimized BF16 Triton LM-head kernel.** The first optional
large-matrix probe exposed a prototype grid.y overflow; mapping output rows
onto grid.x fixed it and the complete rerun passed. This is another reason to
exercise the padded vocabulary dimensions explicitly.

Native FP8 quantization+cuBLASLt also passes same-operand checks. For
N=10240, K=5120, times at M=1, 4, and 16 are 372/333/340 µs; for N=5120,
K=6144, they are 199/200/201 µs.
The corresponding BF16 GEMM controls are 462/525/514 µs and 261/322/325 µs.
This establishes a viable native FP8 building block, not finished checkpoint
FP8 activation calibration or fused-projection semantics.

## b12x and routed MoE

b12x dense A16 and A4 pass at M=1, 2, 4, 5, and 16 with CUTLASS DSL 4.6.2.
The default A16 split-K=4 path is slower than Marlin on these dense shapes.
A 96-case A16 sweep varies N tiles of 64/128, K tiles of 64/128 and split-K
values of 1/2/4/8 for M=1, 5, and 16.
The best evicted A16 configurations reach 279/288/296 µs at M=1, 5, and 16 for
N=17408, K=5120 and 295/300/304 µs for N=5120, K=17408. A K tile of 128 is
substantially better here than the default of 64. Marlin remains faster on these dense shapes,
but tuning closes most of the apparent b12x deficit. These sweep timings use
Python dispatch and a separate fixture, unlike the native default-export check.

More significantly for integration feasibility, **the actual b12x CuTe exports
linked and ran natively**. Generated headers/ARM objects for dense GEMM and
split-K reduction retain the TMA host setup. They require CUTLASS's
`libcuda_dialect_runtime_static.a` plus CUDA; no b12x/CuTe import or JIT occurs
in that probe's execution path. Tests at M=1, 5, and 16 with global scales of
1 and 0.137 all pass.
Default-export evicted times are 430/434/446–449 µs, confirming its slower dense
result is not simply Python overhead. Cross-compiling the host objects from
our normal x86 build environment remains a separate build qualification.

The b12x routed MoE test covers E=256, H=2048, I=512, top-k=8 and
M=1, 2, 4, 5, and 16. Both A16 and A4 pass before and after timing. A16 relative
L2 ranges from 0.0040 to 0.0065; A4 reaches 0.0073 on the M=16 dynamic path.
These use b12x's own reference
composition; they are not independent model references. Timed work includes
route preparation, both expert projections, SiLU and weighted reduction;
router projection/top-k and the shared expert are excluded.

A follow-up also loads all 256 experts from the real cached layer 0, preserving
every per-expert global and hashing all source tensors. All six A16 cases
(M=1, 5, and 16, diverse/shared routes) pass before and after timing, with relative
L2 up to 0.00651 against the complete composition reference. This checks the
fused real-weight operation in addition to the 81 individual projection cases.
The b12x bank is packed `[up, gate]`; our existing qsfi BF16 bank is
`[gate, up]`. This ordering belongs in provider preparation, with a composition
check that applies SiLU to the gate half.

The A16 routing follow-up is particularly relevant to MTP:

| Routes / M | Packed route µs | Direct route µs | Direct calls |
| --- | ---: | ---: | ---: |
| Diverse routes / 1 | 125.9 | 107.4 | 1 |
| Diverse routes / 5 | 392.7 | 375.4 | 1 |
| Diverse routes / 16 | 1078.1 | 1075.1 | 2 |
| Same eight experts / 1 | 126.9 | 105.5 | 1 |
| Same eight experts / 5 | 129.4 | 180.1 | 1 |
| Same eight experts / 16 | 133.1 | 391.1 | 2 |

All 12 cases pass full-output checks. Native direct routing is capped at M=8;
M=16 direct above means two sequential M=8 calls with both timed. The extremes
show that small row count alone does not determine the winner: expert reuse
can pay for grouping at M=5 and 16, while diverse routes retain the simple path's
advantage at M=1 and 5 and are effectively tied at M=16. These measurements use
Python dispatch and 128 MiB eviction; do not interpret the 3 µs difference
between methods for diverse routes at M=16 as a stable performance win.
Revisit with captured real routes and native
launches before introducing a routing heuristic.

## Other candidates and remaining limits

- FlashInfer's pinned Python image lacks its newer `prepare_bf16_fp4_weights`
  API; that failed capability probe is not evidence against current upstream
  W4A16 kernels. Dense W4A4 CUTLASS and cuDNN both passed in the installed API.
  Native CUTLASS already gives a clean qs3 integration boundary, so adopting
  cuDNN solely for a close dense result is not justified yet.
- The installed 0.21 vLLM dispatches the cached 35B mixed W4A16 layer as
  unquantized. The probe stops before allocation. The refreshed 0.29 image
  resolves this dispatch issue; its separate qualification is recorded below.
- Current Triton 3.8 emits SM121 native FP4 MMA. Its aligned/pipelined
  tensor-core variants and direct packed GEMV work. An older claim that Triton
  cannot express this hardware operation would be wrong for this toolchain.
- CUTLASS example 79a and Colfax SM12x CuTe work are useful implementation
  references. Example 79b's FP4-output operation is a different contract.
  GemLite's current NVFP4 GEMV caveats and heuristics were inspected; no GemLite
  performance result is claimed. Details and primary source links are in
  [the alternatives survey](research-notes/triton_alternatives.md) and
  [FlashInfer/vLLM notes](research-notes/flashinfer_vllm.md).
- Quality evaluation, matched quantized vLLM benchmarks, load-time repacking
  benchmarks and end-to-end TPS measurements remain integration gates. The
  whole-model smoke checks below do not close those gates.

## Refreshed vLLM qualification

The separate reference refresh pins official vLLM 0.29.0, Torch 2.13.0+cu130,
FlashInfer 0.6.18 and CUTLASS DSL 4.6.2. Its immutable image and source revision
are in [the reference pins](../2026-09-14-vllm-references/pins.json). Historical
kernel timing results above retain their original environments.

Both 27B configs pass the actual quantization-method constructor checks:
3.8 selects `FlashInferCutlassNvFp4LinearKernel` for W4A4, while 3.6 selects
`MarlinNvFp4LinearKernel` for W4A16. Both retain FP8 methods for FP8 projections.
These preflight checks stop before checkpoint weight allocation.

The pinned 35B NVFP4 checkpoint also loads and generates in eager vLLM with
explicit BF16 KV and FP32 GDN state. Its 40 expert layers resolve to
`ModelOptNvFp4FusedMoE`, `MarlinExperts`, `MARLIN`, and `use_a16=True`.
For prompt `[1,2,3,4]`, it produces `[5,6,24218,10]`, matching the four-token
BF16 smoke continuation. This establishes a usable load/generation path, not
full logit agreement or model quality. First-use compilation occurs in the
prototype; its load/generation durations are not performance baselines.

The pinned 3.8-27B NVFP4 checkpoint also completes eager load and generation,
with 129 fused/dense W4A4 linear modules using FlashInfer CUTLASS and 128 FP8
linear modules. Its small-prompt output is `[5,0,31,0]`; BF16 gives
`[5,0,31,46474]`. At the shared fourth-prediction prefix, BF16 ranks 46474 over
0 by 0.125 logits; NVFP4 reverses those two with a 0.1875 log-probability margin.
That observed flip is not a model-quality verdict or evidence of a kernel bug.
Matched raw-logit checks against the quantized oracle remain necessary.

At 20:37:49 UTC, the download queue had verified complete pinned 35B and 3.8
NVFP4 snapshots; 3.6 NVFP4 was still downloading. It runs independently in
`qs3-nvfp4-downloads-20260914t194122z` and uses the default Hugging Face cache,
without `--local-dir`. [Download status](download-status.json) records the
container, completed checks and partial-file progress. Inspect the current job
with `ssh -F /dev/null sp10@sp10 docker logs --tail 20 qs3-nvfp4-downloads-20260914t194122z`.

## Reproduction and evidence

Restore [probe-sources.tar.gz](probe-sources.tar.gz) into the repository root.
Use the vendored revisions in source-pins.json; b12x's source bundle is generated
from its pinned checkout for remote runs. The archive contains the probe
sources and named `.prototypes/justfile` recipes. Run recipes sequentially with
`./remote.sh prototype NAME`. The gitignored remote helper additionally accepts
`QS3_PROTOTYPE_OUTPUT_PREFIX=nvfp4-` to retrieve only this survey's outputs;
its default retrieval behavior is unchanged.

Main recipes: `nvfp4-initial`, `nvfp4-broad`, `nvfp4-followup`,
`nvfp4-qualification`, `nvfp4-tuning`, `nvfp4-completion`. Raw small evidence is
under [evidence/](evidence/), indexed with hashes in
[evidence-manifest.json](evidence-manifest.json). Complete compile logs, generated
binaries and caches remain in `.prototypes/out/nvfp4-*` locally and on sp10.
Failures are retained alongside successful reruns; a wrapper exit 0 only means
it collected artifacts, so inspect per-case status/correctness fields too.
