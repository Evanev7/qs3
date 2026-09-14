# FlashInfer / vLLM NVFP4 survey

2026-09-14. Source audit plus local native compile/link experiments. Root owns all
sp10 execution; performance results are in its separate output artifacts.

Pins inspected: FlashInfer `b3baedbbef2686df91b6dc43818ee56fe26ceba2`, vLLM
`51a99565c398c8320de8131e07731c75c52eb87c`. Online documentation is newer in places;
do not mistake current docs for features in the pinned container.

## Providers relevant to GB10

| Provider | SM121 route | Preparation and integration |
|---|---|---|
| FlashInfer CUTLASS W4A4 | Explicit supported; native C++ `Sm120` operator on SM121 hardware | Packed row-major A `[M,K/2]`, packed row-major W `[N,K/2]`, 128x4 swizzled E4M3 block scales, scalar FP32 alpha. Compile selected templates, caller-owned workspace; no PyTorch/TVM needed. |
| FlashInfer cuDNN W4A4 | Explicit supported, often auto choice on CUDA13/cuDNN>=9.15 | Same packed+128x4 scales. C++ cuDNN frontend/backend integration possible, but introduces dependency/plan ownership and setup work. cuDNN operation graphs are kernel plans, distinct from CUDA graph capture. |
| FlashInfer b12x W4A4 | Explicit supported, CUDA13+, K multiple32 | Same canonical packed+128x4 scales; CuTe DSL compilation/caching currently Python-facing. Build-time CuTe export needed for Rust runtime. Dense auto deliberately excludes GB10 because CUTLASS/cuDNN usually faster there. |
| FlashInfer TRTLLM W4A4 | Dense guard only SM100/103 | Not a GB10 candidate. Different shuffled weight layout; do not benchmark as though canonical CUTLASS weights work. |
| FlashInfer `cute-dsl` W4A4 | Dense guard only SM100/103 | Different implementation from `b12x`; not a GB10 candidate. |
| FlashInfer BF16×FP4 `cute-dsl` | Guard includes SM121 | **Separate W4A16 API** `prepare_bf16_fp4_weights` then `mm_bf16_fp4`. Uses BF16 activations and backend tile-packed weights; small M<=16 chooses 16×64×128 tile with two MMA warps. Worth comparing to Marlin and Triton W4A16. |
| FlashInfer BF16×FP4 cuDNN | SM121, cuDNN>=9.23.1 | Same W4A16 API; library version may rule out pinned image. |
| vLLM Marlin NVFP4 | >=SM75; W4A16 | Repacks nibbles and E4M3 scales into Marlin format and adjusts global scale, preserving BF16 activations. Native CUDA sources exist, but vLLM wrapper uses PyTorch custom ops. |
| vLLM FlashInfer/CUTLASS MoE | SM12x recognized | Distinct grouped/fused providers. `FLASHINFER_B12X` is opt-in in pinned vLLM due old SM121 MMA guard comment; newer standalone `b12x` integration differs substantially. |

Primary docs: [mm_fp4](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html),
[W4A16 API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_bf16_fp4.html),
[vLLM b12x integration](https://docs.vllm.ai/en/v0.29.0/features/quantization/b12x/).
The newer vLLM backend named `b12x` is not the pinned `flashinfer_b12x` adapter.

## Reference semantics

NVFP4 weights use E2M1 signed nibbles with magnitudes
`0, .5, 1, 1.5, 2, 3, 4, 6`; first K element is the low nibble. A positive
E4M3 block scale covers 16 K elements. Weight storage is 0.5625 byte/element,
including block scales, before scalar/padding overhead: 3.56× smaller than BF16.

Use unambiguous internal scale names:

```
W_real = decode_e2m1(W_bits) * W_block_scale * W_dequant_global
A_block_scale = e4m3(max_abs(A_block) * A_quant_multiplier / 6)
A_bits = e2m1(A / (A_block_scale / A_quant_multiplier))
Y = dot(A_bits * A_block_scale, W_bits * W_block_scale)
    * (W_dequant_global / A_quant_multiplier)
```

Checkpoint ModelOpt `weight_scale_2` and `input_scale` are dequant globals.
Thus `A_quant_multiplier=1/input_scale`, `alpha=input_scale*weight_scale_2`.
Compressed-tensors NVFP4 `weight_global_scale` and `input_global_scale` store
the inverse convention: `W_dequant_global=1/weight_global_scale`, activation
quant multiplier is `input_global_scale`. Preserve metadata format explicitly.

128x4 scale swizzle: pad rows to128 and scale-K to4, reshape linear scale
`[B,R/128,4,32,S/4,4]`, permute `[0,1,4,3,2,5]`, flatten. Quantizer writes this
layout directly. Weight swizzle is load-time work; no per-forward reorder needed.

Correctness has two distinct checks: GEMM against dequantized **same packed
operands**, and quantization distortion against original BF16. Comparing only
against BF16 with a permissive cosine threshold can hide a wrong scale layout.
W4A16 must use unquantized activations in its reference; W4A4 must include
activation quantization in both reference and delivered-latency measurement.

## Fusion scale hazard: inspect real scalars before fusing

Pinned vLLM ModelOpt dense linear collapses per-original-projection globals with
`max(input_scale)` and `max(weight_scale_2)`. It warns if unequal but does not
requantize packed weights or block scales. CT dense likewise takes maxima,
but on the **inverse globals**, then reciprocates the weight maximum. These are
different numerical conventions; copying max indiscriminately is wrong.

Pinned ModelOpt MoE w13 selects the first/gate weight global `[:,0]`, warns if
gate/up differ, and FI preparation takes max activation dequant global over all
experts. FI expects up/gate ordering in its fused first GEMM, so vLLM explicitly
reorders `[gate,up]` to `[up,gate]`.

An unfused gate/up or Q/K/V start keeps every checkpoint scale exact, but will
not necessarily match vLLM's default fused execution when globals differ. The
reference manifest must record this choice. Prefer inspect actual scalar
equality first; unequal globals need an intentional oracle policy or separate
reference projections. A shared input quantizer changes semantics when original
activation globals differ. Per-output-segment epilogue alpha can preserve weight
globals but cannot recover those different activation quantizations.

Do not copy pinned `FlashInferB12xExperts.process_weights_after_loading` blindly:
it folds FP32 weight globals into E4M3 block scales, rounds to E4M3 again, sets
weight globals to1, and forces FC2 activation scale1. Its comment claiming
identical dequantized weights is false for general globals. Example:
`0.125*0.137=0.017125` rounds to E4M3 `0.017578125`, a 2.65% perturbation.
The standalone b12x agent found a newer source-native preparation route retaining
original globals; prefer that for checkpoint-faithful probes.

## Concrete AOT proof and files

`flashinfer_native_compile.cu` instantiates four native CUTLASS tactics:
128×32×128 and128×64×128, swap-AB, each persistent DP and StreamK. All cluster
shapes are1×1×1. Full FI offers eight CTA shapes ×swapAB×scheduler =32 tactics;
benchmark selected shapes rather than compile everything into the runtime first.

`flashinfer_native_quant.cu` directly specializes FI's small-M BF16→NVFP4 kernel
with canonical max/6 scaling, 128x4 SF layout, no 4over6 mode, no PDL in this
probe. It does not invoke Python or JIT. It shares vendor helper code, not a
reimplementation of quantization math.

Local CUDA13.0.88 compile and full executable link **succeeded** for SM121a.
Use `-gencode=arch=compute_121a,code=sm_121a`; `-arch=sm_121a` also generates
fallback PTX that fails FI's architecture-specific guard. Required defines:
`ENABLE_BF16`, `ENABLE_FP4`, `ENABLE_FP8` for quantization,
`CUTLASS_ENABLE_GDC_FOR_SM100=1` for GEMM. Quant header also needs four vendor
common C++ sources: envUtils,logger,stringUtils,tllmException.

`native_flashinfer.py --output DIR` compiles and executes native benchmark:
M1/2/4/5/16, N/K17408/5120,5120/17408,10240/5120,1024/2048,2048/512;
cuBLAS BF16 baseline, each FP4 tactic, FI quantization alone, and quantization
plus selected tactic. Warm and128MiB L2-evicted cases, three FP4 trials. It
validates full outputs against same-quantized operands dequantized to BF16 then
cuBLAS FP32 accumulation. FP4 output is BF16; threshold0.005 relativeL2 allows
rounding but rejects gross scale/layout errors. No CPU/GPU test was run locally.

Other prototype entrypoints (all root-only GPU execution):
`flashinfer_probe.py`, `marlin_probe.py`, `flashinfer_w4a16.py`.
They accept `--output DIR`, save results.jsonl and explicit provider failures;
all include exact-quantized-reference metrics and graph-free event timing.

## Integration recommendation before performance results

1. Add checkpoint-format-aware scalar/layout descriptors; validate individual
   tensors and scale group semantics before device addressing. Keep BF16 output
   and existing attention/GDN schedules.
2. Start dense NVFP4 MLP with a selected native CUTLASS tactic plus native
   activation quantization, or W4A16 winner if measurements favor it. Per-row
   batch buckets1/2/4/5/16 cover decode and intended speculative verification.
3. Keep packed weights resident; load-time scale swizzle/repack; Rust owns
   scratch capacities/plans and C++ only launches selected kernels. Existing
   `qsfi_moe_execute_nvfp4` is still an unsupported stub, not an implementation.
4. MoE needs routed whole-layer measurements including dispatch, quantization,
   SwiGLU, second GEMM, and reduction. Dense GEMM timings cannot establish its
   speed at eight scattered experts per token.
5. Mixed FP8/NVFP4 checkpoints need FP8 projection handling as well. Root found
   official3.8-27B uses FP8 attention and NVFP4 MLP/LMhead. `qscb` currently has
   no FP8 descriptor/storage path. Tensor-scaled FP8 could extend cuBLASLt;
   block-scaled FP8 requires its actual checkpoint block geometry/provider.
   FI already has SM120 groupwise FP8 C++ kernels; verify low-M suitability.
   A BF16 dequant-load fallback preserves quantized weight values but changes
   any W8A8 activation-quantization semantics and storage/performance, so it
   cannot silently stand in for a matched vLLM FP8 reference.

## Bounded FP8 cuBLASLt probe

`fp8_cublaslt.py --output DIR` builds native `fp8_cublaslt.cu`. Local CUDA13
compile/link succeeded. M1/4/16, N/K10240/5120 and5120/6144; E4M3 weight and
activation buffers, caller-owned device FP32 scalar dequant scales, BF16 output.
The probe compares full output against the same FP8 operands dequantized to
BF16 followed by cuBLAS FP32 accumulation. It times fixed-scale activation
conversion, FP8 GEMM, both together, and BF16 baseline, with warm and128MiB
eviction and three trials. It uses the first working cuBLASLt heuristic among
eight requested, not a comprehensive tactic search. The fixed activation global
does **not** include dynamic tensor/row amax reduction cost. This tests tensor
scale feasibility, not block/channel scale support or complete FP8 accuracy.

There is no reason to add a separate provider for this tensor-scaled FP8 path:
extend existing `qscb` linear descriptors with device scale pointers and explicit
scale mode; accept existing `QSFI_DTYPE_FP8_E4M3` for both operands, retain BF16
output initially. Current plan key assumes BF16 inputs and keys only output
dtype; add both input dtypes and scale modes to it. Matrix layouts then use
`CUDA_R_8F_E4M3`; matmul descriptor binds `A_SCALE_POINTER` and `B_SCALE_POINTER`
to **weight and activation** globals respectively, because existing TN layout
uses weight as operand A. Reused plans must refresh those pointers when callers
change buffers, just as executions already accept new tensor addresses. Rust
continues owning scales, activation conversion buffers, and workspace. This
requires no new model scheduling boundary or release-mode stream synchronization.

FP8 fusion differs from NVFP4 in pinned vLLM: CT tensor FP8 weights are actually
requantized to a common maximum scale by `process_fp8_weight_tensor_strategy` /
`requantize_with_max_scale`; NVFP4 warns and collapses globals without that
requantization. Record this distinction in matched-reference metadata.

## Marlin native-launch extraction

`native_marlin.py` now builds a standalone CUDA shared library from the pinned
`marlin_template.h`: BF16 input/output, NVFP4 E2M1 weights, Marlin-encoded E4M3
scales, group16, stage4, FP32 reduction; tiles128×128/256threads and128×64/128threads,
each specialized for M<=8 or M<=16. Local CUDA13 compile and link succeeded.

Python prepares the exact same deterministic packed fixture used by native FI
and Triton, invokes vLLM's existing load-time Marlin weight/scale packing, then
hands raw device pointers to the library. All repeated launches, event recording,
128MiB cache eviction, and synchronization happen in C++; Python is outside every
timed interval. The library has no Torch runtime dependency; a prototype-only
shim replaces the scalar metadata header's `STD_TORCH_CHECK` dependency with a
standard C++ exception. No production source was changed. The two tactics use
one persistent block per SM; this is a bounded tactic comparison, not the entire
vLLM dispatch table.

The library allocates temporary FP32 reduction storage and zeroed locks before
timing, sets a constant kernel shared-memory attribute once per call, and reuses
all buffers over three warm/evicted trials. Final output is checked against full
FP32 GEMM of the same quantized weights and BF16 activations. `native_times_us`
contains warm0,cold0,warm1,cold1,warm2,cold2. A complete AOT integration would
replace Python load-time repacking with Rust/CUDA repacking, retain native
workspace in Rust, and validate the fixed subset of kernel arguments.

Initial root-run Python Marlin artifact
`.prototypes/out/nvfp4-marlin_probe-20260914T191020Z-__cy_juk/results.jsonl`
passed25 cases. For27B gate/down, M1/16 cold GEMM was268–276us versus782–832us
BF16 in that run. These already suggest the large dense result is not dominated
by Python. Small35B expert GEMMs were7–17us, where native timing is necessary.
NativeMarlin results supersede this preliminary timing interpretation when run.

## Executing the actual b12x export

`native_b12x_export.py` links the successful root-produced ARM64 object files
`qs3_b12x_densegemmlaunch_0.o` and `qs3_b12x_densesplitkreduce_1.o` from
`.prototypes/out/nvfp4-b12x_probe-20260914T191625Z-alar0p8x/exports`.
The generated `.h` entrypoints are used intact, so generated host TMA descriptor
construction is preserved. The export fixes N17408,K5120,tile128×64,split4 while
keeping M and device pointers dynamic. C++ allocates split partials and owns all
timed launch/reduction/event work; Python only constructs shared test fixtures.
The execution path imports neither b12x nor CuTe and performs no JIT calls.

CuTe AOT objects refer to `_cuda*` wrapper symbols: they require
`libcuda_dialect_runtime_static.a` (or shared counterpart), plus CUDA libraries.
The helper finds the archive from installed wheel metadata at build time and
records object/archive SHA256s. This is a native runtime dependency, not a Python
dependency. [NVIDIA AOT documentation](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html)
documents the same static-link route.

The local generated-header consumer compiles successfully. ARM export objects
cannot be linked into the local x86 host binary, so root must qualify actual
ARM linkage and GPU execution. The helper covers M1/5/16, both weight global1
and0.137, same-quantized-operands reference, three warm/128MiB-evicted trials.

## Pinned NVIDIA downloads and the 3.6/3.8 activation difference

`../download-pins.json` fixes the three download revisions and records exact
config/quant-config/index SHA256s, indexed tensor counts, payload bytes and
shard file bytes. Order is cached 35B, then 3.8-27B, then 3.6-27B. Metadata-only
HTTP reads fetched no weight payloads; the raw responses are retained under
`download-metadata/` for inspection. Each checkpoint has three weight shards.

| Model | Revision | Authoritative per-layer precision | Total shard bytes |
| --- | --- | --- | ---: |
| NVIDIA 3.6-35B-A3B | `6c7f09d4036e97393f82e9f9ecd1a5c35ca5ee92` | 161 W4A16_NVFP4, 130 FP8 | 23,424,338,320 |
| NVIDIA 3.8-27B | `dbb8f445b3145f8a4c18ddc769f032d57d32867c` | 193 NVFP4 W4A4, 208 FP8 | 21,921,697,280 |
| NVIDIA 3.6-27B | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` | 193 W4A16_NVFP4, 208 FP8 | 21,921,697,184 |

The newly resolved [3.6-27B revision](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/tree/0893e1606ff3d5f97a441f405d5fc541a6bdf404)
declares `W4A16_NVFP4` for its 192 MLP projections plus LM head; its CT
`group_1` omits `input_activations`, consistent with the ModelOpt per-layer map.
The [pinned 3.8-27B](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/tree/dbb8f445b3145f8a4c18ddc769f032d57d32867c)
instead declares `NVFP4` and four-bit input quantization for the same 193 layers.
Both require FP8 attention projections and retain other tensors in BF16. Their
identical tensor-index SHA256 and payload size do not mean identical execution:
activation precision is encoded in quantization metadata, not packed-weight
shape. Use W4A16 for a matched 3.6-27B oracle, W4A4 for a matched 3.8-27B oracle;
running 3.8 weights with BF16 activations is a separate precision experiment.
There is a second metadata difference: 3.6-27B declares
`kv_cache_quant_algo: "FP8"` and an eight-bit CT KV-cache scheme, whereas
3.8-27B declares no KV-cache quantization. In **vLLM v0.29.0**, `auto` actually
adopts this metadata: `EngineArgs.create_engine_config` calls
`resolve_kv_cache_dtype_string`, which recognizes ModelOpt's CT KV scheme.
Running the exact pure resolver functions from source commit
`98dff2a81d747d1dba01a47f939f48c3526d4206` against these configs produced:

| Pinned checkpoint | Requested KV dtype | Resolved KV dtype |
| --- | --- | --- |
| 3.6-35B NVFP4 | `auto` | `auto` → model dtype (BF16 for our runs) |
| 3.8-27B NVFP4 | `auto` | `auto` → model dtype (BF16 for our runs) |
| 3.6-27B NVFP4 | `auto` | `fp8_e4m3` |
| All three | `bfloat16` | `bfloat16` |

Source: [v0.29 resolver](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/utils/torch_utils.py),
[engine setup](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/engine/arg_utils.py).
This source-level check did not execute the models or allocate GPU memory.

Neither 27B tensor index contains separate K/V cache scales: its K/V projection
`input_scale` and `weight_scale` belong to FP8 linear layers, not KV-cache
quantization. The index has no `k_scale`, `v_scale`, `kv_scale`, `output_scale`
or BMM-scale tensors. v0.29's
[BaseKVCacheMethod](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/layers/quantization/kv_cache.py)
therefore initializes missing K/V scales to 1.0 and emits its FP8 E4M3 warning
when that cache path is selected. Do not interpret the KV metadata as calibrated
cache-scale tensors being present.

For a comparison that isolates checkpoint **weight/linear precision**, explicitly
set `kv_cache_dtype="bfloat16"` for all three models and record that override.
Reproducing vLLM's unmodified default for 3.6-27B instead selects FP8 KV with
unit fallback scales and is a separate reference policy. Supporting these
checkpoint weights does not itself require adding FP8 KV to qs3. Recurrent GDN
state precision is a third independent setting and stays explicitly FP32.

`expected_payload_bytes` is the index's tensor payload total;
`expected_shard_bytes` sums HF API shard sizes and includes safetensors headers.
These count complete published checkpoints, including non-text/MTP payloads,
not just bytes the initial text-only runtime will materialize.
