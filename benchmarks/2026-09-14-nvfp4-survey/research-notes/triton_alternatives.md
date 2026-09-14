# Triton and alternative NVFP4 paths, 2026-09-14

## What was actually tested

`triton_aot/probe.py` initially compiled ten kernels with the repository's Triton 3.8.0
wheel, explicitly targeting `GPUTarget("cuda", 121, 32)`. No GPU was used for
this compilation. `triton_aot/run.py --compile-only` also compiled the native
CUDA driver benchmark with `nvcc -arch=sm_121a`.

- Native W4A4 `tl.dot_scaled` produces
  `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`.
- W4A16 `tl.dot_scaled(BF16, None, "bf16", FP4, FP8-scales, "e2m1")`
  produces ordinary BF16 `mma.sync.aligned.m16n8k16` operations. Weights are
  unpacked and block-scaled locally. It retains compressed weight storage;
  it does not materialize a full BF16 matrix in the inference path.
- Tiles: M16 N32 K128, M16 N64 K128, M16 N64 K256; additionally split-K=4
  for N32/K128 and N64/K256. All use four warps and two stages.
- Native W4A4 shared memory is 2/4/8 KiB; W4A16 is 6/6/10 KiB. All are
  single-CTA kernels, no TMEM or global/profile scratch allocations.
- M/N/K are scalar runtime arguments, not a new cubin per token count.
  The bounded benchmark shapes satisfy K divisible by BLOCK_K * SPLIT_K.

The second pass adds seven variants (17 total), also compile-checked:

- Three explicitly suffixed `_aligned` dot controls give the compiler the
  actual cudaMalloc pointer alignment (16 bytes) and N/K divisibility (256).
  This unlocks `cp.async` pipelines: N64/K256 W4A4 changes from zero to 20
  `cp.async` occurrences in PTX; W4A16 changes from zero to 26. Increased
  shared memory reflects staged buffers (11.25/17 KiB respectively). The
  original ten cubins remain controls, so this is a measurable change.
- Four direct W4A16 GEMVs own one output row per CTA and compute 1, 4, or
  16 token rows while reusing decoded weight chunks. They use
  `cvt.rn.f16x2.e2m1x2`, FP8 per-16 scales, and FP32 products/reductions.
  They require no tensor cores, activation quantization, or output scratch.
  K chunks are 512/1024 for M1, and 512 for M4/M16. Alignment hints reduce
  M1 shared reduction storage to 16 bytes and enable vectorized loads.

Three improvements worth comparing before expanding the kernel surface:
alignment/pipelining, split-K for insufficient output-tile parallelism,
and direct W4A16 reductions that avoid padded M16 tensor-core work. Generic
group-M/L2 tile ordering does little when M<=16 gives only one M tile;
scale layout/coalescing is a more credible next optimization in that case.

The minimal kernel is newly written prototype code, not the upstream
`triton_kernels` implementation. The existing copied upstream library has
NVFP4 support too, but pulls in Torch through host modules; this small probe
isolates the compiler capability without that dependency.

The native benchmark sweeps M=1,2,4,5,16 and (N,K)=(17408,5120), (5120,17408),
(10240,5120), (1024,2048). It validates the full output against BF16
cuBLAS GEMM with exactly the same dequantized FP4 inputs, records CUDA-event
latencies, and includes the split-K reduction launch. There are three trials
for warm-cache timing (100 launches/trial) and three for 128MiB L2-evicted
timing (30 launches/trial); eviction runs before the start event and is
excluded from each measured interval. Root owns remote
execution and result interpretation; compilation alone is not performance
or numerical evidence.

```sh
build_tools/.venv/bin/python .prototypes/nvfp4_survey/triton_aot/run.py \
  --output /path/to/results/triton_aot
```

`run.py` performs build-time Python compilation, then invokes a native
executable that loads cubins through the CUDA driver. Inference/timing has
no Python/JIT. Outputs include PTX, cubins, metadata, build command, and CSV.

Benchmark limits: synthetic scale values, tensor-level scales=1, already
quantized inputs, FP32 output, no activation quantization cost, no
provider interleaving, one fixed provider order. The warm small expert
matrix can stay cached; the explicit L2-evicted results distinguish this. cuBLAS baseline uses `cublasGemmEx`, not our cached cuBLASLt plans.
These numbers select kernel candidates; they are not end-to-end TPS.

The third pass adds **the production qscb implementation as a baseline**:
`run.py` links the unchanged repository `qscb.cu`, prepares plans once per
shape/output dtype, and retains 64MiB workspace as in ModelConfig's default.
CSV arms `qscb_bf16_out` and `qscb_f32_out` compare actual runtime BF16
output and identical FP32 output against the original GemmEx baseline.
BF16 validation compares with the BF16-rounded FP32 reference. Source hashes
and baseline output dtypes are included in the manifest. No runtime files
were modified to build these baseline arms.

Optional `--lm-head` appends N248320/K5120 with M=1/5/16;
`--lm-head-only` runs just those cases. They use two warmups, eight warm
launches or five individually L2-evicted launches per trial, with three
trials. The fixture uses approximately 3.5GiB for weights plus small outputs
and workspaces. This measures the geometry and compressed-weight approach;
it does not establish that any given checkpoint actually quantizes LM head.
The first large-N run exposed the CUDA grid.y limit in the GEMV harness:
N248320 exceeds 65535 blocks. GEMV now places output rows on grid.x and
token blocks on grid.y; the matching cubins and native launcher compile
successfully. No GEMV variant is skipped. GPU validation of that corrected
large-N launch remains root's responsibility.

## Sources and integration implications

The [Triton upstream SM120 tests](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/python/test/unit/language/test_matmul.py)
and [FP4 packing issue](https://github.com/triton-lang/triton/issues/9678)
show that native FP4 support must be distinguished from mixed-precision
fallbacks. Keep FP4 packed along K, with a logically transposed RHS and
`rhs_k_pack=True`. Generic statements that Triton always emulates FP4 on
SM121 are contradicted by the compiled PTX above.

The [main block-scaled tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py)
currently gates its demo to compute-capability major 10/11 and uses TMA
descriptors. Its advertised Blackwell support does not make that demo a
drop-in SM121 kernel. The simpler pointer-based dot kernel is AOT-friendly.

The existing local `triton_kernels_gemv/upstream/triton_kernels` has helpers
for NVFP4 local upcasting and GEMM fallback in `_matmul.py` and `_p_matmul.py`.
Its `_upcast_from_mxfp.py` uses native `cvt.rn.f16x2.e2m1x2` for unpacking on
Blackwell, then applies per-16 FP8 scales. This is an attractive basis for a
low-M weight-only GEMV if native W4A4 activation quantization/tile overhead
does not pay off. A direct GEMV should be compared at M1 and small expert
shapes, with a small-M GEMM retained for verification rows.

The production qstriton generator already exports cubins and Rust driver
launchers. It needs an explicit storage/ABI choice for FP8 scale pointers:
its `scalar_rust_type` mapping has no FP8 entry. Expose byte pointers and
bitcast in the kernel, or add a storage-bit mapping. Do not introduce a
Torch dependency into inference. Tensor-level alpha should be applied once
to the accumulator/output, not once per K tile.

The physical compressed weight cost is 0.5 bytes/value plus 1/16 byte/value
for block scales, approximately 0.5625 bytes/value before tensor scales and
padding. Thus the BF16 weight-traffic ratio is about 3.56x, not 4x. W4A16
and W4A4 can both obtain that memory benefit; native tensor-core FP4 changes
compute and activation handling, not the checkpoint storage ratio.

## Other concrete candidates

1. **Direct CUTLASS 79a** is a sensible minimal native baseline already
   vendored at
   `3pty/flashinfer/3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`.
   It uses SM120 block-scaled TensorOps, BF16 output, 128x128x128 tile,
   cluster1x1x1. **79b has FP4 output** and is not the BF16-output baseline.
   Specialize only needed tiles and retain Rust-owned workspace/scheduling.
   [Official source](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).

2. **Marlin W4A16** is already implemented in vendored vLLM, including
   NVFP4 dense and MoE. It merits a low-M comparison, not dismissal because
   the GPU has native FP4. It requires a different load-time packing and
   scale transform: special S0E5M3 scales, permutation, compensation in
   global scale, and workspace. Local paths:
   `3pty/vllm/vllm/model_executor/kernels/linear/nvfp4/marlin.py` and
   `.../layers/quantization/utils/marlin_utils_fp4.py`.
   The BF16 conversion contains rescaling/underflow handling, so reuse its
   semantics rather than assuming stock checkpoint scales can be passed
   unchanged. A native extraction has a larger surface than a small Triton
   fallback. The reference target must select W4A16 too when checking
   matching logits; native W4A4 additionally quantizes activations.

3. **Colfax CuTe DSL SM12x kernels** provide an independent, modern native
   implementation and useful optimization guidance. Their
   [architecture/kernel article](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
   explains that only the per-thread scale fragment layout is hardware
   mandated; global-memory scale layout and TMA are implementation choices.
   Their [optimization article](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)
   and [source](https://github.com/ColfaxResearch/cfx-article-src/tree/master/sm120_nvfp4_gemms)
   are candidates if existing wrappers fail. Published GEMM throughput is
   not evidence for M1-16 on GB10, and DSL export into our AOT lifecycle
   still requires a concrete build/launch experiment.

4. **VincentKaufmann/fp4-cuda-kernel** has a standalone native wrapper and
   quantization/scale layout code, but appears to be essentially a CUTLASS
   example configuration. Its reported measurements start at M256 and
   don't establish a low-M win. Some README claims about unavailable Triton
   or cuBLAS FP4 are stale; its one-level quantization example also should
   not be substituted for checkpoint two-level scale semantics.
   [Repository](https://github.com/VincentKaufmann/fp4-cuda-kernel).

5. **GemLite** now explicitly targets SM120 and supports NVFP4, small-batch
   weight-only execution, split-K and direct GEMV families. Its
   [upstream README](https://github.com/dropbox/gemlite) is a useful candidate
   map. However the current
   [core implementation](https://github.com/dropbox/gemlite/blob/master/gemlite/core.py)
   defaults microscaled M1 to GEMM_SPLITK with an NVFP4 GEMV-failure TODO;
   `enable_activation_scaling` currently returns True unconditionally,
   despite the README advertising batch-size adaptation. Treat these as
   candidate algorithms, not a proven drop-in NVFP4 GEMV.
   Its public FP16 defaults and autotuning wrapper should not be imported
   unchanged into a BF16/AOT runtime. Extract and compile a selected kernel
   after verifying exact NVFP4 storage/scale semantics. Upstream Triton's
   [optimization flags](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/opt_flags.py)
   also distinguish persistent/TMA cases and full-cache-line K blocking;
   datacenter TMEM heuristics cannot simply be reused on SM121.

Recommendation pending GPU results: keep native W4A4 plus weight-only
fallbacks in the candidate set. First prove and measure with canonical
checkpoint packed weights and linear block scales, make any backend
repacking a one-time preparation step, and include quantization plus final
output conversion in the eventual provider comparison. Do not expand the
production provider matrix until an actual low-M measurement chooses it.
