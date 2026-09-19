# Kernel replacement experiments

Target: GB10 / SM121, Qwen3.8-27B NVFP4. QuTLASS prefill is integrated in
`b1db445`; the other providers remain isolated probes. Baseline runtime
and standard benchmark evidence are in [the preceding experiment](../2026-09-19-nvfp4-performance/README.md).

## Source and toolchain pins

| Source | Revision | Use |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | `98dff2a81d747d1dba01a47f939f48c3526d4206` | Fused attention / packed GDN reference |
| [b12x](https://github.com/local-inference-lab/b12x) | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` | Current CuTeDSL FP8 candidates |
| [QuTLASS](https://github.com/IST-DASLab/qutlass) | `e74319e3405ce6d71965732880f5dc1f52371f64` | SM120 NVFP4 GEMM |
| [cuTile Rust](https://github.com/NVlabs/cutile-rs) | `e04245bdcf1f5bfc602a2078168eff90ee40bebb` | Compile-only and native cubin launch |

The b12x and cuTile clones are isolated under `.prototypes/kernel_replacements/vendor`;
production submodule pins are unchanged. No distinct relevant project named
“qute” was identified; CuTe/CuTeDSL candidates are covered here.
Python probes use the pinned vLLM image
`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
b12x uses the previously qualified isolated CUTLASS DSL 4.6.2 overlay at
`~/qs3-nvfp4-toolchain`. Native code uses host CUDA 13.0.88. Source snapshots are
in `sources/`, with `.txt` suffixes to keep them out of the main Nix/Cargo inputs.

## KR01: attention preparation

The copied vLLM kernel replaces two Q/gate copies, two norms, and RoPE with one
launch. Actual geometry: 24 Q heads, 4 KV heads, head dimension 256, rotary
dimension 64, theta 1e7, BF16 storage, Qwen norm weight offset 1.
The reference compiles the actual `qsfi_context.cu` / `qsfi_norm_rope.cu` and
reproduces the runtime's two `cudaMemcpy2DAsync` calls.

[Cached-RoPE results](attention-cache.json) cover M=1,2,4,8,16,128,1024 and
positions 0,137,30000, with FP32 and BF16 caches. Captured timing at position 137:
8.13 → 1.53 µs at M=1; 404.29 → 253.42 µs at M=1024 (FP32 cache).
Gate copies are exact. Q/K are not: the cache construction and reduction differ
from FlashInfer. BF16 caches increase the differences. These timings exclude
cache construction, and are not end-to-end token latency measurements.

Follow-ups replace the cache with FlashInfer-compatible on-the-fly frequencies
and sin/cos, then match its eight-values-per-lane norm reduction. The
[on-the-fly](attention-onfly.json), [matched reduction](attention-matched.json),
and [FMA](attention-fma.json) probes reduce the differences but leave rare
BF16 differences at M1024. Numerical qualification and native AOT launch remain
open. Do not promote the copied cache version based only on its speed.

One early reference timing was invalid because its provider retained the default
stream during graph capture. The retained results select the capture stream for
every reference call. The empty-graph timings were discarded.

## KR03: tensor FP8

[Default b12x results](fp8-default.json) compare identical E4M3 operands and scalar
dequantization scales against qs3's prepared qscb code with 64 MiB workspace.
The cuBLASLt library comes from the container, not the production Nix package.
Both providers are captured; 30 alternating-order samples measure warm and
128 MiB weight-evicted cases. Activation quantization is not included yet.

At M=1, evicted GEMM latency changes from 356.4 → 282.6 µs for N10240/K5120,
199.9 → 190.5 µs for N6144/K5120, and 199.9 → 193.5 µs for N5120/K6144.
M1024 regresses on all three shapes. Outputs differ at small M; b12x defaults
`B12X_DENSE_SPLITK_TURBO=1`, enabling BF16 atomic partial accumulation.
Disabling this knob alone fails preparation: the default selects four slices
on GB10, but the FP32 reducer supports only two. The probe caps
`_select_block_fp8_decode_slices` at two for this experiment.
[The FP32 rerun](fp8-f32.json) retains the large projection's gain: M1/N10240/K5120
350.3 → 283.9 µs evicted, with bit-identical output on that fixture. Smaller
projections gain roughly 2%. Across all shapes, some other rows still differ
slightly due to reduction order. No production b12x source or provider
selection has changed.

[Native AOT qualification](fp8-native/native-results.json) now passes: export
CuTe's GEMM to a `.o`/generated header, compile the two-slice FP32 reducer with
Triton to a cubin, and launch both from C++ on the caller's stream. No b12x,
CuTe, or Triton Python API is called during native execution; Python only
supplies fixtures and timing. Three M1/N10240/K5120 fixtures have 0, 1, and 4
differing BF16 elements versus qscb, respectively (relative L2 <=3.2e-5).
Thus the exact first fixture does not imply general bitwise equivalence.

The native reference still uses the container's cuBLASLt; production-library
and full-model qualification remain open. The native probe includes GEMM and
reduction, but excludes activation quantization and preparation of unit block
scales. Build arguments and hashes are retained beside its results. Reproduce
with `sources/fp8_export.py.txt`, then `sources/fp8_export_native.py.txt`, using
the pinned container and CuTe overlay, `B12X_DENSE_SPLITK_TURBO=0`,
`B12X_COMPILE_DISK_CACHE=0`, and `KR03_OUTPUT=kr03-fp8-export`. The native wrapper
specializes M1/N10240/K5120; small batching needs separate exports/measurements.

## KR04: QuTLASS NVFP4

[Native results](qutlass.csv) include three trials per shape/provider, each
with 30 warmed and 30 weight-evicted samples. Providers 0/1 are FlashInfer N32
DP/Stream-K, 2/3 N64 DP/Stream-K, and 4 the QuTLASS recipe. All 16 shapes
pass the same-quantized-operands FP32-reference check (relative L2 <0.005).
This probe includes raw GEMM and workspace initialization, not activation
quantization. The fixture, reference, warmups, eviction, and timing code are
retained in `sources/qutlass_bench.cu.txt`.

| M | N / K | N32 DP evicted µs | N64 DP evicted µs | QuTLASS evicted µs |
| --- | --- | ---: | ---: | ---: |
| 1 | 17408 / 5120 | 292.5 | 293.6 | 295.6 |
| 1 | 5120 / 17408 | 303.8 | 303.8 | 309.0 |
| 128 | 17408 / 5120 | 307.9 | 302.8 | 311.0 |
| 128 | 5120 / 17408 | 330.6 | 314.0 | 312.1 |
| 512 | 17408 / 5120 | 579.3 | 406.2 | 371.6 |
| 512 | 5120 / 17408 | 622.3 | 415.4 | 382.7 |
| 1024 | 17408 / 5120 | 1060.5 | 683.7 | 563.9 |
| 1024 | 5120 / 17408 | 1160.9 | 743.1 | 616.1 |

The large-prefill implementation passes the complete `./remote.sh test`
workflow ([log](full-tests.log)): 17 Python, 3 Triton launcher, 164 library
(6 existing ignored), 5/3/2/14 other Rust tests, and all four native CUDA suites.
The first run exposed an outdated expected cache count in the expanded test:
adding two row shapes requires seven plans instead of five. This was corrected;
no numerical assertion was relaxed or test skipped.

The [model comparison](model-logits/comparison.json) matches **69/69 complete
248320-logit rows bit-for-bit** against the saved N32DP/64 MiB baseline at a
1024-token prompt, including 68 decode steps. The diagnostic's per-step
downloads make its timings invalid. Raw rows remain in
`sp10:qs3/.prototypes/out/kr04-model-logits-7wh3q12h/rows/`; the source overlay
and hashes are retained here.

The standard committed benchmark passes at `f284ac6`
([log](benchmark.log), [comparison](benchmark-comparison.json)):

| Workload | Baseline prefill ms | New prefill ms | Baseline decode tok/s | New decode tok/s | IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| 102/32 | 128.745 | 129.014 | 10.366 | 10.356 | 36/36 exact |
| 1024/256 | 518.081 | 413.405 | 10.607 | 10.568 | 260/260 exact |

The large workload improves prefill latency **20.2%**; decode remains within
0.4% with unchanged kernel selection. The full benchmark artifacts are
[102/32](../2026-09-19T134627.635456349Z-f284ac6.json) and
[1024/256](../2026-09-19T134653.185391398Z-f284ac6.json).
The first Nix benchmark build exposed that native source filters excluded
`.cuh`; `f284ac6` fixes both filters. The successful run includes the actual
header in the Nix build. Nsight reports its existing unavailable UM/CPU-sampling
features; CUDA kernel traces and unprofiled measurements are present.

It uses the existing packed values, scale layout, global scale, and BF16 output;
there is no Hadamard transformation. Row-dependent selection belongs to
`models/config.nix`, with Rust preparing the chosen plan for each matrix shape.

## KR05: cuTile Rust AOT feasibility

The pinned source (version 0.4.0) exposes `compile_api::KernelCompiler`. It can
emit TileIR without creating a CUDA context; `compile_tile_ir_module` can then
invoke `tileiras` to produce an SM121 cubin. A separate C++ program loaded that
cubin with `cuModuleLoad` and launched `saxpy_entry` with `cuLaunchKernel`:
**64/64 outputs exact**, with no Rust, Python, or compiler at execution time.
This establishes feasibility, not competitive model-kernel performance.

The compiler is isolated under `.prototypes/kernel_replacements/toolchain`:
tileiras 13.3.36, nvcc/nvvm 13.3.73, runtime 13.3.29, nvjitlink 13.3.33.
Use `CUTILE_BYTECODE_VERSION=13.3`, point `CUTILE_TILEIRAS_PATH` at its `bin/tileiras`,
and set `LD_LIBRARY_PATH` to its `lib` directory. Without that library path,
tileiras returned exit 5 with only “failed to compile Tile IR program”.
The host toolkit and driver were not replaced. Cubin launch uses the host CUDA
13.0 Driver API. Generated artifacts live on sp10 under
`.prototypes/out/kr05-cutile-aot/`.

The ordinary cuTile launch API performs compilation on demand. A qs3 integration
should use the compile-only API at build time and retain Rust-owned runtime
buffers, schedules, modules, and launch validation. Start with a small fusion
candidate before attempting to replace tensor-core GEMM. The emitted entry ABI
includes shape/stride/tile parameters even when this particular kernel does not
use all of them; inspect the generated TileIR rather than guessing arguments.

An NVFP4 follow-up using upstream's `linear_tile` example fails compilation for
SM121 with tileiras 13.3.36 at both 16x16x64 and 128x128x128 tile sizes. The
same bytecode compiles for SM100 and SM120. The SM120 cubin fails to load on
GB10 with `CUDA_ERROR_NO_BINARY_FOR_GPU`. An SM120 cubin is therefore not a
working fallback. `sm_121a` is not an accepted tileiras target spelling. An isolated upgrade to **tileiras 13.4.92** resolves this: the same bytecode
compiles for SM121 and launches on GB10. The native NVFP4 driver passes three
M16/N16/K128 fixtures, **768/768 outputs exact**, covering signed FP4 values,
varying block scales, and a non-power-of-two global scale
([log](cutile-nvfp4-native.log)). This uses logical row-major scale storage and
FP32 output, not qs3's swizzled-scale BF16 projection contract.

All new compiler components live under
`.prototypes/kernel_replacements/toolchain-latest/nvidia/cu13/` on sp10.
Run its `bin/tileiras --gpu-name sm_121 --opt-level 3 -o OUTPUT INPUT.bytecode`
with `LD_LIBRARY_PATH` pointing at its `lib` directory. The bytecode version is
still 13.3; the compiler and matching libraries are 13.4.92. Host CUDA is
unchanged. Real-shape performance and a useful fusion remain unmeasured.

## KR06: fused SiLU×up and NVFP4 quantization

The vLLM CUDA kernel is adapted only to separate gate/up pointers. It retains
the BF16 rounding boundary before quantization. The reference calls actual
`qscu_silu_and_mul_bf16` followed by qs3's current FlashInfer quantizer.
[All 42 cases](silu-quant.json) match packed activation bytes and scale bytes
exactly: K5120/17408, M1/2/4/8/16/128/1024, multipliers 1/64/1024.
Padded scale storage is zero-initialized in both paths.

Captured warm timing at K17408 is **5.38 → 1.87 µs** for M1 and
**655.17 → 359.37 µs** for M1024. These are synthetic activation timings;
there is no full-model result or production integration yet. The next check is
a small implementation using FlashInfer's existing packed-vector, SiLU, and
FP4-conversion helpers, avoiding another large helper-header copy. Preserve the
BF16 boundary, compare packed bytes/scales including tails and graph capture,
then run full-model logits, required tests, and the committed benchmark.

Sources and native compilation arguments are archived under `sources/`.
The original probe uses explicit
`-gencode=arch=compute_121a,code=sm_121a` with host CUDA 13.0.88. Its copied vLLM
helper headers come from the pinned vLLM source; Torch-only type includes were
replaced with forward declarations in the isolated probe. Production headers
are unchanged.

## Remaining candidates

- KR02: vLLM packed GDN combines post-convolution Q/K/V loads, L2 normalization,
  decay/beta, and FP32 recurrence. Its batch dimension is independent requests,
  not sequential speculative tokens. Its null state index is zero; qs3 uses
  zero as a valid slot. Preserve explicit source/destination state slots.
  Current b12x selects CuTeDSL recurrence and Triton output norm for the correct
  16-key/48-value-head geometry, but permits only eight state-index columns per
  sequence. Its in-place state/history contract must be adapted before using it
  with qs3's staged state or a future 16-token speculative batch.

Historical findings are preserved in [prior-findings.md](prior-findings.md).
