# Kernel replacement experiments

Target: GB10 / SM121, Qwen3.8-27B NVFP4. These are isolated probes; no new
provider has passed full-model validation or been adopted yet. Baseline runtime
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
and sin/cos, then match its eight-values-per-lane norm reduction. Numerical
qualification and native AOT launch are still in progress. Do not promote the
copied cache version based only on its speed.

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
The next comparison disables this knob and retains FP32 partial sums. No claim
of equivalent numerical behavior or model speedup follows from the default run.

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

## Remaining candidates

- KR04: QuTLASS uses N128 tiles on SM120 (M128 below 512 rows, M256 above).
  Test its raw GEMM with the same packed values/scales. Its Hadamard quantizer
  cannot be substituted into existing checkpoints as a neutral optimization.
- KR02: vLLM packed GDN combines post-convolution Q/K/V loads, L2 normalization,
  decay/beta, and FP32 recurrence. Its batch dimension is independent requests,
  not sequential speculative tokens. Its null state index is zero; qs3 uses
  zero as a valid slot. Preserve explicit source/destination state slots.
- KR06: vLLM SiLU×up+NVFP4 quantization is a plausible launch/memory saving.
  qs3 has separate gate/up buffers; preserve its BF16 intermediate boundary and
  compare packed activation bytes and scales before timing.

Historical findings are preserved in [prior-findings.md](prior-findings.md).
