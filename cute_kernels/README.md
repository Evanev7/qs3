# CuTe kernels

Sources expose a `@cute.jit` host entrypoint named `kernel`, which launches
`@cute.kernel` device functions. qscute specializes and exports the entrypoint
at build time; inference uses its native interface without Python or JIT.

`fp8_decode.py` adapts b12x's SM120 FP8 path for the 27B GDN QKV decode
projection (M=1, N=10240, K=5120). It writes two FP32 split-K partials;
`triton_kernels/fp8_reduce.py` performs the final addition and BF16 rounding.
Rust owns activation quantization, scratch, and both launches. Other shapes
and prefill use cuBLASLt.

`nvfp4_gemm.py` uses b12x's block-scaled SM120 path with tile 32x64x512,
no split-K, and BF16 output. AOT exports cover the 27B gate/up, down and LM-head
shapes for M=1 through 16 on GB10. Rust retains activation quantization and
selects CUTLASS/QuTLASS for larger batches. Weights and swizzled block scales
use the existing NVFP4 layout.

See the [qscute interface](../build_tools/pysrc/qscute/README.md) for spec,
artifact, and linking details.
