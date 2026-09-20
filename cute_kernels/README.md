# CuTe kernels

Sources expose a `@cute.jit` host entrypoint named `kernel`, which launches
`@cute.kernel` device functions. qscute specializes and exports the entrypoint
at build time; inference uses its native interface without Python or JIT.

`fp8_decode.py` adapts b12x's SM120 FP8 path for the 27B GDN QKV decode
projection (M=1, N=10240, K=5120). It writes two FP32 split-K partials;
`triton_kernels/fp8_reduce.py` performs the final addition and BF16 rounding.
Rust owns activation quantization, scratch, and both launches. Other shapes
and prefill use cuBLASLt.

See the [qscute interface](../build_tools/pysrc/qscute/README.md) for spec,
artifact, and linking details.
