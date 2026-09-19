# CuTe kernels

Sources expose a `@cute.jit` host entrypoint named `kernel`, which launches
`@cute.kernel` device functions. qscute specializes and exports the entrypoint
at build time; inference uses its native interface without Python or JIT.

`saxpy.py` exercises pointer types, dynamic scalars, constexprs, and stream
forwarding. It is a compiler smoke fixture, not a model kernel.

See the [qscute interface](../build_tools/pysrc/qscute/README.md) for spec,
artifact, and linking details.
