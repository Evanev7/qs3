# CuTe AOT compiler

Compile a Python source exposing one `@cute.jit` host entrypoint named `kernel`:

```sh
qscute --source cute_kernels/saxpy.py --spec "$spec_json" --target "$target_json" --prefix build/cute/saxpy
```

`CuteSpec` supplies `precision` and byte `alignments` for every `cute.Pointer`
parameter, `constants` for every `cutlass.Constexpr` parameter, and compiler
`options`. Dtype constants use `{ "dtype": "f32" }`. Scalar annotations use
Cutlass numeric types; exactly one parameter must have type `cuda.CUstream`.
The source entrypoint owns launch geometry and passes the stream to its device
kernel. Arguments retain source order; constants disappear from the runtime ABI.

For the SAXPY source, an example spec is:

```json
{
  "precision": {"x": "f32", "y": "f32", "output": "f32"},
  "alignments": {"x": 16, "y": 16, "output": 16},
  "constants": {"BLOCK": 128},
  "options": {"gpu-arch": "sm_121a", "host-target": "linux-aarch64"}
}
```

The target uses the same JSON as qstriton:

```json
{"backend": "cuda", "computeCapability": {"major": 12, "minor": 1}, "warpSize": 32}
```

qscute exports a native `.o` containing the host launcher and device code,
a typed `.rs` launcher, a `.json` provenance manifest, and a `.d` depfile.
The Rust launcher calls CuTe's exported `_mlir_*` entrypoints directly. Their
names come from the compiled function's metadata; arguments are passed by
address in source order, followed by the return-status slot. No C header or
C++ bridge is generated. Link the object with the matching host architecture's
`libcuda_dialect_runtime_static.a`, CUDA driver/runtime libraries, and C++ runtime.
Python and the compiler are build dependencies only.

The Rust launcher uses qs3's `DevicePtr<DType>` markers. Load it on the intended
CUDA device before capture or timing; retain its context and module until queued
work and graphs finish. Rust owns allocations, pointer validation, scheduling,
and graph lifetime. Module loading does not prepare or cache TMA descriptors:
descriptor construction inside a source entrypoint remains host launch work.

CuTe stores the kernel handle in generated global state. Each specialization
therefore allows one live Rust owner, enforced at load time; attempts to load a
second panic. CUDA failures are returned as error codes. Different specializations can coexist.
Include each generated module once, and complete queued work before dropping its
owner. A completed unload permits reloading. No synchronization is added to launch
or drop.

The depfile tracks qscute, qsutil, and Python files below the source directory.
External helper trees need dependency tracking before introducing imports from
them. Compiler version, source hashes, ABI, and artifact hashes are recorded in
the manifest.

`nixsrc/cute.nix` emits Ninja rules for recipes with `provider = "cute"`. It is
not yet connected to the main build, and no model recipe selects CuTe. Local
unit tests cross compile F32/BF16 SAXPY to SM121/AArch64 and type-check the Rust
wrapper. `just build_tools/cute-test` links the exported objects directly into
a Rust executable and exercises F32/BF16 outputs, tail guards, dependent launches
on a nonblocking stream, duplicate-load panics, and module unload/reload.
The full required `./remote.sh test` workflow passes on GB10. Model kernel
integration and qualification remain pending.
