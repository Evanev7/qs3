# CuTe AOT compiler

Compile a Python source exposing one `@cute.jit` host entrypoint named `kernel`:

```sh
qscute --source cute_kernels/fp8_decode.py --spec "$spec_json" --target "$target_json" --prefix build/cute/fp8_decode
```

`CuteSpec` supplies `precision` and byte `alignments` for every `cute.Pointer`
parameter, `constants` for every `cutlass.Constexpr` parameter, and compiler
`options`. Dtype constants use `{ "dtype": "f32" }`. Scalar annotations use
Cutlass numeric types; exactly one parameter must have type `cuda.CUstream`.
The source entrypoint owns launch geometry and passes the stream to its device
kernel. Arguments retain source order; constants disappear from the runtime ABI.

The FP8 decode recipe lives in `models/config.nix`; it specifies E4M3 input and
weight pointers, FP32 scales/partials, and the measured `N=10240, K=5120` shape.
Use `host-target=linux-aarch64` to cross compile on a GPU-free host.

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

`nixsrc/cute.nix` emits Ninja rules for recipes with `provider = "cute"`.
Both the development and Nix builds archive the exported objects in `libqscute.a`.
`qscute-runtime` stages the matching wheel's native runtime archive for Cargo;
inference does not depend on the Python environment.

Unit tests cross compile the real FP8 decode kernel to SM121/AArch64 and type-check
its Rust wrapper. `just build_tools/cute-test` exercises the generated launcher,
using FP8 split outputs and dependent scale inputs to verify exact CPU results,
guards, stream ordering, duplicate loads and reloads. These run in the full
`./remote.sh test` workflow.
