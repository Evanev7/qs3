# Native Operation Boundaries Implementation Plan

**Goal:** Replace one-call Rust descriptor wrappers with direct typed tensor
interfaces, specialize qscb linear output types, internalize fixed Qwen3.6
choices, and remove wildcard imports.

**Architecture:** Model recipes pass `DVec`/`DMat`/head/state views to backend
methods. Backend-private lowering validates and creates raw FFI descriptors at
the point of launch. qscb exposes the actual linear-layer contract rather than a
generic GEMM façade.

**Constraints:** No compatibility aliases, no generic-model abstraction, no
release stream synchronizes, no CMake, and no shared execution-context redesign
in this pass.

### Task 1: Make qscb linear semantics explicit

**Files:** `src/backend/cublas.rs`, `src/backend/tests.rs`, `src/model/mod.rs`,
`src/model/runner/mod.rs`, model runner submodules, `src/ffi/qscb.rs`, `qscb.h`,
`qscb.cu`, `bench_native.cu`, and `tests_cuda_qscb_linear.inc`

- [x] Add failing backend tests for BF16-output and F32-output linear lowering,
  including `[tokens, in] x [out, in] -> [tokens, out]` validation.
- [x] Replace `Bf16Gemm` and `gemm_bf16` with direct `linear_bf16` and
  `linear_f32` methods and a private raw-descriptor builder.
- [x] Rename the qscb native descriptor and entrypoint to linear terminology and
  delete the old names.
- [x] Split the model helper by output type and remove `GemmOut`.
- [x] Run backend and model compile/tests for the green phase.

### Task 2: Collapse qscu wrapper/args pairs

**Files:** `src/backend/cuda.rs`, `src/backend/tests.rs`, model runner submodules

- [x] Add failing tests against the desired private lowering contracts, starting
  with Qwen GDN gated RMSNorm and then each remaining qscu operation.
- [x] Change `Cuda` methods to accept typed operands directly and construct raw
  descriptors internally.
- [x] Remove all one-call public descriptor and `Args` types.
- [x] Update model and vector call sites without introducing forwarding helpers.
- [x] Run backend tests and the relevant vector tests for the green phase.

### Task 3: Internalize Qwen-fixed choices

**Files:** `src/backend/mod.rs`, `src/backend/cuda.rs`, `src/model/runner/gdn.rs`,
`src/model/runner/mod.rs`

- [x] Add failing tests proving Qwen GDN specializations reject non-Qwen shapes
  without accepting activation, scale, or normalization toggles.
- [x] Add only the native Qwen tensor-view constructors that remove duplicated
  shape arithmetic.
- [x] Fix SiLU gating/convolution, recurrence scale and normalization, and fixed
  dimensions inside the specialized backend operations.
- [x] Keep actual batch/sequence/state choices explicit.

### Task 4: Make module dependencies explicit

**Files:** `src/backend/*.rs`, `src/model/runner/*.rs`

- [x] Replace wildcard parent imports with explicit type, constant, and module
  imports.
- [x] Remove enums and shared aliases made obsolete by specialization.
- [x] Use `rg` to confirm removed wrappers, generic GEMM names, `GemmOut`, and
  wildcard imports have no remaining call sites.

### Task 5: Verify the replacement boundary

- [x] Run `cargo fmt --all -- --check`.
- [x] Run fast local compile/unit checks where CUDA linkage permits.
- [x] Run the repository's gitignored `./run_cuda_test.sh` in full after final review fixes.
- [x] Review the final diff specifically for wrapper reintroduction, constants
  leaking back into model calls, unrelated edits, and compatibility shims.
