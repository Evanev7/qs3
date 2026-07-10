# Backend FFI Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make `backend::qscu`, `backend::qscb`, and `backend::qsfi` the sole typed Rust/native boundaries and reduce `ffi` to raw declarations.

**Architecture:** `ffi` contains generated raw bindings, ABI aliases/constants, and direct CUDA Runtime declarations only. Each named backend module owns its native context or borrowed launch state, typed validation and descriptor lowering, status conversion, and direct ABI calls. Model and runtime consumers use backend qsc types rather than forwarding FFI facades.

**Tech Stack:** Rust, bindgen-generated C ABI declarations, CUDA Runtime API, qscu, qscb, qsfi/FlashInfer.

## Global Constraints

- Do not retain compatibility modules, aliases, re-exports, or legacy handle names.
- Keep Qwen3.6 tensor validation in the typed backend methods.
- Preserve existing asynchronous stream behavior and context ownership.
- Run tests through the gitignored `./run_cuda_test.sh`.
- Do not introduce CMake.

---

### Task 1: Establish the module contract

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/backend/mod.rs`

**Interfaces:**
- Consumes: existing `backend::{Cublas, Cuda, FlashInfer}` compile-contract checks.
- Produces: compile-contract checks for `backend::qscu::Qscu`, `backend::qscb::Qscb`, and `backend::qsfi::Qsfi`; no `ffi::{qscu,qscb,qsfi}` modules.

- [x] **Step 1: Change the compile-contract test to import the desired modules and handle names**

  Make the contract refer to `qscu::Qscu`, `qscb::Qscb`, and `qsfi::Qsfi` before those names exist.

- [x] **Step 2: Run `cargo check --tests` and verify the red phase**

  Expected: unresolved imports or types for the desired backend modules and handles.

- [x] **Step 3: Declare the three backend modules without compatibility aliases**

  Rename the current backend files to `qscu.rs`, `qscb.rs`, and `qsfi.rs`, export only `Qscu`, `Qscb`, and `Qsfi`, and update `Operators` accessors to the same names.

- [x] **Step 4: Run `cargo check --tests`**

  Expected: compilation advances to unresolved raw-FFI ownership, proving the public module contract is installed.

### Task 2: Move qscb into its backend boundary

**Files:**
- Modify: `src/backend/qscb.rs`
- Modify: `src/backend/mod.rs`
- Delete: `src/ffi/qscb.rs`

**Interfaces:**
- Consumes: `DMat<BF16>`, `DMat<F32>`, `Workspace`, device ordinal, and CUDA stream.
- Produces: `Qscb::new`, `Qscb::linear_bf16`, and `Qscb::linear_f32`, with private qscb context creation, destruction, last-error handling, and direct `sys::qscb_linear` invocation.

- [x] **Step 1: Move qscb context RAII into `Qscb`**

  Store `NonNull<sys::qscb_context>` directly in `Qscb`; create it from `sys::qscb_context_desc` and destroy it in `Drop`.

- [x] **Step 2: Lower typed operands and invoke `sys::qscb_linear` directly**

  Keep the existing linear shape checks and output specializations, then convert raw status with the shared private status helper.

- [x] **Step 3: Update runtime construction and run `cargo check --tests`**

  Expected: qscb consumers compile without `ffi::qscb`.

### Task 3: Move qscu into its backend boundary

**Files:**
- Modify: `src/backend/qscu.rs`
- Modify: `src/backend/tests.rs`
- Delete: `src/ffi/qscu.rs`

**Interfaces:**
- Consumes: typed tensor/state views, a borrowed CUDA stream, and a borrowed mutable `Qsfi` context where GDN requires it.
- Produces: `Qscu` methods that validate, construct private `sys::qscu_*_desc` values, and directly invoke every qscu ABI function.

- [x] **Step 1: Replace qscu facade descriptor aliases with private `sys` types**

  Descriptor builders return the corresponding generated `sys::qscu_*_desc` type and use raw generated constants internally.

- [x] **Step 2: Replace forwarding calls with direct ABI calls**

  Every launch converts `sys::qscu_*` status through the shared private helper; GDN calls use `Qsfi`'s crate-private raw-context accessor.

- [x] **Step 3: Update backend tests and run `cargo check --tests`**

  Expected: qscu descriptor tests and consumers compile without `ffi::qscu`.

### Task 4: Move qsfi into its backend boundary

**Files:**
- Modify: `src/backend/qsfi.rs`
- Modify: `src/engine/attention.rs`
- Modify: `src/backend/tests.rs`
- Delete: `src/ffi/qsfi.rs`

**Interfaces:**
- Consumes: device ordinal, CUDA stream, attention/page-table descriptors, and typed operation descriptors.
- Produces: `Qsfi`, attention `Plan`, `MoePlan`, plan RAII, native error reporting, and direct qsfi ABI calls.

- [x] **Step 1: Combine the existing typed FlashInfer operations with qsfi context and plan ownership**

  Keep context/plan `Drop` implementations adjacent to the direct ABI calls and retain descriptor validation next to typed constructors.

- [x] **Step 2: Rename `Context` to `Qsfi` and remove the temporary wrapper handle**

  Operation methods execute directly on `Qsfi`; `Operators::qsfi()` returns a mutable borrow of that context rather than constructing another facade.

- [x] **Step 3: Update attention planning/execution consumers and run `cargo check --tests`**

  Expected: attention, model, and backend tests compile without `ffi::qsfi`.

### Task 5: Reduce ffi to raw ABI declarations

**Files:**
- Modify: `src/backend/dtype.rs`
- Modify: `src/backend/tensor.rs`
- Modify: `src/backend/mod.rs`
- Modify: `src/ffi/mod.rs`

**Interfaces:**
- Consumes: bindgen output from `OUT_DIR`, direct CUDA Runtime declarations, and native status codes.
- Produces: a raw-only `ffi` namespace with no qsc contexts, plans, status conversion, descriptor construction, or forwarding calls.

- [x] **Step 1: Expose generated bindings crate-internally to backend modules**

  Keep `ffi_bindings.rs` included only in `ffi::sys`; backend qsc modules call those generated symbols directly.

- [x] **Step 2: Remove status conversion and forwarding responsibilities from ffi**

  Move qsc status conversion into backend and retain `ffi::cuda` solely as direct CUDA Runtime declarations used by allocation and transfer code.

- [x] **Step 3: Delete the three qsc facade files**

  Delete `src/ffi/qscu.rs`, `src/ffi/qscb.rs`, and `src/ffi/qsfi.rs`. Run `rg -n 'ffi::(qscu|qscb|qsfi)' src` and expect no matches.

- [x] **Step 4: Run `cargo fmt --all` and `cargo check --tests`**

  Expected: clean formatting and compilation with no warnings.

### Task 6: Verify the replacement boundary

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-backend-ffi-consolidation.md`

**Interfaces:**
- Consumes: completed consolidated backend.
- Produces: evidence that Rust, native CUDA, end-to-end, and benchmark paths still work.

- [x] **Step 1: Run `git diff --check` and scan for legacy names**

  Run `rg -n '\b(Cublas|Cuda|FlashInfer)\b|ffi::(qscu|qscb|qsfi)' src` and expect no legacy qsc facade matches.

- [x] **Step 2: Run `./run_cuda_test.sh`**

  Expected: all currently enabled Rust, integration, vector, checked-native, and release-native tests pass.

- [x] **Step 3: Build the native benchmark target**

  Run the repository's existing remote benchmark build command and expect `qsfi_bench_native` to link successfully.

- [x] **Step 4: Mark every completed plan checkbox and review the final diff**

  Confirm that no compatibility facade was introduced and each backend module directly owns its ABI calls.
