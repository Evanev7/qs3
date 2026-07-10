# Rust Runtime Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic Rust runtime façade with explicit engine transaction, qscu, qscb, qsfi, Qwen model, and loader boundaries.

**Architecture:** `EngineCore` publishes one coherent active-batch view to an attention session, while a concrete `Engine` exposes the lifecycle without a one-implementation trait. Model execution borrows distinct qscu, qscb, and qsfi operator handles; model and loader source are split by ownership rather than by generic helpers.

**Tech Stack:** Rust 2024, CUDA runtime, qscu CUDA kernels, qscb/cuBLAS, qsfi/FlashInfer, safetensors files, `just`/Ninja remote CUDA test harness.

## Global Constraints

- No CMake.
- Do not preserve old Rust APIs or legacy call sites.
- Keep the runtime Qwen3.6-specific and reject unsupported shapes loudly.
- Validate descriptors before device addressing.
- Do not add release-mode stream synchronization for transactionality.
- Run final verification with `./run_cuda_test.sh`.

---

### Task 1: Concrete engine lifecycle and coherent batch view

**Files:**
- Modify: `src/engine.rs`
- Modify: `src/runtime.rs`
- Modify: `src/lib.rs`
- Modify: `src/model.rs`
- Modify: `tests/engine.rs`

**Interfaces:**
- Produces: inherent `Engine::{new,reset,release_requests,state,begin_append,append_layer,begin_decode,decode_layer,commit_batch,abort_batch}`.
- Produces: `EngineCore::active_batch(&self) -> Result<ActiveBatch<'_>, Status>`.
- Removes: `EngineTrait` and individual `batch_*` getters.

- [ ] **Step 1: Write compile-time and state-view tests**

Add an integration test that imports no trait and calls `Engine::new`, and add a
core unit test with the following assertions after `begin_append`:

```rust
let batch = core.active_batch().unwrap();
assert_eq!(batch.kind, BatchKind::Append);
assert_eq!(batch.request_ids, &[7]);
assert_eq!(batch.tokens, &[10, 11]);
assert_eq!(batch.qo_indptr, &[0, 2]);
assert_eq!(batch.kv_indptr.len(), 2);
assert_eq!(batch.last_page_len, &[2]);
assert_eq!(batch.append_batch_indices, &[0, 0]);
assert_eq!(batch.append_positions, &[0, 1]);
```

- [ ] **Step 2: Verify the tests fail**

Run: `LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib/stubs cargo test --lib engine::tests::active_batch_is_one_coherent_transaction_view`

Expected: compilation fails because `active_batch` and `ActiveBatch` do not exist.

- [ ] **Step 3: Add `ActiveBatch` and the concrete engine methods**

Define the borrowed view in `src/engine.rs`:

```rust
#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveBatch<'a> {
    pub kind: BatchKind,
    pub size: u32,
    pub token_count: u32,
    pub request_ids: &'a [RequestId],
    pub tokens: &'a [i32],
    pub qo_indptr: &'a [i32],
    pub kv_indptr: &'a [i32],
    pub kv_indices: &'a [i32],
    pub last_page_len: &'a [i32],
    pub rope_pos_offset: &'a [i32],
    pub append_batch_indices: &'a [i32],
    pub append_positions: &'a [i32],
}
```

Implement `active_batch` by borrowing the installed candidate and return
`InvalidArgument` when no transaction is active. Replace runtime getter chains
with one local `ActiveBatch`. Move every `EngineTrait` method body into `impl
Engine`, remove the trait and its re-export, and update callers/imports.

- [ ] **Step 4: Run engine tests**

Run: `LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib/stubs cargo test --lib engine::tests`

Expected: all engine unit tests pass.

- [ ] **Step 5: Commit the engine contract**

```bash
git add src/engine.rs src/runtime.rs src/lib.rs src/model.rs tests/engine.rs
git commit -m "refactor: make engine transaction contract concrete"
```

### Task 2: Explicit native backend operator handles

**Files:**
- Create: `src/backend/mod.rs`
- Create: `src/backend/tensor.rs`
- Create: `src/backend/cuda.rs`
- Create: `src/backend/cublas.rs`
- Create: `src/backend/flashinfer.rs`
- Modify: `src/lib.rs`
- Modify: `src/runtime.rs`
- Remove: `src/runtime/kernels.rs`
- Remove: `src/runtime/device_tensor.rs`
- Remove: `src/runtime/dtype.rs`

**Interfaces:**
- Produces: `Operators<'a>` with short-lived `cuda()`, `cublas()`, and `flashinfer()` borrows.
- Produces: backend-specific validated descriptor types colocated with their executor.
- Removes: `KernelOps` and `runtime::kernels`.

- [ ] **Step 1: Add backend classification tests**

Add module tests that construct `Operators` from a stream and mutable qscb/qsfi
contexts, then type-check representative descriptors against only their owning
handle:

```rust
fn cuda_owns_embedding(_: &cuda::Cuda<'_>, _: cuda::EmbeddingGatherBf16) {}
fn cublas_owns_gemm(_: &mut cublas::Cublas<'_>, _: cublas::Bf16Gemm) {}
fn flashinfer_owns_norm(_: &mut flashinfer::FlashInfer<'_>, _: flashinfer::RmsNormBf16) {}
```

Keep existing descriptor validation assertions when relocating tests.

- [ ] **Step 2: Verify the new module is absent**

Run: `cargo check --lib`

Expected: compilation fails after adding `mod backend` references because the
backend modules do not yet exist.

- [ ] **Step 3: Move typed tensors and descriptor builders by owner**

Move `DType`, `DVec`, `DMat`, `DTensor3`, `Bf16Heads`, and shared workspace/view
types to `backend/tensor.rs`. Move qscu descriptors and execution methods to
`backend/cuda.rs`, `Bf16Gemm` to `backend/cublas.rs`, and norm/RoPE/MoE
descriptors to `backend/flashinfer.rs`. Preserve constructor validation exactly.
Define the operator group without forwarding methods:

```rust
pub(crate) struct Operators<'a> {
    stream: &'a ffi::CudaStream,
    flashinfer: &'a mut qsfi::Context,
    cublas: &'a mut qscb::Context,
}
```

`EngineInner::operators` constructs this group by borrowing its stream, qscb
context, and qsfi context. Update every call site to select the real backend,
for example `ops.cublas().gemm_bf16(&desc)` and
`ops.flashinfer().rmsnorm_bf16(&desc)`.

- [ ] **Step 4: Remove the unified façade and run descriptor tests**

Delete `runtime::kernels`, its old files, and all imports. Run:

`LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib/stubs cargo test --lib backend`

Expected: backend unit tests pass and no `KernelOps` symbol remains (`rg KernelOps src` exits 1).

- [ ] **Step 5: Commit backend ownership**

```bash
git add src/backend src/lib.rs src/runtime.rs src/model.rs
git add -u src/runtime
git commit -m "refactor: expose explicit native backend operators"
```

### Task 3: Split Qwen model ownership

**Files:**
- Create: `src/model/mod.rs`
- Create: `src/model/config.rs`
- Create: `src/model/weights.rs`
- Create: `src/model/state.rs`
- Create: `src/model/scratch.rs`
- Create: `src/model/runner.rs`
- Create: `src/model/runner/attention.rs`
- Create: `src/model/runner/gdn.rs`
- Create: `src/model/runner/mlp.rs`
- Create: `src/model/tests.rs`
- Remove: `src/model.rs`

**Interfaces:**
- `config` produces validated `QwenConfig`, schedule indices, and `EngineConfig`.
- `weights` produces owning `QwenWeights` and immutable per-layer pointer views.
- `state` produces staged GDN slots with explicit `commit` and `reset`.
- `scratch` produces named typed views over grow-only buffers.
- `runner` is the only module that coordinates engine and GDN transactions.

- [ ] **Step 1: Pin module-level contracts with unit tests**

Relocate existing config, weight, slot-map, vector, and runner tests into their
owning modules. Add a state test that proves staging is not committed implicitly:

```rust
let mut slots = GdnSlotMap::new(3).unwrap();
let before = slots.layer_slots(0).unwrap();
slots.stage_for_batch(&[0], &[1]).unwrap();
assert_eq!(slots.layer_slots(0).unwrap(), before);
slots.commit();
assert_ne!(slots.layer_slots(0).unwrap(), before);
```

- [ ] **Step 2: Move configuration and weights first**

Create `config.rs` with Qwen shape/schedule policy and `weights.rs` with device
ownership and pointer views. Export only:

```rust
pub use config::{QwenConfig, QwenMoeConfig};
pub use runner::{ModelRunner, QwenRequest, QwenResult};
pub use weights::QwenWeights;
```

Use `pub(super)` only for contracts required by sibling model modules; keep fields
private when a typed accessor can express the invariant.

- [ ] **Step 3: Move state and scratch ownership**

Move `GdnSlotMap`/`GdnState` to `state.rs` and `DeviceBuffer`/`RunnerScratch` to
`scratch.rs`. Give scratch named view methods so runner modules do not calculate
byte offsets. Retain device/stream binding validation and grow-only allocation.

- [ ] **Step 4: Split runner execution by Qwen block**

Move prefix synchronization and transaction coordination to `runner.rs`.
`attention.rs`, `gdn.rs`, and `mlp.rs` receive `&mut Operators`, typed weight
views, and typed scratch/state views; they do not own requests or commit batches.
Delete the old monolithic file rather than including it textually.

- [ ] **Step 5: Run all model tests**

Run: `LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib/stubs cargo test --lib model`

Expected: model unit tests pass, including vector tests on a CUDA host.

- [ ] **Step 6: Commit model ownership**

```bash
git add src/model src/lib.rs src/weight_loader
git add -u src/model.rs
git commit -m "refactor: split qwen model by ownership"
```

### Task 4: Separate loader format validation from transfer

**Files:**
- Create: `src/loader/mod.rs`
- Create: `src/loader/error.rs`
- Create: `src/loader/format.rs`
- Create: `src/loader/transfer.rs`
- Create: `src/loader/plan.rs`
- Create: `src/loader/tests.rs`
- Modify: `src/lib.rs`
- Modify: `src/model/config.rs`
- Remove: `src/weight_loader.rs`
- Remove: `src/weight_loader/materialize.rs`

**Interfaces:**
- `format::validate_qwen36_bf16(model_dir) -> LoadResult<ValidatedQwen36>` performs no CUDA allocation.
- `plan::QwenBf16LoadPlan<B: WeightLoadBackend>` owns the backend until materialization handoff.
- `transfer::WeightLoadBackend` owns allocation, I/O, seal, and cleanup only.

- [ ] **Step 1: Add an allocation-free validation test**

Use a fake transfer backend with an allocation counter. Validate malformed and
valid temporary safetensors metadata through `format` and assert the counter stays
zero. Retain tests for duplicate, missing, unexpected, dtype, shape, overlap, and
out-of-range failures.

- [ ] **Step 2: Verify format tests fail before the split**

Run: `cargo test --lib loader::tests::format_validation_never_allocates`

Expected: compilation fails because `loader::format` does not exist.

- [ ] **Step 3: Move parsing and validation into `format`**

Move JSON helpers, config parsing, safetensors index/header parsing, span checks,
and Qwen tensor-table validation without importing `ffi::cuda`. The output owns
all validated paths, offsets, dtypes, shapes, and tensor targets needed by a plan.

- [ ] **Step 4: Move transfer backends and materialization**

Move the trait, managed UMA backend, pinned ring backend, and their stats into
`transfer.rs`. Move plan construction and `into_qwen_model` into `plan.rs`.
Ensure backend `Drop` still frees all allocations before handoff and the final
allocation list is drained exactly once into `QwenWeights`.

- [ ] **Step 5: Run loader tests**

Run: `LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib/stubs cargo test --lib loader`

Expected: all non-ignored loader tests pass.

- [ ] **Step 6: Commit loader ownership**

```bash
git add src/loader src/lib.rs src/model
git add -u src/weight_loader.rs src/weight_loader/materialize.rs
git commit -m "refactor: separate weight format and transfer layers"
```

### Task 5: Architectural cleanup and end-to-end verification

**Files:**
- Modify: `src/bin/qs3_model_bench.rs`
- Modify: `tests/engine.rs`
- Modify: `tests/model.rs`
- Modify: `AGENTS.md` only if a new durable fact is discovered during verification.

**Interfaces:**
- Removes all legacy symbol paths and compatibility shims.
- Keeps public entrypoints limited to the concrete engine and Qwen model API.

- [ ] **Step 1: Remove stale architecture and compatibility symbols**

Run:

```bash
rg 'EngineTrait|KernelOps|runtime::kernels|weight_loader' src tests
```

Expected: no matches. Update tests and the benchmark to the replacement API;
delete stale imports rather than re-exporting aliases.

- [ ] **Step 2: Format and run static checks**

Run: `cargo fmt --all -- --check`

Expected: pass.

Run: `git diff --check`

Expected: pass.

- [ ] **Step 3: Run the repository test script**

Run: `./run_cuda_test.sh`

Expected: remote Ninja build, Rust library build, vector generation, all Rust
tests, checked native tests, and release native tests pass.

- [ ] **Step 4: Review architecture against the design**

Check that qscu/qscb/qsfi operations have exactly one owning Rust module, engine
transactions remain candidate-first, loader validation precedes allocation, and
no release synchronization was introduced. Fix every finding and rerun Steps 1-3.

- [ ] **Step 5: Commit final cleanup**

```bash
git add src tests docs/superpowers
git commit -m "refactor: clarify qwen runtime architecture"
```
