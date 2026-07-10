# Rust Runtime Architecture Design

## Goal

Reshape the completed Qwen3.6 end-to-end Rust path around ownership and backend
contracts. The result should make it obvious that FlashInfer owns paged attention,
MoE, norm, and RoPE integration; qscu owns small CUDA kernels; qscb owns cuBLAS
GEMM; the engine owns transactional request/KV-cache state; and the model runner
owns Qwen3.6 execution.

This is an API replacement. There is no compatibility layer for the current
`EngineTrait`, monolithic `KernelOps`, or their call sites.

## Chosen architecture

The crate has four internal layers, ordered from policy to mechanism:

1. `model` contains Qwen3.6 configuration, weights, persistent recurrent state,
   scratch storage, and the execution runner. It is the only layer that knows the
   decoder schedule or composes complete Qwen blocks.
2. `engine` contains a pure transactional `EngineCore` and an attention session.
   `EngineCore` owns request IDs, tokens, pages, staged batches, commit, abort,
   reset, and release. The attention session owns device batch metadata,
   FlashInfer plans, and per-layer paged K/V buffers.
3. `backend` contains typed device views and three explicit operator handles:
   `Cuda<'a>` for qscu, `Cublas<'a>` for qscb, and `FlashInfer<'a>` for qsfi.
   Descriptor builders live with the backend that consumes them. `Operators<'a>`
   owns the shared borrows and creates one short-lived backend handle at a time,
   so qscu operations that use the qsfi error/context object cannot alias a
   simultaneous FlashInfer borrow. It does not forward backend methods.
4. `ffi` is the thin unsafe binding layer. It performs native descriptor checks
   that protect device addressing and translates native status values, but it
   contains no model scheduling policy.

The weight loader is a peer construction path rather than an execution layer. It
has separate `format` validation and `transfer` backends, then consumes a fully
validated plan into `QwenWeights`.

## Engine contract

`Engine` becomes a concrete type with inherent methods. Removing `EngineTrait`
eliminates an abstraction with one implementation and makes the lifecycle
protocol discoverable on the owning type.

The public lifecycle remains explicit:

- `begin_append` / `begin_decode` construct candidate host state, upload device
  metadata, and create or reuse a FlashInfer plan.
- `attention` accepts a typed Qwen attention layer operation for the active
  batch. It validates all shapes before K/V addressing, appends K/V, executes
  attention, and advances layer progress.
- `commit_batch` installs the candidate only after all required attention layers
  completed. `abort_batch` discards the candidate. Release-mode commits do not
  add a stream synchronization.

`EngineCore` stays independent of CUDA and FlashInfer. Its batch-facing data is
published as a single borrowed `ActiveBatch` view instead of a collection of
loosely related getters. The attention session consumes that view and owns all
device mirrors.

## Backend contracts

The current `KernelOps` obscures which implementation runs an operation. It is
replaced by:

- `Cuda<'a>`: qscu embedding, activation, routing, sampling, output gating, and
  Qwen3.6 GDN preparation/recurrence launches. It borrows the stream and, for
  the current qscu GDN ABI, the qsfi context used for stream/error state.
- `Cublas<'a>`: qscb BF16 GEMM. It owns a mutable qscb context borrow.
- `FlashInfer<'a>`: qsfi RMSNorm, fused add/RMSNorm, RoPE, MoE planning/execution,
  and any FlashInfer-owned GDN operation. It owns a mutable qsfi context borrow.

Each operation is represented by a validated backend-specific descriptor type.
Construction is safe and rejects bad shapes, strides, pointers, dtypes, or modes.
Execution is unsafe because device pointer validity and asynchronous lifetime
remain caller contracts. There is no generic kernel trait and no forwarding
method whose only job is to make a call site read fluently.

FlashInfer attention planning remains inside the engine attention session because
its cache and plan lifetime are part of the engine transaction. The general
FlashInfer operator handle is for model-level primitives and MoE.

## Model boundaries

`model` is decomposed into modules with concrete ownership roles:

- `config`: public `QwenConfig`/`QwenMoeConfig` plus the fixed Qwen3.6 schedule
  and shape validation.
- `weights`: `QwenWeights`, layer weight ownership, and immutable bound pointer
  views. Random fixtures live under test support rather than the production
  weight representation.
- `state`: GDN slot mapping and persistent recurrent buffers. State exposes
  staging and commit/abort semantics matching the attention engine.
- `scratch`: grow-only device buffers and the named scratch layout.
- `runner`: public request/result types, exact-prefix synchronization, batch
  execution, and layer orchestration.
- `runner/attention`, `runner/gdn`, and `runner/mlp`: Qwen block operations. These
  modules consume typed weights, scratch views, and explicit backend handles.

The runner is the transaction coordinator. A batch is committed only when both
attention-engine state and GDN staged state can advance. On a deterministic
validation failure it aborts both. Once a native mutation launch succeeds,
existing non-rollbackable device semantics remain explicit; the runner must
rebuild or overwrite the affected prefix before reuse.

## Loader boundaries

Loader work is split into:

- `loader/format`: JSON helpers, Qwen3.6 text config, safetensors index/header
  parsing, tensor-table construction, and complete pre-allocation validation.
- `loader/transfer`: `WeightLoadBackend`, managed-UMA and pinned-upload
  implementations, allocation ownership, I/O, sealing, and cleanup.
- `loader/plan`: validated Qwen BF16 plan and materialization into weights.

`format` cannot allocate CUDA memory. `plan` can only be built from a completely
validated format description. Materialization owns its transfer backend until
the final allocation list is drained into device buffers, preserving cleanup on
all pre-handoff failures.

## Error and validation policy

Public errors continue to use the compact `Status` enum. Internal construction
uses `Result` and checked arithmetic. Contract violations are rejected before
native device addressing. Debug-only allocator invariants and debug stream
completion checks remain debug-only. Unsupported non-Qwen3.6 shapes fail loudly;
there is no generic model/runtime compatibility layer.

## Testing

Existing public CUDA integration and vector tests remain the behavioral baseline.
New or relocated unit tests cover:

- the concrete engine lifecycle after `EngineTrait` removal;
- `ActiveBatch` as a coherent transactional view;
- correct classification of every descriptor and operation under qscu, qscb, or
  qsfi;
- Qwen layer schedule, weight binding, GDN staging/commit, and scratch growth in
  their owning modules;
- loader format validation without CUDA allocation and backend handoff cleanup.

Verification uses `./run_cuda_test.sh`, the repository's gitignored remote test
script. Formatting and a local `cargo check` may be used as fast feedback, but do
not replace that script.

## Scope limits

This refactor does not add NVFP4, generic model support, new native kernels,
release-mode stream synchronization, CMake, ABI/versioning machinery, or legacy
entrypoints. Native CUDA source organization changes only where necessary to
keep Rust backend ownership truthful.
