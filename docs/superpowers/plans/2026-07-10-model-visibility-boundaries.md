# Model Visibility Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove accidental model-wide visibility while retaining narrow parent interfaces for runner recipe modules.

**Architecture:** `ModelRunner` owns private state and private orchestration helpers. `runner::{attention,gdn,mlp}` expose only the operations invoked by their parent as `pub(super)`; helpers used within one recipe stay private. White-box tests become descendants of `runner` so test placement does not widen production visibility.

**Tech Stack:** Rust module privacy, existing CUDA-backed model tests.

## Global Constraints

- A scoped visibility must have a concrete consumer in the named parent module.
- Use private items when descendants can already access them.
- Use `pub(super)` only for a child-to-parent runner operation.
- Do not add test-only production accessors.
- Preserve the existing test behavior and run verification with `./run_cuda_test.sh`.

---

### Task 1: Prove the current tests force model-wide visibility

**Files:**
- Modify: `src/model/runner/mod.rs`

**Interfaces:**
- Consumes: existing `model::tests` white-box accesses.
- Produces: compiler evidence identifying sibling-test dependencies on runner internals.

- [x] **Step 1: Make `ModelRunner` fields and parent-defined helpers private**

  Remove `pub(in crate::model)` from the twelve fields and twenty parent-defined helper methods without changing their signatures.

- [x] **Step 2: Run `cargo check --tests` and verify the red phase**

  Expected: privacy errors originate from `model::tests` and from parent calls into child recipe modules.

### Task 2: Put white-box tests under their owner

**Files:**
- Move: `src/model/tests.rs` to `src/model/runner/tests.rs`
- Modify: `src/model/mod.rs`
- Modify: `src/model/runner/mod.rs`

**Interfaces:**
- Consumes: existing model and runner tests unchanged in behavior.
- Produces: `runner::tests`, which can inspect private `ModelRunner` state through descendant privacy.

- [x] **Step 1: Move the test module declaration from `model` to `model::runner`**

  Declare `#[cfg(test)] mod tests;` in `runner/mod.rs` and remove it from `model/mod.rs`.

- [x] **Step 2: Update test imports for the new parent module**

  Import model-level helpers through `crate::model` and runner-owned types through `super`; do not add production visibility.

- [x] **Step 3: Run `cargo check --tests`**

  Expected: field and parent-helper privacy errors disappear; only genuine child-to-parent method visibility remains.

### Task 3: Narrow child recipe interfaces

**Files:**
- Modify: `src/model/runner/attention.rs`
- Modify: `src/model/runner/gdn.rs`
- Modify: `src/model/runner/mlp.rs`

**Interfaces:**
- Consumes: private `ModelRunner` state and parent helper methods.
- Produces: `pub(super)` recipe entrypoints used by `runner/mod.rs`; private operation-local helpers.

- [x] **Step 1: Make every child method private and run `cargo check --tests`**

  Expected: the compiler identifies the attention, GDN, and post-attention MLP entrypoints called by the parent or sibling runner tests.

- [x] **Step 2: Restore `pub(super)` only on methods with demonstrated parent/sibling consumers**

  Keep all helpers used solely inside their defining child module private.

- [x] **Step 3: Run `cargo check --tests`**

  Expected: clean compilation with no `pub(in crate::model)` declarations.

### Task 4: Verify visibility and behavior

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-model-visibility-boundaries.md`

**Interfaces:**
- Consumes: narrowed model module graph.
- Produces: structural and behavioral verification evidence.

- [x] **Step 1: Run structural scans**

  Run `rg -n 'pub\\(in ' src` and inspect every remaining match; the model and backend refactor should contribute none.

- [x] **Step 2: Run local checks**

  Run `cargo fmt --all -- --check`, `cargo check --tests`, and `git diff --check` with no errors or warnings.

- [x] **Step 3: Run `./run_cuda_test.sh`**

  Expected: all enabled Rust, integration, vector, checked-native, and release-native tests pass.

- [x] **Step 4: Mark the completed checklist and review the final diff**

  Confirm every remaining restricted visibility has a named production consumer rather than a test-only justification.
