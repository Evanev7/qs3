# Native Operation Boundaries Design

## Goal

Make the boundary between the Qwen3.6 model recipe and its native backends
visible in the Rust types. Model code should pass typed device tensors directly;
qscu, qscb, and qsfi should construct their raw FFI descriptors at the launch
site. Fixed Qwen3.6 choices should not travel through generic-looking argument
objects.

This replaces the current one-call Rust descriptor wrappers. No compatibility
aliases or deprecated entrypoints are retained.

## Boundary rule

The model layer composes operations. It owns the sequence of projections,
normalization, recurrence, attention, and residual work. Cohesive functions such
as `execute_gdn_layer` remain cohesive even when they are long.

The backend layer accepts native Rust device views such as `DVec`, `DMat`,
`Bf16Heads`, recurrent state views, and the few enums that represent genuine
runtime choices. A backend method validates those views, lowers them into the
corresponding raw FFI descriptor, and launches it immediately. Raw descriptors
do not escape the backend module.

This removes chains such as:

```text
GdnRmsNormGatedBf16Args
  -> GdnRmsNormGatedBf16
  -> qscu_gdn_rmsnorm_gated_bf16_desc
  -> native launch
```

and replaces them with:

```text
typed DVec/DMat/head views
  -> Cuda::qwen36_gdn_gated_rmsnorm
  -> raw qscu descriptor + native launch
```

The unsafe boundary stays on execution methods because device allocation,
asynchronous lifetime, and aliasing remain caller obligations. Shape, stride,
mode, and finite-value validation remains safe Rust logic and is tested without
launching CUDA.

## qscb is a linear backend

The operation currently called `gemm_bf16` is not a general GEMM interface. Its
model contract is the linear-layer calculation:

```text
input [tokens, in_features]
  x weight^T [out_features, in_features]
  -> output [tokens, out_features]
```

qscb therefore exposes this as a linear operation. Rust has two explicit
specializations:

- `linear_bf16`: BF16 input and weight, BF16 output.
- `linear_f32`: BF16 input and weight, F32 output.

Both take `DMat` operands and a `Workspace` directly. The output type is encoded
by the method signature, so `GemmOut`, `Bf16OrF32Mat` at model call sites, and a
public `Bf16Gemm` wrapper disappear. Alpha and beta are not model-level options;
the Qwen path uses the native defaults currently encoded by the wrapper.

The qscb C entrypoint and descriptor are renamed from generic GEMM terminology
to linear terminology. The old names are removed.

## Qwen-specific qscu operations

qscu remains a thin collection of CUDA implementations, but its Rust entrypoints
state which choices belong to Qwen3.6:

- GDN causal convolution fixes SiLU activation and the Qwen packed width.
- GDN post-convolution preparation fixes the currently used log-decay output and
  leaves only actual optional outputs or modes visible.
- GDN recurrence fixes the Qwen scale, Q/K normalization, and state-update mode.
- GDN gated RMSNorm fixes SiLU gating and the Qwen value dimension.
- Full-attention output gating and shared-expert combination keep their
  Qwen-specific names and accept matrices directly.

The fixed dimensions are checked inside these methods using the canonical Qwen
constants. Call sites still construct tensor views with their true shapes, but
they do not pass the same constants again as apparent operation configuration.
Where a useful Qwen view constructor eliminates repeated shape arithmetic, it
lives next to the view type and returns that native view rather than another
descriptor wrapper.

Generic small CUDA operations such as embedding gather, SiLU-and-multiply,
logit soft cap, greedy argmax, and router top-k also take their typed operands
directly. They retain only genuine runtime parameters.

## Imports and module readability

Backend and runner modules use explicit imports. Wildcard parent imports are
removed so each source file shows its dependencies and the constants it owns.
This is a readability constraint, not a reason to create forwarding helpers.

## Scoped visibility must describe a real boundary

Private is the default for runner state and helpers. A restricted visibility
such as `pub(super)` is appropriate only when a child module deliberately
supplies an operation to its parent—for example, `runner::attention` supplying
the attention-layer recipe used by `runner` orchestration. Internal helper
methods within that child remain private.

`pub(in crate::model)` is not used merely so sibling tests can inspect runner
state or invoke intermediate stages. White-box runner tests live below the
`runner` module that owns those details. This lets `ModelRunner` fields and
parent-defined helpers remain private while preserving direct tests for tensor
recipes. A future `pub(in ...)` declaration must identify a consumer that
cannot be represented by ordinary private descendant access or a genuine
parent interface.

## Backend modules are the FFI boundaries

The separate `ffi::{qscu,qscb,qsfi}` facade does not own a useful independent
responsibility after typed operation lowering moved into `backend`. The backend
modules therefore take the names of the native libraries and become the sole
Rust/native boundaries:

```text
model and runtime
  -> backend::{qscu,qscb,qsfi} typed interfaces
  -> raw ffi::sys declarations
  -> native ABI
```

`backend::qscu` owns typed CUDA-kernel operations, descriptor construction, and
their raw launches. `backend::qscb` owns typed linear operations and the qscb
context lifetime. `backend::qsfi` owns FlashInfer operations, plans, context
lifetime, and raw launches. Generated bindings, raw ABI aliases and constants,
and direct CUDA Runtime declarations remain in a raw-only `ffi` namespace. It
owns no contexts, plans, status conversion, descriptor construction, or
forwarding qsc calls. No `ffi::{qscu,qscb,qsfi}` facade or compatibility
re-export is retained.

The Rust handle names follow the native components: `Qscu`, `Qscb`, and `Qsfi`.
This naming describes concrete native ownership rather than implying pluggable
CUDA, cuBLAS, or FlashInfer implementations.

## Context ownership

`Operators` remains the shared borrowing context for the CUDA stream, qscb
context, and qsfi context. This pass changes the concrete handle and module
names without changing the runtime's ownership graph. qscu continues borrowing
the qsfi context for FlashInfer-owned GDN launches; moving ownership into a
single long-lived context can be evaluated separately.

## Testing

Backend unit tests exercise private lowering functions with fake non-null device
pointers. They verify operation semantics, output specialization, Qwen-fixed
shape constraints, strides, and modes without launching CUDA. Existing vector
and end-to-end tests exercise the real launch methods.

Final verification uses `./run_cuda_test.sh`.
