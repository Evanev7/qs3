# AOT kernels

`models/config.nix` selects the specializations. `just ninja` writes the build
edges; Ninja runs `qstriton` to produce cubins and Rust launchers. Inference uses
neither Python nor JIT compilation.

## GDN prefill

`gdn_prefill_*.py` adapt the FLA kernels and fused post-convolution preparation
from vLLM commit `98dff2a81d747d1dba01a47f939f48c3526d4206`:
`vllm/third_party/flash_linear_attention/ops`, including its GDN preparation kernel.
See `LICENSE.vllm` and the source headers. The seven stages preserve the upstream
64-token arithmetic and BF16 intermediates. Autotuning and unused TMA code are
removed. Fixed algorithm choices live in the kernel sources;
`models/gdn_prefill.nix` lists each stage's model geometry, storage types, and
compiler options explicitly.

`ModelRunner` owns the loaded prefill modules. Its scratch contains only the
reusable buffers and their capacity; the backend receives that storage when it
validates and launches the stages in `src/backend/gdn_prefill.rs`.
Prefill reads the live recurrent state and writes
the staged state; decode retains its existing recurrent kernel. These internal
64-token chunks are independent of any future scheduler prefill chunk size.

## Sampling

Configure a runner before starting a request (or after reset/release):

```rust
runner.set_sampling(SamplingParams {
    temperature: 0.8,
    top_k: 20,
    top_p: 0.95,
    seed: 42,
})?;
```

The default is greedy. Temperature zero uses the existing greedy kernel and
ignores filtering. Positive temperature divides logits, applies top-k, applies
top-p to the renormalized survivors, then samples with Gumbel-Max. Top-k zero and
top-p one disable those filters. Top-p must be in `(0, 1]`; top-k cannot exceed
the vocabulary. Boundary ties favor lower token IDs. Invalid logits, scaling
overflow, an empty candidate set, or a sampled padded tokenizer ID fail the run.
Negative infinity masks individual logits. Original prediction logits are kept.

`sampling_gumbel.py` and `sampling_filter.py` adapt vLLM commit
`51a99565c398c8320de8131e07731c75c52eb87c`, respectively
`vllm/v1/worker/gpu/sample/gumbel.py` and
`vllm/v1/sample/ops/topk_topp_triton.py`. See `LICENSE.vllm` and the source headers.
The filter's uniform-row fallback is tightened to honor top-k/top-p counts.
The Gumbel transform uses the upstream FP32 `log1p(-u)` tail correction.

Rust owns persistent buffers and module lifetimes. The seed and absolute input
position determine the noise; reset, rebuild, or retry repeats a draw at the same
position. Request IDs do not change the seed. Changing settings requires ending
the live request first. The sampler reads position from device memory, and its
launch sequence is tested under CUDA graph replay with changing positions.

The runner still downloads the selected ID at its existing completion point.
Connecting that device ID directly to subsequent graph-based model decode is
separate runner work. Core benchmarks remain greedy.
