# Exact 27B GDN geometry probe

Source `4a6eb30`, sp10 GB10, CUDA 13. The disposable AOT build changes the native
GDN value-head constant from 32 to 48 and tests 16 Q/K heads, 128-dimensional
heads, width-four convolution over 10240 channels, and 6144 output channels.
Production validation remains unchanged by this standalone probe.

Both checked and release builds pass:

- Analytic one-token decode and two-token prefill recurrence, BF16 and FP32 state.
  The active value head is 47, mapping to Q/K head 15 with group size three.
  Every output/state element is checked, including untouched zeros.
- CPU-reference causal convolution, post-convolution Q/K/V extraction,
  normalization and decay/beta materialization, and gated RMSNorm.
- Host descriptor rejection; checked builds also reject invalid state indices,
  sequence pointers and convolution metadata before unsafe device addressing.

`probe.py` builds and runs from a normal checked-out native build. It copies
source into `.prototypes/qwen27_gdn/` and applies only the specified geometry and
test-state adaptations. No runtime Python or JIT is introduced into qs3.
An initial FP32 adapter incorrectly kept a BF16 state descriptor; correcting that
descriptor made both storage tests pass. Rust views, scratch/state allocation,
production dispatch and full 27B inference still require integration.
