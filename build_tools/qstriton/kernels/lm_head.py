# Adapted from b12x 75ffee6375b0577ce2c8d6931ffacefda3ecbdd6,
# b12x/gemm/bf16_vocab_projection/_kernel.py. Apache-2.0; see LICENSE.b12x.
# Changes: source parameter renamed activation; accumulation is a constexpr.
# The row reduction algorithm is unchanged.

import triton
import triton.language as tl


@triton.jit
def row_kernel(
    activation: tl.tensor,
    weight: tl.tensor,
    output: tl.tensor,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    values = tl.load(activation + offsets, mask=mask, other=0.0).to(ACC)
    weights = tl.load(
        weight + row * K + offsets,
        mask=mask,
        other=0.0,
    ).to(ACC)
    # ty cannot model Triton's JITFunction.__call__ self type for this reduction.
    tl.store(output + row, tl.sum(values * weights, axis=0))  # ty: ignore[invalid-argument-type]
