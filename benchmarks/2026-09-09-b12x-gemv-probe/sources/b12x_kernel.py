# Copied from b12x 75ffee6375b0577ce2c8d6931ffacefda3ecbdd6
# b12x/gemm/bf16_vocab_projection/_kernel.py; Apache-2.0, see LICENSE.b12x.
# Only the Torch import and host wrappers are omitted. Kernel bodies are unchanged.
"""Triton kernels for single-row BF16 vocabulary projection."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _row_kernel(
    source,
    weight,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    values = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(
        weight + row * K + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output + row, tl.sum(values * weights, axis=0))


@triton.jit
def _row_loop_kernel(
    source,
    weight,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((), tl.float32)
    for start in range(0, K, BLOCK_K):
        positions = start + offsets
        mask = positions < K
        values = tl.load(source + positions, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(
            weight + row * K + positions,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values * weights, axis=0)
    tl.store(output + row, accumulator)
