"""Custom qs3 kernel; not adapted from an upstream implementation.

Reduce the two FP32 decode partials with a single BF16 rounding.
"""

import triton as tr
import triton.language as tl


@tr.jit
def kernel(partials, output, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first = tl.load(partials + i, i < N, 0)
    second = tl.load(partials + N + i, i < N, 0)
    tl.store(output + i, first + second, i < N)
