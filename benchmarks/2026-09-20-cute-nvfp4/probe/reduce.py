"""Prototype-only FP32 split-two reduction; scheduling belongs to the harness."""
import triton as tr
import triton.language as tl

@tr.jit
def kernel(partials, output, total: tl.int32, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(partials + i, i < total, 0)
    b = tl.load(partials + total + i, i < total, 0)
    tl.store(output + i, a + b, i < total)
