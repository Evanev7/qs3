# Triton JIT call/annotation types are not Python typing types.
# ty: ignore[invalid-argument-type]
import triton
import triton.language as tl


@triton.jit
def kernel(
    local_max: tl.tensor,
    local_ids: tl.tensor,
    output: tl.tensor,
    BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(local_max + offsets, offsets < BLOCKS, other=-float("inf"))
    ids = tl.load(local_ids + offsets, offsets < BLOCKS, other=0x7FFFFFFF)
    maximum = tl.max(values, axis=0)
    # Match argmax's lower-index tie break, including ties across blocks.
    winner = tl.min(tl.where(values == maximum, ids, 0x7FFFFFFF), axis=0)
    invalid = (tl.min(ids, axis=0) < 0) | (maximum == -float("inf"))
    tl.store(output, tl.where(invalid, -1, winner))
