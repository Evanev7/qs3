# Triton JIT call/annotation types are not Python typing types.
# ty: ignore[invalid-type-form]
import triton
import triton.language as tl


@triton.jit
def kernel(
    logits: tl.tensor,
    output: tl.tensor,
    vocab: tl.int32,
    temperature: tl.float32,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + offsets, offsets < vocab, other=-float("inf"))
    scaled = values / temperature
    # Keep invalid inputs out of the pivot search. Gumbel checks the original
    # row (including scaling overflow) and returns an error sentinel afterward.
    scaled = tl.where(scaled < float("inf"), scaled, -float("inf"))
    tl.store(output + offsets, scaled, offsets < VOCAB)
