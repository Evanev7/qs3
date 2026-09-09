"""Single-token Qwen BF16 projection; compiled by compile.py, never imported at inference."""
import triton
import triton.language as tl


@triton.jit
def gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, K)
    x = tl.load(X + cols).to(tl.float32)
    w = tl.load(W + rows[:, None] * K + cols[None, :],
                mask=rows[:, None] < N, other=0).to(tl.float32)
    y = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rows, y, mask=rows < N)
