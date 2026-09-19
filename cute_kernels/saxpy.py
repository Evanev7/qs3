"""AOT smoke kernel: runtime length/alpha and compile-time launch width."""

import cutlass
from cuda.bindings import (
    driver as cuda,  # ty: ignore[unresolved-import]  # Binary extension, no stubs.
)
from cutlass import cute


@cute.kernel
def saxpy(
    x: cute.Pointer,
    y: cute.Pointer,
    output: cute.Pointer,
    n: cutlass.Int32,
    alpha: cutlass.Float32,
    BLOCK: cutlass.Constexpr,
):
    layout = cute.make_layout(n)
    tx = cute.make_tensor(x, layout)
    ty = cute.make_tensor(y, layout)
    out = cute.make_tensor(output, layout)
    i = cute.arch.block_idx()[0] * BLOCK + cute.arch.thread_idx()[0]
    if i < n:
        out[i] = (alpha * tx[i].to(cutlass.Float32) + ty[i].to(cutlass.Float32)).to(
            output.value_type
        )


@cute.jit
def kernel(
    x: cute.Pointer,
    y: cute.Pointer,
    output: cute.Pointer,
    n: cutlass.Int32,
    alpha: cutlass.Float32,
    stream: cuda.CUstream,
    BLOCK: cutlass.Constexpr,
):
    saxpy(x, y, output, n, alpha, BLOCK).launch(
        grid=((n + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1), stream=stream
    )
