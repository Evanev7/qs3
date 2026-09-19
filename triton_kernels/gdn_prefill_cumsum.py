# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Adapted from pinned vLLM 98dff2a81d747d1dba01a47f939f48c3526d4206: cumsum.py::chunk_local_cumsum_scalar_kernel.
# Model geometry is specialized by Nix; algorithm choices are fixed below.
import triton
import triton.language as tl

exp = tl.exp
exp2 = tl.exp2
make_tensor_descriptor = tl.make_tensor_descriptor


@triton.jit(do_not_specialize=["T"])
def kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T: tl.int32,  # ty: ignore[invalid-type-form]
    H: tl.constexpr,
):
    # Fixed Qwen prefill algorithm; only model geometry is specialized.
    BT: tl.constexpr = tl.constexpr(64)
    HEAD_FIRST: tl.constexpr = tl.constexpr(False)
    IS_VARLEN: tl.constexpr = tl.constexpr(False)
    REVERSE: tl.constexpr = tl.constexpr(False)

    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)  # ty: ignore[invalid-argument-type]
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)  # ty: ignore[invalid-argument-type]
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
