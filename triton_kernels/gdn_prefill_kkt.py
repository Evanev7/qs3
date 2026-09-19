# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Adapted from pinned vLLM 98dff2a81d747d1dba01a47f939f48c3526d4206: chunk_scaled_dot_kkt.py::chunk_scaled_dot_kkt_fwd_kernel.
# Model geometry is specialized by Nix; algorithm choices are fixed below.
import triton
import triton.language as tl

exp = tl.exp
exp2 = tl.exp2
make_tensor_descriptor = tl.make_tensor_descriptor


@triton.jit(do_not_specialize=["T"])
def kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T: tl.int32,  # ty: ignore[invalid-type-form]
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
):
    # Fixed Qwen prefill algorithm; only model geometry is specialized.
    BK: tl.constexpr = tl.constexpr(64)
    BT: tl.constexpr = tl.constexpr(64)
    CAST_DOT_TO_K_DTYPE: tl.constexpr = tl.constexpr(False)
    IS_VARLEN: tl.constexpr = tl.constexpr(False)
    USE_G: tl.constexpr = tl.constexpr(True)

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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)  # ty: ignore[invalid-argument-type]
    for i_k in range(tl.cdiv(K, BK)):  # ty: ignore[invalid-argument-type]
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        if CAST_DOT_TO_K_DTYPE:
            # RDNA: force operands to k's native dtype so WMMA is used.
            b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
        else:
            # Keep the promoted precision of the beta-scaled operand (WGMMA/MFMA).
            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
