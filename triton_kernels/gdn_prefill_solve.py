# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Adapted from pinned vLLM 98dff2a81d747d1dba01a47f939f48c3526d4206: solve_tril.py::merge_16x16_to_64x64_inverse_kernel.
# Model geometry is specialized by Nix; algorithm choices are fixed below.
import triton
import triton.language as tl

exp = tl.exp
exp2 = tl.exp2


@triton.jit(do_not_specialize=["T"])
def kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T: tl.int32,  # ty: ignore[invalid-type-form]
    H: tl.constexpr,
):
    # Fixed Qwen prefill algorithm; only model geometry is specialized.
    BT: tl.constexpr = tl.constexpr(64)
    DOT_PRECISION: tl.constexpr = tl.constexpr("ieee")
    IS_VARLEN: tl.constexpr = tl.constexpr(False)

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

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_A_33 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_A_44 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)  # ty: ignore[invalid-argument-type]
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)  # ty: ignore[invalid-argument-type]
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)  # ty: ignore[invalid-argument-type]
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)  # ty: ignore[invalid-argument-type]
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_11,
        b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
