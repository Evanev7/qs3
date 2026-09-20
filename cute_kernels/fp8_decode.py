# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Adapted from b12x 0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68.
from typing import cast

import cutlass
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings import (
    driver as cuda,  # ty: ignore[unresolved-import]  # Binary extension, no stubs.
)
from cutlass import Float32, Int32, Uint32, cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op


def _reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _convert_layout_acc_mn(acc.layout, transpose=transpose)
    )


@cute.jit
def _emit_plain_fp8_dense_mma_k_block(
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
) -> None:
    acc = cast(cute.Tensor, accumulators[None, mt, nt])
    a_frag = cute.flatten(
        cute.recast_tensor(tCrA[None, mt, k_block_idx], cutlass.Uint32)
    )
    b_frag = cute.flatten(
        cute.recast_tensor(tCrB[None, nt, k_block_idx], cutlass.Uint32)
    )
    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_frag[0],
        a_frag[1],
        a_frag[2],
        a_frag[3],
        b_frag[0],
        b_frag[1],
    )
    acc[0] = d0
    acc[1] = d1
    acc[2] = d2
    acc[3] = d3


@dsl_user_op
def mma_m16n8k32_f32_e4m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Plain (non-block-scaled) SM120 FP8 E4M3 warp MMA `m16n8k32`.

    d{0..3} are the accumulator IN-OUT (C on input, D on output)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),  # ty: ignore[unresolved-attribute]  # MLIR extension type.
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        "\n        mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n        {$0, $1, $2, $3},\n        {$4, $5, $6, $7},\n        {$8, $9},\n        {$0, $1, $2, $3};\n        ",
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return (Float32(r0), Float32(r1), Float32(r2), Float32(r3))


def _convert_layout_acc_mn(
    acc_layout: cute.Layout | cute.ComposedLayout, transpose: bool = False
) -> cute.Layout | cute.ComposedLayout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if cutlass.const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


@cute.jit
def _accumulate_stage(acc: cute.Tensor, stage: cute.Tensor):
    # Match b12x's block-FP8 path with unit block scales: accumulate four
    # m16n8k32 instructions, then add that K=128 partial to the running sum.
    for i in cutlass.range_constexpr(cute.size(acc)):
        acc[i] = cutlass.Float32(acc[i]) + cutlass.Float32(stage[i])
        stage[i] = 0.0


class Fp8Decode:
    """SM121, M=1, 32x64x128 tiles, two FP32 split-K outputs.

    Four MMA warps consume a four-stage TMA pipeline driven by one load warp.
    Rust quantizes activations before this launch and reduces partials after it.
    """

    def __init__(self):
        self.ab_stage = 4
        self.num_m_tiles = 1
        self.num_n_tiles = 4
        self.a_layout = utils.LayoutEnum.ROW_MAJOR
        self.b_layout = utils.LayoutEnum.ROW_MAJOR

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        input_scale: cute.Tensor,
        weight_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mma_op = cute.nvgpu.warp.MmaMXF8Op(
            cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((2, 2, 1)),
            permutation_mnk=sm120_utils.get_permutation_mnk((32, 64, 128), 32, True),
        )
        a_smem = self._smem_layout((32, 128, 4))
        b_smem = self._smem_layout((64, 128, 4))
        tma_a, tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            a,
            cute.slice_(a_smem, (None, None, 0)),
            (32, 128),
        )
        tma_b, tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            b,
            cute.slice_(b_smem, (None, None, 0)),
            (64, 128),
        )
        # Keep the prototype's scheduler order, including its split-K grid axis.
        params = utils.PersistentTileSchedulerParams(  # ty: ignore[deprecated]  # Pinned b12x scheduler.
            (1, cute.size(b, mode=[0]) // 64, 2),
            (1, 1, 1),
            swizzle_size=1,
        )

        self.kernel(
            tma_a,
            tensor_a,
            tma_b,
            tensor_b,
            c,
            tiled_mma,
            cute.make_layout((1, 1, 1)),
            a_smem,
            b_smem,
            params,
            input_scale,
            weight_scale,
        ).launch(
            grid=(1, 2, cute.size(b, mode=[0]) // 64),
            block=(160, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @staticmethod
    def _smem_layout(shape: tuple[int, int, int]):
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, cutlass.Float8E4M3FN, 128
            ),
            cutlass.Float8E4M3FN,
        )
        return cute.tile_to_shape(atom, shape, order=(0, 1, 2))

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        directC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,  # ty: ignore[deprecated]
        input_scale: cute.Tensor,
        weight_scale: cute.Tensor,
    ):
        alpha_value = cutlass.Float32(input_scale[0]) * cutlass.Float32(weight_scale[0])
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        b_tma_smem_layout = b_smem_layout
        tma_copy_bytes = cute.size_in_bytes(
            cutlass.Float8E4M3FN, a_smem_layout
        ) + cute.size_in_bytes(cutlass.Float8E4M3FN, b_tma_smem_layout)
        smem = cutlass.utils.SmemAllocator()
        mainloop_pipeline_array_ptr = smem.allocate_array(cutlass.Int64, 8)
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 4
        )
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        sA = smem.allocate_tensor(
            cutlass.Float8E4M3FN,
            a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            cutlass.Float8E4M3FN,
            b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_((32, 64, 128), (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_((32, 64, 128), (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(tidx)
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        c_identity = cute.make_identity_tensor((32, 64))
        acc_shape = thr_mma.partition_C(c_identity).shape
        accumulators = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        stage_accumulators = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        split_k_acc_mn = _reshape_acc_to_mn(accumulators)
        split_k_c_identity = cute.make_identity_tensor(
            cute.slice_((32, 64, 128), (None, None, 0))
        )
        split_k_coord_mn = _reshape_acc_to_mn(thr_mma.partition_C(split_k_c_identity))
        cute.arch.sync_threads()
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        block_idx = cute.arch.block_idx()
        k_tile_start = Int32(0)
        k_tile_iter_cnt = k_tile_cnt
        k_tiles_per_split = k_tile_cnt // 2
        k_tile_start = Int32(block_idx[1]) * Int32(k_tiles_per_split)
        k_tile_iter_cnt = k_tiles_per_split
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, block_idx, cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        if warp_idx < 4:
            cute.arch.setmaxregister_increase(232)
            num_k_blocks = cute.size(tCrA, mode=[2])
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                cutlass.Float8E4M3FN,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                cutlass.Float8E4M3FN,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                accumulators.fill(0.0)
                stage_accumulators.fill(0.0)
                mainloop_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_iter_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )
                for k_tile in cutlass.range(0, k_tile_iter_cnt - 1, 1, unroll=2):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )
                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                _emit_plain_fp8_dense_mma_k_block(
                                    stage_accumulators,
                                    tCrA,
                                    tCrB,
                                    _mt,
                                    _nt,
                                    k_block_idx,
                                )
                        if k_block_idx == num_k_blocks - 1:
                            _accumulate_stage(accumulators, stage_accumulators)
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            _emit_plain_fp8_dense_mma_k_block(
                                stage_accumulators, tCrA, tCrB, _mt, _nt, k_block_idx
                            )
                    if k_block_idx == num_k_blocks - 1:
                        _accumulate_stage(accumulators, stage_accumulators)
                split_idx = Int32(block_idx[1])
                for acc_m in cutlass.range_constexpr(
                    cute.size(split_k_acc_mn, mode=[0])
                ):
                    for acc_n in cutlass.range_constexpr(
                        cute.size(split_k_acc_mn, mode=[1])
                    ):
                        coord = cast(
                            tuple[Int32, Int32], split_k_coord_mn[acc_m, acc_n]
                        )
                        m_coord = tile_coord_mnl[0] * Int32(32) + coord[0]
                        n_coord = tile_coord_mnl[1] * Int32(64) + coord[1]
                        if m_coord < Int32(
                            cute.size(directC_mnl, mode=[0])
                        ) and n_coord < Int32(cute.size(directC_mnl, mode=[1])):
                            directC_mnl[m_coord, n_coord, split_idx] = (
                                alpha_value * split_k_acc_mn[acc_m, acc_n]
                            )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        elif warp_idx == 4:
            cute.arch.setmaxregister_decrease(40)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[None, tile_coord_mnl[0], None, tile_coord_mnl[2]]
                tBgB_nkl = tBgB[None, tile_coord_mnl[1], None, tile_coord_mnl[2]]
                mainloop_producer_state.reset_count()
                for _k_tile in cutlass.range(0, k_tile_iter_cnt, 1, unroll=2):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    k_tile_global = k_tile_start + mainloop_producer_state.count
                    tBgB_k = tBgB_nkl[None, k_tile_global]
                    tBsB_pipe = tBsB[None, mainloop_producer_state.index]
                    tAgA_k = tAgA_mkl[None, k_tile_global]
                    tAsA_pipe = tAsA[None, mainloop_producer_state.index]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)


@cute.jit
def kernel(
    x: cute.Pointer,
    w: cute.Pointer,
    partials: cute.Pointer,
    input_scale: cute.Pointer,
    weight_scale: cute.Pointer,
    stream: cuda.CUstream,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
):
    assert N == 10240 and K == 5120, (
        "only the measured 27B QKV decode shape is qualified"
    )
    # Preserve a 2D TMA descriptor: a Python-literal M=1 collapses it to the
    # 1D TMA lowering that faults on SM121 with this compiler. No external M argument.
    m = cutlass.Int32(1)
    a = cute.make_tensor(x, cute.make_ordered_layout((m, K, 1), order=(1, 0, 2)))
    b = cute.make_tensor(w, cute.make_ordered_layout((N, K, 1), order=(1, 0, 2)))
    c = cute.make_tensor(partials, cute.make_ordered_layout((m, N, 2), order=(1, 0, 2)))
    xs = cute.make_tensor(input_scale, cute.make_layout((1,)))
    ws = cute.make_tensor(weight_scale, cute.make_layout((1,)))
    Fp8Decode()(a, b, c, xs, ws, stream)
