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


# Extracted from b12x 0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68.
# NVFP4 W4A4, M<=16, tile 32x64x512, BF16 output, no split-K.
# Scale layout helpers, from utils.py:
# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast

import cutlass
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings import (
    driver as cuda,  # ty: ignore[unresolved-import] # Binary extension, no stubs.
)
from cutlass import Int32, cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: tuple[int, int, int],
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFA based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """
    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"
    blk_mn = 128
    blk_sf = min(4, tile_shape_mnk[2] // sf_vec_size)
    blk_elems = blk_mn * 4
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)
    assert tile_shape_mnk[0] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[0] must be divisible by 16"
    )
    sfa_tile_m = max(blk_mn, ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)
    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)
    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (k_basic_block_stride, mma_nsf, sfa_tile_m // blk_mn * blk_elems)
    sSFA_shape = (sSFA_shapeM, sSFA_shapeK)
    sSFA_stride = (sSF_strideM, sSF_strideK)
    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)
    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )
    return sfa_smem_layout_staged


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: tuple[int, int, int],
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFB based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """
    blk_mn = 128
    blk_sf = min(4, tile_shape_mnk[2] // sf_vec_size)
    blk_elems = blk_mn * 4
    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"
    assert tile_shape_mnk[1] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[1] must be divisible by 16"
    )
    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)
    sfb_tile_n = max(blk_mn, ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)
    sSFA_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)
    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (k_basic_block_stride, mma_nsf, sfb_tile_n // blk_mn * blk_elems)
    sSFA_shape = (sSFA_shapeN, sSFA_shapeK)
    sSFA_stride = (sSF_strideN, sSF_strideK)
    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)
    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )
    return sfb_smem_layout_staged


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


class DenseGemmKernel:
    def __init__(self):
        self.mma_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            cutlass.Float4E2M1FN, cutlass.Float32, cutlass.Float8E4M3FN
        )
        atom_shape = (2, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            (32, 64, 512),
            16,
            cutlass.const_expr(
                cutlass.Float4E2M1FN == cutlass.Float8E4M3FN
                or cutlass.Float4E2M1FN == cutlass.Float6E3M2FN
                or cutlass.Float4E2M1FN == cutlass.Float6E2M3FN
            ),
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op, atom_layout, permutation_mnk=permutation_mnk
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        mma_m, mma_n, mma_k = (16, 8, 64)
        self.num_m_tiles = (32, 64, 512)[0] // (mma_m * atom_shape[0])
        self.num_n_tiles = (32, 64, 512)[1] // (mma_n * atom_shape[1])
        self.num_k_blocks = (32, 64, 512)[2] // mma_k
        self.cta_layout_mnk = cute.make_layout((1, 1, 1))
        sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
            self.tiled_mma, (32, 64, 512), 16, 1
        )
        sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
            self.tiled_mma, (32, 64, 512), 16, 1
        )
        self.ab_stage, self.epi_stage = self._compute_stages(
            (32, 64, 512),
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            (32, 64),
            cutlass.BFloat16,
            101376,
            1,
            False,
        )
        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
        )
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            (32, 64, 512),
            (32, 64),
            cutlass.Float4E2M1FN,
            self.a_layout,
            cutlass.Float4E2M1FN,
            self.b_layout,
            self.ab_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            16,
            self.tiled_mma,
            False or False,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, 16)
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)
        b_logical_shape = cast(tuple, b.shape)
        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_logical_shape, 16)
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged, ((32, 64, 512)[1], (32, 64, 512)[2]), 1
        )
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged, ((32, 64, 512)[0], (32, 64, 512)[2]), 1
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            (128, 512),
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            (128, 512),
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c, self.epi_smem_layout_staged, (32, 64)
        )
        tile_sched_params, grid = self._compute_grid(
            c, (32, 64, 512), max_active_clusters, 1, False, False
        )

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
        ).launch(grid=grid, block=[160, 1, 1], cluster=[1, 1, 1], stream=stream)

    def _partition_fragment_SFA(
        self, sfa_tensor: cute.Tensor, thr_mma: cute.ThrMma, tidx: int
    ):
        return sm120_utils.partition_fragment_SFA(sfa_tensor, thr_mma, tidx)

    def _partition_fragment_SFB(
        self, sfb_tensor: cute.Tensor, thr_mma: cute.ThrMma, tidx: int
    ):
        return sm120_utils.partition_fragment_SFB(sfb_tensor, thr_mma, tidx)

    def _get_layoutSFA_TV(self, tiled_mma: cute.TiledMma):
        return sm120_utils.get_layoutSFA_TV(tiled_mma)

    def _get_layoutSFB_TV(self, tiled_mma: cute.TiledMma):
        return sm120_utils.get_layoutSFB_TV(tiled_mma)

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,  # ty: ignore[deprecated] # Pinned CuTe 4.6 scheduler.
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        alpha_value = cast(cutlass.Float32, alpha[0])
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        b_tma_smem_layout = b_smem_layout
        tma_copy_bytes = (
            cute.size_in_bytes(cutlass.Float4E2M1FN, b_tma_smem_layout)
            + cute.size_in_bytes(cutlass.Float8E4M3FN, sfa_smem_layout)
            + cute.size_in_bytes(cutlass.Float8E4M3FN, sfb_smem_layout)
        )
        tma_copy_bytes += cute.size_in_bytes(cutlass.Float4E2M1FN, a_smem_layout)
        smem = cutlass.utils.SmemAllocator()
        mainloop_pipeline_array_ptr = smem.allocate_array(
            cutlass.Int64, self.ab_stage * 2
        )
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
        if cute.size((1, 1, 1)) > 1:
            cute.arch.cluster_arrive_relaxed()
        sA = smem.allocate_tensor(
            cutlass.Float4E2M1FN,
            a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            cutlass.Float4E2M1FN,
            b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            cutlass.Float8E4M3FN, sfa_smem_layout_staged, byte_alignment=1024
        )
        sSFB = smem.allocate_tensor(
            cutlass.Float8E4M3FN, sfb_smem_layout_staged, byte_alignment=1024
        )
        sC = smem.allocate_tensor(
            cutlass.BFloat16,
            epi_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=epi_smem_layout_staged.inner,
        )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_((32, 64, 512), (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_((32, 64, 512), (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(mSFA_mkl, (128, 512), (None, None, None))
        gSFB_nkl = cute.local_tile(mSFB_nkl, (128, 512), (None, None, None))
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_((32, 64, 512), (None, None, 0)), (None, None, None)
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
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        if cute.size((1, 1, 1)) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        block_idx = cute.arch.block_idx()
        k_tile_start = Int32(0)
        k_tile_iter_cnt = k_tile_cnt
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
                cutlass.Float4E2M1FN,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                cutlass.Float4E2M1FN,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[None, None, *tile_coord_mnl]
                sfa_tile_offset = tile_coord_mnl[0] % 4
                sfb_tile_offset = tile_coord_mnl[1] % 2
                sSFA_tile = cute.local_tile(
                    sSFA,
                    cute.slice_((32, 64, 512), (None, 0, None)),
                    (sfa_tile_offset, 0, None),
                )
                tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA_tile)
                tCrSFA_tile = self._partition_fragment_SFA(
                    sSFA_tile[None, None, 0], thr_mma, tidx
                )
                tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA_tile)
                sSFB_tile = cute.local_tile(
                    sSFB,
                    cute.slice_((32, 64, 512), (0, None, None)),
                    (sfb_tile_offset, 0, None),
                )
                tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(sSFB_tile)
                tCrSFB_tile = self._partition_fragment_SFB(
                    sSFB_tile[None, None, 0], thr_mma, tidx
                )
                tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB_tile)
                accumulators.fill(0.0)
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
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
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
                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)
                cute.copy(
                    smem_tiled_copy_SFA, tCsSFA_p_filtered, tCrSFA_copy_view_filtered
                )
                cute.copy(
                    smem_tiled_copy_SFB, tCsSFB_p_filtered, tCrSFB_copy_view_filtered
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
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    accumulators[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    accumulators[None, _mt, _nt],
                                )
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
                        if k_block_idx == num_k_blocks - 1:
                            tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                            tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                            tCrSFA_copy_view_filtered = cute.filter_zeros(
                                tCrSFA_tile_copy_view
                            )
                            tCrSFB_copy_view_filtered = cute.filter_zeros(
                                tCrSFB_tile_copy_view
                            )
                            cute.copy(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered,
                                tCrSFA_copy_view_filtered,
                            )
                            cute.copy(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered,
                                tCrSFB_copy_view_filtered,
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
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_tile[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_tile[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                accumulators[None, _mt, _nt],
                                tCrA[None, _mt, k_block_idx],
                                tCrB[None, _nt, k_block_idx],
                                accumulators[None, _mt, _nt],
                            )
                _is_m_major = self.c_layout.is_m_major_c()
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2), cutlass.BFloat16
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 2),
                    cutlass.BFloat16,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_Atom
                )
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.Float32)
                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, (32, 64))
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )
                epi_rest_m = bSG_gD.shape[1][0]
                epi_rest_n = bSG_gD.shape[1][1]
                epi_tile_m = (32, 64)[0]
                epi_tile_n = (32, 64)[1]
                mma_tile_m = (32, 64, 512)[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = (32, 64, 512)[1] // cute.size(tRS_rAcc, mode=[2])
                has_multi_epi_store = True
                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 4 * 32
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage, producer_group=tma_store_producer_group
                )
                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        MmaMPerEpiM = epi_tile_m // mma_tile_m
                        MmaNPerEpiN = epi_tile_n // mma_tile_n
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = epi_n * MmaNPerEpiN + mma_n_in_epi
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[None, mma_m_in_epi, mma_n_in_epi]
                                tRS_rAcc_slice = tRS_rAcc[None, mma_m, mma_n]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = tRS_rAcc_slice[elem_idx]
                        gmem_coord = (epi_m, epi_n)
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, cutlass.BFloat16
                        )
                        acc_vec = tRS_rD.load()
                        acc_vec = epilogue_op(
                            (alpha_value * acc_vec).to(cutlass.BFloat16)
                        )
                        tRS_rD_out.store(acc_vec)
                        epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        if has_multi_epi_store:
                            self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[None, None, None, epi_buffer],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[None, epi_buffer],
                                bSG_gD[None, gmem_coord],
                            )
                            if has_multi_epi_store:
                                tma_store_pipeline.producer_commit()
                                tma_store_pipeline.producer_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                if has_multi_epi_store:
                    tma_store_pipeline.producer_tail()
        elif warp_idx == 4:
            cute.arch.setmaxregister_decrease(40)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[None, tile_coord_mnl[0], None, tile_coord_mnl[2]]
                tBgB_nkl = tBgB[None, tile_coord_mnl[1], None, tile_coord_mnl[2]]
                sfa_tile_coord_m = tile_coord_mnl[0] // 4
                tAgSFA_mkl = tAgSFA[None, sfa_tile_coord_m, None, tile_coord_mnl[2]]
                sfb_tile_coord_n = tile_coord_mnl[1] // 2
                tBgSFB_nkl = tBgSFB[None, sfb_tile_coord_n, None, tile_coord_mnl[2]]
                mainloop_producer_state.reset_count()
                for _k_tile in cutlass.range(0, k_tile_iter_cnt, 1, unroll=2):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    k_tile_global = k_tile_start + mainloop_producer_state.count
                    tBgB_k = tBgB_nkl[None, k_tile_global]
                    tBsB_pipe = tBsB[None, mainloop_producer_state.index]
                    tAgA_k = tAgA_mkl[None, k_tile_global]
                    tAsA_pipe = tAsA[None, mainloop_producer_state.index]
                    tAgSFA_k = tAgSFA_mkl[None, k_tile_global]
                    tAsSFA_pipe = tAsSFA[None, mainloop_producer_state.index]
                    tBgSFB_k = tBgSFB_nkl[None, k_tile_global]
                    tBsSFB_pipe = tBsSFB[None, mainloop_producer_state.index]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
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
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        sf_dtype,
        sfa_smem_layout,
        sfb_smem_layout,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
        b_packed: bool = False,
        epi_stage_cap: int = 0,
        decode_stage3: bool = False,
    ) -> tuple:
        epi_stage_max = (
            tile_shape_mnk[1] // epi_tile[1] * (tile_shape_mnk[0] // epi_tile[0])
        )
        epi_stage = min(epi_stage_max, 4)
        if epi_stage_cap:
            epi_stage = max(1, min(epi_stage, epi_stage_cap))
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        raw_ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        ab_stage = max(1, min(raw_ab_stage, 4))
        if tile_shape_mnk[0] in (16, 64) and tile_shape_mnk[1] == 128:
            ab_stage = max(1, min(raw_ab_stage, 5))
        if b_packed:
            ab_stage = max(1, min(raw_ab_stage, 5))
        if decode_stage3 and occupancy >= 2 and (tile_shape_mnk[0] <= 16):
            ab_stage = max(1, min(raw_ab_stage, 3))
        return (ab_stage, epi_stage)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
        block_fp8: bool = False,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )
        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )
        if block_fp8:
            sfa_smem_layout_staged = cute.make_layout((1, 1, ab_stage))
            sfb_smem_layout_staged = cute.make_layout((1, 1, ab_stage))
        else:
            sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
                tiled_mma, tile_shape_mnk, sf_vec_size, ab_stage
            )
            sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
                tiled_mma, tile_shape_mnk, sf_vec_size, ab_stage
            )
        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )
        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
        split_k_slices: int,
        large_m_unroll: bool,
        split_k_all_m: bool = False,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[0, (None, None, None)].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(  # ty: ignore[deprecated] # Pinned CuTe 4.6 scheduler.
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=16
            if tile_shape_mnk == (128, 128, 64) and (not large_m_unroll)
            else 1,
        )
        if cutlass.const_expr(split_k_slices > 1 or split_k_all_m):
            grid = (
                num_ctas_mnl[0] if split_k_all_m else 1,
                split_k_slices,
                num_ctas_mnl[1],
            )
        else:
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            )
        return (tile_sched_params, grid)

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c, epi_smem_layout_staged, epi_tile: tuple
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), tensor_c, epi_smem_layout, epi_tile
        )
        return (tma_atom_c, tma_tensor_c)

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor, smem_layout_staged, smem_tile: tuple, mcast_dim: int, internal_type=None
    ) -> tuple:
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )
        return (tma_atom, tma_tensor)


@cute.jit
def kernel(
    a: cute.Pointer,
    b: cute.Pointer,
    sfa: cute.Pointer,
    sfb: cute.Pointer,
    output: cute.Pointer,
    alpha: cute.Pointer,
    m: cutlass.Int32,
    stream: cuda.CUstream,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
):
    a_tensor = cute.make_tensor(a, cute.make_ordered_layout((m, K, 1), order=(1, 0, 2)))
    b_tensor = cute.make_tensor(b, cute.make_ordered_layout((N, K, 1), order=(1, 0, 2)))
    sa = cute.make_tensor(sfa, cute.make_layout((1,)))
    sb = cute.make_tensor(sfb, cute.make_layout((1,)))
    c = cute.make_tensor(output, cute.make_ordered_layout((m, N, 1), order=(1, 0, 2)))
    scale = cute.make_tensor(alpha, cute.make_layout((1,)))
    DenseGemmKernel()(a_tensor, b_tensor, sa, sb, c, scale, 48, stream)
