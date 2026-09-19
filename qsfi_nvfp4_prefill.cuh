/*
 * Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// QuTLASS SM120 prefill recipe, adapted from qutlass/csrc/gemm.cu at
// e74319e3405ce6d71965732880f5dc1f52371f64. Rust owns plans and buffers.
#pragma once
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <stdexcept>

namespace qsfi_nvfp4_prefill {
using Tile = cute::Shape<cute::_256, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Output = cutlass::bfloat16_t;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    Tile,
    Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    Output,
    cutlass::layout::RowMajor,
    8,
    Output,
    cutlass::layout::RowMajor,
    8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    Element,
    cutlass::layout::RowMajor,
    32,
    Element,
    cutlass::layout::ColumnMajor,
    32,
    float,
    Tile,
    Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::
    GemmUniversal<cute::Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

inline void check(cutlass::Status status)
{
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(cutlassGetStatusString(status));
}

inline size_t launch(
    void* out,
    const void* a,
    const void* b,
    const void* as,
    const void* bs,
    const float* alpha,
    int m,
    int n,
    int k,
    int batch_count,
    flashinfer::gemm::CutlassGemmConfig,
    char* workspace,
    size_t capacity,
    cudaStream_t stream,
    int* occupancy
)
{
    if (batch_count != 1 || occupancy != nullptr)
        throw std::runtime_error("QuTLASS prefill requires one matrix and no occupancy query");
    using Config = Mainloop::Sm1xxBlkScaledConfig;
    auto stride_a = cutlass::make_cute_packed_stride(Kernel::StrideA {}, { m, k, 1 });
    auto stride_b = cutlass::make_cute_packed_stride(Kernel::StrideB {}, { n, k, 1 });
    auto stride_d = cutlass::make_cute_packed_stride(Kernel::StrideD {}, { m, n, 1 });
    auto layout_a = Config::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_b = Config::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm,
        { m, n, k, 1 },
        { static_cast<Gemm::ElementA const*>(a),
          stride_a,
          static_cast<Gemm::ElementB const*>(b),
          stride_b,
          static_cast<cutlass::float_ue4m3_t const*>(as),
          layout_a,
          static_cast<cutlass::float_ue4m3_t const*>(bs),
          layout_b },
        { {}, static_cast<Output const*>(out), stride_d, static_cast<Output*>(out), stride_d }
    };
    args.epilogue.thread.alpha_ptr = alpha;
    const size_t bytes = Gemm::get_workspace_size(args);
    if (out == nullptr)
        return bytes;
    if (bytes > capacity)
        throw std::runtime_error("QuTLASS prefill workspace too small");
    Gemm gemm;
    check(gemm.can_implement(args));
    check(gemm.initialize(args, workspace, stream));
    check(gemm.run(stream));
    return bytes;
}
} // namespace qsfi_nvfp4_prefill
