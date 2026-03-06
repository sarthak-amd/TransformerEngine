/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <hip/hip_runtime.h>

#include <iostream>

#include <transformer_engine/transformer_engine.h>
#include "../common.h"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp"
#include "ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp"

namespace transformer_engine {
namespace grouped_gemm {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename TEScalar> struct TETypeToCKType;
template <> struct TETypeToCKType<transformer_engine::fp32>    { using type = float; };
template <> struct TETypeToCKType<transformer_engine::fp8e4m3> { using type = ck_tile::fp8_t; };
template <> struct TETypeToCKType<transformer_engine::fp8e5m2> { using type = ck_tile::bf8_t; };
template <> struct TETypeToCKType<transformer_engine::fp16>    { using type = ck_tile::half_t; };
template <> struct TETypeToCKType<transformer_engine::bf16>    { using type = ck_tile::bfloat16_t; };

// Treat TE tensors as generalized 2D matrices by flattening:
// (D1, D2, ..., Dn) -> (D1*...*D(n-1), Dn), consistent with TE Tensor::flat_*_dim.
static inline bool get_flat_2d_dims(const transformer_engine::Tensor& t,
                                    int64_t& d0, int64_t& d1) {
  // Require at least a matrix (rank >= 2). Higher ranks are flattened.
  if (t.shape().size() < 2)
    return false;
  d0 = static_cast<int64_t>(t.flat_first_dim());
  d1 = static_cast<int64_t>(t.flat_last_dim());
  return true;
}

static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data;  // rowwise data view
}

static inline const transformer_engine::SimpleTensor& scale_inv_view(const transformer_engine::Tensor& t) {
  return t.scale_inv;  // dequantization scaling factor
}

// Primus-Turbo-like FP16/BF16 tile configs
// Selection rule:
//   if (N % 256 == 0) use 256x256x64
//   else if (N % 128 == 0) use 256x128x64
//   else use 256x128x64 with N padding enabled
struct TileCfg_256x256x64 {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 64;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01      = 4;
};

struct TileCfg_256x128x64 : TileCfg_256x256x64 {
  static constexpr ck_tile::index_t N_Tile = 128;
};

struct TileCfg_256x128x64_padding : TileCfg_256x128x64 {
  static constexpr bool kPadN = true;
};

// Primus-Turbo-like FP8/BF8 tile configs
// Selection rule:
//   if (N % 256 == 0) use 256x256x128
//   else if (N % 128 == 0) use 256x128x128
//   else use 256x128x128 with N padding enabled
struct TileCfg_256x256x128 {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 128;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 32;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01      = 4;
};

struct TileCfg_256x128x128 : TileCfg_256x256x128 {
  static constexpr ck_tile::index_t N_Tile = 128;
};

struct TileCfg_256x128x128_padding : TileCfg_256x128x128 {
  static constexpr bool kPadN = true;
};

// Fallback FP8/BF8 tile family for normalized (bf8_t, fp8_t) pair during backprop.
// That is, while there is a supported WarpGemmMfma_f32_32x32x32_fp8_bf8,
// there is no such thing as WarpGemmMfma_f32_32x32x32_bf8_fp8,
// so we need to fall back to WarpGemmMfma_f32_32x32x16_bf8_fp8
// by selecting K_Warp_Tile = 16
struct TileCfg_256x256x128_k16 {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 128;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01      = 4;
};

struct TileCfg_256x128x128_k16 : TileCfg_256x256x128_k16 {
  static constexpr ck_tile::index_t N_Tile = 128;
};

struct TileCfg_256x128x128_k16_padding : TileCfg_256x128x128_k16 {
  static constexpr bool kPadN = true;
};

template <typename TEType> struct GemmTilePolicy;

// FP16/BF16 – K=64 tiles
template <>
struct GemmTilePolicy<fp16> {
  using Tile256x256 = TileCfg_256x256x64;
  using Tile256x128 = TileCfg_256x128x64;
  using TilePadding = TileCfg_256x128x64_padding;
};

template <>
struct GemmTilePolicy<bf16> {
  using Tile256x256 = TileCfg_256x256x64;
  using Tile256x128 = TileCfg_256x128x64;
  using TilePadding = TileCfg_256x128x64_padding;
};

// FP8/BF8 – K=128 tiles
template <>
struct GemmTilePolicy<fp8e4m3> {
  using Tile256x256 = TileCfg_256x256x128;
  using Tile256x128 = TileCfg_256x128x128;
  using TilePadding = TileCfg_256x128x128_padding;
};

template <>
struct GemmTilePolicy<fp8e5m2> {
  using Tile256x256 = TileCfg_256x256x128;
  using Tile256x128 = TileCfg_256x128x128;
  using TilePadding = TileCfg_256x128x128_padding;
};

// Fallback policy for normalized mixed pair:
// AType = bf8_t, BType = fp8_t
struct GemmTilePolicyBF8FP8Fallback {
  using Tile256x256 = TileCfg_256x256x128_k16;
  using Tile256x128 = TileCfg_256x128x128_k16;
  using TilePadding = TileCfg_256x128x128_k16_padding;
};

// This class instantiates CK_Tile's grouped GEMM pipeline.
// See e.g. https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/03_gemm/universal_gemm_invoker.hpp for reference.
// Currently the only quantization mode supported is tensor quant
template <typename AType, typename BType, typename CType,
          typename ALayout, typename BLayout, typename CLayout,
          typename TileCfg, bool useTensorQuant,
          ck_tile::memory_operation_enum MemOp,
          typename AccType = float>
struct Runner {
  using GemmShape = ck_tile::TileGemmShape<
      ck_tile::sequence<TileCfg::M_Tile, TileCfg::N_Tile, TileCfg::K_Tile>,
      ck_tile::sequence<TileCfg::M_Warp, TileCfg::N_Warp, TileCfg::K_Warp>,
      ck_tile::sequence<TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile>>;

  using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
      GemmShape, TileCfg::TilePartitionerGroupNum, TileCfg::TilePartitionerM01>;

  using AQLayout = RowMajor;
  using BQLayout = RowMajor;
  static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::TensorQuant;

  using UniversalTraits = std::conditional_t<
      useTensorQuant,
      ck_tile::TileGemmQuantTraits<
          TileCfg::kPadM, TileCfg::kPadN, TileCfg::kPadK,
          false, false, ALayout, BLayout, CLayout,
          QuantMode, AQLayout, BQLayout,
          false, TileCfg::DoubleSmemBuffer, false>,
      ck_tile::PersistentTileGemmUniversalTraits<
          TileCfg::kPadM, TileCfg::kPadN, TileCfg::kPadK,
          TileCfg::DoubleSmemBuffer, ALayout, BLayout, CLayout>>;

  static constexpr ck_tile::GemmPipelineScheduler Scheduler =
      ck_tile::GemmPipelineScheduler::Intrawave;

  using Problem = std::conditional_t<
      useTensorQuant,
      ck_tile::GemmRowColTensorQuantPipelineProblem<
          AType, BType, AccType,
          AccType, GemmShape, UniversalTraits,
          false, AccType>,
      ck_tile::UniversalGemmPipelineProblem<
          AType, BType, AccType,
          GemmShape, UniversalTraits, Scheduler>>;

  using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

  using Epilogue = ck_tile::CShuffleEpilogue<
      ck_tile::CShuffleEpilogueProblem<
          AType, BType, ck_tile::tuple<>, AccType,
          CType, ck_tile::tuple<>, CLayout,
          ck_tile::element_wise::PassThrough,
          Partitioner::MPerBlock, Partitioner::NPerBlock,
          TileCfg::M_Warp, TileCfg::N_Warp,
          TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile,
          Problem::TransposeC, MemOp>>;

  using Kernel = std::conditional_t<
      useTensorQuant,
      ck_tile::QuantGroupedGemmKernel<
          Partitioner, Pipeline, Epilogue, QuantMode>,
      ck_tile::GroupedGemmKernel<
          Partitioner, Pipeline, Epilogue>>;
};

template <typename AType, typename BType, typename CType,
          typename ALayout, typename BLayout, typename CLayout,
          typename TileCfg,
          bool useTensorQuant,
          ck_tile::memory_operation_enum MemOp,
          typename AccType = float>
static bool run_grouped_impl(const NVTETensor* A_use,
                             const NVTETensor* B_use,
                             NVTETensor* D,
                             int group_num,
                             bool transA_use,
                             bool transB_use,
                             void* workspace,
                             size_t workspace_bytes,
                             hipStream_t stream)
{
  using RunnerT = Runner<AType, BType, CType, ALayout, BLayout, CLayout, TileCfg, useTensorQuant, MemOp, AccType>;
  using Kernel  = typename RunnerT::Kernel;

  using HostArgs = std::conditional_t<
      useTensorQuant,
      ck_tile::QuantGroupedGemmHostArgs,
      ck_tile::GroupedGemmHostArgs<0>>;

  const size_t needed = Kernel::GetWorkSpaceSize(group_num);
  if (!workspace || workspace_bytes < needed) {
    NVTE_ERROR("ck_tile_grouped_gemm: insufficient workspace. Needed bytes=", needed);
    return false;
  }

  thread_local std::vector<HostArgs> descs;
  descs.clear();
  descs.reserve(group_num);

  for (int i = 0; i < group_num; ++i) {
    const transformer_engine::Tensor* const A_te =
        transformer_engine::convertNVTETensorCheck(A_use[i]);
    const transformer_engine::Tensor* const B_te =
        transformer_engine::convertNVTETensorCheck(B_use[i]);
    transformer_engine::Tensor* D_te =
        transformer_engine::convertNVTETensorCheck(D[i]);

    const auto& a = data_view(*A_te);
    const auto& b = data_view(*B_te);
    const auto& d = data_view(*D_te);

    int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;
    if (!get_flat_2d_dims(*A_te, Ad0, Ad1) ||
        !get_flat_2d_dims(*B_te, Bd0, Bd1) ||
        !get_flat_2d_dims(*D_te, Dd0, Dd1)) {
      NVTE_ERROR("ck_tile_grouped_gemm: expected all groups to be rank>=2 (2D or higher).");
      return false;
    }

    const int64_t M  = transA_use ? Ad1 : Ad0;
    const int64_t K  = transA_use ? Ad0 : Ad1;
    const int64_t N  = transB_use ? Bd0 : Bd1;
    const int64_t Kb = transB_use ? Bd1 : Bd0;

    if (Kb != K) {
      NVTE_ERROR("ck_tile_grouped_gemm: K mismatch between A and B in group ", i);
      return false;
    }

    if (Dd0 != M || Dd1 != N) {
      NVTE_ERROR("ck_tile_grouped_gemm: D shape mismatch in group ", i);
      return false;
    }

    const ck_tile::index_t stride_A = Ad1;
    const ck_tile::index_t stride_B = Bd1;
    const ck_tile::index_t stride_E = Dd1;

    if constexpr (useTensorQuant) {
      ck_tile::index_t AQK       = 1;
      ck_tile::index_t BQK       = 1;
      ck_tile::index_t stride_AQ = 1;
      ck_tile::index_t stride_BQ = 1;

      const auto& aq = scale_inv_view(*A_te);
      const auto& bq = scale_inv_view(*B_te);

      descs.emplace_back(
          a.dptr,
          b.dptr,
          d.dptr,
          aq.dptr,
          bq.dptr,
          1,
          M,
          N,
          K,
          AQK,
          BQK,
          stride_A,
          stride_B,
          stride_E,
          stride_AQ,
          stride_BQ);
    } else {
      descs.emplace_back(
          a.dptr,
          b.dptr,
          std::array<const void*, 0>{},
          d.dptr,
          1,
          M,
          N,
          K,
          stride_A,
          stride_B,
          std::array<ck_tile::index_t, 0>{},
          stride_E);
    }
  }

  const dim3 grids = Kernel::GridSize(descs);
  auto kargs       = Kernel::MakeKargs(descs);
  if (!Kernel::IsSupportedArgument(kargs)) {
    NVTE_ERROR("ck_tile_grouped_gemm: CK_Tile kernel arguments not supported for this config.");
    return false;
  }

  HIP_CHECK_ERROR(hipMemcpyAsync(workspace,
                                 kargs.data(),
                                 kargs.size() * sizeof(typename decltype(kargs)::value_type),
                                 hipMemcpyHostToDevice,
                                 stream));

  const ck_tile::stream_config s{stream};
  const dim3 blocks = Kernel::BlockSize();

  ck_tile::launch_kernel(
      s,
      ck_tile::make_kernel<1>(
          Kernel{}, grids, blocks, 0,
          ck_tile::cast_pointer_to_constant_address_space(workspace),
          group_num));

  return true;
}

}  // namespace grouped_gemm
}  // namespace transformer_engine

bool ck_tile_grouped_gemm(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,
                          hipStream_t stream) {
  if (group_num <= 0)
    return true;

  using namespace transformer_engine;
  using namespace transformer_engine::grouped_gemm;

  // Workspace pointer + bytes
  void*  ws_ptr   = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr   = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }


  // Normalize operand order to match upstream CUTLASS path.
  // TE grouped GEMM frontend passes (A=weights, B=inputs) with layout "TN",
  // i.e. effectively W^T * X. The backend kernels expect inputs first
  // (X * W^T), so swap A/B and their transpose flags while preserving
  // the same mathematical operation.
  const NVTETensor* A_use = B;
  const NVTETensor* B_use = A;
  const bool transA_use = transB;
  const bool transB_use = transA;

  const auto a_dtype = convertNVTETensorCheck(A_use[0])->dtype();
  const auto b_dtype = convertNVTETensorCheck(B_use[0])->dtype();

  // D dtype + N (assume uniform N across groups)
  Tensor* D0_te = convertNVTETensorCheck(D[0]);
  const auto d_dtype = D0_te->dtype();

  int64_t ref_d0 = 0, ref_d1 = 0;
  if (!get_flat_2d_dims(*D0_te, ref_d0, ref_d1)) {
    NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for D[0]");
    return false;
  }
  const ck_tile::index_t N = static_cast<ck_tile::index_t>(ref_d1);

  auto choose_tile = [&](auto policy_tag, auto&& run_tile) -> bool {
    using Policy = decltype(policy_tag);
    if ((N % 256) == 0) return run_tile(typename Policy::Tile256x256{});
    if ((N % 128) == 0) return run_tile(typename Policy::Tile256x128{});
    return run_tile(typename Policy::TilePadding{});
  };

  auto dispatch_pair = [&](auto ATag, auto BTag) -> bool {
    using teA = decltype(ATag);
    using teB = decltype(BTag);

    using AType = typename TETypeToCKType<teA>::type;
    using BType = typename TETypeToCKType<teB>::type;

    constexpr bool TensorQuantMode =
        std::is_same_v<teA, fp8e4m3> || std::is_same_v<teA, fp8e5m2> ||
        std::is_same_v<teB, fp8e4m3> || std::is_same_v<teB, fp8e5m2>;

    constexpr bool UseBF8FP8Fallback =
        std::is_same_v<AType, ck_tile::bf8_t> &&
        std::is_same_v<BType, ck_tile::fp8_t>;

    using DefaultPolicy = std::conditional_t<
        TensorQuantMode,
        GemmTilePolicy<fp8e4m3>,
        GemmTilePolicy<teA>>;

    using Policy = std::conditional_t<
        UseBF8FP8Fallback,
        GemmTilePolicyBF8FP8Fallback,
        DefaultPolicy>;

    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
      using CType = typename TETypeToCKType<d_te_type>::type;

      auto run_tile = [&](auto tile_tag) -> bool {
        using TileCfgSel = decltype(tile_tag);

        TRANSFORMER_ENGINE_SWITCH_CONDITION(transA_use, kTransA, {
          using ALayout = std::conditional_t<kTransA, ColMajor, RowMajor>;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(transB_use, kTransB, {
            using BLayout = std::conditional_t<kTransB, ColMajor, RowMajor>;

            if (accumulate) {
              return run_grouped_impl<
                  AType, BType, CType,
                  ALayout, BLayout, RowMajor,
                  TileCfgSel, TensorQuantMode,
                  ck_tile::memory_operation_enum::atomic_add>(
                  A_use, B_use, D, group_num,
                  kTransA, kTransB,
                  ws_ptr, ws_bytes, stream);
            } else {
              return run_grouped_impl<
                  AType, BType, CType,
                  ALayout, BLayout, RowMajor,
                  TileCfgSel, TensorQuantMode,
                  ck_tile::memory_operation_enum::set>(
                  A_use, B_use, D, group_num,
                  kTransA, kTransB,
                  ws_ptr, ws_bytes, stream);
            }
          });
        });
      };

      return choose_tile(Policy{}, run_tile);
    });

    return false;
  };

  switch (a_dtype) {
    case DType::kFloat16:
      if (b_dtype == DType::kFloat16) return dispatch_pair(fp16{}, fp16{});
      break;

    case DType::kBFloat16:
      if (b_dtype == DType::kBFloat16) return dispatch_pair(bf16{}, bf16{});
      break;

    case DType::kFloat8E4M3:
      if (b_dtype == DType::kFloat8E4M3) return dispatch_pair(fp8e4m3{}, fp8e4m3{});
      if (b_dtype == DType::kFloat8E5M2) return dispatch_pair(fp8e4m3{}, fp8e5m2{});
      break;

    case DType::kFloat8E5M2:
      if (b_dtype == DType::kFloat8E5M2) return dispatch_pair(fp8e5m2{}, fp8e5m2{});
      if (b_dtype == DType::kFloat8E4M3) return dispatch_pair(fp8e5m2{}, fp8e4m3{});
      break;

    default:
      break;
  }

  NVTE_ERROR("ck_tile_grouped_gemm: unsupported dtype pair for CK path.");
  return false;
}