#include "ck_grouped_gemm_common.h"

#include <array>
#include <type_traits>
#include <vector>
#include <memory>

#include <transformer_engine/transformer_engine.h>

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

static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data;
}

static inline const transformer_engine::SimpleTensor& scale_inv_view(const transformer_engine::Tensor& t) {
  return t.scale_inv;
}

template <typename AType, typename BType, typename CType,
          typename ALayout, typename BLayout, typename CLayout,
          typename TileCfg,
          ck_tile::memory_operation_enum MemOp,
          typename AccType = float>
class QuantGroupedGemmRunner : public RunnerInterface {
public:
    // hard-coded for tensor quant for now
    static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::TensorQuant;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileCfg::M_Tile, TileCfg::N_Tile, TileCfg::K_Tile>,
        ck_tile::sequence<TileCfg::M_Warp, TileCfg::N_Warp, TileCfg::K_Warp>,
        ck_tile::sequence<TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile>>;

    using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape, TileCfg::TilePartitionerGroupNum, TileCfg::TilePartitionerM01>;

    using AQLayout = RowMajor;
    using BQLayout = RowMajor;

    using UniversalTraits = 
        ck_tile::TileGemmQuantTraits<
            TileCfg::kPadM, TileCfg::kPadN, TileCfg::kPadK,
            false, false, ALayout, BLayout, CLayout,
            QuantMode, AQLayout, BQLayout,
            false, TileCfg::DoubleSmemBuffer, false>;
    
    using Problem =  ck_tile::GemmRowColTensorQuantPipelineProblem<
            AType, BType, AccType,
            AccType, GemmShape, UniversalTraits,
            false, AccType>;

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
    
    using Kernel = ck_tile::QuantGroupedGemmKernel<Partitioner, Pipeline, Epilogue, QuantMode>;

    using HostArgs = ck_tile::QuantGroupedGemmHostArgs;

public:
    static std::vector<HostArgs> build_descs(const GroupedGemmRunContext& ctx) {
       const size_t needed = Kernel::GetWorkSpaceSize(ctx.group_num);
       if (!ctx.workspace || ctx.workspace_bytes < needed) {
           NVTE_ERROR("ck_tile_grouped_gemm: insufficient workspace. Needed bytes=", needed);
       }
      std::vector<HostArgs> descs;
      descs.reserve(ctx.group_num);
      for (int i = 0; i < ctx.group_num; ++i) {
          const transformer_engine::Tensor* const A_te =
              transformer_engine::convertNVTETensorCheck(ctx.A[i]);
          const transformer_engine::Tensor* const B_te =
              transformer_engine::convertNVTETensorCheck(ctx.B[i]);
          transformer_engine::Tensor* D_te =
              transformer_engine::convertNVTETensorCheck(ctx.D[i]);

          const auto& a = data_view(*A_te);
          const auto& b = data_view(*B_te);
          const auto& d = data_view(*D_te);

          int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;
          if (!get_flat_2d_dims(*A_te, Ad0, Ad1) ||
              !get_flat_2d_dims(*B_te, Bd0, Bd1) ||
              !get_flat_2d_dims(*D_te, Dd0, Dd1)) {
              NVTE_ERROR("ck_tile_grouped_gemm: expected all groups to be rank>=2.");
          }

          const int64_t M  = ctx.transA ? Ad1 : Ad0;
          const int64_t K  = ctx.transA ? Ad0 : Ad1;
          const int64_t N  = ctx.transB ? Bd0 : Bd1;
          const int64_t Kb = ctx.transB ? Bd1 : Bd0;

          if (Kb != K) {
              NVTE_ERROR("ck_tile_grouped_gemm: K mismatch between A and B in group ", i);
          }

          if (Dd0 != M || Dd1 != N) {
              NVTE_ERROR("ck_tile_grouped_gemm: D shape mismatch in group ", i);
          }

          const ck_tile::index_t stride_A = Ad1;
          const ck_tile::index_t stride_B = Bd1;
          const ck_tile::index_t stride_E = Dd1;
          
          // Hard-coded to tensor quant for the moment
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
        }

        return descs;
    };
    bool run(const ck_tile::stream_config& stream_cfg,
                const GroupedGemmRunContext& ctx) override {
        auto descs = build_descs(ctx);

        constexpr int kBlockPerCu = 1;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::GridSize(descs);
        auto kargs = Kernel::MakeKargs(descs);
        if (!Kernel::IsSupportedArgument(kargs)) {
            NVTE_ERROR("ck_tile_grouped_gemm: CK_Tile kernel arguments not supported for this config.");
        }

        HIP_CHECK_ERROR(hipMemcpyAsync(ctx.workspace,
                                        kargs.data(),
                                        kargs.size() * sizeof(typename decltype(kargs)::value_type),
                                        hipMemcpyHostToDevice,
                                        ctx.stream));

        ck_tile::launch_kernel(
            stream_cfg, ck_tile::make_kernel<kBlockPerCu>(
                            Kernel{}, grids, blocks, 0,
                            ck_tile::cast_pointer_to_constant_address_space(ctx.workspace),
                            ctx.group_num));
        return true;
    };
};

// Primus-Turbo-style extern template declarations
#define DECL_CK_QUANT_GG_RUNNER_EXTERN(AType, BType, CType, ALayout, BLayout, CLayout, TileCfg, MemOp)        \
        extern template class QuantGroupedGemmRunner<AType, BType, CType, ALayout, BLayout, CLayout, TileCfg, MemOp, float>;

#define DECL_CK_QUANT_GG_RUNNER(AType, BType, CType, ALayout, BLayout, CLayout, TileCfg, MemOp)               \
        template class QuantGroupedGemmRunner<AType, BType, CType, ALayout, BLayout, CLayout, TileCfg, MemOp, float>;

#define APPLY_CK_GG_ALL_LAYOUT(MACRO, AType, BType, CType, TileCfg, MemOp)   \
    MACRO(AType, BType, CType, RowMajor, ColMajor, RowMajor, TileCfg, MemOp) \
    MACRO(AType, BType, CType, RowMajor, RowMajor, RowMajor, TileCfg, MemOp) \
    MACRO(AType, BType, CType, ColMajor, RowMajor, RowMajor, TileCfg, MemOp)

}
}