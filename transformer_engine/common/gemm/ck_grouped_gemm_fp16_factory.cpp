/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

 #include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp16_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

template <typename teA, typename teB, typename ALayout, typename BLayout>
std::unique_ptr<RunnerInterface> make_fp16_runner_typed(DType d_dtype, const GroupedGemmRunContext& ctx) {
  std::unique_ptr<RunnerInterface> runner = nullptr;
  using AType = typename TETypeToCKType<teA>::type;
  using BType = typename TETypeToCKType<teB>::type;
  using CLayout = RowMajor;
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using CType = typename TETypeToCKType<d_te_type>::type;
        if (ctx.N % 256 == 0) {
            using TileCfg = TileCfg_256x256x64;
            if (ctx.accumulate) {
              using Runner = GroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = GroupedGemmRunner<AType, BType, CType,
                                    ALayout, BLayout, CLayout,
                                    TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
            
        } else if (ctx.N % 128 == 0) {
            using TileCfg = TileCfg_256x128x64;
            if (ctx.accumulate) {
              using Runner = GroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = GroupedGemmRunner<AType, BType, CType,
                                    ALayout, BLayout, CLayout,
                                    TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
        } else {
            using TileCfg = TileCfg_256x128x64_padding;
            if (ctx.accumulate) {
              using Runner = GroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = GroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
        }
    });
    return runner;
}

std::unique_ptr<RunnerInterface> make_fp16_runner(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx) {

  if (!ctx.transA && !ctx.transB) {
    using ALayout = RowMajor;
    using BLayout = RowMajor;

    switch (a_dtype) {
      case DType::kFloat16:
        if (b_dtype == DType::kFloat16) {
          return make_fp16_runner_typed<fp16, fp16, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kBFloat16:
        if (b_dtype == DType::kBFloat16) {
          return make_fp16_runner_typed<bf16, bf16, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      default:
        break;
    }
  } else if (!ctx.transA && ctx.transB) {
    using ALayout = RowMajor;
    using BLayout = ColMajor;

    switch (a_dtype) {
      case DType::kFloat16:
        if (b_dtype == DType::kFloat16) {
          return make_fp16_runner_typed<fp16, fp16, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kBFloat16:
        if (b_dtype == DType::kBFloat16) {
          return make_fp16_runner_typed<bf16, bf16, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      default:
        break;
    }
  } else if (ctx.transA && !ctx.transB) {
    using ALayout = ColMajor;
    using BLayout = RowMajor;

    switch (a_dtype) {
      case DType::kFloat16:
        if (b_dtype == DType::kFloat16) {
          return make_fp16_runner_typed<fp16, fp16, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kBFloat16:
        if (b_dtype == DType::kBFloat16) {
          return make_fp16_runner_typed<bf16, bf16, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      default:
        break;
    }
  } else {
    return nullptr;
  }
  return nullptr;
}

}  // namespace grouped_gemm
}  // namespace transformer_engine