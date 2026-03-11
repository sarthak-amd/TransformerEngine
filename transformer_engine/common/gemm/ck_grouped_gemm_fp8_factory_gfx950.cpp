/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#if !__HIP_DEVICE_COMPILE__ || defined(__gfx950__)

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp8_gfx950_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

template <typename teA, typename teB, typename ALayout, typename BLayout>
std::unique_ptr<RunnerInterface> make_fp8_runner_typed_gfx950(DType d_dtype, const GroupedGemmRunContext& ctx) {
  std::unique_ptr<RunnerInterface> runner = nullptr;
  using AType = typename TETypeToCKType<teA>::type;
  using BType = typename TETypeToCKType<teB>::type;
  using CLayout = RowMajor;
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using CType = typename TETypeToCKType<d_te_type>::type;
        if (ctx.N % 256 == 0) {
          if constexpr (std::is_same_v<AType, ck_tile::bf8_t> && std::is_same_v<BType, ck_tile::fp8_t>) {
            using TileCfg = TileCfg_GFX950_128x128x128;
            if (ctx.accumulate) {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
          } else {
            using TileCfg = TileCfg_GFX950_128x128x128;
            if (ctx.accumulate) {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
          }
        } else if (ctx.N % 128 == 0) {
           if constexpr (std::is_same_v<AType, ck_tile::bf8_t> && std::is_same_v<BType, ck_tile::fp8_t>) {
             using TileCfg = TileCfg_GFX950_128x128x128;
            if (ctx.accumulate) {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
           } else {
            using TileCfg = TileCfg_GFX950_128x128x128;
            if (ctx.accumulate) {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::atomic_add>;
              runner = std::make_unique<Runner>();
            } else {
              using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                  ALayout, BLayout, CLayout,
                                                  TileCfg, ck_tile::memory_operation_enum::set>;
              runner = std::make_unique<Runner>();
            }
           }
        } else {
            if constexpr (std::is_same_v<AType, ck_tile::bf8_t> && std::is_same_v<BType, ck_tile::fp8_t>) {
              using TileCfg = TileCfg_GFX950_128x128x128;
              if (ctx.accumulate) {
                using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                    ALayout, BLayout, CLayout,
                                                    TileCfg, ck_tile::memory_operation_enum::atomic_add>;
                runner = std::make_unique<Runner>();
              } else {
                using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                    ALayout, BLayout, CLayout,
                                                    TileCfg, ck_tile::memory_operation_enum::set>;
                runner = std::make_unique<Runner>();
              }
            } else {
              using TileCfg = TileCfg_GFX950_128x128x128;
              if (ctx.accumulate) {
                using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                    ALayout, BLayout, CLayout,
                                                    TileCfg, ck_tile::memory_operation_enum::atomic_add>;
                runner = std::make_unique<Runner>();
              } else {
                using Runner = QuantGroupedGemmRunner<AType, BType, CType,
                                                    ALayout, BLayout, CLayout,
                                                    TileCfg, ck_tile::memory_operation_enum::set>;
                runner = std::make_unique<Runner>();
              }
            }
        }
    });
    return runner;
}

std::unique_ptr<RunnerInterface> make_fp8_runner_gfx950(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx) {

  if (!ctx.transA && !ctx.transB) {
    using ALayout = RowMajor;
    using BLayout = RowMajor;

    switch (a_dtype) {
      case DType::kFloat8E4M3:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx950<fp8e4m3, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx950<fp8e4m3, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kFloat8E5M2:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx950<fp8e5m2, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx950<fp8e5m2, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      default:
        break;
    }
  } else if (!ctx.transA && ctx.transB) {
    using ALayout = RowMajor;
    using BLayout = ColMajor;

    switch (a_dtype) {
      case DType::kFloat8E4M3:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx950<fp8e4m3, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx950<fp8e4m3, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kFloat8E5M2:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx950<fp8e5m2, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx950<fp8e5m2, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      default:
        break;
    }
  } else if (ctx.transA && !ctx.transB) {
    using ALayout = ColMajor;
    using BLayout = RowMajor;

    switch (a_dtype) {
      case DType::kFloat8E4M3:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx950<fp8e4m3, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx950<fp8e4m3, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kFloat8E5M2:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx950<fp8e5m2, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx950<fp8e5m2, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
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

}
}

#endif