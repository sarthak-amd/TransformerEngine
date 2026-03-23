/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#if !__HIP_DEVICE_COMPILE__ || defined(__gfx942__)

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp8_gfx942_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

template <typename teA, typename teB, typename ALayout, typename BLayout>
std::unique_ptr<RunnerInterface> make_fp8_runner_typed_gfx942(
    DType d_dtype,
    const GroupedGemmRunContext& ctx) {
  std::unique_ptr<RunnerInterface> runner = nullptr;
  using AType = typename TETypeToCKType<teA>::type;
  using BType = typename TETypeToCKType<teB>::type;
  using CLayout = RowMajor;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
    using CType = typename TETypeToCKType<d_te_type>::type;
    using TileCfg = TileCfg_GFX942_128x128x128_32x32x16_2x2x1;
    using Runner = QuantGroupedGemmRunner<
        AType, BType, CType,
        ALayout, BLayout, CLayout,
        TileCfg, ck_tile::memory_operation_enum::set>;
    runner = std::make_unique<Runner>();
  });

  return runner;
}

std::unique_ptr<RunnerInterface> make_fp8_runner_gfx942(DType a_dtype,
                                                        DType b_dtype,
                                                        DType d_dtype,
                                                        const GroupedGemmRunContext& ctx) {
  if (!ctx.transA && !ctx.transB) {
    using ALayout = RowMajor;
    using BLayout = RowMajor;

    switch (a_dtype) {
      case DType::kFloat8E4M3:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx942<fp8e4m3, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx942<fp8e4m3, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kFloat8E5M2:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx942<fp8e5m2, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx942<fp8e5m2, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
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
          return make_fp8_runner_typed_gfx942<fp8e4m3, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx942<fp8e4m3, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kFloat8E5M2:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx942<fp8e5m2, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx942<fp8e5m2, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
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
          return make_fp8_runner_typed_gfx942<fp8e4m3, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx942<fp8e4m3, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
        }
        break;

      case DType::kFloat8E5M2:
        if (b_dtype == DType::kFloat8E4M3) {
          return make_fp8_runner_typed_gfx942<fp8e5m2, fp8e4m3, ALayout, BLayout>(d_dtype, ctx);
        } else if (b_dtype == DType::kFloat8E5M2) {
          return make_fp8_runner_typed_gfx942<fp8e5m2, fp8e5m2, ALayout, BLayout>(d_dtype, ctx);
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

#endif