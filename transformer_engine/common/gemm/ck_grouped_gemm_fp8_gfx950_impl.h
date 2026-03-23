/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include "ck_grouped_gemm_common.h"
#include "ck_grouped_gemm_fp8_runner_common.h"

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

// -------------------------------------
// GFX950-specific tile configs: FP8/BF8
// -------------------------------------

struct TileCfg_GFX950_128x128x128_16x16x128_2x2x1 {
  static constexpr ck_tile::index_t M_Tile = 128;
  static constexpr ck_tile::index_t N_Tile = 128;
  static constexpr ck_tile::index_t K_Tile = 128;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 128;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 16;
  static constexpr ck_tile::index_t TilePartitionerM01      = 8; 
};

// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E4M3 * FP8_E4M3 = FP32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E4M3 * FP8_E4M3 = BF16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E5M2 * FP8_E5M2 = FP32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E5M2 * FP8_E5M2 = BF16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E5M2 * FP8_E4M3 = F16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E5M2 * FP8_E4M3 = F32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E5M2 * FP8_E4M3 = BF16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E4M3 * FP8_E5M2 = F16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E4M3 * FP8_E5M2 = F32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

// FP8_E4M3 * FP8_E5M2 = BF16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER_EXTERN, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)

} // namespace grouped_gemm
} // namespace transformer_engine
