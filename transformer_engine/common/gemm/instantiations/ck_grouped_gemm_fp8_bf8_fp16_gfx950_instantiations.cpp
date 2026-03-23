/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#if !__HIP_DEVICE_COMPILE__ || defined(__gfx950__)
#include "../ck_grouped_gemm_fp8_gfx950_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

// FP8_E4M3 * FP8_E5M2 = F16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_GFX950_128x128x128_16x16x128_2x2x1, ck_tile::memory_operation_enum::atomic_add)

} // namespace grouped_gemm
} // namespace transformer_engine
#endif
