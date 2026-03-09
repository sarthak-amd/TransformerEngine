#include "ck_grouped_gemm_common.h"

namespace transformer_engine {
namespace grouped_gemm {

// FP8_E4M3 * FP8_E4M3 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x256x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x256x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E4M3 * FP8_E4M3 = FP32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_256x256x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_256x128x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_256x256x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_256x128x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::fp8_t, float, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E5M2 * FP8_E5M2 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x256x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x256x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E5M2 * FP8_E5M2 = FP32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_256x256x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_256x128x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_256x256x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_256x128x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::bf8_t, float, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E5M2 * FP8_E4M3 = F16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x256x128_k16, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x256x128_k16, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E5M2 * FP8_E4M3 = F32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_256x256x128_k16, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_256x128x128_k16, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_256x128x128_k16_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_256x256x128_k16, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_256x128x128_k16, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, float, TileCfg_256x128x128_k16_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E4M3 * FP8_E5M2 = F16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x256x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x256x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::atomic_add)

// FP8_E4M3 * FP8_E5M2 = F32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_256x256x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_256x128x128, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_256x256x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_256x128x128, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::fp8_t, ck_tile::bf8_t, float, TileCfg_256x128x128_padding, ck_tile::memory_operation_enum::atomic_add)

} // namespace grouped_gemm
} // namespace transformer_engine