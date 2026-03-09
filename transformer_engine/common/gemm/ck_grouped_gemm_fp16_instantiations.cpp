#include "ck_grouped_gemm_common.h"

namespace transformer_engine {
namespace grouped_gemm {

// FP16 * FP16 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, TileCfg_256x256x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, TileCfg_256x128x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, TileCfg_256x256x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, TileCfg_256x128x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::atomic_add)

// FP16 * FP16 = FP32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, float, TileCfg_256x256x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, float, TileCfg_256x128x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, float, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, float, TileCfg_256x256x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, float, TileCfg_256x128x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::half_t, ck_tile::half_t, float, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::atomic_add)

// BF16 * BF16 = FP16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::half_t, TileCfg_256x256x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::half_t, TileCfg_256x128x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::half_t, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::half_t, TileCfg_256x256x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::half_t, TileCfg_256x128x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::half_t, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::atomic_add)

// BF16 * BF16 = FP32
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, float, TileCfg_256x256x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, float, TileCfg_256x128x64, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, float, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, float, TileCfg_256x256x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, float, TileCfg_256x128x64, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_GG_RUNNER, ck_tile::bfloat16_t, ck_tile::bfloat16_t, float, TileCfg_256x128x64_padding, ck_tile::memory_operation_enum::atomic_add)

} // namespace grouped_gemm
} // namespace transformer_engine