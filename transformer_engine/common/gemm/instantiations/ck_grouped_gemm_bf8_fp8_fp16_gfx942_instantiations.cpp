#if !__HIP_DEVICE_COMPILE__ || defined(__gfx942__)
#include "../ck_grouped_gemm_fp8_gfx942_impl.h"

namespace transformer_engine {
namespace grouped_gemm {

// FP8_E5M2 * FP8_E4M3 = F16
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x256x128_k16, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16, ck_tile::memory_operation_enum::set)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16_padding, ck_tile::memory_operation_enum::set)

APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x256x128_k16, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16, ck_tile::memory_operation_enum::atomic_add)
APPLY_CK_GG_ALL_LAYOUT(DECL_CK_QUANT_GG_RUNNER, ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, TileCfg_256x128x128_k16_padding, ck_tile::memory_operation_enum::atomic_add)

} // namespace grouped_gemm
} // namespace transformer_engine
#endif