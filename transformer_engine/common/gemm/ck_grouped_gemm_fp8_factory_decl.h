#pragma once

#include "ck_grouped_gemm_common.h"
#include <memory>

namespace transformer_engine {
namespace grouped_gemm {

enum class GPUArch {
  GFX942,
  GFX950,
  UNKNOWN
};

GPUArch detect_gpu_arch();

std::unique_ptr<RunnerInterface> make_fp8_runner_gfx942(
    DType a_dtype,
    DType b_dtype,
    DType d_dtype,
    const GroupedGemmRunContext& ctx);

std::unique_ptr<RunnerInterface> make_fp8_runner_gfx950(
    DType a_dtype,
    DType b_dtype,
    DType d_dtype,
    const GroupedGemmRunContext& ctx);

}  // namespace grouped_gemm
}  // namespace transformer_engine