/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_fp8_factory_decl.h"

#include <hip/hip_runtime.h>
#include <string>

namespace transformer_engine {
namespace grouped_gemm {

GPUArch detect_gpu_arch() {
  int device = 0;
  HIP_CHECK_ERROR(hipGetDevice(&device));

  hipDeviceProp_t props{};
  HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

  const std::string arch(props.gcnArchName);

  if (arch.find("gfx942") != std::string::npos) {
    return GPUArch::GFX942;
  }
  if (arch.find("gfx950") != std::string::npos) {
    return GPUArch::GFX950;
  }
  return GPUArch::UNKNOWN;
}

std::unique_ptr<RunnerInterface> make_fp8_runner(
    DType a_dtype,
    DType b_dtype,
    DType d_dtype,
    const GroupedGemmRunContext& ctx) {
  switch (detect_gpu_arch()) {
    case GPUArch::GFX942:
      return make_fp8_runner_gfx942(a_dtype, b_dtype, d_dtype, ctx);
    case GPUArch::GFX950:
      return make_fp8_runner_gfx950(a_dtype, b_dtype, d_dtype, ctx);
    default:
      NVTE_ERROR("ck_tile_grouped_gemm: available architectures = {gfx942, gfx950}");
      return nullptr;
  }
}

}  // namespace grouped_gemm
}  // namespace transformer_engine