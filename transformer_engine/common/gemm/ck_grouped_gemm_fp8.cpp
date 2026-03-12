/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"

namespace transformer_engine {
namespace grouped_gemm {

bool ck_tile_grouped_gemm_fp8_dispatch(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx) {
    const ck_tile::stream_config s{ctx.stream};

    auto runner = make_fp8_runner(
        a_dtype, b_dtype, d_dtype, ctx);

    if (!runner) {
        return false;
    }

    return runner->run(s, ctx);
}

}  // namespace grouped_gemm
}  // namespace transformer_engine
