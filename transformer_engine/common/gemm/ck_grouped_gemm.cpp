/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_grouped_gemm_common.h"
#include <iostream>
bool ck_tile_grouped_gemm(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,
                          hipStream_t stream) {
  if (group_num <= 0) {
    return true;
  }

  using namespace transformer_engine;
  using namespace transformer_engine::grouped_gemm;

  void* ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }

  // Normalize operand order to match upstream CUTLASS path.
  // TE grouped GEMM frontend passes (A=weights, B=inputs) with layout "TN",
  // i.e. effectively W^T * X. The backend kernels expect inputs first
  // (X * W^T), so swap A/B and their transpose flags while preserving
  // the same mathematical operation.
  const NVTETensor* A_use = B;
  const NVTETensor* B_use = A;
  const bool transA_use = transB;
  const bool transB_use = transA;

  const auto a_dtype = convertNVTETensorCheck(A_use[0])->dtype();
  const auto b_dtype = convertNVTETensorCheck(B_use[0])->dtype();

  Tensor* D0_te = convertNVTETensorCheck(D[0]);
  const auto d_dtype = D0_te->dtype();

  int64_t ref_d0 = 0, ref_d1 = 0;
  if (!get_flat_2d_dims(*D0_te, ref_d0, ref_d1)) {
    NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for D[0]");
    return false;
  }

  // construct run context
  GroupedGemmRunContext ctx = {A_use, B_use, D, ref_d1, group_num, transA_use, transB_use, accumulate, ws_ptr, ws_bytes, stream};

  if (ck_tile_grouped_gemm_fp16_dispatch(a_dtype, b_dtype, d_dtype, ctx)) {
    return true;
  }

  if (ck_tile_grouped_gemm_fp8_dispatch(a_dtype, b_dtype, d_dtype, ctx)) {
    return true;
  }

  NVTE_ERROR("ck_tile_grouped_gemm: unsupported dtype pair for CK path.");
  return false;
}