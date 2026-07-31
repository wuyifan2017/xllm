/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/string.h>

#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::cuda {

// Native DeepSeek block-wise FP8 GEMM via the mate `gemm_ops` TVM-FFI module
// (muDNN groupwise matmul, GROUP_BLOCK (1,128,128)). Avoids materializing a
// BF16 weight: the FP8 weight is
// read directly and 128x128 block scales are applied inside the kernel.
//
// Layout contract (matches mate.gemm.gemm_fp8_nt_groupwise):
//   a       [M, K]              float8_e4m3fn, contiguous
//   b       [N, K] (NT)         float8_e4m3fn, contiguous
//   a_scale [M, ceil(K/128)]    float32,       contiguous (K-major)
//   b_scale [ceil(N/128),
//            ceil(K/128)]        float32,       contiguous (K-major)
//   out     [M, N]              bf16/fp16
torch::Tensor gemm_fp8_nt_groupwise(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output) {
  const int64_t m = a.size(0);
  const int64_t n = b.size(0);

  torch::Tensor out;
  if (output.has_value() && output.value().defined()) {
    out = output.value();
  } else {
    out = torch::empty({m, n}, a.options().dtype(output_dtype));
  }

  MusaTvmffiStreamGuard stream_guard(a.device());

  // The cached mate module lives at ${FLASHINFER_OPS_PATH}/gemm_ops/gemm_ops.so
  // and exports the typed function "gemm_fp8_nt_groupwise".
  static const std::string kGemmOpsUri = "gemm_ops";
  get_function(kGemmOpsUri, "gemm_fp8_nt_groupwise")(
      to_ffi_tensor(a),
      to_ffi_tensor(b),
      to_ffi_tensor(a_scale),
      to_ffi_tensor(b_scale),
      /*scale_major_mode=*/std::string("K"),
      /*mma_sm=*/static_cast<int64_t>(1),
      /*scale_granularity_mnk=*/
      std::make_tuple(static_cast<int64_t>(1),
                      static_cast<int64_t>(128),
                      static_cast<int64_t>(128)),
      to_ffi_tensor(out),
      /*backend=*/std::string("mudnn"));

  // muDNN may complete the eager FP8 GEMM after the TVM-FFI call returns.
  // Ensure PyTorch consumers cannot reuse or read `out` before that write is
  // visible. Both helpers are capture-aware no-ops, so graph capture/replay
  // remains asynchronous; the pool-stream sync covers the null-current-stream
  // fallback selected by MusaTvmffiStreamGuard.
  sync_current_musa_stream(a.device());
  sync_musa_ffi_stream(a.device());

  return out;
}

}  // namespace xllm::kernel::cuda
