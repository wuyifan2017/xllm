/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/layers/common/linear.h"
#if defined(USE_MUSA)
#include "layers/musa/linear_block_fp8.h"
#endif

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>

#include "core/common/macros.h"
#include "core/layers/common/quant_utils.h"
#include "core/platform/mx_quantization.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#endif

namespace xllm {
namespace layer {

namespace {

bool mx_env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string(value) == "1";
}

// ============================================================================
// FP8 Fused Weight Utilities
// ============================================================================
// Unlike INT8/SmoothQuant (per-channel), FP8 usually uses per-tensor scaling.
// When fusing separate layers (e.g., gate_proj + up_proj) into one, we cannot
// simply concatenate them if they have different scaling factors. We must
// requantize all partitions to align with a single global max_scale.

struct Fp8PartitionInfo {
  std::vector<float> scales;
  std::vector<int64_t> logical_widths;

  bool empty() const { return scales.empty(); }
  size_t size() const { return scales.size(); }
};

inline float compute_max_scale(const std::vector<float>& scales) {
  if (scales.empty()) {
    return 1.0f;
  }
  return *std::max_element(scales.begin(), scales.end());
}

// Detect if the checkpoint contains valid separate scales for each partition.
// The check on the last element serves as a heuristic to ensure the scales
// are fully populated and not just initialized to a sentinel/minimum value.
inline bool is_unfused_checkpoint(const std::vector<float>& scales) {
  return scales.size() > 1 &&
         scales.back() > std::numeric_limits<float>::lowest();
}

bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

// Realigns FP8 partitions to a unified global scale to enable fusion.
// Logic:
// 1. Recover original values (FP8 -> FP16) using partition-specific scales.
// 2. Re-quantize (FP16 -> FP8) using the new global max_scale.
void requantize_fp8_weight(torch::Tensor& weight,
                           const std::vector<float>& partition_scales,
                           const std::vector<int64_t>& logical_widths,
                           float max_scale) {
  if (partition_scales.size() != logical_widths.size()) {
    return;
  }

  int64_t start = 0;
  for (size_t idx = 0; idx < logical_widths.size(); ++idx) {
    int64_t logical_width = logical_widths[idx];
    if (logical_width == 0) {
      continue;
    }
    int64_t end = start + logical_width;

    // Dequantize: FP8 -> FP16 with original scale
    auto weight_slice = weight.slice(0, start, end);
    auto weight_fp16 = weight_slice.to(torch::kFloat16) * partition_scales[idx];

    // Requantize: FP16 -> FP8 with unified max_scale
    auto scale_tensor = torch::tensor(
        {max_scale}, weight_fp16.options().dtype(torch::kFloat32));
    auto weight_quantized =
        torch::empty_like(weight_slice, torch::kFloat8_e4m3fn);

    xllm::kernel::StaticScaledFp8QuantParams quant_params;
    quant_params.output = weight_quantized;
    quant_params.input = weight_fp16;
    quant_params.scale = scale_tensor;
    xllm::kernel::static_scaled_fp8_quant(quant_params);

    weight.slice(0, start, end).copy_(weight_quantized);
    start = end;
  }
}

// Load max input scale from multiple prefixes
torch::Tensor load_max_input_scale(const StateDict& state_dict,
                                   const std::vector<std::string>& prefixes) {
  torch::Tensor max_scale;
  for (const auto& prefix : prefixes) {
    auto scale_tensor = state_dict.get_tensor(prefix + "input_scale");
    if (scale_tensor.defined()) {
      auto scale_val = scale_tensor.flatten().max();
      if (!max_scale.defined()) {
        max_scale = scale_val;
      } else {
        max_scale = torch::max(max_scale, scale_val);
      }
    }
  }
  return max_scale;
}

// ============================================================================
// FP8 Forward Helper
// ============================================================================
// Performs FP8 W8A8 quantized linear: input quantization + scaled matmul.
// Consolidates repeated logic from Column/QKV/RowParallelLinear forward paths.
//
// Performance Optimization:
// - If input is already FP8 (from fused RMSNorm+FP8 quantization), skip
//   quantization step and use input_scale directly. This avoids redundant
//   memory reads/writes.
// - For non-FP8 inputs, quantization is performed based on input_scale:
//   - input_scale provided: static quantization (faster, no absmax compute)
//   - input_scale not provided: dynamic quantization (computes absmax)

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  // Flatten input to 2D for matmul
  auto input_2d = input.view({-1, input.size(-1)});

  torch::Tensor quantized_input;
  torch::Tensor a_scale;

  // Check if input is already FP8 quantized (from fused RMSNorm+FP8)
  if (input.dtype() == torch::kFloat8_e4m3fn) {
    // Input is already FP8, use directly (skip quantization)
    // This is the fast path when using fused RMSNorm+FP8 quantization
    CHECK(input_scale.has_value())
        << "input_scale must be provided when input is already FP8";
    quantized_input = input_2d;
    a_scale = input_scale.value();
  } else {
    // Input is not FP8, perform quantization
    // (static if input_scale provided, dynamic otherwise)
    xllm::kernel::Fp8ScaledQuantizeParams quantize_params;
    quantize_params.input = input_2d;
    quantize_params.output = std::nullopt;
    quantize_params.scale = input_scale;

    std::tie(quantized_input, a_scale) =
        xllm::kernel::fp8_scaled_quantize(quantize_params);
  }

  // FP8 scaled matmul
  xllm::kernel::Fp8ScaledMatmulParams matmul_params;
  matmul_params.a = quantized_input;
  matmul_params.b = weight;
  matmul_params.a_scale = a_scale;
  matmul_params.b_scale = weight_scale;
  matmul_params.bias = bias;
  matmul_params.output = std::nullopt;
  matmul_params.output_dtype = output_dtype;
  matmul_params.input_shape = input.sizes().vec();

  return xllm::kernel::fp8_scaled_matmul(matmul_params);
}

#if defined(USE_NPU)
namespace {
// Shared QuantBatchMatmulV5 call for the MXFP8 linear paths. The E8M0 bytes
// are passed as uint8 tensors; the GE dtype codes select the interpretation.
at::Tensor mxfp8_quant_matmul(const torch::Tensor& quantized_input,
                              const torch::Tensor& input_scale,
                              const torch::Tensor& weight,
                              const torch::Tensor& weight_scale,
                              const std::optional<torch::Tensor>& bias,
                              at::ScalarType output_dtype) {
  check_mx_quantization_supported();
#if defined(XLLM_HAS_NPU_MX_MATMUL)
  constexpr int64_t kFloat8E4M3FN = 292;
  constexpr int64_t kFloat8E5M2 = 291;
  constexpr int64_t kFloat4E2M1X2 = 296;  // packed fp4 (2 vals/byte)
  const int64_t weight_dtype =
      weight.scalar_type() == torch::kFloat8_e5m2 ? kFloat8E5M2 : kFloat8E4M3FN;
  std::optional<torch::Tensor> bias_fp32;
  if (bias.has_value() && bias->defined()) {
    bias_fp32 = bias->scalar_type() == torch::kFloat32
                    ? bias
                    : std::optional<torch::Tensor>(bias->to(torch::kFloat32));
  }
  return at_npu::native::custom_ops::npu_quant_matmul(
      quantized_input,
      weight,
      weight_scale,
      std::nullopt,
      input_scale,
      bias_fp32,
      static_cast<int64_t>(output_dtype),
      /*x1_dtype=*/292,
      /*x2_dtype=*/weight_dtype,
      /*pertoken_scale_dtype=*/293,
      /*scale_dtype=*/293,
      c10::IntArrayRef({1, 1, 32}),
      std::nullopt);
#else
  LOG(FATAL) << "MXFP8 matmul requires Torch-NPU/CANN MX support.";
  return {};
#endif
}
}  // namespace

// W4A8 MXFP4 linear: fp4 packed weights + fp8 activations.
at::Tensor mxfp4_quant_matmul(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight_packed,
    const torch::Tensor& weight_scale,
    at::ScalarType output_dtype,
    int64_t input_dtype = 292,
    const std::optional<torch::Tensor>& bias = std::nullopt) {
  check_mx_quantization_supported();
#if defined(XLLM_HAS_NPU_MX_MATMUL)
  const int64_t weight_dtype = 296;
  std::optional<torch::Tensor> bias_bf16;
  if (bias.has_value() && bias->defined()) {
    bias_bf16 = bias->reshape({1, -1}).to(torch::kBFloat16);
  }
  return at_npu::native::custom_ops::npu_quant_matmul(
      quantized_input,
      weight_packed,
      weight_scale,
      std::nullopt,
      input_scale,
      bias_bf16,
      static_cast<int64_t>(output_dtype),
      /*x1_dtype=*/input_dtype,
      /*x2_dtype=*/weight_dtype,
      /*pertoken_scale_dtype=*/293,
      /*scale_dtype=*/293,
      c10::IntArrayRef({1, 1, 32}),
      std::nullopt);
#else
  LOG(FATAL) << "MXFP4 matmul requires Torch-NPU/CANN MX support.";
  return {};
#endif
}

// MXFP8 W8A8 linear for dense Qwen3.5 projections. Expects load-time
// prepacked tensors (prepack_mxfp8_linear_weight): weight [K, N] float8 and
// weight_scale [K/64, N, 2] byte E8M0 values, the layout QuantBatchMatmulV5
// consumes. Activations are quantized on the fly to [M, K/64, 2].
torch::Tensor mxfp8_linear_forward(const torch::Tensor& input,
                                   const torch::Tensor& weight,
                                   const torch::Tensor& weight_scale,
                                   const std::optional<torch::Tensor>& bias,
                                   at::ScalarType output_dtype) {
  TORCH_CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn ||
                  weight.scalar_type() == torch::kFloat8_e5m2,
              "MXFP8 weight must be FLOAT8_E4M3FN or FLOAT8_E5M2");
  // Weights arrive prepacked at load time (prepack_mxfp8_linear_weight):
  // [K, N] float8 values and [K/64, N, 2] uint8 E8M0 scale bytes, exactly
  // what QuantBatchMatmulV5 consumes, so the hot path issues no copies.
  TORCH_CHECK(weight.dim() == 2 && weight_scale.dim() == 3,
              "MXFP8 weight/scale must be prepacked [K, N] + [K/64, N, 2]");
  const int64_t k = weight.size(0);
  const int64_t n = weight.size(1);
  TORCH_CHECK(weight_scale.size(0) == k / 64 && weight_scale.size(1) == n &&
                  weight_scale.size(2) == 2,
              "MXFP8 weight/scale shape mismatch");
  TORCH_CHECK(k % 64 == 0, "MXFP8 linear requires K divisible by 64, got ", k);

  const auto input_shape = input.sizes().vec();
  auto input_2d = input.view({-1, k});
  auto quantized = xllm::kernel::npu::mxfp8_dynamic_quant(input_2d);
  auto output = mxfp8_quant_matmul(std::get<0>(quantized),
                                   std::get<1>(quantized),
                                   weight,
                                   weight_scale,
                                   bias,
                                   output_dtype);
  auto output_shape = input_shape;
  output_shape.back() = n;
  return output.view(output_shape);
}

// GEMM-only variant for callers that fuse the activation quant (for example
// the SiLU-gate + MX-quant fused MLP path): consumes a pre-quantized
// (activation, scale) pair produced by mxfp8_dynamic_quant or
// mxfp8_swiglu_quant.
torch::Tensor mxfp8_linear_forward_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  TORCH_CHECK(weight.dim() == 2 && weight_scale.dim() == 3,
              "MXFP8 weight/scale must be prepacked [K, N] + [K/64, N, 2]");
  const int64_t k = weight.size(0);
  const int64_t n = weight.size(1);
  TORCH_CHECK(quantized_input.dim() == 2 && quantized_input.size(1) == k,
              "MXFP8 quantized input must be [M, ",
              k,
              "]");
  auto output = mxfp8_quant_matmul(
      quantized_input, input_scale, weight, weight_scale, bias, output_dtype);
  auto output_shape = quantized_input.sizes().vec();
  output_shape.back() = n;
  return output.view(output_shape);
}
#endif

// Preserve logical checkpoint dimensions while arranging packed FP4 NZ storage.
#if defined(USE_NPU)

namespace {
void prepack_mxfp4_nz_weight(torch::Tensor& weight) {
  check_mx_quantization_supported();
  static const bool enabled = mx_env_enabled("XLLM_MXFP4_W_NZ");
  if (!enabled ||
      at_npu::native::get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ) {
    return;
  }
  const int64_t n = weight.size(0);
  const int64_t k_half = weight.size(1);
  if (n % 16 != 0 || k_half % 32 != 0) {
    return;
  }
  // uint8 format_cast uses byte tiles; rearrange packed FP4's half-byte tiles
  // once so the native NZ kernel reads exactly the checkpoint values.
  auto packed = weight.to(torch::kCPU)
                    .reshape({n, k_half / 16, 16})
                    .permute({1, 0, 2})
                    .contiguous()
                    .reshape({k_half / 32, n / 16, 16, 32})
                    .permute({1, 2, 0, 3})
                    .contiguous()
                    .reshape({n, k_half})
                    .to(weight.device());
  weight.set_data(
      at_npu::native::npu_format_cast(packed, ACL_FORMAT_FRACTAL_NZ));
}

void cache_mxfp4_prefill_fp8(const torch::Tensor& weight,
                             torch::Tensor& expanded_weight) {
  check_mx_quantization_supported();
  if (expanded_weight.defined()) {
    return;
  }
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1) * 2;
  CHECK_EQ(at_npu::native::get_npu_format(weight), ACL_FORMAT_ND)
      << "Prepare lossless FP8 cache before FP4 NZ conversion";
  const auto packed = weight.to(torch::kCPU).contiguous();
  const auto lut = torch::tensor({0.0f,
                                  0.5f,
                                  1.0f,
                                  1.5f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  6.0f,
                                  -0.0f,
                                  -0.5f,
                                  -1.0f,
                                  -1.5f,
                                  -2.0f,
                                  -3.0f,
                                  -4.0f,
                                  -6.0f});
  auto decoded =
      torch::empty({n, k}, packed.options().dtype(torch::kFloat8_e4m3fn));
  constexpr int64_t kRowsPerDecode = 256;
  for (int64_t start = 0; start < n; start += kRowsPerDecode) {
    const int64_t count = std::min(kRowsPerDecode, n - start);
    const auto bytes = packed.narrow(/*dim=*/0, start, count);
    const auto low = torch::bitwise_and(bytes, 15).to(torch::kLong);
    const auto high = torch::bitwise_right_shift(bytes, 4).to(torch::kLong);
    const auto codes = torch::stack({low, high}, /*dim=*/-1).reshape({-1});
    decoded.narrow(/*dim=*/0, start, count)
        .copy_(lut.index_select(/*dim=*/0, codes)
                   .reshape({count, k})
                   .to(torch::kFloat8_e4m3fn));
  }
  expanded_weight = at_npu::native::npu_format_cast(
      decoded.transpose(0, 1).contiguous().to(weight.device()),
      ACL_FORMAT_FRACTAL_NZ);
  LOG_FIRST_N(INFO, 4) << "MXFP4 lossless prefill FP8 cache="
                       << expanded_weight.sizes();
}
}  // namespace

torch::Tensor mxfp4_linear_forward_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale,
    torch::Tensor& weight_packed,
    const torch::Tensor& weight_scale,
    torch::Tensor& expanded_weight,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  TORCH_CHECK(quantized_input.dim() == 2 && input_scale.dim() == 3,
              "MXFP4 input must be [M, K] with scale [M, K/64, 2]");
  const int64_t k = quantized_input.size(1);
  TORCH_CHECK(k % 64 == 0 && input_scale.size(0) == quantized_input.size(0) &&
                  input_scale.size(1) == k / 64 && input_scale.size(2) == 2,
              "MXFP4 input/scale shape mismatch");
  TORCH_CHECK(weight_packed.dim() == 2 && weight_scale.dim() == 2,
              "MXFP4 checkpoint weight must be [N, K/2], scale [N, K/32]");
  const int64_t n = weight_packed.size(0);
  TORCH_CHECK(weight_packed.size(1) == k / 2 && weight_scale.size(0) == n &&
                  weight_scale.size(1) == k / 32,
              "MXFP4 checkpoint weight/scale shape mismatch: weight=",
              weight_packed.sizes(),
              ", scale=",
              weight_scale.sizes(),
              ", input=",
              quantized_input.sizes());
  TORCH_CHECK(output_dtype == torch::kBFloat16,
              "MXFP4 FP8xFP4 pseudo quant matmul requires BF16 output");
  static const bool prefill_fp8 = mx_env_enabled("XLLM_MXFP4_PREFILL_FP8");
  static const bool native_control =
      mx_env_enabled("XLLM_MXFP4_PREFILL_FP8_OFF");
  if (prefill_fp8) {
    cache_mxfp4_prefill_fp8(weight_packed, expanded_weight);
  }
  prepack_mxfp4_nz_weight(weight_packed);
  if (prefill_fp8 && !native_control && quantized_input.size(0) >= 1024) {
    const auto scale =
        weight_scale.view({n, k / 64, 2}).transpose(0, 1).contiguous();
    return mxfp8_quant_matmul(quantized_input,
                              input_scale,
                              expanded_weight,
                              scale,
                              bias,
                              output_dtype);
  }
  // The A8W4 kernel requires transposeX2=true, with matching scale
  // strides. These views preserve the checkpoint nibble and E8M0 ordering.
  auto weight_transposed = weight_packed.transpose(0, 1);
  auto scale_transposed = weight_scale.view({n, k / 64, 2}).transpose(0, 1);
  auto output = mxfp4_quant_matmul(quantized_input,
                                   input_scale,
                                   weight_transposed,
                                   scale_transposed,
                                   output_dtype,
                                   /*input_dtype=*/292,
                                   bias);
  return output.view({quantized_input.size(0), n});
}

torch::Tensor attention_mxfp4_forward_quantized(
    const torch::Tensor& input,
    const torch::Tensor& input_scale,
    torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    torch::ScalarType output_dtype) {
  check_mx_quantization_supported();
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(input.size(1), weight.size(1) * 2);
  CHECK_EQ(weight_scale.size(0), weight.size(0));
  CHECK_EQ(weight_scale.size(1) * 32, input.size(1));
  prepack_mxfp4_nz_weight(weight);
  const int64_t n = weight.size(0);
  const int64_t k = input.size(1);
  return mxfp4_quant_matmul(input,
                            input_scale,
                            weight.transpose(0, 1),
                            weight_scale.view({n, k / 64, 2}).transpose(0, 1),
                            output_dtype,
                            /*input_dtype=*/292,
                            bias)
      .view({input.size(0), n});
}
#endif

// MXFP4 W4A8 linear forward: fp4 packed weights.
#if defined(USE_NPU)
torch::Tensor mxfp4_linear_forward(
    const torch::Tensor& input,
    torch::Tensor& weight_packed,  // uint8 [N, K/2] or [K/2, N]
    torch::Tensor& weight_scale,   // uint8 [N, K/32] or [K/64, N, 2]
    torch::Tensor& decoded_weight,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  if (weight_packed.device() != input.device()) {
    weight_packed.set_data(weight_packed.to(input.device()));
  }
  if (weight_scale.device() != input.device()) {
    weight_scale.set_data(weight_scale.to(input.device()));
  }
  (void)decoded_weight;
  TORCH_CHECK(weight_packed.dim() == 2 && weight_scale.dim() == 2,
              "MXFP4 linear requires checkpoint weight and scale layout");
  const int64_t k_native = weight_packed.size(1) * 2;
  auto input_2d_native = input.view({-1, k_native});
  auto quantized_native =
      xllm::kernel::npu::mxfp8_dynamic_quant(input_2d_native);
  auto output_native =
      mxfp4_linear_forward_quantized(std::get<0>(quantized_native),
                                     std::get<1>(quantized_native),
                                     weight_packed,
                                     weight_scale,
                                     decoded_weight,
                                     bias,
                                     output_dtype);
  auto output_shape_native = input.sizes().vec();
  output_shape_native.back() = weight_packed.size(0);
  return output_native.view(output_shape_native);
}

// One-time load-phase layout prepack for MXFP8 linears. Checkpoint weights
// are [N, K] with [N, K/32] E8M0 scale bytes; npu_quant_matmul consumes
// [K, N] weights with [K/64, N, 2] scales. Repacking per forward call (as
// the original integration did) costs ~19 GB of HBM copy traffic per decode
// step on Qwen3.8-27B and erases the FP8 weight-size win, so the transform
// runs once after load instead. Must run after any load-time weight row
// reorders; idempotent via the scale dimension check (2 -> 3).
void prepack_mxfp8_linear_weight(torch::Tensor& weight,
                                 torch::Tensor& weight_scale) {
  check_mx_quantization_supported();
  if (!weight.defined() || !weight_scale.defined() || weight_scale.dim() != 2) {
    return;  // not loaded, or already prepacked (dim == 3)
  }
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn ||
        weight.scalar_type() == torch::kFloat8_e5m2)
      << "MXFP8 weights must use E4M3FN or E5M2.";
  CHECK_EQ(k % 64, 0) << "Packed MXFP8 requires K divisible by 64.";
  CHECK_EQ(weight_scale.size(0), n);
  CHECK_EQ(weight_scale.size(1), k / 32)
      << "MXFP8 requires one E8M0 scale per block of 32 weights.";
  bool e8m0_storage = weight_scale.scalar_type() == torch::kUInt8;
#if defined(XLLM_HAS_FLOAT8_E8M0)
  e8m0_storage =
      e8m0_storage || weight_scale.scalar_type() == torch::kFloat8_e8m0fnu;
#endif
  CHECK(e8m0_storage) << "MXFP8 scale must be E8M0 or raw E8M0 bytes.";
  weight.set_data(weight.transpose(0, 1).contiguous());
  weight_scale.set_data(
      weight_scale.view({n, k / 64, 2}).transpose(0, 1).contiguous());
  // Optional FRACTAL_NZ metadata tag on prepacked [K, N] weights: the cast is
  // storage-preserving and lets npu_quant_matmul select its NZ code path.
  // Gated by env so it stays opt-in until validated end to end.
  static const bool tag_fractal_nz = std::getenv("XLLM_MXFP8_W_NZ") != nullptr;
  if (tag_fractal_nz) {
    weight.set_data(
        at_npu::native::npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ));
  }
}

#endif

bool wants_mmrs(RowParallelReduceMode reduce_mode) {
  return reduce_mode == RowParallelReduceMode::MATMUL_REDUCE_SCATTER;
}

void log_mmrs_quant_skip(RowParallelReduceMode reduce_mode,
                         const FlashComm1Context* fc1_ctx,
                         const char* quant_path,
                         const torch::Tensor& input) {
  if (!wants_mmrs(reduce_mode)) {
    return;
  }
  LOG_FIRST_N(WARNING, 16)
      << "FC1 MMRS skipped in row-parallel " << quant_path
      << " path: fused matmul_reduce_scatter is currently wired only for "
         "non-quant linear. input="
      << input.sizes() << ", sequence_sharded="
      << (fc1_ctx != nullptr && is_sequence_sharded(*fc1_ctx))
      << ", enable_mmrs_fusion="
      << (fc1_ctx != nullptr && fc1_ctx->enable_mmrs_fusion);
}

bool is_fp8_channelwise_w8a8(const QuantArgs& quant_args) {
#if defined(USE_DCU)
  // Compressed-tensors FP8 W8A8 currently pairs dynamic activations with
  // per-output-channel weight scales on DCU.
  return quant_args.quant_method() == kQuantMethodFp8 &&
         quant_args.activation_dynamic();
#else
  (void)quant_args;
  return false;
#endif
}

void check_fp8_activation_dynamic_supported(const QuantArgs& quant_args) {
  if (!quant_args.activation_dynamic()) {
    return;
  }
  CHECK(is_fp8_channelwise_w8a8(quant_args))
      << "DCU FP8 currently supports only compressed-tensors dynamic "
         "channelwise W8A8.";
}

torch::Dtype get_w8a8_deq_scale_dtype(const torch::TensorOptions& options) {
  const torch::Dtype dtype = c10::typeMetaToScalarType(options.dtype());
  if (dtype == torch::kFloat16) {
    return torch::kInt64;
  }
  if (dtype == torch::kBFloat16) {
    return torch::kFloat32;
  }
  LOG(WARNING) << "W8A8 deq_scale defaults to float32 for dtype " << dtype;
  return torch::kFloat32;
}

struct W8A8LinearParamRefs {
  torch::Tensor& weight;
  bool& weight_is_loaded;
  torch::Tensor& input_scale;
  bool& input_scale_is_loaded;
  torch::Tensor& input_offset;
  bool& input_offset_is_loaded;
  torch::Tensor& deq_scale;
  bool& deq_scale_is_loaded;
  torch::Tensor& quant_bias;
  bool& quant_bias_is_loaded;
  torch::Tensor& weight_scale;
  bool& weight_scale_is_loaded;
  torch::Tensor& weight_offset;
  bool& weight_offset_is_loaded;
  torch::Tensor& smooth;
  bool& smooth_is_loaded;
};

void ensure_w8a8_params_for_linear_load(
    torch::nn::Module* module,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const std::optional<std::string>& resolved_weight_quant_method,
    int64_t shared_input_param_size,
    W8A8LinearParamRefs refs) {
#if defined(USE_MUSA)
  musa::check_quantization_supported(quant_args, resolved_weight_quant_method);
#endif
#if !defined(USE_NPU) && !defined(USE_MLU) && !defined(USE_DCU)
  CHECK(!is_w8a8_dynamic_quant(resolved_weight_quant_method))
      << "Dynamic W8A8 Linear is unsupported on this backend";
#endif
  std::vector<weight::LazyParameterSpec> specs;
  auto push = [&](torch::Tensor& tensor,
                  bool& tensor_is_loaded,
                  const char* name,
                  std::vector<int64_t> sizes,
                  const torch::TensorOptions& tensor_options) {
    specs.push_back(weight::LazyParameterSpec{
        &tensor, &tensor_is_loaded, name, std::move(sizes), tensor_options});
  };

  if (!is_w8a8_quant(resolved_weight_quant_method) &&
      !is_w8a8_dynamic_quant(resolved_weight_quant_method) &&
      !is_w8a8_mxfp8_quant(resolved_weight_quant_method)) {
    if (!quant_args.quant_descs().empty() ||
        quant_args.is_compressed_tensors_w8a8_dynamic()) {
      // Quant args indicated a checkpoint that may be quantized, so the
      // constructor initialized weights as kInt8. If the actual checkpoint is
      // not resolved to a W8A8 method, re-register the weight in the original
      // dtype so the subsequent load can copy checkpoint weights correctly.
      CHECK(refs.weight.defined())
          << "weight must be registered before lazy quant fallback";
      const int64_t out_features = refs.weight.size(0);
      const int64_t in_features = refs.weight.size(1);
      specs.reserve(1);
      push(refs.weight,
           refs.weight_is_loaded,
           "weight",
           {out_features, in_features},
           options);
      weight::ensure_parameter_storage(module, specs);
    }
    return;
  }

  CHECK(refs.weight.defined())
      << "weight must be registered before lazy quant init";
  const int64_t out_features = refs.weight.size(0);
  const int64_t in_features = refs.weight.size(1);

  specs.reserve(5);
  if (is_w8a8_mxfp8_quant(resolved_weight_quant_method)) {
    // MXFP8 stores one E8M0 byte per 32 values along K. Keep the checkpoint
    // layout here; the matmul wrapper repacks it to [K/64, N, 2].
    push(refs.weight,
         refs.weight_is_loaded,
         "weight",
         {out_features, in_features},
         options.dtype(torch::kFloat8_e4m3fn));
    push(refs.weight_scale,
         refs.weight_scale_is_loaded,
         "weight_scale",
         {out_features, (in_features + 31) / 32},
         options.dtype(torch::kUInt8));
    weight::ensure_parameter_storage(module, specs);
    return;
  } else if (is_w8a8_quant(resolved_weight_quant_method)) {
    push(refs.input_scale,
         refs.input_scale_is_loaded,
         "input_scale",
         {shared_input_param_size},
         options.dtype(torch::kFloat32));
    push(refs.input_offset,
         refs.input_offset_is_loaded,
         "input_offset",
         {shared_input_param_size},
         options.dtype(torch::kInt8));
    push(refs.deq_scale,
         refs.deq_scale_is_loaded,
         "deq_scale",
         {out_features},
         options.dtype(get_w8a8_deq_scale_dtype(options)));
    push(refs.quant_bias,
         refs.quant_bias_is_loaded,
         "quant_bias",
         {out_features},
         options.dtype(torch::kInt32));
  } else {
    push(refs.weight_scale,
         refs.weight_scale_is_loaded,
         "weight_scale",
         {out_features},
         options.dtype(torch::kFloat32));
    push(refs.weight_offset,
         refs.weight_offset_is_loaded,
         "weight_offset",
         {out_features},
         options.dtype(torch::kFloat32));
  }
  if (quant_args.is_compressed_tensors_w8a8_dynamic() &&
      quant_args.preserve_smooth()) {
#if !defined(USE_MLU) && !defined(USE_DCU)
    NOT_IMPLEMENTED_WITH_MSG(
        "Preserved compressed-tensors smooth is unsupported on this backend");
#endif
    push(refs.smooth,
         refs.smooth_is_loaded,
         "smooth",
         {in_features},
         options.dtype(torch::kFloat32));
  }
  weight::ensure_parameter_storage(module, specs);
}

bool tensors_allclose_as_fp32(const torch::Tensor& lhs,
                              const torch::Tensor& rhs) {
  return torch::allclose(lhs.to(torch::kFloat32), rhs.to(torch::kFloat32));
}

bool load_shared_tensor_from_prefixes_or_fail(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes,
    const std::string& name,
    torch::Tensor& tensor,
    bool& tensor_is_loaded) {
  // W8A8 fused input_scale/offset should be same
  if (tensor_is_loaded || !tensor.defined()) {
    return tensor_is_loaded;
  }
  torch::Tensor first_candidate;
  std::string first_prefix;
  for (const auto& prefix : prefixes) {
    auto candidate = state_dict.get_tensor(prefix + name);
    if (!candidate.defined()) {
      continue;
    }
    auto flattened = candidate.flatten();
    if (!first_candidate.defined()) {
      first_candidate = flattened;
      first_prefix = prefix;
      continue;
    }
    CHECK_EQ(flattened.sizes(), first_candidate.sizes())
        << "Shared tensor size for " << name << ": prefix '" << prefix
        << "' has shape " << flattened.sizes() << ", but prefix '"
        << first_prefix << "' has shape " << first_candidate.sizes() << ".";
    CHECK(tensors_allclose_as_fp32(flattened, first_candidate))
        << "Shared tensor value for " << name << ": prefix '" << prefix
        << "' differs from prefix '" << first_prefix << "'.";
  }
  if (!first_candidate.defined()) {
    return false;
  }
  CHECK_EQ(first_candidate.numel(), tensor.numel())
      << "Tensor size mismatch for shared: " << state_dict.prefix() << name;
  tensor.copy_(first_candidate.view(tensor.sizes()));
  tensor_is_loaded = true;
  return true;
}

void collapse_shared_tensor_to_scalar_or_fail(torch::Tensor& tensor,
                                              const char* name) {
  // W8A8 fused input_scale/offset should be same
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK_GT(tensor.numel(), 0) << name << " must contain at least one element.";
  if (tensor.numel() <= 1) {
    return;
  }
  auto flattened = tensor.flatten();
  auto first = flattened.slice(0, 0, 1).expand_as(flattened);
  CHECK(tensors_allclose_as_fp32(flattened, first))
      << "Shared tensor value for " << name
      << " in fused static W8A8 should be same.";
  tensor = tensor.flatten().slice(0, 0, 1);
}

torch::Tensor npu_w8a8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& input_scale,
    const torch::Tensor& input_offset,
    const torch::Tensor& deq_scale,
    const std::optional<torch::Tensor>& quant_bias,
    at::ScalarType output_dtype) {
  xllm::kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;
  quant_params.scale = input_scale;
  quant_params.zero_point = input_offset;
  // quant_params.output_dtype = at::ScalarType::QInt8;
  quant_params.axis = -1;

  auto quantized_input = xllm::kernel::quantize(quant_params);

  xllm::kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = deq_scale;
  quant_matmul_params.bias = quant_bias;
  quant_matmul_params.output_dtype = output_dtype;

  return xllm::kernel::quant_matmul(quant_matmul_params);
}

#if defined(USE_MLU) || defined(USE_DCU)
torch::Tensor loaded_ct_smooth(const QuantArgs& args,
                               const torch::Tensor& smooth,
                               bool is_loaded) {
  CHECK(!args.preserve_smooth() || is_loaded)
      << "Missing compressed-tensors smooth";
  return smooth.defined() && is_loaded ? smooth : torch::Tensor();
}
#endif

torch::Tensor scaled_w8a8_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& bias,
    torch::ScalarType output_dtype,
    const LinearExtraArgs& extra = LinearExtraArgs()) {
  kernel::ScaledQuantizeParams quantize;
  const bool reshape_input = input.dim() != 2;
  quantize.x = reshape_input ? input.reshape({-1, input.size(-1)}) : input;
  quantize.smooth = smooth;
  quantize.act_mode = extra.act_mode;
  quantize.is_gated = extra.is_gated;
  auto [quantized, token_scale] = kernel::scaled_quantize(quantize);
  kernel::ScaledMatmulParams matmul;
  matmul.a = quantized;
  matmul.b = weight;
  matmul.a_scale = token_scale;
  matmul.b_scale = scale;
  matmul.bias = bias;
  matmul.output_dtype = output_dtype;
  matmul.a_quant_bit_size = 8;
  matmul.beta = 0.0;
  torch::Tensor output = kernel::scaled_matmul(matmul);
  if (!reshape_input) {
    return output;
  }
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  shape.back() = weight.size(0);
  return output.reshape(shape);
}
}  // namespace

ColumnParallelLinearImpl::ColumnParallelLinearImpl(const ModelContext& context)
    : ColumnParallelLinearImpl(
          context.get_model_args().hidden_size(),
          context.get_model_args().vocab_size(),
          /*bias=*/false,
          /*gather_output=*/true,
          // Desc-driven lm_head quantization: a checkpoint that marks
          // lm_head.weight as W8A8_MXFP8 in quant_model_description.json
          // activates the mxfp8 path; FLOAT/absent entries keep bf16.
          context.get_quant_args(),
          context.get_parallel_args().lm_head_group_ != nullptr
              ? context.get_parallel_args().lm_head_group_
              : context.get_parallel_args().tp_group_,
          context.get_tensor_options()) {}

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args,
    int32_t output_replicas)
    : gather_output_(gather_output),
      device_(options.device()),
      process_group_(process_group),
      quant_args_(quant_args),
      options_(options),
      linear_extra_args_(linear_extra_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
#if defined(USE_MUSA)
  musa::check_quantization_supported(quant_args_);
#endif
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  int32_t valid_output_replicas = output_replicas;
  if (valid_output_replicas <= 0 || world_size_ % valid_output_replicas != 0 ||
      (valid_output_replicas != 1 && gather_output)) {
    valid_output_replicas = 1;
  }
  weight_rank_ = rank_ / valid_output_replicas;
  weight_world_size_ = world_size_ / valid_output_replicas;
  CHECK(out_features % weight_world_size_ == 0)
      << "out_features " << out_features
      << " not divisible by weight_world_size " << weight_world_size_;
  const int64_t out_features_per_partition = out_features / weight_world_size_;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ =
        register_parameter("per_channel_scale",
                           torch::empty({out_features_per_partition},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    smooth_ = register_parameter(
        "smooth",
        torch::empty({in_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::register_block_fp8_parameters(*this,
                                        out_features_per_partition,
                                        in_features,
                                        options,
                                        weight_,
                                        weight_scale_inv_);
#endif
  } else if (quant_args_.quant_method() == kQuantMethodW8A8Mxfp8) {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({out_features_per_partition, (in_features + 31) / 32},
                     options.dtype(torch::kUInt8)),
        /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t weight_scale_size =
        is_fp8_channelwise_w8a8(quant_args_) ? out_features_per_partition : 1;
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({weight_scale_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  torch::Tensor output;

  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";

    output = scaled_w8a8_linear(input,
                                qweight_,
                                per_channel_scale_,
                                smooth_,
                                bias,
                                output_dtype_,
                                linear_extra_args_);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    output = musa::block_fp8_or_bf16_forward(
        input, weight_, weight_scale_inv_, bias, matmul_output_buffer_);
#endif
#if defined(USE_NPU)
  } else if ((is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
              quant_args_.quant_method() == "w4a8_mxfp4") &&
             weight_.scalar_type() == torch::kUInt8) {
    // W4A8 MXFP4: fp4 packed weights (uint8 [N, K/2]) + fp8 activations.
    // The packed weight is loaded as-is from the checkpoint's
    // 'weight_packed' key; no prepack transpose (the op handles [N, K/2]).
    output = mxfp4_linear_forward(input,
                                  weight_,
                                  weight_scale_,
                                  mxfp4_weight_decoded_,
                                  bias,
                                  output_dtype_);
  } else if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    // Lazy one-shot [N,K]->[K,N] prepack: load_state_dict runs once per
    // checkpoint shard, so the transform cannot happen at load time. Runs
    // during the first (eager prefill) forward, before any ACL graph
    // capture; idempotent via the scale dimension check.
    prepack_mxfp8_linear_weight(weight_, weight_scale_);
    output = mxfp8_linear_forward(
        input, weight_, weight_scale_, bias, output_dtype_);
#endif
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    check_fp8_activation_dynamic_supported(quant_args_);
    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_MLU) || defined(USE_DCU)
    CHECK(weight_is_loaded_) << "Missing compressed-tensors weight";
    const auto smooth =
        loaded_ct_smooth(quant_args_, smooth_, smooth_is_loaded_);
    output = scaled_w8a8_linear(input,
                                weight_,
                                weight_scale_,
                                smooth,
                                bias,
                                output_dtype_,
                                linear_extra_args_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale_, bias, output_dtype_);
#endif
  } else {
#if defined(USE_MUSA)
    output = musa::matmul_forward(input, weight_, bias, matmul_output_buffer_);
#else
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
#endif
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
}

torch::Tensor ColumnParallelLinearImpl::forward_quantized(
    const W8A8DynamicInput& input) {
  CHECK(uses_w8a8_dynamic_quant())
      << "forward_quantized requires w8a8_dynamic quant method.";
  CHECK(input.activation.defined())
      << "forward_quantized requires a quantized activation.";
  CHECK(input.per_token_scale.defined())
      << "forward_quantized requires a per-token scale.";
  CHECK_EQ(input.activation.dim(), 2)
      << "forward_quantized requires a 2D activation.";
  CHECK_EQ(input.activation.size(0), input.per_token_scale.numel())
      << "forward_quantized requires one scale per activation row.";

#if defined(USE_NPU)
  torch::Tensor output = npu_w8a8_dynamic_quantized_linear_forward(
      input.activation.to(device_),
      input.per_token_scale.to(device_),
      weight_,
      w8a8_dynamic_weight_scale(),
      bias(),
      output_dtype_);
  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
#else
  LOG(FATAL) << "forward_quantized is only supported on NPU.";
  return torch::Tensor();
#endif
}

bool ColumnParallelLinearImpl::uses_w8a8_dynamic_quant() const {
  return is_w8a8_dynamic_quant(resolved_weight_quant_method_);
}

torch::Tensor ColumnParallelLinearImpl::w8a8_dynamic_weight_scale() const {
  CHECK(uses_w8a8_dynamic_quant())
      << "w8a8_dynamic_weight_scale requires w8a8_dynamic quant method.";
  CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
      << "weight_scale is required for w8a8_dynamic quant matmul.";
  return weight_scale_;
}

std::optional<torch::Tensor> ColumnParallelLinearImpl::bias() const {
  if (bias_.defined()) {
    return bias_;
  }
  return std::nullopt;
}

// load the weight from the checkpoint
torch::Tensor ColumnParallelLinearImpl::forward_mx_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale) {
#if defined(USE_NPU)
  const bool use_mxfp8 = is_w8a8_mxfp8_quant(resolved_weight_quant_method_);
  const bool use_mxfp4 = is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
                         quant_args_.quant_method() == "w4a8_mxfp4";
  TORCH_CHECK(use_mxfp8 || use_mxfp4,
              "forward_mx_quantized requires MXFP8 or MXFP4 quant method");
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  torch::Tensor output;
  if (use_mxfp8) {
    // Lazy one-shot prepack; see ColumnParallelLinearImpl::forward.
    prepack_mxfp8_linear_weight(weight_, weight_scale_);
    output = mxfp8_linear_forward_quantized(quantized_input,
                                            input_scale,
                                            weight_,
                                            weight_scale_,
                                            bias,
                                            output_dtype_);
  } else {
    output = mxfp4_linear_forward_quantized(quantized_input,
                                            input_scale,
                                            weight_,
                                            weight_scale_,
                                            mxfp4_weight_decoded_,
                                            bias,
                                            output_dtype_);
  }
  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
#else
  (void)quantized_input;
  (void)input_scale;
  TORCH_CHECK(false, "forward_mx_quantized is only supported on NPU");
  return torch::Tensor();
#endif
}

bool ColumnParallelLinearImpl::uses_mxfp8_quant() const {
  return is_w8a8_mxfp8_quant(resolved_weight_quant_method_);
}
bool ColumnParallelLinearImpl::uses_mxfp4_quant() const {
  return is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
         quant_args_.quant_method() == "w4a8_mxfp4";
}

void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_world_size_ == 1 ? 0 : weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);

  // MXFP4: checkpoint stores packed FP4 weights as 'weight_packed'
  // (uint8 [N, K/2]) + 'weight_scale' (uint8 [N, K/32]). Load these
  // directly and skip the standard weight path.
  if (is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
      quant_args_.quant_method() == "w4a8_mxfp4") {
    check_mx_quantization_supported();
    CHECK_EQ(world_size_, 1) << "Packed MXFP4 loading currently requires TP1.";
    try {
      weight_.set_data(
          state_dict.get_tensor("weight_packed").to(options_.device()));
      mxfp4_weight_decoded_ = torch::Tensor();
      weight_is_loaded_ = true;
      if (weight_scale_is_loaded_ == false) {
        weight_scale_ =
            state_dict.get_tensor("weight_scale").to(options_.device());
        weight_scale_is_loaded_ = true;
      }
      return;  // Skip standard weight loading
    } catch (...) {
      // Fall through to standard loading if weight_packed not found
    }
  }
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 0);
    LOAD_SHARDED_WEIGHT(per_channel_scale, 0);
    // for input, there is one smooth value
    LOAD_WEIGHT(smooth);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::maybe_resolve_block_fp8_unquantized(state_dict,
                                              /*prefixes=*/nullptr,
                                              options_,
                                              weight_,
                                              weight_is_loaded_,
                                              weight_scale_inv_is_loaded_,
                                              block_fp8_resolved_unquantized_);
    if (!block_fp8_resolved_unquantized_) {
      LOAD_SHARDED_WEIGHT(weight_scale_inv, 0);
    }
    LOAD_SHARDED_WEIGHT(weight, 0);
#endif
  } else if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 0);
    if (is_fp8_channelwise_w8a8(quant_args_)) {
      LOAD_SHARDED_WEIGHT(weight_scale, 0);
    } else {
      LOAD_WEIGHT(weight_scale);
    }
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_SHARDED_WEIGHT(deq_scale, 0);
    LOAD_SHARDED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
    if (smooth_.defined()) {
      LOAD_WEIGHT(smooth);
    }
    if (weight_offset_.defined()) {
      LOAD_SHARDED_WEIGHT(weight_offset, 0);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_world_size_ == 1 ? 0 : weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, &prefixes, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    // Find the first available "smooth" tensor in prefixes (e.g.,
    // "gate.smooth", "up_proj.smooth", etc.)
    for (const auto& prefix : prefixes) {
      auto smooth_tensor_candidate = state_dict.get_tensor(prefix + "smooth");
      if (smooth_tensor_candidate.defined()) {
        // Copy the found smooth tensor to the module parameter
        CHECK_EQ(smooth_.sizes(), smooth_tensor_candidate.sizes())
            << "smooth weight size mismatch for " << state_dict.prefix()
            << "smooth";
        smooth_.copy_(smooth_tensor_candidate);
        smooth_is_loaded_ = true;
        break;
      }
    }
    LOAD_FUSED_WEIGHT(qweight, 0);
    LOAD_FUSED_WEIGHT(per_channel_scale, 0);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::maybe_resolve_block_fp8_unquantized(state_dict,
                                              /*prefixes=*/&prefixes,
                                              options_,
                                              weight_,
                                              weight_is_loaded_,
                                              weight_scale_inv_is_loaded_,
                                              block_fp8_resolved_unquantized_);
    if (!block_fp8_resolved_unquantized_) {
      LOAD_FUSED_WEIGHT(weight_scale_inv, 0);
    }
    LOAD_FUSED_WEIGHT(weight, 0);
#endif
  } else if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    LOAD_FUSED_WEIGHT(weight_scale, 0);
  } else if (is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
             quant_args_.quant_method() == "w4a8_mxfp4") {
    check_mx_quantization_supported();
    CHECK_EQ(world_size_, 1) << "Packed MXFP4 loading currently requires TP1.";
    std::vector<torch::Tensor> packed_parts;
    std::vector<torch::Tensor> scale_parts;
    for (const auto& prefix : prefixes) {
      // MXFP4 packed tensors are already checkpoint-partitioned on TP axes;
      // unlike ordinary dense weights they are not registered in the
      // sharded-weight metadata, so get_sharded_tensor() returns undefined.
      auto packed = state_dict.get_tensor(prefix + "weight_packed");
      auto scale = state_dict.get_tensor(prefix + "weight_scale");
      if (packed.defined() && scale.defined()) {
        packed_parts.push_back(packed);
        scale_parts.push_back(scale);
      }
    }
    // HF shards split gate/up across files. Each load_state_dict call may
    // therefore contain only one fused prefix; accumulate rows across calls.
    if (packed_parts.empty()) {
      return;
    }
    mxfp4_weight_decoded_ = torch::Tensor();
    auto packed_cat = torch::cat(packed_parts, 0).to(options_.device());
    auto scale_cat = torch::cat(scale_parts, 0).to(options_.device());
    if (weight_is_loaded_) {
      weight_.set_data(torch::cat({weight_, packed_cat}, 0));
      weight_scale_ = torch::cat({weight_scale_, scale_cat}, 0);
    } else {
      weight_.set_data(packed_cat);
      weight_scale_ = scale_cat;
    }
    weight_is_loaded_ = true;
    weight_scale_is_loaded_ = true;
    LOG_FIRST_N(INFO, 4) << "Loaded fused MXFP4 packed weight="
                         << weight_.sizes()
                         << " scale=" << weight_scale_.sizes();
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    if (is_fp8_channelwise_w8a8(quant_args_)) {
      LOAD_FUSED_WEIGHT(weight, 0);
      LOAD_FUSED_WEIGHT(weight_scale, 0);
    } else {
      // FP8 fused layer loading: each partition may have its own per-tensor
      // scale (unfused checkpoint). We must requantize all partitions with
      // max_scale.

      // Step 1: Collect partition info BEFORE LOAD_FUSED_WEIGHT (clears list)
      Fp8PartitionInfo partition_info;
      if (!weight_scale_is_loaded_) {
        for (const auto& prefix : prefixes) {
          auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
          if (scale_tensor.defined()) {
            partition_info.scales.push_back(
                scale_tensor.flatten().item<float>());
          }
          auto weight_tensor = state_dict.get_sharded_tensor(
              prefix + "weight", 0, rank, world_size);
          if (weight_tensor.defined()) {
            partition_info.logical_widths.push_back(weight_tensor.size(0));
          }
        }
      }

      // Step 2: Load fused weight
      LOAD_FUSED_WEIGHT(weight, 0);

      // Step 3: Requantize if needed (unfused checkpoint case)
      if (!weight_scale_is_loaded_ && !partition_info.empty()) {
        float max_scale = compute_max_scale(partition_info.scales);

        if (is_unfused_checkpoint(partition_info.scales) && weight_.defined() &&
            partition_info.logical_widths.size() ==
                partition_info.scales.size()) {
          requantize_fp8_weight(weight_,
                                partition_info.scales,
                                partition_info.logical_widths,
                                max_scale);
        }

        weight_scale_.fill_(max_scale);
        weight_scale_is_loaded_ = true;
      }

      // Step 4: Load input_scale for static activation quantization
      if (input_scale_.defined() && !input_scale_is_loaded_) {
        auto max_input_scale = load_max_input_scale(state_dict, prefixes);
        if (max_input_scale.defined()) {
          input_scale_.copy_(max_input_scale.view({1}));
          input_scale_is_loaded_ = true;
        }
      }
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    // Fused static W8A8 quantizes the shared input only once, so keep a single
    // input_scale/input_offset slot and pull the first available tensor.
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_FUSED_WEIGHT(deq_scale, 0);
    LOAD_FUSED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    LOAD_FUSED_WEIGHT(weight_scale, 0);
    if (smooth_.defined()) {
      load_smooth_from_prefixes(
          state_dict, prefixes, smooth_, smooth_list_, smooth_is_loaded_);
    }
    if (weight_offset_.defined()) {
      LOAD_FUSED_WEIGHT(weight_offset, 0);
    }
  } else {
    LOAD_FUSED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

std::optional<torch::Tensor> ColumnParallelLinearImpl::get_input_scale() const {
  if (quant_args_.quant_method() == kQuantMethodFp8 &&
      !quant_args_.activation_dynamic() && input_scale_.defined()) {
    return input_scale_;
  }
  return std::nullopt;
}

// load_state_dict for merged weights with variable shard sizes
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    int32_t shard_tensor_count,
    const std::vector<int64_t>& shard_sizes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);

  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});

  // load and merge the weights on dim 0 with variable shard sizes
  if (quant_args_.quant_method() == "smoothquant") {
    // For smoothquant, load quantized weights with variable shard sizes
    LOAD_MERGED_WEIGHT_V2(qweight, 0);
    LOAD_MERGED_WEIGHT_V2(per_channel_scale, 0);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::maybe_resolve_block_fp8_unquantized(state_dict,
                                              /*prefixes=*/nullptr,
                                              options_,
                                              weight_,
                                              weight_is_loaded_,
                                              weight_scale_inv_is_loaded_,
                                              block_fp8_resolved_unquantized_);
    if (!block_fp8_resolved_unquantized_) {
      LOAD_WEIGHT(weight_scale_inv);
      if (weight_scale_inv_is_loaded_) {
        CHECK_EQ(world_size, 1)
            << "block-fp8 merged-variable-shard scale loading requires TP=1";
      }
    }
    LOAD_MERGED_WEIGHT_V2(weight, 0);
#endif
  } else {
    if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_MERGED_WEIGHT_V2(weight_scale, 0);
    } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_WEIGHT(input_scale);
      LOAD_WEIGHT(input_offset);
      LOAD_MERGED_WEIGHT_V2(deq_scale, 0);
      LOAD_MERGED_WEIGHT_V2(quant_bias, 0);
    } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_MERGED_WEIGHT_V2(weight_scale, 0);
      if (smooth_.defined()) {
        LOAD_WEIGHT(smooth);
      }
      if (weight_offset_.defined()) {
        LOAD_MERGED_WEIGHT_V2(weight_offset, 0);
      }
    } else {
      // For regular weights, use the new merged weight loading with variable
      // shard sizes
      LOAD_MERGED_WEIGHT_V2(weight, 0);
    }
  }

  if (bias_.defined()) {
    // For bias, we might need to handle it differently based on the use case
    // For now, we'll use the same approach if bias is also sharded
    LOAD_MERGED_WEIGHT_V2(bias, 0);
  }
}

QKVParallelLinearImpl::QKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args,
    const LinearExtraArgs& linear_extra_args)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()),
      quant_args_(quant_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  (void)linear_extra_args;
#if defined(USE_MUSA)
  musa::check_quantization_supported(quant_args_);
#endif
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ =
        register_parameter("per_channel_scale",
                           torch::empty({out_features_per_partition},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    smooth_ = register_parameter(
        "smooth",
        torch::empty({hidden_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::register_block_fp8_parameters(*this,
                                        out_features_per_partition,
                                        hidden_size,
                                        options,
                                        weight_,
                                        weight_scale_inv_);
#endif
  } else if (quant_args_.quant_method() == kQuantMethodW8A8Mxfp8) {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({out_features_per_partition, (hidden_size + 31) / 32},
                     options.dtype(torch::kUInt8)),
        /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t weight_scale_size =
        is_fp8_channelwise_w8a8(quant_args_) ? out_features_per_partition : 3;
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({weight_scale_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    // Also create {3} for Q/K/V, will use max() after loading
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({3}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor QKVParallelLinearImpl::forward_mx_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale) {
#if defined(USE_NPU)
  TORCH_CHECK(is_w8a8_mxfp8_quant(resolved_weight_quant_method_),
              "forward_mx_quantized requires w8a8_mxfp8 quant method");
  prepack_mxfp8_linear_weight(weight_, weight_scale_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  return mxfp8_linear_forward_quantized(quantized_input,
                                        input_scale,
                                        weight_,
                                        weight_scale_,
                                        bias,
                                        output_dtype_);
#else
  (void)quantized_input;
  (void)input_scale;
  TORCH_CHECK(false, "forward_mx_quantized is only supported on NPU");
  return torch::Tensor();
#endif
}

bool QKVParallelLinearImpl::uses_mxfp8_quant() const {
  return is_w8a8_mxfp8_quant(resolved_weight_quant_method_);
}

torch::Tensor QKVParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;

  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";
    CHECK(smooth_.defined()) << "smooth is required for smoothquant.";

    output = scaled_w8a8_linear(
        input, qweight_, per_channel_scale_, smooth_, bias, output_dtype_);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    output = musa::block_fp8_or_bf16_forward(
        input, weight_, weight_scale_inv_, bias, matmul_output_buffer_);
#endif
#if defined(USE_NPU)
  } else if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    // Lazy one-shot prepack; see ColumnParallelLinearImpl::forward.
    prepack_mxfp8_linear_weight(weight_, weight_scale_);
    output = mxfp8_linear_forward(
        input, weight_, weight_scale_, bias, output_dtype_);
#endif
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    check_fp8_activation_dynamic_supported(quant_args_);
    auto a_scale = input_scale_.defined()
                       ? std::optional<torch::Tensor>(input_scale_)
                       : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, a_scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_MLU) || defined(USE_DCU)
    CHECK(weight_is_loaded_) << "Missing compressed-tensors weight";
    const auto smooth =
        loaded_ct_smooth(quant_args_, smooth_, smooth_is_loaded_);
    output = scaled_w8a8_linear(
        input, weight_, weight_scale_, smooth, bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale_, bias, output_dtype_);
#endif
  } else {
#if defined(USE_MUSA)
    output = musa::matmul_forward(input, weight_, bias, matmul_output_buffer_);
#else
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;

    output = xllm::kernel::matmul(matmul_params);
#endif
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

void QKVParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, &prefixes, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});
#if defined(USE_MUSA)
  const bool is_block_fp8 = musa::is_block_fp8_quant(quant_args_);
  if (is_block_fp8) {
    musa::maybe_resolve_block_fp8_unquantized(state_dict,
                                              /*prefixes=*/&prefixes,
                                              options_,
                                              weight_,
                                              weight_is_loaded_,
                                              weight_scale_inv_is_loaded_,
                                              block_fp8_resolved_unquantized_);
  }
#endif
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_QKV_WEIGHT(qweight, 0, num_kv_head_replicas_);
    LOAD_QKV_WEIGHT(per_channel_scale, 0, num_kv_head_replicas_);
    load_shared_tensor_from_prefixes_or_fail(
        state_dict, prefixes, "smooth", smooth_, smooth_is_loaded_);
  } else {
    LOAD_QKV_WEIGHT(weight, 0, num_kv_head_replicas_);
  }
  if (bias_.defined()) {
    LOAD_QKV_WEIGHT(bias, 0, num_kv_head_replicas_);
  }
  if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight_scale, 0);
#if defined(USE_MUSA)
  } else if (is_block_fp8) {
    if (!block_fp8_resolved_unquantized_) {
      LOAD_QKV_WEIGHT(weight_scale_inv, 0, num_kv_head_replicas_);
    }
#endif
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    if (is_fp8_channelwise_w8a8(quant_args_)) {
      LOAD_QKV_WEIGHT(weight_scale, 0, num_kv_head_replicas_);
      return;
    }

    // Build partition info for Q/K/V
    Fp8PartitionInfo partition_info;
    int64_t num_heads_per_partition = num_heads_ / world_size_;
    int64_t num_kv_heads_per_partition = num_kv_heads_ / world_size_;
    partition_info.logical_widths = {
        num_heads_per_partition * head_size_,      // Q
        num_kv_heads_per_partition * head_size_,   // K
        num_kv_heads_per_partition * head_size_};  // V

    for (const auto& prefix : prefixes) {
      auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
      if (scale_tensor.defined()) {
        partition_info.scales.push_back(scale_tensor.flatten().item<float>());
      }
    }

    // Requantize if unfused checkpoint (multiple scales)
    if (partition_info.scales.size() > 1 && weight_.defined()) {
      float max_scale = compute_max_scale(partition_info.scales);

      if (is_unfused_checkpoint(partition_info.scales)) {
        requantize_fp8_weight(weight_,
                              partition_info.scales,
                              partition_info.logical_widths,
                              max_scale);
      }
      weight_scale_.fill_(max_scale);
    } else if (partition_info.scales.size() == 1) {
      weight_scale_.fill_(partition_info.scales[0]);
    } else {
      LOAD_FUSED_WEIGHT(weight_scale, 0);
    }

    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_FUSED_WEIGHT(input_scale, 0);
    }

    // For per-tensor quantization with fused QKV, replace scale tensors with
    // scalar max values to avoid recomputing max() in every forward() call.
    // Only apply for per-tensor quantization.
    // Per-channel/per-block quantization should NOT take max.
    if (weight_scale_.defined() && weight_scale_.numel() > 1) {
      weight_scale_ = weight_scale_.max();
    }
    if (input_scale_.defined() && input_scale_.numel() > 1) {
      input_scale_ = input_scale_.max();
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    // input_scale/input_offset are shared activation-quant params and should
    // not inherit the KV-head replication logic used by output-channel tensors.
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_QKV_WEIGHT(deq_scale, 0, num_kv_head_replicas_);
    LOAD_QKV_WEIGHT(quant_bias, 0, num_kv_head_replicas_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_QKV_WEIGHT(weight_scale, 0, num_kv_head_replicas_);
    if (smooth_.defined()) {
      load_smooth_from_prefixes(
          state_dict, prefixes, smooth_, smooth_list_, smooth_is_loaded_);
    }
    if (weight_offset_.defined()) {
      LOAD_QKV_WEIGHT(weight_offset, 0, num_kv_head_replicas_);
    }
  }
}

void QKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  const int32_t shard_tensor_count = 3;
  const int64_t shard_size = num_heads_ * head_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});
#if defined(USE_MUSA)
  const bool is_block_fp8 = musa::is_block_fp8_quant(quant_args_);
  if (is_block_fp8) {
    musa::maybe_resolve_block_fp8_unquantized(state_dict,
                                              /*prefixes=*/nullptr,
                                              options_,
                                              weight_,
                                              weight_is_loaded_,
                                              weight_scale_inv_is_loaded_,
                                              block_fp8_resolved_unquantized_);
  }
#endif
  CHECK_EQ(num_heads_, num_kv_heads_);
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_MERGED_WEIGHT(qweight, 0);
    LOAD_MERGED_WEIGHT(per_channel_scale, 0);
    LOAD_WEIGHT(smooth);
  } else {
    LOAD_MERGED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT(bias, 0);
  }
  if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    LOAD_MERGED_WEIGHT(weight_scale, 0);
#if defined(USE_MUSA)
  } else if (is_block_fp8) {
    if (!block_fp8_resolved_unquantized_) {
      CHECK_EQ(world_size, 1)
          << "block-fp8 merged QKV scale loading requires TP=1";
      LOAD_WEIGHT(weight_scale_inv);
    }
#endif
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    const std::vector<std::string> shared_input_prefixes{""};
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             shared_input_prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             shared_input_prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_SHARDED_WEIGHT(deq_scale, 0);
    LOAD_SHARDED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
    if (smooth_.defined()) {
      LOAD_WEIGHT(smooth);
    }
    if (weight_offset_.defined()) {
      LOAD_SHARDED_WEIGHT(weight_offset, 0);
    }
  }
}

std::optional<torch::Tensor> QKVParallelLinearImpl::get_input_scale() const {
  if (quant_args_.quant_method() == kQuantMethodFp8 &&
      !quant_args_.activation_dynamic() && input_scale_.defined()) {
    // input_scale_ is already reduced to per-tensor scale in load_state_dict.
    return input_scale_;
  }
  return std::nullopt;
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : input_is_parallelized_(input_is_parallelized),
      enable_result_reduction_(enable_result_reduction),
      quant_args_(quant_args),
      options_(options),
      process_group_(process_group),
      linear_extra_args_(linear_extra_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
#if defined(USE_MUSA)
  musa::check_quantization_supported(quant_args_);
#endif
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ = register_parameter(
        "per_channel_scale",
        torch::empty({out_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    smooth_ = register_parameter("smooth",
                                 torch::empty({in_features_per_partition},
                                              options.dtype(torch::kFloat32)),
                                 /*requires_grad=*/false);
    // Output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::register_block_fp8_parameters(*this,
                                        out_features,
                                        in_features_per_partition,
                                        options,
                                        weight_,
                                        weight_scale_inv_);
#endif
  } else if (quant_args_.quant_method() == kQuantMethodW8A8Mxfp8) {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({out_features, (in_features_per_partition + 31) / 32},
                     options.dtype(torch::kUInt8)),
        /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t weight_scale_size =
        is_fp8_channelwise_w8a8(quant_args_) ? out_features : 1;
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({weight_scale_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  const RowParallelReduceMode reduce_mode =
      enable_result_reduction_ ? RowParallelReduceMode::ALL_REDUCE
                               : RowParallelReduceMode::NONE;
  return forward_impl(input, reduce_mode);
}

torch::Tensor RowParallelLinearImpl::mmrs_weight_transposed() const {
  CHECK(weight_.defined()) << "weight is required for MMRS.";
  const bool valid = mmrs_weight_t_.defined() &&
                     mmrs_weight_t_.device() == weight_.device() &&
                     mmrs_weight_t_.scalar_type() == weight_.scalar_type() &&
                     mmrs_weight_t_.size(0) == weight_.size(1) &&
                     mmrs_weight_t_.size(1) == weight_.size(0);
  if (!valid) {
    mmrs_weight_t_ = weight_.transpose(0, 1).contiguous();
  }
  return mmrs_weight_t_;
}

torch::Tensor RowParallelLinearImpl::forward(
    torch::Tensor input,
    RowParallelReduceMode reduce_mode) {
#if !defined(USE_NPU)
  reduce_mode = enable_result_reduction_ ? RowParallelReduceMode::ALL_REDUCE
                                         : RowParallelReduceMode::NONE;
#endif
  return forward_impl(input, reduce_mode);
}

torch::Tensor RowParallelLinearImpl::forward_impl(
    torch::Tensor input,
    RowParallelReduceMode reduce_mode) {
#if defined(USE_NPU)
  const bool use_fc1_reduce =
      reduce_mode == RowParallelReduceMode::REDUCE_SCATTER ||
      reduce_mode == RowParallelReduceMode::MATMUL_REDUCE_SCATTER;
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
#else
  const bool use_fc1_reduce = false;
  const FlashComm1Context* fc1_ctx = nullptr;
#endif
  auto bias = bias_.defined() && rank_ == 0
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;

  const bool skip_scatter =
      use_fc1_reduce && fc1_ctx && is_sequence_sharded(*fc1_ctx);

  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    log_mmrs_quant_skip(reduce_mode, fc1_ctx, "smoothquant", input);
    CHECK(smooth_.defined()) << "smooth is required for smoothquant.";
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";

    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    output = scaled_w8a8_linear(input,
                                qweight_,
                                per_channel_scale_,
                                smooth_,
                                bias,
                                output_dtype_,
                                linear_extra_args_);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    log_mmrs_quant_skip(reduce_mode, fc1_ctx, "block_fp8", input);
    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    output = musa::block_fp8_or_bf16_forward(
        input, weight_, weight_scale_inv_, bias, matmul_output_buffer_);
#endif
#if defined(USE_NPU)
  } else if ((is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
              quant_args_.quant_method() == "w4a8_mxfp4") &&
             weight_.scalar_type() == torch::kUInt8) {
    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    output = mxfp4_linear_forward(input,
                                  weight_,
                                  weight_scale_,
                                  mxfp4_weight_decoded_,
                                  bias,
                                  output_dtype_);
  } else if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    log_mmrs_quant_skip(reduce_mode, fc1_ctx, "mxfp8", input);
    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    // Lazy one-shot prepack; see ColumnParallelLinearImpl::forward.
    prepack_mxfp8_linear_weight(weight_, weight_scale_);
    output = mxfp8_linear_forward(
        input, weight_, weight_scale_, bias, output_dtype_);
#endif
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    log_mmrs_quant_skip(reduce_mode, fc1_ctx, "fp8", input);
    check_fp8_activation_dynamic_supported(quant_args_);

    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }

    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    log_mmrs_quant_skip(reduce_mode, fc1_ctx, "w8a8", input);
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_MLU) || defined(USE_DCU)
    CHECK(weight_is_loaded_) << "Missing compressed-tensors weight";
    const auto smooth =
        loaded_ct_smooth(quant_args_, smooth_, smooth_is_loaded_);
    output = scaled_w8a8_linear(input,
                                weight_,
                                weight_scale_,
                                smooth,
                                bias,
                                output_dtype_,
                                linear_extra_args_);
#elif defined(USE_NPU)
    // FC1 fused int8 MMRS: per-token quantize the (padded) activation, then let
    // torch_npu fuse matmul + reduce_scatter. Numerically equivalent to the
    // symmetric w8a8_dynamic path (per-channel weight scale, no weight offset).
    if (wants_mmrs(reduce_mode) && fc1_ctx && is_sequence_sharded(*fc1_ctx) &&
        fc1_ctx->enable_mmrs_fusion && !bias.has_value() && input.defined() &&
        input.dim() == 2 && input.size(0) == fc1_ctx->original_num_tokens) {
      torch::Tensor mmrs_input = input;
      if (fc1_ctx->pad_size > 0) {
        mmrs_input = pad_rows_by_copy(input, fc1_ctx->padded_num_tokens);
      }
      xllm::kernel::NpuQuantizeParams q_params;
      q_params.input = mmrs_input;
      torch::Tensor q_input;
      std::optional<torch::Tensor> pertoken_scale;
      std::tie(q_input, pertoken_scale) = xllm::kernel::dynamic_quant(q_params);
      if (pertoken_scale.has_value() && pertoken_scale->defined()) {
        const std::vector<int64_t> output_shape = {
            fc1_ctx->padded_local_num_tokens, weight_.size(0)};
        xllm::kernel::MatmulReduceScatterParams mmrs_params;
        mmrs_params.a = q_input;
        mmrs_params.b = mmrs_weight_transposed();
        mmrs_params.bias = std::nullopt;
        mmrs_params.process_group = process_group_;
        mmrs_params.comm_mode = fc1_ctx->mmrs_comm_mode;
        mmrs_params.x1_scale = pertoken_scale->reshape({-1, 1}).to(at::kFloat);
        mmrs_params.x2_scale = weight_scale_.reshape({1, -1}).to(torch::kFloat);
        mmrs_params.output_dtype = output_dtype_;
        try {
          output = xllm::kernel::matmul_reduce_scatter(mmrs_params);
        } catch (const c10::Error& error) {
          LOG_FIRST_N(WARNING, 8)
              << "FC1 w8a8 MMRS call failed; fallback reduction will run: "
              << error.what_without_backtrace();
          output = torch::Tensor();
        }
        if (output.defined() &&
            output.sizes() == torch::IntArrayRef(output_shape)) {
          retain_quantized_mmrs_launch_tensors(q_input,
                                               mmrs_params.x1_scale.value(),
                                               mmrs_params.x2_scale.value());
          return output;
        }
        if (output.defined()) {
          LOG_FIRST_N(WARNING, 8)
              << "FC1 w8a8 MMRS returned unexpected shape; fallback reduction "
                 "will run. returned="
              << output.sizes() << ", expected_local=" << output_shape;
          output = torch::Tensor();
        }
      }
    }
    if (!output.defined()) {
      log_mmrs_quant_skip(reduce_mode, fc1_ctx, "w8a8_dynamic", input);
      output = npu_w8a8_dynamic_linear_forward(
          input, weight_, weight_scale_, bias, output_dtype_);
    }
#endif
  } else {
    if (!input_is_parallelized_ && !skip_scatter) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
#if defined(USE_NPU)
    if (wants_mmrs(reduce_mode) && fc1_ctx && is_sequence_sharded(*fc1_ctx) &&
        fc1_ctx->enable_mmrs_fusion) {
      bool can_try_mmrs = input.defined() && weight_.defined() &&
                          input.dim() == 2 &&
                          input.size(0) == fc1_ctx->original_num_tokens &&
                          (!bias.has_value() || fc1_ctx->pad_size == 0);
      if (can_try_mmrs) {
        torch::Tensor mmrs_input = input;
        if (fc1_ctx->pad_size > 0) {
          mmrs_input = pad_rows_by_copy(input, fc1_ctx->padded_num_tokens);
        }

        auto output_shape = mmrs_input.sizes().vec();
        output_shape[0] = fc1_ctx->padded_local_num_tokens;
        output_shape[1] = weight_.size(0);

        xllm::kernel::MatmulReduceScatterParams mmrs_params;
        mmrs_params.a = mmrs_input;
        mmrs_params.b = mmrs_weight_transposed();
        mmrs_params.bias = bias;
        mmrs_params.process_group = process_group_;
        mmrs_params.comm_mode = fc1_ctx->mmrs_comm_mode;
        try {
          output = xllm::kernel::matmul_reduce_scatter(mmrs_params);
        } catch (const c10::Error& error) {
          LOG_FIRST_N(WARNING, 8)
              << "FC1 MMRS call failed; fallback reduction will run: "
              << error.what_without_backtrace();
          output = torch::Tensor();
        }
        if (output.defined() &&
            output.sizes() == torch::IntArrayRef(output_shape)) {
          return output;
        }
        if (output.defined()) {
          LOG_FIRST_N(WARNING, 8)
              << "FC1 MMRS returned non-local shape; fallback reduction will "
                 "run. input="
              << input.sizes() << ", weight=" << weight_.sizes()
              << ", returned_output=" << output.sizes()
              << ", expected_local_output=" << output_shape;
          output = torch::Tensor();
        }
      } else {
        LOG_FIRST_N(WARNING, 8)
            << "FC1 MMRS skipped for unsupported row-parallel shape; fallback "
               "to matmul + reduce_scatter. input="
            << input.sizes() << ", weight=" << weight_.sizes()
            << ", original_num_tokens=" << fc1_ctx->original_num_tokens
            << ", pad_size=" << fc1_ctx->pad_size
            << ", has_bias=" << bias.has_value()
            << ", input_dim=" << input.dim();
      }

      if (!output.defined()) {
        xllm::kernel::MatmulParams matmul_params;
        matmul_params.a = input;
        matmul_params.b = weight_;
        matmul_params.bias = bias;
        output = xllm::kernel::matmul(matmul_params);
      }
    } else {
      if (wants_mmrs(reduce_mode)) {
        LOG_FIRST_N(WARNING, 16)
            << "FC1 MMRS skipped before row-parallel matmul: fc1_ctx="
            << (fc1_ctx != nullptr) << ", sequence_sharded="
            << (fc1_ctx != nullptr && is_sequence_sharded(*fc1_ctx))
            << ", enable_mmrs_fusion="
            << (fc1_ctx != nullptr && fc1_ctx->enable_mmrs_fusion)
            << ", reduce_mode=" << static_cast<int>(reduce_mode)
            << ", input=" << input.sizes();
      }
      xllm::kernel::MatmulParams matmul_params;
      matmul_params.a = input;
      matmul_params.b = weight_;
      matmul_params.bias = bias;
      output = xllm::kernel::matmul(matmul_params);
    }
#else
#if defined(USE_MUSA)
    output = musa::matmul_forward(input, weight_, bias, matmul_output_buffer_);
#else
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
#endif
#endif
  }

  return apply_row_reduce(output, reduce_mode);
}

torch::Tensor RowParallelLinearImpl::apply_row_reduce(
    torch::Tensor output,
    RowParallelReduceMode reduce_mode) {
#if defined(USE_NPU)
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
#else
  const FlashComm1Context* fc1_ctx = nullptr;
#endif
  if (reduce_mode == RowParallelReduceMode::NONE) {
    return output;
  }

  if ((reduce_mode == RowParallelReduceMode::REDUCE_SCATTER ||
       reduce_mode == RowParallelReduceMode::MATMUL_REDUCE_SCATTER) &&
      fc1_ctx) {
    FlashComm1Context ctx_copy = *fc1_ctx;
    ctx_copy.tp_group = process_group_;
    return maybe_pad_and_reduce(output, ctx_copy, reduce_mode);
  }

  if (enable_result_reduction_ && world_size_ > 1) {
    output = xllm::parallel_state::reduce(output, process_group_);
  }
  return output;
}

torch::Tensor RowParallelLinearImpl::forward_mx_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale) {
  const RowParallelReduceMode reduce_mode =
      enable_result_reduction_ ? RowParallelReduceMode::ALL_REDUCE
                               : RowParallelReduceMode::NONE;
  return forward_mx_quantized(quantized_input, input_scale, reduce_mode);
}

torch::Tensor RowParallelLinearImpl::forward_mx_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale,
    RowParallelReduceMode reduce_mode) {
#if defined(USE_NPU)
  auto bias = bias_.defined() && rank_ == 0
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;
  torch::Tensor output;
  if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    // Lazy one-shot prepack; see ColumnParallelLinearImpl::forward.
    prepack_mxfp8_linear_weight(weight_, weight_scale_);
    output = mxfp8_linear_forward_quantized(quantized_input,
                                            input_scale,
                                            weight_,
                                            weight_scale_,
                                            bias,
                                            output_dtype_);
  } else if (is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
             quant_args_.quant_method() == "w4a8_mxfp4") {
    TORCH_CHECK(weight_.scalar_type() == torch::kUInt8,
                "MXFP4 forward_mx_quantized requires packed uint8 weights");
    output = mxfp4_linear_forward_quantized(quantized_input,
                                            input_scale,
                                            weight_,
                                            weight_scale_,
                                            mxfp4_weight_decoded_,
                                            bias,
                                            output_dtype_);
  } else {
    TORCH_CHECK(false,
                "forward_mx_quantized requires MXFP8 or MXFP4 quant method");
  }
  return apply_row_reduce(output, reduce_mode);
#else
  (void)quantized_input;
  (void)input_scale;
  (void)reduce_mode;
  TORCH_CHECK(false, "forward_mx_quantized is only supported on NPU");
  return torch::Tensor();
#endif
}

bool RowParallelLinearImpl::uses_mxfp8_quant() const {
  return is_w8a8_mxfp8_quant(resolved_weight_quant_method_);
}

bool RowParallelLinearImpl::uses_mxfp4_quant() const {
  return is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
         quant_args_.quant_method() == "w4a8_mxfp4";
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  // The transposed weight is derived state and must be rebuilt after reload.
  mmrs_weight_t_ = torch::Tensor();
  mxfp4_qmm_weight_ = torch::Tensor();
  mxfp4_qmm_scale_ = torch::Tensor();
  mxfp4_weight_decoded_ = torch::Tensor();
  const int64_t rank = world_size_ == 1 ? 0 : rank_;
  const int64_t world_size = world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);

  // MXFP4 row-parallel projections (for example mlp.down_proj) store packed
  // FP4 weights under weight_packed rather than the standard weight key.
  if (is_w4a8_mxfp4_quant(resolved_weight_quant_method_) ||
      quant_args_.quant_method() == "w4a8_mxfp4") {
    check_mx_quantization_supported();
    CHECK_EQ(world_size_, 1) << "Packed MXFP4 loading currently requires TP1.";
    try {
      weight_.set_data(
          state_dict.get_tensor("weight_packed").to(options_.device()));
      weight_is_loaded_ = true;
      if (!weight_scale_is_loaded_) {
        weight_scale_ =
            state_dict.get_tensor("weight_scale").to(options_.device());
        weight_scale_is_loaded_ = true;
      }
      return;
    } catch (...) {
      // Fall through to standard loading if this module is ignored or lacks
      // packed weights.
    }
  }

  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});

  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 1);
    LOAD_WEIGHT(per_channel_scale);
    LOAD_SHARDED_WEIGHT(smooth, 0);
#if defined(USE_MUSA)
  } else if (musa::is_block_fp8_quant(quant_args_)) {
    musa::maybe_resolve_block_fp8_unquantized(state_dict,
                                              /*prefixes=*/nullptr,
                                              options_,
                                              weight_,
                                              weight_is_loaded_,
                                              weight_scale_inv_is_loaded_,
                                              block_fp8_resolved_unquantized_);
    if (!block_fp8_resolved_unquantized_) {
      LOAD_SHARDED_WEIGHT(weight_scale_inv, 1);
    }
    LOAD_SHARDED_WEIGHT(weight, 1);
#endif
  } else if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_SHARDED_WEIGHT(weight_scale, 1);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_WEIGHT(deq_scale);
    if (rank_ == 0) {
      LOAD_WEIGHT(quant_bias);
    } else if (quant_bias_.defined()) {
      quant_bias_.zero_();
      quant_bias_is_loaded_ = true;
    }
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    if (smooth_.defined()) {
      LOAD_SHARDED_WEIGHT(smooth, 0);
    }
    if (weight_offset_.defined()) {
      LOAD_WEIGHT(weight_offset);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 1);
  }

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Linear layer with row parallelism.
ReplicatedLinearImpl::ReplicatedLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : quant_args_(quant_args),
      options_(options),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  (void)linear_extra_args;
#if defined(USE_MUSA)
  musa::check_quantization_supported(quant_args_);
#endif
  if (quant_args_.quant_method() == kQuantMethodW8A8Mxfp8) {
    weight_ =
        register_parameter("weight",
                           torch::empty({out_features, in_features},
                                        options.dtype(torch::kFloat8_e4m3fn)),
                           /*requires_grad=*/false);
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({out_features, (in_features + 31) / 32},
                                        options.dtype(torch::kUInt8)),
                           /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // Replicated projections are mixed in DeepSeek checkpoints: attention
    // low-rank projections can be FP8, while router gates remain BF16. Keep the
    // storage in runtime dtype initially and switch to FP8 lazily after seeing
    // the checkpoint tensor dtype in load_state_dict().
    weight_ =
        register_parameter("weight",
                           torch::empty({out_features, in_features}, options),
                           /*requires_grad=*/false);
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features}, options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ =
        register_parameter("weight",
                           torch::empty({out_features, in_features}, options),
                           /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor ReplicatedLinearImpl::forward(torch::Tensor input) {
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
#if defined(USE_NPU)
  if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
        << "weight_scale is required for MXFP8 replicated linear.";
    // Lazy one-shot prepack; see ColumnParallelLinearImpl::forward.
    prepack_mxfp8_linear_weight(weight_, weight_scale_);
    return mxfp8_linear_forward(
        input, weight_, weight_scale_, bias, output_dtype_);
  }
#endif
  if (is_fp8_dtype(weight_.scalar_type())) {
    check_fp8_activation_dynamic_supported(quant_args_);
    CHECK(weight_scale_.defined())
        << "weight_scale is required for FP8 replicated linear.";
    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    return fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  }
  if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    return npu_w8a8_linear_forward(input,
                                   weight_,
                                   input_scale_,
                                   input_offset_,
                                   deq_scale_,
                                   quant_bias,
                                   input.scalar_type());
  }
  if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_MLU) || defined(USE_DCU)
    CHECK(weight_is_loaded_) << "Missing compressed-tensors weight";
    const auto smooth =
        loaded_ct_smooth(quant_args_, smooth_, smooth_is_loaded_);
    return scaled_w8a8_linear(
        input, weight_, weight_scale_, smooth, bias, input.scalar_type());
#elif defined(USE_NPU)
    return npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale_, bias, input.scalar_type());
#endif
  }
#if defined(USE_MUSA)
  return musa::matmul_forward(input, weight_, bias, matmul_output_buffer_);
#else
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_;
  matmul_params.bias = bias;

  auto output = xllm::kernel::matmul(matmul_params);
  return output;
#endif
}

bool ReplicatedLinearImpl::uses_w8a8_dynamic_quant() const {
  return is_w8a8_dynamic_quant(resolved_weight_quant_method_);
}

torch::Tensor ReplicatedLinearImpl::w8a8_dynamic_weight_scale() const {
  CHECK(uses_w8a8_dynamic_quant())
      << "w8a8_dynamic_weight_scale requires w8a8_dynamic quant method.";
  CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
      << "weight_scale is required for w8a8_dynamic quant matmul.";
  return weight_scale_;
}

at::ScalarType ReplicatedLinearImpl::output_dtype() const {
  return output_dtype_;
}

std::optional<torch::Tensor> ReplicatedLinearImpl::bias() const {
  if (bias_.defined()) {
    return bias_;
  }
  return std::nullopt;
}

// load the weight from the checkpoint
void ReplicatedLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_,
                          smooth_,
                          smooth_is_loaded_});

#if defined(USE_MUSA)
  musa::check_replicated_weight_supported(state_dict);
#else
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    torch::Tensor checkpoint_weight = state_dict.get_tensor("weight");
    if (checkpoint_weight.defined() &&
        is_fp8_dtype(checkpoint_weight.scalar_type())) {
      const int64_t out_features = weight_.size(0);
      const int64_t in_features = weight_.size(1);
      const int64_t weight_scale_size =
          is_fp8_channelwise_w8a8(quant_args_) ? out_features : 1;
      std::vector<weight::LazyParameterSpec> specs;
      specs.reserve(quant_args_.activation_dynamic() ? 2 : 3);
      specs.push_back(weight::LazyParameterSpec{
          &weight_,
          &weight_is_loaded_,
          "weight",
          {out_features, in_features},
          options_.dtype(checkpoint_weight.scalar_type())});
      specs.push_back(
          weight::LazyParameterSpec{&weight_scale_,
                                    &weight_scale_is_loaded_,
                                    "weight_scale",
                                    {weight_scale_size},
                                    options_.dtype(torch::kFloat32)});
      if (!quant_args_.activation_dynamic()) {
        specs.push_back(
            weight::LazyParameterSpec{&input_scale_,
                                      &input_scale_is_loaded_,
                                      "input_scale",
                                      {1},
                                      options_.dtype(torch::kFloat32)});
      }
      weight::ensure_parameter_storage(this, specs);
    }
  }
#endif

  LOAD_WEIGHT(weight);
  if (is_w8a8_mxfp8_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(weight_scale);
  } else if (is_fp8_dtype(weight_.scalar_type())) {
    LOAD_WEIGHT(weight_scale);
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_WEIGHT(deq_scale);
    LOAD_WEIGHT(quant_bias);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(weight_scale);
    if (smooth_.defined()) {
      LOAD_WEIGHT(smooth);
    }
    if (weight_offset_.defined()) {
      LOAD_WEIGHT(weight_offset);
    }
  }
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm
