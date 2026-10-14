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

#include "dense_mlp.h"

#include <glog/logging.h>

#include "common/flash_comm1_context.h"
#include "kernels/ops_api.h"
#include "platform/platform.h"

#if defined(USE_NPU)
#include <atomic>
#include <cstdlib>

#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif

namespace xllm {
namespace layer {

namespace {

#if defined(USE_NPU)
W8A8DynamicInput quantize_and_gather_w8a8_dynamic_input(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_ctx) {
  xllm::kernel::NpuQuantizeParams quantize_params;
  quantize_params.input = input;

  torch::Tensor quantized_input;
  std::optional<torch::Tensor> per_token_scale;
  std::tie(quantized_input, per_token_scale) =
      xllm::kernel::dynamic_quant(quantize_params);
  CHECK(per_token_scale.has_value() && per_token_scale->defined())
      << "dynamic_quant must return per-token scale before AllGather.";

  torch::Tensor gathered_input = gather_sequence(quantized_input, fc1_ctx);
  torch::Tensor gathered_scale =
      gather_sequence(per_token_scale->reshape({-1, 1}), fc1_ctx).reshape({-1});
  return W8A8DynamicInput{gathered_input, gathered_scale};
}
#endif

#if defined(USE_NPU)
// Process-wide decision for the fused SiLU-gate + MX-quant path:
// 0 = undecided, 1 = gate on the left half, 2 = gate on the right half,
// 3 = fused op unusable (fall back to act + per-linear quant).
std::atomic<int32_t> g_mxfp8_swiglu_fused_mode{0};
constexpr int32_t kMxfp8SwigluLeft = 1;
constexpr int32_t kMxfp8SwigluRight = 2;
constexpr int32_t kMxfp8SwigluOff = 3;
std::atomic<int32_t> g_mxfp4_swiglu_fused_mode{0};
constexpr int32_t kMxfp4SwigluLeft = 1;
constexpr int32_t kMxfp4SwigluRight = 2;
constexpr int32_t kMxfp4SwigluOff = 3;
#endif

}  // namespace

DenseMLPImpl::DenseMLPImpl(int64_t hidden_size,
                           int64_t intermediate_size,
                           bool is_gated,
                           bool has_bias,
                           const std::string& hidden_act,
                           bool enable_result_reduction,
                           const QuantArgs& quant_args,
                           ProcessGroup* process_group,
                           const torch::TensorOptions& options,
                           const std::string& module_prefix,
                           double swiglu_limit,
                           bool apply_fc1_sequence_parallel)
    : is_gated_(is_gated),
      intermediate_size_(intermediate_size),
      process_group_(process_group),
      hidden_act_(hidden_act),
      swiglu_limit_(swiglu_limit),
      apply_fc1_sequence_parallel_(apply_fc1_sequence_parallel) {
  // Check if using w8a8 smoothquant quantization
  is_smoothquant_ = quant_args.quant_method() == kQuantMethodSmoothquant;

  if (is_smoothquant_) {
    // Safety check: only w8a8 smoothquant is supported
    if (quant_args.bits() != 8 || !quant_args.activation_dynamic()) {
      LOG(FATAL)
          << "DenseMLP w8a8 mode only supports w8a8 smoothquant quantization. "
          << "Got bits=" << quant_args.bits()
          << ", activation_dynamic=" << quant_args.activation_dynamic();
    }
  }

  // Determine extra args based on quantization mode
  LinearExtraArgs gate_up_proj_extra_args("none", false);
  LinearExtraArgs down_proj_extra_args("none", false);
  if (is_smoothquant_) {
    // For per-token smoothquant, use specific args
    down_proj_extra_args = LinearExtraArgs(hidden_act_, is_gated_);
  }

  // 1. gate + up
  int64_t out_feature = is_gated_ ? intermediate_size_ * 2 : intermediate_size_;
  gate_up_proj_ =
      register_module("gate_up_proj",
                      ColumnParallelLinear(hidden_size,
                                           out_feature,
                                           /*bias=*/has_bias,
                                           /*gather_output=*/false,
                                           quant_args,
                                           process_group_,
                                           options,
                                           gate_up_proj_extra_args));

  act_ =
      register_module("act", Activation(hidden_act_, is_gated_, swiglu_limit_));

  // 2. down
  const auto down_proj_quant_args =
      module_prefix.empty()
          ? quant_args
          : quant_args.for_module(module_prefix + ".down_proj");
  down_proj_ = register_module("down_proj",
                               RowParallelLinear(intermediate_size_,
                                                 hidden_size,
                                                 /*bias=*/has_bias,
                                                 /*input_is_parallelized=*/true,
                                                 enable_result_reduction,
                                                 down_proj_quant_args,
                                                 process_group_,
                                                 options,
                                                 down_proj_extra_args));
}

torch::Tensor DenseMLPImpl::forward_from_gate_up(torch::Tensor gate_up) {
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  const bool use_fc1_sequence_parallel =
      apply_fc1_sequence_parallel_ && fc1_ctx && is_sequence_sharded(*fc1_ctx);

  if (is_smoothquant_) {
    if (use_fc1_sequence_parallel) {
      return down_proj_->forward(gate_up,
                                 row_parallel_reduce_mode_for_fc1(*fc1_ctx));
    }

    return down_proj_->forward(gate_up);
  }

#if defined(USE_NPU)
  // Fused SiLU-gate + MX-quant path: replaces act + down_proj quant with one
  // op (npu_swiglu_mx_quant), removing one bf16 [M, I] round trip per layer.
  // Only taken when the load-time verification accepted the fused op.
  const int32_t mxfp8_mode = g_mxfp8_swiglu_fused_mode.load();
  if ((mxfp8_mode == kMxfp8SwigluLeft || mxfp8_mode == kMxfp8SwigluRight) &&
      down_proj_ && down_proj_->uses_mxfp8_quant()) {
    auto fused = xllm::kernel::npu::mxfp8_swiglu_quant(
        gate_up,
        /*activate_left=*/mxfp8_mode == kMxfp8SwigluLeft,
        /*clamp_limit=*/swiglu_limit_);
    if (use_fc1_sequence_parallel) {
      return down_proj_->forward_mx_quantized(
          std::get<0>(fused),
          std::get<1>(fused),
          row_parallel_reduce_mode_for_fc1(*fc1_ctx));
    }
    return down_proj_->forward_mx_quantized(std::get<0>(fused),
                                            std::get<1>(fused));
  }
  const int32_t mxfp4_mode = g_mxfp4_swiglu_fused_mode.load();
  if ((mxfp4_mode == kMxfp4SwigluLeft || mxfp4_mode == kMxfp4SwigluRight) &&
      down_proj_ && down_proj_->uses_mxfp4_quant()) {
    auto fused = xllm::kernel::npu::mxfp4_swiglu_quant(
        gate_up,
        /*activate_left=*/mxfp4_mode == kMxfp4SwigluLeft,
        /*clamp_limit=*/swiglu_limit_);
    if (use_fc1_sequence_parallel) {
      return down_proj_->forward_mx_quantized(
          std::get<0>(fused),
          std::get<1>(fused),
          row_parallel_reduce_mode_for_fc1(*fc1_ctx));
    }
    return down_proj_->forward_mx_quantized(std::get<0>(fused),
                                            std::get<1>(fused));
  }
#endif

  torch::Tensor output;
  if (!Platform::is_npu() && !Platform::is_musa()) {
    const int64_t batch_size = gate_up.sizes()[0];
    output = torch::empty(
        {batch_size, intermediate_size_ / process_group_->world_size()},
        gate_up.options());
  }

  act_->forward(gate_up, output);

  if (use_fc1_sequence_parallel) {
    return down_proj_->forward(output,
                               row_parallel_reduce_mode_for_fc1(*fc1_ctx));
  }
  return down_proj_->forward(output);
}

torch::Tensor DenseMLPImpl::forward_mx_quantized(
    const torch::Tensor& quantized_input,
    const torch::Tensor& input_scale) {
  torch::Tensor gate_up =
      gate_up_proj_->forward_mx_quantized(quantized_input, input_scale);
  return forward_from_gate_up(gate_up);
}

bool DenseMLPImpl::gate_up_uses_mxfp8() const {
#if defined(XLLM_HAS_NPU_MXFP8)
  // The fused Add+RMSNorm+MX quant op produces the FP8 activation format
  // required by both MXFP8 W8A8 and MXFP4 W4A8 gate/up projections.
  return gate_up_proj_ && (gate_up_proj_->uses_mxfp8_quant() ||
                           gate_up_proj_->uses_mxfp4_quant());
#else
  return false;
#endif
}

torch::Tensor DenseMLPImpl::forward(const torch::Tensor& hidden_states) {
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  const bool use_fc1_sequence_parallel =
      apply_fc1_sequence_parallel_ && fc1_ctx && is_sequence_sharded(*fc1_ctx);
  torch::Tensor gate_up;
#if defined(USE_NPU)
  const bool use_quantized_allgather = use_fc1_sequence_parallel &&
                                       hidden_states.dim() == 2 &&
                                       gate_up_proj_->uses_w8a8_dynamic_quant();
  if (use_quantized_allgather) {
    const W8A8DynamicInput quantized_input =
        quantize_and_gather_w8a8_dynamic_input(hidden_states, *fc1_ctx);
    gate_up = gate_up_proj_->forward_quantized(quantized_input);
  }
#endif
  if (!gate_up.defined()) {
    torch::Tensor h = hidden_states;
    if (use_fc1_sequence_parallel) {
      h = gather_sequence(hidden_states, *fc1_ctx);
    }
    gate_up = gate_up_proj_->forward(h);
  }

  return forward_from_gate_up(gate_up);
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict) {
  gate_proj_weight_seen_ =
      gate_proj_weight_seen_ || state_dict.has("gate_proj.weight");
  up_proj_weight_seen_ =
      up_proj_weight_seen_ || state_dict.has("up_proj.weight");
  gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix("down_proj."));
  maybe_decide_mxfp8_swiglu_fusion();
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict,
                                   const std::vector<std::string>& gate_up_name,
                                   const std::string& down_name) {
  if (is_gated_) {
    CHECK_EQ(gate_up_name.size(), 2);
    gate_proj_weight_seen_ =
        gate_proj_weight_seen_ || state_dict.has(gate_up_name[0] + "weight");
    up_proj_weight_seen_ =
        up_proj_weight_seen_ || state_dict.has(gate_up_name[1] + "weight");
    gate_up_proj_->load_state_dict(state_dict, gate_up_name);
  } else {
    CHECK_EQ(gate_up_name.size(), 1);
    up_proj_weight_seen_ =
        up_proj_weight_seen_ || state_dict.has(gate_up_name[0] + "weight");
    gate_up_proj_->load_state_dict(
        state_dict.get_dict_with_prefix(gate_up_name[0]));
  }
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix(down_name));
  maybe_decide_mxfp8_swiglu_fusion();
}

void DenseMLPImpl::maybe_decide_mxfp8_swiglu_fusion() {
#if defined(USE_NPU)
  if (!is_gated_ || (hidden_act_ != "silu" && hidden_act_ != "swiglu")) {
    g_mxfp8_swiglu_fused_mode.store(kMxfp8SwigluOff);
    g_mxfp4_swiglu_fused_mode.store(kMxfp4SwigluOff);
    return;
  }
  const bool use_mxfp8 = down_proj_->uses_mxfp8_quant();
  const bool use_mxfp4 = down_proj_->uses_mxfp4_quant();
  const char* disable_mxfp4_fusion =
      std::getenv("XLLM_DISABLE_MXFP4_SWIGLU_FUSION");
  if (use_mxfp4 && disable_mxfp4_fusion != nullptr &&
      disable_mxfp4_fusion[0] == '1') {
    g_mxfp4_swiglu_fused_mode.store(kMxfp4SwigluOff);
    LOG_FIRST_N(INFO, 1)
        << "DenseMLP: MXFP4 SwiGLU fusion disabled for diagnosis";
  }
  if (!use_mxfp8 && !use_mxfp4) {
    // load_state_dict runs once per checkpoint shard; down_proj may not be
    // resolved or loaded on this call. Keep the decision pending -- the
    // undecided mode falls through to the unfused path, and a genuinely
    // non-mxfp8 down_proj never enters the fused branch anyway.
    return;
  }
  const int64_t world_size =
      process_group_ == nullptr
          ? 1
          : std::max<int64_t>(1, process_group_->world_size());
  const int64_t i_local = intermediate_size_ / world_size;
  if (i_local <= 0 || i_local % 64 != 0) {
    g_mxfp8_swiglu_fused_mode.store(kMxfp8SwigluOff);
    g_mxfp4_swiglu_fused_mode.store(kMxfp4SwigluOff);
    return;
  }
  try {
    const auto options = down_proj_->weight().options().dtype(torch::kBFloat16);
    auto sample = torch::randn({8, 2 * i_local}, options) * 0.5;
    torch::Tensor ref_act;
    act_->forward(sample, ref_act);  // ground truth incl. swish clamp
    const int64_t verdict = xllm::kernel::npu::verify_mxfp8_swiglu_quant(
        sample, ref_act, swiglu_limit_);
    const bool valid = verdict == 0 || verdict == 1;
    if (use_mxfp8 && g_mxfp8_swiglu_fused_mode.load() == 0) {
      if (verdict == 1) {
        g_mxfp8_swiglu_fused_mode.store(kMxfp8SwigluLeft);
      } else if (verdict == 0) {
        g_mxfp8_swiglu_fused_mode.store(kMxfp8SwigluRight);
      } else {
        g_mxfp8_swiglu_fused_mode.store(kMxfp8SwigluOff);
      }
    }
    if (use_mxfp4 && g_mxfp4_swiglu_fused_mode.load() == 0) {
      if (valid) {
        const bool activate_left = verdict == 1;
        // The FP8 verifier establishes the gate ordering and activation
        // semantics; W4A8 retains FP8 activations for the FP4 weight matmul.
        auto probe = xllm::kernel::npu::mxfp4_swiglu_quant(
            sample, activate_left, swiglu_limit_);
        TORCH_CHECK(
            std::get<0>(probe).defined() && std::get<1>(probe).defined() &&
                std::get<0>(probe).scalar_type() == torch::kFloat8_e4m3fn &&
                std::get<0>(probe).size(1) == i_local,
            "W4A8 SwiGLU quant must return FP8 [M, I] activations");
        g_mxfp4_swiglu_fused_mode.store(activate_left ? kMxfp4SwigluLeft
                                                      : kMxfp4SwigluRight);
        LOG(INFO) << "DenseMLP: fused SiLU-gate + MXFP4 quant enabled (gate="
                  << (activate_left ? "left" : "right") << ")";
      } else {
        g_mxfp4_swiglu_fused_mode.store(kMxfp4SwigluOff);
      }
    }
    if (!valid) {
      LOG(WARNING)
          << "DenseMLP: fused SiLU-gate + MX-quant failed verification; "
             "falling back to separate act + quant";
    }
  } catch (const std::exception& e) {
    if (use_mxfp8) {
      g_mxfp8_swiglu_fused_mode.store(kMxfp8SwigluOff);
    }
    if (use_mxfp4) {
      g_mxfp4_swiglu_fused_mode.store(kMxfp4SwigluOff);
    }
    LOG(WARNING) << "DenseMLP: fused SiLU-gate + MX-quant disabled: "
                 << e.what();
  }
#endif
}

void DenseMLPImpl::verify_loaded_weights(const std::string& prefix) const {
  if (!gate_up_proj_->is_weight_loaded()) {
    if (is_gated_) {
      CHECK(gate_proj_weight_seen_)
          << "weight is not loaded for " << prefix + "gate_proj.weight";
    }
    CHECK(up_proj_weight_seen_)
        << "weight is not loaded for " << prefix + "up_proj.weight";
  }
  CHECK(gate_up_proj_->is_weight_loaded())
      << "weight is not loaded for " << prefix
      << (is_gated_ ? "{gate_proj,up_proj}.weight" : "up_proj.weight");
  CHECK(down_proj_->is_weight_loaded())
      << "weight is not loaded for " << prefix + "down_proj.weight";
}

std::optional<torch::Tensor> DenseMLPImpl::get_fp8_input_scale() const {
  if (gate_up_proj_) {
    return gate_up_proj_->get_input_scale();
  }
  return std::nullopt;
}

}  // namespace layer
}  // namespace xllm
