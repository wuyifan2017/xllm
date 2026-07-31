/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

// MUSA builds place torch_musa kernel sources under kernels/musa/ but expose
// them in the xllm::kernel::cuda namespace so layers/runtime can share the
// CUDA graph code path. Native MUSA symbols live in xllm::kernel::musa.

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype);

// Fused token-replace for schedule-overlap decode path.
// For each position i: if dst[i] < 0, set dst[i] = src[(-dst[i]) - 1].
// Otherwise dst[i] is left unchanged.  Modifies dst in-place.
// Declared in the musa namespace (not cuda) because the call site
// (worker_impl.cpp) is compiled without the musamapping plugin, so
// the cuda->musa token rewrite would not apply there.
void replace_token(torch::Tensor& dst,
                   torch::Tensor& src,
                   bool synchronize_stream = true);

}  // namespace xllm::kernel::musa

namespace xllm::kernel::cuda {

// TODO: add head_size parameter
void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox);

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void mul_sigmoid_gate_inplace(torch::Tensor& out, const torch::Tensor& gate);

void fused_shared_expert_gate_inplace(torch::Tensor& shared_output,
                                      const torch::Tensor& hidden_states,
                                      const torch::Tensor& gate_weight);

void reshape_paged_cache(torch::Tensor slot_ids,
                         torch::Tensor keys,
                         torch::Tensor values,
                         torch::Tensor key_cache,
                         torch::Tensor value_cache);

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype);

void batch_prefill(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& mask = std::nullopt);

void batch_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse);

void batch_prefill_non_causal(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    const std::optional<torch::Tensor>& mask = std::nullopt);

void batch_chunked_prefill(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    bool causal = true,
    const torch::Tensor& paged_kv_indptr_host = torch::Tensor(),
    const torch::Tensor& paged_kv_indices_host = torch::Tensor(),
    const torch::Tensor& paged_kv_last_page_len_host = torch::Tensor());

void batch_chunked_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    bool causal = true,
    const torch::Tensor& paged_kv_indptr_host = torch::Tensor(),
    const torch::Tensor& paged_kv_indices_host = torch::Tensor(),
    const torch::Tensor& paged_kv_last_page_len_host = torch::Tensor());

void batch_decode(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    bool use_tensor_core,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    const torch::Tensor& paged_kv_indptr_host = torch::Tensor(),
    const torch::Tensor& paged_kv_indices_host = torch::Tensor(),
    const torch::Tensor& paged_kv_last_page_len_host = torch::Tensor());
void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                torch::Tensor& output,
                torch::Tensor& output_lse);

// Dense ragged FA3 prefill (Mate mutlass flash_attn_varlen).
// Specialized for bf16, head_dim=256, GQA ratios 6 and 8
// (Qwen3.5-27B/35B TP=1).
void fa3_prefill(const torch::Tensor& query,
                 const torch::Tensor& key,
                 const torch::Tensor& value,
                 const torch::Tensor& cu_seqlens_q,
                 const torch::Tensor& cu_seqlens_k,
                 int64_t max_seqlen_q,
                 int64_t max_seqlen_k,
                 int64_t window_left,
                 int64_t window_right,
                 double sm_scale,
                 torch::Tensor& output,
                 torch::Tensor& output_lse);

// Paged-KV FA3 prefill (Mate flash_attn_with_kvcache). Unlike the
// dense-ragged variant above, this consumes the KV cache populated by
// reshape_paged_cache and the rectangular page table. It is used when a
// prefill extends an existing cached prefix.
torch::Tensor fa3_prefill_scheduler_metadata(
    const torch::Device& device,
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    int32_t window_size_right,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k_new,
    const torch::Tensor& seqused_k);

void fa3_prefill_paged(const torch::Tensor& query,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache,
                       const torch::Tensor& cu_seqlens_q,
                       const torch::Tensor& cu_seqlens_k_new,
                       const torch::Tensor& seqused_k,
                       const torch::Tensor& page_table,
                       const torch::Tensor& scheduler_metadata,
                       int64_t max_seqlen_q,
                       int64_t window_left,
                       int64_t window_right,
                       double sm_scale,
                       torch::Tensor& output,
                       torch::Tensor& output_lse);

// Piecewise-graph-aware dense ragged FA3 prefill. During capture this splits
// the graph and registers a replay runner; eager calls dispatch directly.
void fa3_prefill_with_optional_piecewise_capture(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t window_left,
    int64_t window_right,
    double sm_scale,
    torch::Tensor& output,
    torch::Tensor& output_lse);

torch::Tensor fa3_decode_scheduler_metadata(const torch::Device& device,
                                            int32_t batch_size,
                                            int32_t num_heads_q,
                                            int32_t num_heads_kv,
                                            int32_t head_dim_qk,
                                            int32_t head_dim_vo,
                                            int32_t max_seqlen_q,
                                            int32_t max_seqlen_k,
                                            int32_t window_size_left,
                                            int32_t window_size_right,
                                            const torch::Tensor& cu_seqlens_q,
                                            const torch::Tensor& seqused_k);

void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);

void fused_add_rms_norm(torch::Tensor& input,
                        torch::Tensor& residual,
                        torch::Tensor& weight,
                        double epsilon);

void gemma_rms_norm(torch::Tensor output,
                    torch::Tensor input,
                    torch::Tensor weight,
                    double eps);

void fused_add_gemma_rms_norm(torch::Tensor& input,
                              torch::Tensor& residual,
                              torch::Tensor& weight,
                              double epsilon);

void fused_qk_norm_rope(torch::Tensor& qkv,
                        int64_t num_heads_q,
                        int64_t num_heads_k,
                        int64_t num_heads_v,
                        int64_t head_dim,
                        double eps,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin_cache,
                        bool interleaved,
                        const torch::Tensor& position_ids,
                        int64_t k_head_offset = 0);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias,
                     std::optional<torch::Tensor> output_buf = std::nullopt);

void gdn_fused_qkvzba_split_contiguous(const torch::Tensor& mixed_qkvz,
                                       const torch::Tensor& mixed_ba,
                                       torch::Tensor& mixed_qkv,
                                       torch::Tensor& z,
                                       torch::Tensor& b,
                                       torch::Tensor& a,
                                       int64_t num_heads_qk,
                                       int64_t num_heads_v,
                                       int64_t head_qk,
                                       int64_t head_v);

void partial_rotary_embedding_inplace(torch::Tensor& positions,
                                      torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      int64_t head_size,
                                      int64_t rotary_dim,
                                      bool is_neox);

void cutlass_scaled_mm(torch::Tensor& c,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

void static_scaled_fp8_quant(torch::Tensor& out,
                             torch::Tensor const& input,
                             torch::Tensor const& scale);

std::tuple<torch::Tensor, torch::Tensor> fp8_scaled_quantize(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output = std::nullopt,
    const std::optional<torch::Tensor>& scale = std::nullopt);

void rms_norm_static_fp8_quant(torch::Tensor& out,
                               torch::Tensor& input,
                               torch::Tensor& weight,
                               torch::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::Tensor& out,
                                         torch::Tensor& input,
                                         torch::Tensor& residual,
                                         torch::Tensor& weight,
                                         torch::Tensor& scale,
                                         double epsilon);

torch::Tensor fp8_scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Native DeepSeek block-wise FP8 GEMM (mate/muDNN groupwise, GROUP_BLOCK
// (1,128,128)). See kernels/musa/fp8_block_gemm.cpp for the layout contract.
torch::Tensor gemm_fp8_nt_groupwise(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Fused DeepSeek per-token-group FP8 activation quantization (bf16 -> e4m3,
// group=128 along K). Returns {q [M,K] e4m3, scale [M, K/128] fp32}. See
// kernels/musa/fp8_act_quant.cu.
std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size);

// MUSA-specific Qwen3.5 MoE preprocess. Token-major contiguous preprocess
// returns {fp8_rows, scales, src_to_dst,
// expert_counts} while fusing expert placement, hidden-state replication, and
// g128 FP8 quantization.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_preprocess_fp8(const torch::Tensor& input,
                         const torch::Tensor& topk_ids,
                         int64_t num_experts,
                         int64_t group_size);

// BF16 token-major contiguous MoE preprocess. Returns
// {padded_hidden, row_expert_ids, original_to_padded, group_m_counts}; each
// expert occupies an aligned block and padding rows in row_expert_ids are -1.
// group_m_counts sums to the padded M and can be passed directly to Mate's
// m-grouped contiguous GEMM. Fused preprocess avoids the per-layer
// sort/index/cumsum sequence in the long-prefill path.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_preprocess_bf16(const torch::Tensor& input,
                          const torch::Tensor& topk_ids,
                          int64_t num_experts,
                          int64_t alignment);

// Decode-only fixed-block Ragged MoE helpers. Each routed assignment owns one
// 128-row block and stores its valid row at the block start. This keeps graph
// shapes static while allowing Mate's Ragged kernel to skip all padding rows.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_ragged_preprocess_fp8(const torch::Tensor& input,
                                const torch::Tensor& topk_ids,
                                int64_t group_size,
                                int64_t alignment);

std::tuple<torch::Tensor, torch::Tensor> fused_moe_ragged_preprocess_bf16(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    int64_t alignment);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_decode_preprocess_bf16(const torch::Tensor& input,
                                 const torch::Tensor& topk_ids,
                                 int64_t num_experts,
                                 int64_t alignment);

torch::Tensor fused_moe_ragged_swiglu_bf16(const torch::Tensor& input,
                                           int64_t alignment);

torch::Tensor fused_moe_indexed_swiglu_bf16(const torch::Tensor& input,
                                            const torch::Tensor& valid_rows);

std::tuple<torch::Tensor, torch::Tensor> fused_moe_ragged_swiglu_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size,
    int64_t alignment);

torch::Tensor fused_moe_ragged_combine(const torch::Tensor& down,
                                       const torch::Tensor& topk_weights,
                                       int64_t num_tokens,
                                       int64_t alignment);

// Token-major FP8 MoE decode path. The AOT MUBINs
// are specialized for MP31 and decode batch sizes 1 through 8; helper artifacts
// provide routing alignment, SwiGLU, and final top-k reduction.
bool musa_fused_moe_aot_available(int64_t num_tokens);

bool musa_fused_moe_bf16_aot_available(int64_t num_tokens);

void prepare_musa_fused_moe_aot(const torch::Device& device);

void prepare_musa_fused_moe_bf16_aot(const torch::Device& device);

torch::Tensor musa_fused_moe_aot_fp8(const torch::Tensor& hidden_states,
                                     const torch::Tensor& w13,
                                     const torch::Tensor& w13_scale,
                                     const torch::Tensor& w2,
                                     const torch::Tensor& w2_scale,
                                     const torch::Tensor& topk_weights,
                                     const torch::Tensor& topk_ids);

torch::Tensor musa_fused_moe_aot_bf16(const torch::Tensor& hidden_states,
                                      const torch::Tensor& w13,
                                      const torch::Tensor& w2,
                                      const torch::Tensor& topk_weights,
                                      const torch::Tensor& topk_ids);

std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    torch::Device device);

std::pair<torch::Tensor, torch::Tensor> compute_topk_general(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device);

torch::Tensor air_log_softmax_last_dim(const torch::Tensor& input,
                                       const torch::Tensor& temperatures);

void fused_qk_norm_rope(torch::Tensor& qkv,
                        int64_t num_heads_q,
                        int64_t num_heads_k,
                        int64_t num_heads_v,
                        int64_t head_dim,
                        double eps,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin_cache,
                        bool interleaved,
                        const torch::Tensor& position_ids);

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func);

torch::Tensor random_sample(const torch::Tensor& probs);

// Target-only speculative rejection sampling for the common MTP K=1 case.
// draft_probs contains the selected draft-token probability with shape [B, 1]
// and target_probs has shape [B, 1, V].  The returned tensor is [B, 2] with
// rejected suffix positions masked to -1.
torch::Tensor rejection_sample_target_only_k1(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& recovery_exponential,
    const torch::Tensor& bonus_token_ids);

// Mate grouped MoE GEMM entry points.  The MUSA Qwen3.5 MoE path uses the
// masked layout for both BF16 and block-wise FP8 expert weights.  Keeping the
// wrapper in the MUSA API makes the layer independent of the Python Mate
// package while still using the production Mate grouped-GEMM kernels.
torch::Tensor masked_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& token_counts,
                                   torch::ScalarType output_dtype,
                                   int64_t expected_tokens);

torch::Tensor masked_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& token_counts,
                                  torch::ScalarType output_dtype,
                                  int64_t expected_tokens);

// Compact BF16 grouped GEMM. Input rows must be sorted by expert and
// token_counts must contain the number of consecutive rows for each expert.
torch::Tensor contiguous_moe_gemm_bf16(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& token_counts,
                                       torch::ScalarType output_dtype);

// Fixed-block BF16 Ragged GEMM. row_expert_ids has one entry per input row;
// valid expert ids occur in aligned blocks and padding rows contain -1.
torch::Tensor ragged_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& row_expert_ids,
                                   torch::ScalarType output_dtype,
                                   int64_t alignment);

// Compact FP8 grouped GEMM. Input rows must be sorted by expert and
// token_counts must contain the number of consecutive rows for each expert.
// Unlike the masked layout, this path allocates and computes only valid routed
// assignments.
torch::Tensor contiguous_moe_gemm_fp8(const torch::Tensor& input,
                                      const torch::Tensor& input_scale,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& weight_scale,
                                      const torch::Tensor& token_counts,
                                      torch::ScalarType output_dtype);

// Fixed-block FP8 Ragged GEMM. row_expert_ids has one entry per input row;
// valid expert ids occur at aligned block starts and padding rows contain -1.
torch::Tensor ragged_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& row_expert_ids,
                                  torch::ScalarType output_dtype,
                                  int64_t alignment);

// MUSA top-k kernel fuses softmax, top-k selection, and selected
// weight renormalization. This is intended for small decode graph buckets;
// large prefill continues to use the existing route.
std::tuple<torch::Tensor, torch::Tensor> musa_moe_topk_softmax(
    const torch::Tensor& router_logits,
    int64_t topk);
bool musa_moe_topk_softmax_available();

// Graph-safe routed-index construction. Returns {src_dst, dst_src,
// expert_sizes}, where src_dst maps each original assignment to its compact
// expert-grouped row and dst_src is the inverse mapping.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts);

torch::Tensor moe_combine_result(const torch::Tensor& gemm2,
                                 const torch::Tensor& reduce_weight,
                                 int64_t N,
                                 int32_t topk);

torch::Tensor moe_combine_result_indexed(const torch::Tensor& gemm2_sorted,
                                         const torch::Tensor& sorted_positions,
                                         const torch::Tensor& reduce_weight,
                                         int64_t N,
                                         int32_t topk);

}  // namespace xllm::kernel::cuda

#include "core/kernels/musa/attention_runner.h"
#include "core/kernels/musa/gdn_ops.h"
