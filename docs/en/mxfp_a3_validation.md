# MXFP4/MXFP8 Integration And A3 Regression

Prepared on 2026-10-14. This branch ports the selected A5 MX optimizations onto
upstream xLLM `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb` and xllm-ops
`68033814eea35b96addec2f8cef12785accb364f`. The final ops commit is recorded by
`third_party/xllm_ops`. A3 hardware validation is pending and must run on another
machine.

## Included Changes

- Dense MXFP8 weight/scale loading, one-time layout preparation, shared input
  RMSNorm/MX quantization, post-attention RMSNorm/MLP fusion and SwiGLU fusion.
- Packed MXFP4 loading without retaining the original BF16 parameter storage,
  optional native FP4 NZ layout and lossless expanded-FP8 prefill weight cache.
- Independent build capability detection, E8M0 compatibility and a symbol-free
  implementation for toolchains without MX support.
- MX Torch adapters, public declarations and build capability detection are
  owned by xllm-ops (`torch_adapter/` and `cmake/mxfp8.cmake`). xLLM consumes
  them through the pinned submodule, its existing host library, ACLNN helper
  and platform guard; model/layer integration remains in xLLM.
- Model-load and direct-call architecture checks: MXFP4/MXFP8 require Ascend950.
  Ordinary A3 BF16/INT8 paths retain upstream behavior; MX is rejected rather
  than silently converted to BF16.
- A5 MTP conv/norm optimization in both ops headers, preserving upstream kernel
  signatures and all non-A5 preprocessed operations.

MXFP4 packed loading currently requires TP1. Set `XLLM_MXFP4_W_NZ=1` to enable
FP4 NZ. Unsupported NZ shapes retain ND. Set `XLLM_MXFP4_PREFILL_FP8=1` to enable
the additional weight cache; only matrices with at least 1024 rows use it.
`XLLM_MXFP4_PREFILL_FP8_OFF=1` retains the native prefill control with the cache
allocated. These MXFP4 switches require the value `1`; `0` is disabled.
The cache consumes additional resident memory, approximately 15.93 GB for the
historically tested Qwen3.5-27B TP1 checkpoint. Capacity and throughput must be
remeasured for the final branch.

This branch does not import the entire experiment history, full-attention
calibration hooks, generated OPP packages, model checkpoints or diagnostic
prefill seed experiments. It retains upstream GDN scheduling and attention
lifecycle. Historical performance/accuracy numbers do not establish acceptance
of this new combined source version.

## Retrieve The Candidate

```bash
git clone --branch codex/mxfp-a5-a3-validation-20261014 \
  https://github.com/wuyifan2017/xllm.git xllm-mxfp-candidate
cd xllm-mxfp-candidate
git submodule update --init --recursive
git rev-parse HEAD
git -C third_party/xllm_ops rev-parse HEAD
```

The ops submodule URL points to the same account so a recursive checkout can
retrieve the exact candidate gitlink. Access to every other upstream submodule
is still required.

## A3 Validation Contract

1. Record hostname, device model/health, driver, CANN, Torch/Torch-NPU, compiler,
   source revisions, dirty state, binary hashes and OPP hashes. Use assigned idle
   devices and isolated build/install paths; preserve active services.
2. Build an upstream control and candidate with identical dependencies and
   compiler flags. The control uses the two upstream commits named above.
   Verify the candidate capability result on the real A3 toolchain. Old MX
   headers must select unavailable wrappers, without unresolved MX symbols.
3. Run the framework tests, including `HFModelLoaderTest.*` and the existing
   linear/MLP tests. Build the changed MTP operator for the detected A3 compute
   unit and run its numerical/state/graph regression tests.
4. Serve the same BF16 model and a supported INT8 model with the same TP, cache,
   chunking, graph/eager and sampling settings. Freeze deterministic prompts and
   seeds. Check actual generation semantics, completion counts/token lengths,
   numerical outputs where available and clean shutdown. A readiness endpoint
   alone is insufficient.
5. Repeat short/long prefill, concurrency 1 and multiple requests, graph replay
   and supported MTP modes. Include repeated requests, model reload and shutdown
   after workloads. Compare results to the control and preserve raw outputs.
6. Attempt MXFP4 and MXFP8 model loading on A3. Both must reject with an explicit
   Ascend950 requirement before allocating quantized model weights. Confirm
   ordinary BF16/INT8 still loads afterward.
7. Publish completed/total test counts, failures, exact configuration and raw
   artifact hashes. Mark A3 unaffected only after its required tests pass.

Source preprocessing and syntax checks are preparation evidence. They do not
replace a clean build, full server link, operator execution or A3 generation.

## Preparation Results

The isolated preparation environment used the existing A5 CANN/Torch-NPU
headers without replacing any running server or installed operator package.

- Changed translation units: 18/18 syntax checks passed with detected MX
  capability macros explicitly enabled and disabled.
- Production compressed-tensors parser functions: 12/12 focused tests passed,
  including MXFP4, MXFP8, ordinary INT8 and conflicting quantization groups.
  This harness does not run the full `HFModelLoaderTest` executable.
- Real CMake capability module: enabled and disabled configurations passed.
- Production architecture predicate and direct-call guard: 6/6 cases passed
  using injected SoC names, including Ascend950, Ascend910 and missing names.
- Ops non-A5 preprocessing: 4/4 complete token-stream comparisons passed for
  generic/arch35 headers in vector/cube modes after expanding helper aliases.

Logs are retained under
`/data/w30082966/mxfp_upstream_merge_20261014/validation` in `wyf_A5_Perf`.
No clean full build, full server link or hardware execution was performed for
this merged revision. A3 regression remains pending.
