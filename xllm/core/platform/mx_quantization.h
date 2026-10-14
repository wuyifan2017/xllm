/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>

#include "core/platform/platform.h"

namespace xllm {

inline void check_mx_quantization_supported() {
  CHECK(Platform::is_ascend950())
      << "MXFP4/MXFP8 quantization requires Ascend950 (A5); "
         "A2/A3 and other backends are unsupported. No BF16 fallback is used.";
}

}  // namespace xllm
