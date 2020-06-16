// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <cuda_runtime_api.h>
#include <tensorview/tensorview.h>
#include <torch/script.h>
namespace spconv {

void fused_conv_cuda(torch::Tensor output, torch::Tensor features,
                     torch::Tensor filters, torch::Tensor indicesIn,
                     torch::Tensor indicesOut, int nHot);

void fused_conv_backward_cuda(torch::Tensor features, torch::Tensor din,
                              torch::Tensor dout, torch::Tensor filters,
                              torch::Tensor dfilters, torch::Tensor indicesIn,
                              torch::Tensor indicesOut, int nHot);

} // namespace spconv
