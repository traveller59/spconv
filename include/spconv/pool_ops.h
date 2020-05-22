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

#ifndef SPARSE_POOL_OP_H_
#define SPARSE_POOL_OP_H_

#include <spconv/maxpool.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>
#include <utility/timer.h>

namespace spconv {
torch::Tensor indiceMaxPool(torch::Tensor features, torch::Tensor indicePairs,
                            torch::Tensor indiceNum, int64_t numAct);

torch::Tensor indiceMaxPoolBackward(torch::Tensor features,
                                    torch::Tensor outFeatures,
                                    torch::Tensor outGrad,
                                    torch::Tensor indicePairs,
                                    torch::Tensor indiceNum);

} // namespace spconv

#endif