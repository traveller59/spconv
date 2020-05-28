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

#ifndef SPARSE_REORDERING_FUNCTOR_H_
#define SPARSE_REORDERING_FUNCTOR_H_
#include <tensorview/tensorview.h>
#include <torch/script.h>

namespace spconv {

void batch_sparse_gather_cuda(torch::Tensor buffer, torch::Tensor features,
                              torch::Tensor indices, int size);
void batch_sparse_scatter_add_cuda(torch::Tensor buffer,
                                   torch::Tensor outFeatures,
                                   torch::Tensor indices, int size);

void sparse_gather_cuda(torch::Tensor buffer, torch::Tensor features,
                        torch::Tensor indices, int size);
void sparse_scatter_add_cuda(torch::Tensor buffer, torch::Tensor outFeatures,
                             torch::Tensor indices, int size);

void sparse_gather_cpu(torch::Tensor buffer, torch::Tensor features,
                       torch::Tensor indices, int size);
void sparse_scatter_add_cpu(torch::Tensor buffer, torch::Tensor outFeatures,
                            torch::Tensor indices, int size);

} // namespace spconv

#endif