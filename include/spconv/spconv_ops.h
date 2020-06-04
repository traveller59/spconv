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

#ifndef SPARSE_CONV_OP_H_
#define SPARSE_CONV_OP_H_

#include <spconv/indice.h>
#include <spconv/reordering.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>
#include <utility/timer.h>

namespace spconv {

enum ConvAlgo { kNative = 0, kBatch = 1, kBatchGemmGather = 2 };

// torch.jit's doc says only support int64, so we need to convert to int32.
std::vector<torch::Tensor>
getIndicePairs(torch::Tensor indices, int64_t batchSize,
               std::vector<int64_t> outSpatialShape,
               std::vector<int64_t> spatialShape,
               std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
               std::vector<int64_t> padding, std::vector<int64_t> dilation,
               std::vector<int64_t> outPadding, int64_t _subM,
               int64_t _transpose, int64_t _useHash);

torch::Tensor indiceConvBatch(torch::Tensor features, torch::Tensor filters,
                              torch::Tensor indicePairs,
                              torch::Tensor indiceNum, int64_t numActOut,
                              int64_t _inverse, int64_t _subM,
                              bool batchScatter);

torch::Tensor indiceConv(torch::Tensor features, torch::Tensor filters,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM,
                         int64_t algo);
std::vector<torch::Tensor>
indiceConvBackward(torch::Tensor features, torch::Tensor filters,
                   torch::Tensor outGrad, torch::Tensor indicePairs,
                   torch::Tensor indiceNum, int64_t _inverse, int64_t _subM,
                   int64_t algo);

std::vector<torch::Tensor>
indiceConvBackwardBatch(torch::Tensor features, torch::Tensor filters,
                        torch::Tensor outGrad, torch::Tensor indicePairs,
                        torch::Tensor indiceNum, int64_t _inverse,
                        int64_t _subM, bool batchScatter);
} // namespace spconv

#endif