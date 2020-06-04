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

#ifndef SPARSE_CONV_INDICE_FUNCTOR_H_
#define SPARSE_CONV_INDICE_FUNCTOR_H_
#include <tensorview/tensorview.h>
#include <torch/script.h>

namespace spconv {
int create_conv_indice_pair_p1_cuda(
    torch::Tensor indicesIn, torch::Tensor indicePairs, torch::Tensor indiceNum,
    torch::Tensor indicePairUnique, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
    bool transpose);

int create_conv_indice_pair_p2_cuda(
    torch::Tensor indicesIn, torch::Tensor indicesOut, torch::Tensor gridsOut,
    torch::Tensor indicePairs, torch::Tensor indiceNum,
    torch::Tensor indicePairUnique, std::vector<int64_t> outSpatialShape,
    bool transpose, bool resetGrid, bool useHash);

int create_submconv_indice_pair_cuda(
    torch::Tensor indicesIn, torch::Tensor gridsOut, torch::Tensor indicePairs,
    torch::Tensor indiceNum, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
    bool transpose, bool resetGrid, bool useHash);

int create_conv_indice_pair_cpu(
    torch::Tensor indicesIn, torch::Tensor indicesOut, torch::Tensor gridsOut,
    torch::Tensor indicePairs, torch::Tensor indiceNum,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outSpatialShape, bool transpose, bool resetGrid,
    bool useHash);

int create_submconv_indice_pair_cpu(
    torch::Tensor indicesIn, torch::Tensor gridsOut, torch::Tensor indicePairs,
    torch::Tensor indiceNum, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
    bool transpose, bool resetGrid, bool useHash);

} // namespace spconv

#endif