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

#ifndef PILLAR_SCATTER_OP_H_
#define PILLAR_SCATTER_OP_H_

#include <spconv/pillar_scatter_functor.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>
#include <utility/timer.h>

namespace spconv {
// torch.jit's doc says only support int64, so we need to convert to int32.

template <typename T>
torch::Tensor pointPillarScatter(torch::Tensor features, torch::Tensor coors,
                                 torch::Tensor shape) {
  TV_ASSERT_RT_ERR(shape.device().type() == torch::kCPU, "error");
  TV_ASSERT_RT_ERR(features.device().type() == torch::kCUDA, "error");
  TV_ASSERT_RT_ERR(shape.dim() == 1, "error");
  TV_ASSERT_RT_ERR(shape.size(0) == 4, "error");
  TV_ASSERT_RT_ERR(features.dim() >= 3, "error");
  TV_ASSERT_RT_ERR(features.size(0) == 1, "feature first dim must be 1");
  TV_ASSERT_RT_ERR(coors.size(0) == 1, "coors first dim must be 1");
  TV_ASSERT_RT_ERR(features.size(2) == coors.size(2), "err");

  tv::check_torch_dtype<int>(shape);
  tv::check_torch_dtype<T>(coors);
  auto shapeData = shape.data_ptr<int>();
  torch::Tensor canvas =
      torch::zeros({shapeData[0], shapeData[1], shapeData[2], shapeData[3]},
                   features.options());
  TV_ASSERT_RT_ERR(shapeData[1] == features.size(1), "error");
#ifdef TV_CUDA
  functor::PointPillarScatter<tv::GPU, T, int> ftor;
  ftor(tv::TorchGPU(), tv::torch2tv<T>(canvas),
       tv::torch2tv<const T>(features.squeeze()),
       tv::torch2tv<const T>(coors.squeeze()));
#endif
  return canvas;
}

} // namespace spconv

#endif