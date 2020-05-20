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

#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <spconv/pillar_scatter_functor.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/kernel_utils.h>
#include <tensorview/mp_helper.h>
#include <tensorview/tensorview.h>
#include <type_traits>
#include <utility/timer.h>

namespace spconv {
template <typename T, typename Index>
__global__ void pointPillarsScatterKernel(tv::TensorView<T> canvas,
                                          tv::TensorView<const T> features,
                                          tv::TensorView<const T> coors) {
  auto numFeatures = features.dim(0);
  auto numPoints = features.dim(1);
  for (int i : tv::KernelLoopX<int>(numPoints)) {
    for (int ifeature : tv::KernelLoopY<int>(numFeatures)) {
      canvas(int(coors(0, i)), ifeature, int(coors(2, i)), int(coors(3, i))) =
          features(ifeature, i);
    }
  }
}
namespace functor {
template <typename T, typename Index>
struct PointPillarScatter<tv::GPU, T, Index> {
  void operator()(const tv::GPU &d, tv::TensorView<T> canvas,
                  tv::TensorView<const T> features,
                  tv::TensorView<const T> coors) {
    auto grid = dim3(tv::cuda::DivUp(features.dim(1), 32),
                     tv::cuda::DivUp(features.dim(0), 32));
    pointPillarsScatterKernel<T, Index>
        <<<grid, dim3(32, 32), 0, d.getStream()>>>(canvas, features, coors);
    TV_CHECK_CUDA_ERR();
  }
};
} // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T, Index)                                    \
  template struct functor::PointPillarScatter<tv::GPU, T, Index>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPECS_T_INDEX(T, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
} // namespace spconv