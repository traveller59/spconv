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

#ifndef POINTPILLARS_SCATTER_FUNCTOR_H_
#define POINTPILLARS_SCATTER_FUNCTOR_H_
#include <tensorview/tensorview.h>

namespace spconv {
namespace functor {
template <typename Device, typename T, typename Index>
struct PointPillarScatter {
  void operator()(const Device &d, tv::TensorView<T> canvas,
                  tv::TensorView<const T> features,
                  tv::TensorView<const T> coors);
};

} // namespace functor
} // namespace spconv

#endif