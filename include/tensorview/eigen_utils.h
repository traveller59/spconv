// Copyright 2019-2020 Yan Yan
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
#include "tensor.h"
#include "tensorview.h"
#include <eigen3/Eigen/Dense>

namespace tv {

template <typename T, int Row = Eigen::Dynamic, int Col = Eigen::Dynamic>
Eigen::Map<Eigen::Matrix<T, Row, Col, Eigen::RowMajor>>
tv2eigen(TensorView<T> view) {
  TV_ASSERT_INVALID_ARG(view.ndim() <= 2 && view.ndim() > 0, "error");
  if (Row != Eigen::Dynamic) {
    TV_ASSERT_INVALID_ARG(view.dim(0) == Row, "error");
  }
  if (Col != Eigen::Dynamic) {
    TV_ASSERT_INVALID_ARG(view.dim(1) == Col, "error");
  }
  int row = 1;
  if (view.ndim() == 2) {
    row = view.dim(0);
  }
  Eigen::Map<Eigen::Matrix<T, Row, Col, Eigen::RowMajor>> eigen_map(
      view.data(), row, view.dim(1));
  return eigen_map;
}

} // namespace tv
