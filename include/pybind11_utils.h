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
#include <tensorview/tensorview.h>
#include <tensorview/tensor.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace tv {

template <typename T> TensorView<T> arrayt2tv(py::array_t<T> arr) {
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return TensorView<T>(arr.mutable_data(), shape);
}

template <typename T> TensorView<const T> carrayt2tv(py::array_t<T> arr) {
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return TensorView<const T>(arr.data(), shape);
}

template <typename T> TensorView<T> vector2tv(std::vector<T> &arr) {
  return TensorView<T>(arr.data(), {arr.size()});
}

template <typename T>
TensorView<T> vector2tv(std::vector<T> &arr, Shape shape) {
  TV_ASSERT_INVALID_ARG(shape.prod() == arr.size(), "error");
  return TensorView<T>(arr.data(), shape);
}

template <typename T> TensorView<const T> vector2tv(const std::vector<T> &arr) {
  return TensorView<const T>(arr.data(), {arr.size()});
}

template <typename T>
std::vector<T> shape2stride(const std::vector<T> &shape, T itemsize) {
  T p = T(1);
  std::vector<T> res;
  for (auto iter = shape.rbegin(); iter != shape.rend(); ++iter) {
    res.push_back(p * itemsize);
    p *= *iter;
  }
  std::reverse(res.begin(), res.end());
  return res;
}

tv::DType get_array_tv_dtype(const py::array& arr){
  // 
  switch (arr.dtype().kind()){
    case 'b': return tv::bool_;
    case 'i': {
      switch (arr.itemsize()){
        case 1: return tv::int8;
        case 2: return tv::int16;
        case 4: return tv::int32;
        case 8: return tv::int64;
        default: break;
      }
    }
    case 'u': {
      switch (arr.itemsize()){
        case 1: return tv::uint8;
        case 2: return tv::uint16;
        case 4: return tv::uint32;
        case 8: return tv::uint64;
        default: break;
      }
    }
    case 'f': {
      switch (arr.itemsize()){
        case 4: return tv::float32;
        case 8: return tv::float64;
        default: break;
      }
    }
  }
  TV_THROW_RT_ERR("unknown dtype", arr.dtype().kind(), arr.itemsize());
}


Tensor array2tensor(py::array& arr) {
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return tv::from_blob(arr.mutable_data(), shape, get_array_tv_dtype(arr), -1);
}


} // namespace tv
