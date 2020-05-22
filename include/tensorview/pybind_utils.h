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

#include <algorithm>
#include <array>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace tv {

template <typename Tarr> bool is_c_style(const Tarr &arr) {
  return bool(arr.flags() & py::array::c_style);
}

template <typename T, int Rank = -1>
TensorView<T, Rank> arrayt2tv(py::array_t<T> arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  if (Rank >= 0) {
    TV_ASSERT_INVALID_ARG(shape.ndim() == Rank, "error");
  }
  return TensorView<T, Rank>(arr.mutable_data(), shape);
}

template <typename T, int Rank = -1>
TensorView<const T> carrayt2tv(py::array_t<T> arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  Shape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  if (Rank >= 0) {
    TV_ASSERT_INVALID_ARG(shape.ndim() == Rank, "error");
  }
  return TensorView<const T, Rank>(arr.data(), shape);
}

template <typename Tarr> tv::DType get_array_tv_dtype(const Tarr &arr) {
  switch (arr.dtype().kind()) {
  case 'b':
    return tv::bool_;
  case 'i': {
    switch (arr.itemsize()) {
    case 1:
      return tv::int8;
    case 2:
      return tv::int16;
    case 4:
      return tv::int32;
    case 8:
      return tv::int64;
    default:
      break;
    }
  }
  case 'u': {
    switch (arr.itemsize()) {
    case 1:
      return tv::uint8;
    case 2:
      return tv::uint16;
    case 4:
      return tv::uint32;
    case 8:
      return tv::uint64;
    default:
      break;
    }
  }
  case 'f': {
    switch (arr.itemsize()) {
    case 2:
      return tv::float16;
    case 4:
      return tv::float32;
    case 8:
      return tv::float64;
    default:
      break;
    }
  }
  }
  TV_THROW_RT_ERR("unknown dtype", arr.dtype().kind(), arr.itemsize());
}

template <typename Tarr> Tensor array2tensor(Tarr &arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  TensorShape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return tv::from_blob(arr.mutable_data(), shape, get_array_tv_dtype(arr), -1);
}

template <typename T> Tensor arrayt2tensor(py::array_t<T> &arr) {
  TV_ASSERT_INVALID_ARG(is_c_style(arr), "array must be c-contiguous array");
  TensorShape shape;
  for (int i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return tv::from_blob(arr.mutable_data(), shape, tv::type_v<T>, -1);
}

template <typename TDType> py::dtype tv_dtype_to_py(TDType d) {
  switch (d) {
  case float32:
    return py::dtype("float32");
  case float64:
    return py::dtype("float64");
  case float16:
    return py::dtype("float16");
  case int32:
    return py::dtype("int32");
  case int16:
    return py::dtype("int16");
  case int8:
    return py::dtype("int8");
  case int64:
    return py::dtype("int64");
  case uint32:
    return py::dtype("uint32");
  case uint16:
    return py::dtype("uint16");
  case uint8:
    return py::dtype("uint8");
  case uint64:
    return py::dtype("uint64");
  case bool_:
    return py::dtype("bool_");
  default:;
  }
  TV_THROW_INVALID_ARG("unknown dtype", d);
}

// add template to define function in header
template <typename Ttensor> py::array tensor2array(Ttensor &tensor) {
  // you cant call this function during GIL released.
  TV_ASSERT_INVALID_ARG(tensor.device() == -1, "must be cpu tensor");
  auto shape = tensor.shape();
  std::vector<int> shape_vec(shape.begin(), shape.end());
  auto dtype = tv_dtype_to_py(tensor.dtype());
  // construct py::array will copy content from ptr.
  // its expected because we can't transfer ownership from
  // c++ tv::Tensor to numpy array when c++ object is deleted.
  return py::array(dtype, shape_vec, {}, tensor.raw_data());
}

} // namespace tv
