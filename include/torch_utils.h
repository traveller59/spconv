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
#include <tensorview/mp_helper.h>
#include <tensorview/tensorview.h>

#include <ATen/ATen.h>
#include <torch/script.h>
#ifdef TV_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace tv {

#ifdef TV_CUDA
struct TorchGPU : public tv::GPU {
  virtual cudaStream_t getStream() const override {
    return at::cuda::getCurrentCUDAStream();
  }
};
#endif
template <typename T> void check_torch_dtype(const torch::Tensor &tensor) {
  switch (tensor.scalar_type()) {
  case at::ScalarType::Double: {
    auto val = std::is_same<std::remove_const_t<T>, double>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Float: {
    auto val = std::is_same<std::remove_const_t<T>, float>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Int: {
    auto val = std::is_same<std::remove_const_t<T>, int>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Half: {
    auto val = std::is_same<std::remove_const_t<T>, at::Half>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  case at::ScalarType::Long: {
    auto val = std::is_same<std::remove_const_t<T>, long>::value;
    TV_ASSERT_RT_ERR(val, "error");
    break;
  }
  default:
    TV_ASSERT_RT_ERR(false, "error");
  }
}
namespace detail {
template <typename T> struct TypeToTorchDtypeTraits;

template <> struct TypeToTorchDtypeTraits<int32_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt32;
};

template <> struct TypeToTorchDtypeTraits<int64_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt64;
};

template <> struct TypeToTorchDtypeTraits<float> {
  static constexpr decltype(torch::kInt32) value = torch::kFloat32;
};
template <> struct TypeToTorchDtypeTraits<double> {
  static constexpr decltype(torch::kInt32) value = torch::kFloat64;
};
template <> struct TypeToTorchDtypeTraits<at::Half> {
  static constexpr decltype(torch::kInt32) value = torch::kHalf;
};

} // namespace detail

template <typename T>
constexpr decltype(torch::kInt32) torch_type_v =
    detail::TypeToTorchDtypeTraits<T>::value;

template <typename T> tv::TensorView<T> torch2tv(const torch::Tensor &tensor) {
  check_torch_dtype<T>(tensor);
  tv::Shape shape;
  for (auto i : tensor.sizes()) {
    shape.push_back(i);
  }
  return tv::TensorView<T>(tensor.data_ptr<std::remove_const_t<T>>(), shape);
}
namespace detail {
template <> struct TypeToString<at::Half> {
  static constexpr const char *value = "half";
};
} // namespace detail
template <class... Ts, typename F>
void dispatch_torch(at::ScalarType t, F &&f) {
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  spconv::tv::mp_for_each<spconv::mp_list<Ts...>>([=, &notFound, &f](auto I) {
    if (torch_type_v<decltype(I)> == t) {
      std::forward<F>(f)(decltype(I)());
      notFound = false;
    }
  });
  if (notFound) {
    std::stringstream ss;
    spconv::tv::mp_for_each<spconv::mp_list<Ts...>>([=, &ss](auto I) {
      ss << tv::detail::TypeToString<decltype(I)>::value << " ";
    });
    TV_THROW_RT_ERR("unknown type", t, ", available: ", ss.str());
  }
}

} // namespace tv