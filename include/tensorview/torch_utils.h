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
#include "mp_helper.h"
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
namespace detail {
template <typename T> struct TypeToTorchDtypeTraits;

template <> struct TypeToTorchDtypeTraits<int32_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt32;
};
template <> struct TypeToTorchDtypeTraits<int16_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt16;
};
template <> struct TypeToTorchDtypeTraits<int8_t> {
  static constexpr decltype(torch::kInt8) value = torch::kInt8;
};
template <> struct TypeToTorchDtypeTraits<int64_t> {
  static constexpr decltype(torch::kInt32) value = torch::kInt64;
};
template <> struct TypeToTorchDtypeTraits<uint8_t> {
  static constexpr decltype(torch::kInt32) value = torch::kUInt8;
};
template <> struct TypeToTorchDtypeTraits<bool> {
  static constexpr decltype(torch::kInt32) value = torch::kBool;
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

using all_torch_types_t = std::tuple<float, double, int8_t, int16_t, int32_t,
                                     int64_t, uint8_t, bool, at::Half>;

} // namespace detail

template <typename T>
constexpr decltype(torch::kInt32) torch_type_v =
    detail::TypeToTorchDtypeTraits<T>::value;

template <class... Ts, typename F>
void dispatch_torch(at::ScalarType t, F &&f) {
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  tv::mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    if (detail::TypeToTorchDtypeTraits<TV_DECLTYPE(I)>::value == t) {
      std::forward<F>(f)(TV_DECLTYPE(I)());
      notFound = false;
    }
  });
  if (notFound) {
    std::stringstream ss;
    tv::mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << tv::detail::TypeToString<TV_DECLTYPE(I)>::value << " ";
    });
    TV_THROW_RT_ERR("unknown type", t, ", available:", ss.str());
  }
}

template <class T> struct DispatchTorch;

template <template <class...> class T, class... Args>
struct DispatchTorch<T<Args...>> {
  template <typename F> inline void operator()(at::ScalarType t, F &&f) {
    return dispatch_torch<Args...>(t, std::forward<F>(f));
  }
};

template <typename T> void check_torch_dtype(const torch::Tensor &tensor) {
  DispatchTorch<detail::all_torch_types_t>()(tensor.scalar_type(), [&](auto I) {
    using Ttensor = TV_DECLTYPE(I);
    constexpr bool val = std::is_same<std::remove_cv_t<T>, Ttensor>::value;
    TV_ASSERT_RT_ERR(val, "error");
  });
}

template <typename T, int Rank = -1,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = int>
TensorView<T, Rank, PtrTraits, Tindex> torch2tv(const torch::Tensor &tensor) {
  using tv_shape_t =
      typename TensorView<T, Rank, PtrTraits, Tindex>::tv_shape_t;
  check_torch_dtype<T>(tensor);
  // TODO stride
  if (Rank > 0) {
    TV_ASSERT_INVALID_ARG(tensor.dim() == Rank, "error");
  }
  tv_shape_t shape;
  for (auto i : tensor.sizes()) {
    shape.push_back(i);
  }
  return tv::TensorView<T, Rank, PtrTraits, Tindex>(
      tensor.data_ptr<std::remove_const_t<T>>(), shape);
}

template <typename T>
torch::Tensor torch_slice_first_axis(torch::Tensor tensor, T start, T end) {
  // only torch >= 1.5 have tensor slice.
  torch::Tensor res;
  auto tensor_shape = tensor.sizes();
  std::vector<int64_t> shape(tensor_shape.begin(), tensor_shape.end());
  shape[0] = end - start;
  uint8_t *ptr = reinterpret_cast<uint8_t *>(tensor.data_ptr());
  res = torch::from_blob(ptr + start * tensor.stride(0) * tensor.itemsize(),
                         torch::IntArrayRef(shape), tensor.options());
  return res;
}

namespace detail {
template <> struct TypeToString<at::Half> {
  static constexpr const char *value = "half";
};
} // namespace detail
} // namespace tv