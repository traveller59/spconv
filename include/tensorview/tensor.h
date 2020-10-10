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

/*
tv::Tensor is a lightweight header-only tensor container
without template and annoying dependencies. no algorithm is implemented.
it should only be used when you want a no-template simple container but
dont want to link with libtorch.

If you can use libtorch, dont use tv::Tensor.
*/

#pragma once
#include "mp_helper.h"
#include "tensorview.h"
#include <cstring>
#include <iomanip>
#include <memory>
#include <type_traits>
#ifdef TV_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

namespace tv {
enum DType {
  float32,
  int32,
  int16,
  int8,
  float64,
  bool_,
  uint8,
  float16,
  int64,
  uint16,
  uint32,
  uint64
};

namespace detail {

using dtype_collection_t =
    tv::mp_list_c<int, float32, int32, int16, int8, float64, bool_, uint8,
                  float16, int64, uint16, uint32, uint64>;

#ifdef TV_CUDA
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;
#else
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;
#endif

template <typename T> class TensorStorage {
public:
  TensorStorage(size_t size, int device = -1, bool managed = false,
                bool pinned = false)
      : mSize(size), device_(device), managed_(managed), pinned_(pinned) {
    if (size == 0) {
      mPtr = nullptr;
    } else {
      if (device == -1) {
        if (pinned_) {
#ifdef TV_CUDA
          checkCudaErrors(cudaMallocHost(&mPtr, size * sizeof(T)));
#else
          TV_THROW_INVALID_ARG("you need to define TV_CUDA to use pinned");
#endif
        } else {
          mPtr = new T[size];
        }
      } else {
#ifdef TV_CUDA
        // we should select device in external
        /*
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (device >= deviceCount) {
          TV_THROW_INVALID_ARG("you provide device ", device,
                               " but you only have ", deviceCount, " device.");
        }
        cudaSetDevice(device);
        */
        if (managed) {
          checkCudaErrors(cudaMallocManaged(&this->mPtr, size * sizeof(T)));
        } else {
          checkCudaErrors(cudaMalloc(&mPtr, size * sizeof(T)));
        }
#else
        TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
      }
    }
  }
  TensorStorage(T *ptr, size_t size, int device)
      : mSize(size), mPtr(ptr), from_blob_(true), device_(device) {}

  virtual ~TensorStorage() {
    if (empty()) {
      return;
    }
    if (from_blob_) {
      return;
    }
    if (device_ == -1) {
      if (pinned_) {
#ifdef TV_CUDA
        cudaFreeHost(mPtr);
#endif
      } else {
        delete[] mPtr;
      }
    } else {
#ifdef TV_CUDA
      cudaFree(mPtr);
#endif
    }
  };

  inline size_t size() const { return mSize; }

  T *data() { return mPtr; }
  const T *data() const { return mPtr; }

  bool empty() const { return mPtr == nullptr || mSize == 0; }
  bool managed() const { return managed_; }
  bool pinned() const { return pinned_; }

  int device() const { return device_; }
  void zero_() {
    if (device_ == -1) {
      std::memset(data(), 0, mSize);
      // std::fill(data(), data() + mSize, 0);
    } else {
#ifdef TV_CUDA
      checkCudaErrors(cudaMemset(data(), 0, mSize / sizeof(T)));
#else
      TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
    }
  }

private:
  size_t mSize = 0;
  T *mPtr = nullptr;
  bool from_blob_ = false;
  int device_ = -1;
  bool managed_ = false;
  bool pinned_ = false;
};

template <typename T> size_t sizeof_dtype(T dtype) {
  switch (dtype) {
  case float32:
    return sizeof(float);
  case int8:
    return sizeof(int8_t);
  case int16:
    return sizeof(int16_t);
  case int32:
    return sizeof(int32_t);
  case float64:
    return sizeof(double);
  case int64:
    return sizeof(int64_t);
  case bool_:
    return sizeof(bool);
  case uint8:
    return sizeof(uint8_t);
  case uint16:
    return sizeof(uint16_t);
  case uint32:
    return sizeof(uint32_t);
  case uint64:
    return sizeof(uint64_t);
  case float16:
    return 2;
  default:
    TV_THROW_RT_ERR("unsupported dtype");
  }
  return 0;
}

template <typename T> std::string typeString(T t) {
  switch (t) {
  case DType::bool_:
    return "bool";
  case DType::float32:
    return "float32";
  case DType::int8:
    return "int8";
  case DType::int16:
    return "int16";
  case DType::int32:
    return "int32";
  case DType::float64:
    return "float64";
  case DType::int64:
    return "int64";
  case DType::uint8:
    return "uint8";
  case DType::uint16:
    return "uint16";
  case DType::uint32:
    return "uint32";
  case DType::uint64:
    return "uint64";
  case DType::float16:
    return "half";
  default:
    return "";
  }
}

template <typename T> struct TypeToDtypeTraits;

template <> struct TypeToDtypeTraits<int32_t> {
  static constexpr DType dtype = int32;
};

#ifdef TV_CUDA
template <> struct TypeToDtypeTraits<__half> {
  static constexpr DType dtype = float16;
};
#endif

template <> struct TypeToDtypeTraits<float> {
  static constexpr DType dtype = float32;
};
template <> struct TypeToDtypeTraits<double> {
  static constexpr DType dtype = float64;
};
template <> struct TypeToDtypeTraits<int16_t> {
  static constexpr DType dtype = int16;
};
template <> struct TypeToDtypeTraits<int8_t> {
  static constexpr DType dtype = int8;
};
template <> struct TypeToDtypeTraits<int64_t> {
  static constexpr DType dtype = int64;
};
template <> struct TypeToDtypeTraits<uint8_t> {
  static constexpr DType dtype = uint8;
};
template <> struct TypeToDtypeTraits<uint16_t> {
  static constexpr DType dtype = uint16;
};
template <> struct TypeToDtypeTraits<uint32_t> {
  static constexpr DType dtype = uint32;
};
template <> struct TypeToDtypeTraits<uint64_t> {
  static constexpr DType dtype = uint64;
};
template <> struct TypeToDtypeTraits<bool> {
  static constexpr DType dtype = bool_;
};
template <> struct TypeToDtypeTraits<const int32_t> {
  static constexpr DType dtype = int32;
};

#ifdef TV_CUDA
template <> struct TypeToDtypeTraits<const __half> {
  static constexpr DType dtype = float16;
};
#endif

template <> struct TypeToDtypeTraits<const float> {
  static constexpr DType dtype = float32;
};
template <> struct TypeToDtypeTraits<const double> {
  static constexpr DType dtype = float64;
};
template <> struct TypeToDtypeTraits<const int16_t> {
  static constexpr DType dtype = int16;
};
template <> struct TypeToDtypeTraits<const int8_t> {
  static constexpr DType dtype = int8;
};
template <> struct TypeToDtypeTraits<const int64_t> {
  static constexpr DType dtype = int64;
};
template <> struct TypeToDtypeTraits<const uint8_t> {
  static constexpr DType dtype = uint8;
};
template <> struct TypeToDtypeTraits<const uint16_t> {
  static constexpr DType dtype = uint16;
};
template <> struct TypeToDtypeTraits<const uint32_t> {
  static constexpr DType dtype = uint32;
};
template <> struct TypeToDtypeTraits<const uint64_t> {
  static constexpr DType dtype = uint64;
};
template <> struct TypeToDtypeTraits<const bool> {
  static constexpr DType dtype = bool_;
};

} // namespace detail

template <class T> constexpr DType type_v = detail::TypeToDtypeTraits<T>::dtype;

template <class... Ts, typename F> bool dispatch_noexcept(DType t, F &&f) {
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    if (type_v<TV_DECLTYPE(I)> == t && notFound) {
      std::forward<F>(f)(TV_DECLTYPE(I)());
      notFound = false;
    }
  });
  return !notFound;
}

template <class... Ts, typename F> void dispatch(DType t, F &&f) {
  if (!dispatch_noexcept<Ts...>(t, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << detail::TypeToString<TV_DECLTYPE(I)>::value << " ";
    });
    TV_THROW_RT_ERR("unknown type", detail::typeString(t),
                    ", available:", ss.str());
  }
}

template <typename T, T... Is, typename F> void dispatch_scalar(T idx, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<T, Is...>>([=, &notFound, &f](auto I) {
    if (T(I) == idx && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  if (notFound) {
    std::stringstream ss;
    mp_for_each<mp_list_c<T, Is...>>([=, &ss](auto I) { ss << T(I) << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

template <int... Is, typename F> bool dispatch_int_noexcept(int idx, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<int, Is...>>([=, &notFound, &f](auto I) {
    if (TV_DECLTYPE(I)::value == idx && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  return !notFound;
}

template <int... Is, typename F, class BinaryPredicate>
bool dispatch_int_noexcept(int idx, BinaryPredicate p, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<int, Is...>>([=, &notFound, &f](auto I) {
    if (p(idx, TV_DECLTYPE(I)::value) && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  return !notFound;
}

template <int... Is, typename F> void dispatch_int(int idx, F &&f) {
  if (!dispatch_int_noexcept<Is...>(idx, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list_c<int, Is...>>(
        [=, &ss](auto I) { ss << TV_DECLTYPE(I)::value << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

template <int... Is, typename F, class BinaryPredicate>
void dispatch_int(int idx, BinaryPredicate p, F &&f) {
  // BinaryPredicate: BinaryPredicate(idx, candidate)
  if (!dispatch_int_noexcept<Is...>(idx, p, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list_c<int, Is...>>(
        [=, &ss](auto I) { ss << TV_DECLTYPE(I)::value << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

// Ts is pack of mp_list_c
template <class... Ts, typename Iterator, typename F>
bool dispatch_container_noexcept(Iterator begin, Iterator end, F &&f) {
  static_assert(sizeof...(Ts) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    using val_lst_t = TV_DECLTYPE(I);
    auto val_lst_size = mp_size<val_lst_t>::value;
    bool equal = true;
    std::size_t count = 0;
    auto iter = begin;
    mp_for_each<val_lst_t>([&](auto E) {
      if (iter == end || !equal) {
        return;
      }
      if (count >= val_lst_size) {
        TV_THROW_INVALID_ARG("iterator length invalid:", val_lst_size);
      }
      constexpr auto c = TV_DECLTYPE(E)::value;
      if (c != *iter) {
        equal = false;
      }
      ++count;
      std::advance(iter, 1);
    });
    if (count != val_lst_size || iter != end) {
      equal = false;
    }
    if (equal && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });

  return !notFound;
}

template <class... Ts, typename Iterator, typename F>
void dispatch_container(Iterator begin, Iterator end, F &&f) {
  if (!dispatch_container_noexcept<Ts...>(begin, end, std::forward<F>(f))) {
    std::stringstream ss;
    ss << "unknown value [";
    for (auto iter = begin; iter != end; std::advance(iter, 1)) {
      ss << *iter << ",";
    }
    ss << "], available: ";
    mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << "[";
      mp_for_each<TV_DECLTYPE(I)>(
          [=, &ss](auto E) { ss << TV_DECLTYPE(E)::value << ","; });
      ss << "]";
    });
    TV_THROW_RT_ERR(ss.str());
  }
}

/*
template <int... Is, typename F> void dispatch_int(int idx, F &&f) {
  return dispatch_scalar<int, Is...>(idx, f);
}
*/

template <class T> struct Dispatch;

template <template <class...> class T, class... Args>
struct Dispatch<T<Args...>> {
  template <typename F> inline void operator()(DType t, F &&f) {
    return dispatch<Args...>(t, std::forward<F>(f));
  }
};

template <class T> struct DispatchContainer;

template <template <class...> class T, class... Args>
struct DispatchContainer<T<Args...>> {
  template <typename Iterator, typename F>
  inline void operator()(Iterator begin, Iterator end, F &&f) {
    return dispatch_container<Args...>(begin, end, std::forward<F>(f));
  }
};

template <class T> struct DispatchContainerNoexcept;

template <template <class...> class T, class... Args>
struct DispatchContainerNoexcept<T<Args...>> {
  template <typename Iterator, typename F>
  inline bool operator()(Iterator begin, Iterator end, F &&f) {
    return dispatch_container_noexcept<Args...>(begin, end, std::forward<F>(f));
  }
};

template <class T> struct DispatchInt;

// Args should be std::integral_constant<int, value>
// you need to use type_container<std::integral_constant<int, value>...>
// as template parameter of DispatchInt.
// tv::mp_list_c is ok.
template <template <class...> class T, class... Args>
struct DispatchInt<T<Args...>> {
  template <typename F> inline void operator()(int t, F &&f) {
    return dispatch_int<Args::value...>(t, std::forward<F>(f));
  }
  template <typename F, typename BinaryPredicate>
  inline void operator()(int t, BinaryPredicate p, F &&f) {
    return dispatch_int<Args::value...>(t, p, std::forward<F>(f));
  }
};

constexpr size_t kTensorMaxDim = 10;
using TensorShape = ShapeBase<kTensorMaxDim, int64_t>;

struct Tensor {
  Tensor() {}
  Tensor(TensorShape shape, TensorShape stride, DType dtype, int device = -1,
         bool pinned = false, bool managed = false)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed, pinned);
    shape_ = shape;
    stride_ = stride;
  }

  Tensor(TensorShape shape, DType dtype, int device = -1, bool pinned = false,
         bool managed = false)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed, pinned);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
  }
  Tensor(void *ptr, TensorShape shape, TensorShape stride, DType dtype,
         int device = -1)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = stride;
  }
  Tensor(void *ptr, TensorShape shape, DType dtype, int device = -1)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
  }

  Tensor(const void *ptr, TensorShape shape, TensorShape stride, DType dtype,
         int device = -1)
      : dtype_(dtype), writeable_(false) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(ptr)),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = stride;
  }
  Tensor(const void *ptr, TensorShape shape, DType dtype, int device = -1)
      : dtype_(dtype), writeable_(false) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(ptr)),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
  }

  Tensor(std::initializer_list<int32_t> init)
      : Tensor({int(init.size())}, tv::int32) {
    std::copy(init.begin(), init.end(), data<int32_t>());
  }
  Tensor(std::initializer_list<int64_t> init)
      : Tensor({int(init.size())}, tv::int64) {
    std::copy(init.begin(), init.end(), data<int64_t>());
  }
  Tensor(std::initializer_list<float> init)
      : Tensor({int(init.size())}, tv::float32) {
    std::copy(init.begin(), init.end(), data<float>());
  }
  Tensor(std::initializer_list<double> init)
      : Tensor({int(init.size())}, tv::float64) {
    std::copy(init.begin(), init.end(), data<double>());
  }

  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = int,
            typename std::enable_if<(Rank > 0), int>::type = 0>
  TensorView<T, Rank, PtrTraits, Tindex> tview() {
    using tv_shape_t =
        typename TensorView<T, Rank, PtrTraits, Tindex>::tv_shape_t;
    writable_check();
    static_assert(Rank == -1 || Rank > 0, "error");
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    tv_shape_t shape(Rank), stride(Rank);
    for (int i = 0; i < Rank; ++i) {
      shape[i] = shape_[i];
      stride[i] = stride_[i];
    }
    return TensorView<T, Rank, PtrTraits, Tindex>(
        reinterpret_cast<T *>(data<T>()), shape, stride);
  }
  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = int,
            typename std::enable_if<Rank == -1, int>::type = 0>
  TensorView<T, Rank, PtrTraits, Tindex> tview() {
    writable_check();
    static_assert(Rank == -1 || Rank > 0, "error");
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    ShapeBase<TV_MAX_DIM, Tindex> shape(ndim()), stride(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
      shape[i] = shape_[i];
      stride[i] = stride_[i];
    }
    return TensorView<T, Rank, PtrTraits, Tindex>(
        reinterpret_cast<T *>(data<T>()), shape, stride);
  }

  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = int,
            typename std::enable_if<(Rank > 0), int>::type = 0>
  TensorView<const std::remove_const_t<T>, Rank, PtrTraits, Tindex>
  tview() const {
    static_assert(Rank == -1 || Rank > 0, "error");
    if (Rank > 0) {
      TV_ASSERT_RT_ERR(Rank == ndim(), "error");
    }
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");

    ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex> shape(Rank), stride(Rank);
    for (int i = 0; i < Rank; ++i) {
      shape[i] = shape_[i];
      stride[i] = stride_[i];
    }
    return TensorView<const std::remove_const_t<T>, Rank, PtrTraits, Tindex>(
        reinterpret_cast<const std::remove_const_t<T> *>(data<T>()), shape,
        stride);
  }
  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = int,
            typename std::enable_if<Rank == -1, int>::type = 0>
  TensorView<const std::remove_const_t<T>, Rank, PtrTraits, Tindex>
  tview() const {
    static_assert(Rank == -1 || Rank > 0, "error");
    if (Rank > 0) {
      TV_ASSERT_RT_ERR(Rank == ndim(), "error");
    }
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");

    ShapeBase<TV_MAX_DIM, Tindex> shape(ndim()), stride(ndim());
    for (int i = 0; i < int(ndim()); ++i) {
      shape[i] = shape_[i];
      stride[i] = stride_[i];
    }
    return TensorView<const std::remove_const_t<T>, Rank, PtrTraits, Tindex>(
        reinterpret_cast<const std::remove_const_t<T> *>(data<T>()), shape,
        stride);
  }

  template <class... Inds> Tensor view(Inds... newShapes) const {
    static_assert(sizeof...(newShapes) > 0, "dont support empty for now");
    TensorShape shape{int(newShapes)...};
    bool found_minus_1 = false;
    for (size_t i = 0; i < shape.ndim(); ++i) {
      if (!found_minus_1) {
        if (shape[i] == -1) {
          shape[i] = 1;
          shape[i] = size() / shape.size();
          found_minus_1 = true;
        } else {
          TV_ASSERT_INVALID_ARG(shape[i] > 0,
                                "shape except -1 must larger than 0");
        }
      } else {
        TV_ASSERT_INVALID_ARG(shape[i] > 0, "multiple -1 in your argument.");
      }
    }
    TV_ASSERT_RT_ERR(shape.size() == size(), "error");
    Tensor res(*this);
    res.shape_ = shape;
    res.stride_ = shape.stride_rowmajor();
    return res;
  }

  Tensor view(TensorShape shape) const {
    TV_ASSERT_RT_ERR(shape.size() == size(), "error");
    Tensor res(*this);
    res.shape_ = shape;
    res.stride_ = shape.stride_rowmajor();
    return res;
  }

  Tensor operator[](int64_t index) {
    TV_ASSERT_INVALID_ARG(ndim() > 1, "error");
    if (index < 0) {
      index += dim(0);
    }
    TV_ASSERT_INVALID_ARG(index < dim(0), "error");
    Tensor res = Tensor();
    res.storage_ = storage_;
    res.shape_ = shape_.subshape(1);
    res.offset_ = offset_ + index * stride_[0];
    res.stride_ = stride_.subshape(1);
    res.writeable_ = writeable_;
    return res;
  }

  Tensor squeeze() const { return view(shape_.squeeze()); }

  Tensor squeeze(int axis) const {
    if (axis < 0) {
      axis = ndim() + axis;
    }
    return view(shape_.squeeze(axis));
  }

  Tensor unsqueeze(int axis) const {
    if (axis < 0) {
      axis = ndim() + axis;
    }
    return view(shape_.unsqueeze(axis));
  }

  bool pinned() const { return storage_->pinned(); }

  Tensor slice_first_axis(int start, int end) const {
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    if (start < 0) {
      start = shape_[0] + start;
    }
    if (end < 0) {
      end = shape_[0] + end;
    }
    TV_ASSERT_INVALID_ARG(start < shape_[0], "start must small than dim 0");
    TV_ASSERT_INVALID_ARG(start < end, "start must small than end");
    size_t new_offset = start * shape_.prod(1) * itemsize();
    Tensor res(*this);
    TensorShape newshape(shape_);
    newshape[0] = end - start;
    res.shape_ = newshape;
    res.stride_ = stride_;
    res.offset_ = new_offset;
    return res;
  }

  bool empty() const { return storage_->empty(); }
  DType dtype() const { return dtype_; }
  int device() const { return storage_->device(); }
  size_t ndim() const { return shape_.ndim(); }

  const TensorShape &shape() const { return shape_; }
  const TensorShape &sizes() const { return shape_; }
  const TensorShape &stride() const { return stride_; }

  int dim(int idx) const {
    if (idx < 0) {
      TV_ASSERT_RT_ERR(shape_.size() + idx < shape_.size(), idx, shape_);
      return shape_[shape_.size() + idx];
    } else {
      TV_ASSERT_RT_ERR(idx < int(shape_.size()), idx, shape_);
      return shape_[idx];
    }
  }
  const uint8_t *raw_data() const { return storage_->data() + offset_; }
  size_t raw_size() const { return size() * itemsize(); }
  size_t size() const { return shape_.size(); }
  size_t size(int64_t idx) const { return dim(idx); }
  size_t itemsize() const { return detail::sizeof_dtype(dtype_); }
  Tensor &zero_() {
    writable_check();
    storage_->zero_();
    return *this;
  }
  uint8_t *raw_data() {
    writable_check();
    return storage_->data() + offset_;
  }
  template <typename T> Tensor &fill_(T value) {
    writable_check();
    TV_ASSERT_RT_ERR(device() == -1, "error");
    Dispatch<detail::all_tensor_types_t>()(dtype_, [&](auto I) {
      using Treal = TV_DECLTYPE(I);
      if (std::is_convertible<T, Treal>::value) {
        auto ptr = reinterpret_cast<Treal *>(raw_data());
        std::fill(ptr, ptr + size(), Treal(value));
      } else {
        TV_THROW_INVALID_ARG("not convertable from", type_s<T>, "to",
                             type_s<Treal>);
      }
    });
    return *this;
  }

  template <typename T> T *data() {
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    writable_check();
    return reinterpret_cast<T *>(raw_data());
  }

  template <typename T> const T *data() const {
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    return reinterpret_cast<const T *>(raw_data());
  }

  template <typename T> T *data_ptr() { return data<T>(); }

  template <typename T> const T *data_ptr() const { return data<T>(); }

  void *data_ptr() { return reinterpret_cast<void *>(raw_data()); }

  const void *data_ptr() const {
    return reinterpret_cast<const void *>(raw_data());
  }

  void copy_(const Tensor &tensor) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_RT_ERR(!empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(dtype() == tensor.dtype(), "must have same dtype",
                     detail::typeString(dtype()),
                     detail::typeString(tensor.dtype()));
    if (device() == -1 && tensor.device() == -1) {
#ifdef TV_CUDA
      host2host(storage_->data(), tensor.raw_data(),
                size() * detail::sizeof_dtype(dtype_));
#else
      std::copy(tensor.raw_data(),
                tensor.raw_data() + size() * detail::sizeof_dtype(dtype_),
                storage_->data());
#endif
    }
#ifdef TV_CUDA
    else if (device() >= 0 && tensor.device() == -1) {
      host2dev(storage_->data(), tensor.raw_data(),
               size() * detail::sizeof_dtype(dtype_));
    } else if (device() == -1 && tensor.device() >= 0) {
      dev2host(storage_->data(), tensor.raw_data(),
               size() * detail::sizeof_dtype(dtype_));
    } else if (device() >= 0 && tensor.device() >= 0) {
      dev2dev(storage_->data(), tensor.raw_data(),
              size() * detail::sizeof_dtype(dtype_));
    }
#endif
    else {
      TV_THROW_RT_ERR("only support cpu tensor");
    }
  }

#ifdef TV_CUDA
  void copy_(const Tensor &tensor, cudaStream_t stream) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_RT_ERR(!empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(dtype() == tensor.dtype(), "must have same dtype",
                     detail::typeString(dtype()),
                     detail::typeString(tensor.dtype()));
    if (device() == -1 && tensor.device() == -1) {
      host2host(storage_->data(), tensor.raw_data(),
                size() * detail::sizeof_dtype(dtype_), stream);
    } else if (device() >= 0 && tensor.device() == -1) {
      host2dev(storage_->data(), tensor.raw_data(),
               size() * detail::sizeof_dtype(dtype_), stream);
    } else if (device() == -1 && tensor.device() >= 0) {
      dev2host(storage_->data(), tensor.raw_data(),
               size() * detail::sizeof_dtype(dtype_), stream);
    } else if (device() >= 0 && tensor.device() >= 0) {
      dev2dev(storage_->data(), tensor.raw_data(),
              size() * detail::sizeof_dtype(dtype_), stream);
    } else {
      TV_THROW_RT_ERR("only support cpu tensor");
    }
  }
#endif

  Tensor cpu() const {
    if (storage_->device() == -1) {
      // cpu() should always copy tensor.
      return clone();
    }
    Tensor res(shape_, stride_, dtype_, -1, storage_->managed());
    res.copy_(*this);
    return res;
  }

  template <typename T> void copy_(const TensorView<T> &tensor, int device) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    Tensor src = from_blob(tensor, device);
    return copy_(src);
  }

  Tensor &operator=(const Tensor &tensor) {
    dtype_ = tensor.dtype_;
    storage_ = tensor.storage_;
    shape_ = tensor.shape_;
    writeable_ = tensor.writeable_;
    offset_ = tensor.offset_;
    stride_ = tensor.stride_;
    return *this;
  }

  Tensor(const Tensor &tensor) {
    dtype_ = tensor.dtype_;
    storage_ = tensor.storage_;
    shape_ = tensor.shape_;
    writeable_ = tensor.writeable_;
    offset_ = tensor.offset_;
    stride_ = tensor.stride_;
  }

  Tensor clone(bool pinned = false) const {
    TV_ASSERT_RT_ERR(!empty(), "clone a empty tensor");
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    Tensor newtensor(shape_, stride_, dtype_, device(), pinned,
                     storage_->managed());
    newtensor.copy_(*this);
    return newtensor;
  }

  Tensor astype(DType dtype) {
    if (dtype == dtype_) {
      return clone();
    }
    TV_ASSERT_INVALID_ARG(device() == -1, "only support cpu tensor");
    TV_ASSERT_INVALID_ARG(!empty(), "can't be used in empty tensor");
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    auto tensor = Tensor();
    Dispatch<detail::all_tensor_types_t>()(dtype, [&](auto Idst) {
      using Tdst = TV_DECLTYPE(Idst);
      Dispatch<detail::all_tensor_types_t>()(this->dtype_, [&](auto Icur) {
        using Tcur = TV_DECLTYPE(Icur);
        if (std::is_convertible<Tcur, Tdst>::value) {
          auto ptr = this->data<Tcur>();
          tensor = Tensor(this->shape_, this->stride_, dtype, this->device(),
                          this->pinned(), this->storage_->managed());
          std::copy(ptr, ptr + this->size(), tensor.data<Tdst>());
        } else {
          TV_THROW_INVALID_ARG("not convertable from", type_s<Tcur>, "to",
                               type_s<Tdst>);
        }
      });
    });
    return tensor;
  }

  template <class... Ts, typename F> inline void dispatch(F &&f) {
    return tv::dispatch<Ts...>(dtype_, std::forward<F>(f));
  }

protected:
  inline void writable_check() {
    TV_ASSERT_RT_ERR(writeable_,
                     "you cant do non-const operation when not writable");
  }

  DType dtype_;
  std::shared_ptr<detail::TensorStorage<uint8_t>> storage_;
  TensorShape shape_;
  size_t offset_ = 0;
  TensorShape stride_;

private:
  bool writeable_ = true;
  bool contiguous_ = true;
};

template <typename Os> Os &operator<<(Os &os, const Tensor &tensor) {
  TV_ASSERT_INVALID_ARG(tensor.device() == -1, "must be cpu tensor");
  Dispatch<detail::all_tensor_types_t>()(tensor.dtype(), [&](auto I) {
    using T = TV_DECLTYPE(I);
    std::stringstream ss;
    if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      ss << std::setprecision(4);
    }
    os << tensor.tview<T, -1, DefaultPtrTraits, int64_t>().repr(ss);
  });
  return os;
}

inline Tensor from_blob(void *ptr, TensorShape shape, DType dtype, int device) {
  return Tensor(ptr, shape, dtype, device);
}

inline Tensor from_blob(const void *ptr, TensorShape shape, DType dtype,
                        int device) {
  return Tensor(ptr, shape, dtype, device);
}

} // namespace tv