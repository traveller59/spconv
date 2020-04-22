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
#include "tensorview.h"
#include <memory>
#include <spconv/mp_helper.h>
#ifdef SPCONV_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

namespace tv
{
enum DType
{
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

namespace detail
{

template <typename T>
class TensorStorage
{
public:
  TensorStorage(size_t size, int device = -1, bool managed = false)
      : mSize(size), device_(device), managed_(managed)
  {
    if (size == 0)
    {
      mPtr = nullptr;
    }
    else
    {
      if (device == -1)
      {
#ifdef SPCONV_CUDA
        checkCudaErrors(cudaMallocHost(&mPtr, size * sizeof(T)));
#else
        mPtr = new T[size];
#endif
      }
      else
      {
#ifdef SPCONV_CUDA
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (device >= deviceCount)
        {
          TV_ASSERT_INVALID_ARG("you provide device ", device,
                                " but you only have ", deviceCount, " device.");
        }
        cudaSetDevice(device);
        if (managed)
        {
          checkCudaErrors(cudaMallocManaged(&this->mPtr, size * sizeof(T)));
        }
        else
        {
          checkCudaErrors(cudaMalloc(&mPtr, size * sizeof(T)));
        }
#else
        TV_ASSERT_INVALID_ARG(false, "don't compiled with cuda");
#endif
      }
    }
  }
  TensorStorage(T *ptr, size_t size, int device)
      : mSize(size), mPtr(ptr), from_blob_(true), device_(device) {}

  virtual ~TensorStorage()
  {
    if (empty())
    {
      return;
    }
    if (from_blob_)
    {
      return;
    }
    if (device_ == -1)
    {
#ifdef SPCONV_CUDA
      cudaFreeHost(mPtr);
#else
      delete[] mPtr;
#endif
    }
    else
    {
#ifdef SPCONV_CUDA
      cudaFree(mPtr);
#endif
    }
  };

  inline size_t size() const { return mSize; }

  T *data() { return mPtr; }
  const T *data() const { return mPtr; }

  bool empty() const { return mPtr == nullptr || mSize == 0; }
  bool managed() const { return managed_; }
  int device() const { return device_; }
  void zero_()
  {
    if (device_ == -1)
    {
      std::memset(data(), 0, mSize);
      // std::fill(data(), data() + mSize, 0);
    }
    else
    {
#ifdef SPCONV_CUDA
      checkCudaErrors(cudaMemset(data(), 0, mSize / sizeof(T)));
#else
      TV_ASSERT_INVALID_ARG(false, "don't compiled with cuda");
#endif
    }
  }

private:
  T *mPtr = nullptr;
  size_t mSize = 0;
  int device_ = -1;
  bool from_blob_ = false;
  bool managed_ = false;
};

size_t sizeof_dtype(DType dtype)
{
  switch (dtype)
  {
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
#ifdef SPCONV_CUDA
  case float16:
    return sizeof(__half);
#endif
  default:
    TV_THROW_RT_ERR("unsupported dtype");
  }
  return 0;
}

std::string typeString(DType t)
{
  switch (t)
  {
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
#ifdef SPCONV_CUDA
  case DType::float16:
    return "half";
#endif
  default:
    return "";
  }
}

template <typename T>
struct TypeToDtypeTraits;

template <>
struct TypeToDtypeTraits<int32_t>
{
  static constexpr DType dtype = int32;
};

#ifdef SPCONV_CUDA
template <>
struct TypeToDtypeTraits<__half>
{
  static constexpr DType dtype = float16;
};
#endif

template <>
struct TypeToDtypeTraits<float>
{
  static constexpr DType dtype = float32;
};
template <>
struct TypeToDtypeTraits<double>
{
  static constexpr DType dtype = float64;
};
template <>
struct TypeToDtypeTraits<int16_t>
{
  static constexpr DType dtype = int16;
};
template <>
struct TypeToDtypeTraits<int8_t>
{
  static constexpr DType dtype = int8;
};
template <>
struct TypeToDtypeTraits<int64_t>
{
  static constexpr DType dtype = int64;
};
template <>
struct TypeToDtypeTraits<uint8_t>
{
  static constexpr DType dtype = uint8;
};
template <>
struct TypeToDtypeTraits<uint16_t>
{
  static constexpr DType dtype = uint16;
};
template <>
struct TypeToDtypeTraits<uint32_t>
{
  static constexpr DType dtype = uint32;
};
template <>
struct TypeToDtypeTraits<uint64_t>
{
  static constexpr DType dtype = uint64;
};

} // namespace detail

template <class T>
constexpr DType type_v = detail::TypeToDtypeTraits<T>::dtype;

struct Tensor
{
  Tensor() {}
  Tensor(Shape shape, DType dtype, int device = -1, bool managed = false)
      : dtype_(dtype)
  {
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed);
    shape_ = shape;
  }
  Tensor(void *ptr, Shape shape, DType dtype, int device = -1) : dtype_(dtype)
  {
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
  }

  template <typename T>
  TensorView<T> tview()
  {
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    TV_ASSERT_RT_ERR(shape_.size() == storage_->size() / sizeof(T), "error");
    return TensorView<T>(reinterpret_cast<T *>(storage_->data()), shape_);
  }
  template <typename T>
  TensorView<T> tview() const
  {
    TV_ASSERT_RT_ERR(shape_.size() == storage_->size() / sizeof(T), "error");
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    return TensorView<const std::remove_const_t<T>>(
        reinterpret_cast<const std::remove_const_t<T> *>(storage_->data()),
        shape_);
  }
  bool empty() const { return storage_->empty(); }
  DType dtype() const { return dtype_; }
  int device() const { return storage_->device(); }
  const Shape &shape() const { return shape_; }
  int dim(int idx) const
  {
    TV_ASSERT_RT_ERR(idx < shape_.size(), "error");
    return shape_[idx];
  }
  const uint8_t *raw_data() const { return storage_->data(); }
  size_t size() const { return shape_.size(); }
  Tensor &zero_()
  {
    storage_->zero_();
    return *this;
  }
  uint8_t *raw_data() { return storage_->data(); }
  template <typename T>
  Tensor &fill_(T value)
  {
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    auto ptr = reinterpret_cast<T *>(raw_data());
    std::fill(ptr, ptr + size(), value);
    return *this;
  }

  template <typename T>
  T *data()
  {
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    return reinterpret_cast<T *>(raw_data());
  }

  template <typename T>
  const T *data() const
  {
    TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "error");
    return reinterpret_cast<const T *>(raw_data());
  }

  void copy_(const Tensor &tensor)
  {
    TV_ASSERT_RT_ERR(!empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(dtype() == tensor.dtype(), "must have same dtype");
    if (device() == -1 && tensor.device() == -1)
    {
#ifdef SPCONV_CUDA
      host2host(storage_->data(), tensor.raw_data(),
                size() * detail::sizeof_dtype(dtype_));
#else
      std::copy(tensor.raw_data(),
                tensor.raw_data() + size() * detail::sizeof_dtype(dtype_),
                storage_->data());
#endif
    }
#ifdef SPCONV_CUDA
    else if (device() >= 0 && tensor.device() == -1)
    {
      // host2dev
      host2dev(storage_->data(), tensor.raw_data(),
               size() * detail::sizeof_dtype(dtype_));
    }
    else if (device() == -1 && tensor.device() >= 0)
    {
      // dev2host
      dev2host(storage_->data(), tensor.raw_data(),
               size() * detail::sizeof_dtype(dtype_));
    }
    else if (device() >= 0 && tensor.device() >= 0)
    {
      // dev2dev
      dev2dev(storage_->data(), tensor.raw_data(),
              size() * detail::sizeof_dtype(dtype_));
    }
#endif
    else
    {
      TV_ASSERT_RT_ERR(false, "only support cpu tensor");
    }
  }

  Tensor cpu() const
  {
    if (storage_->device() == -1)
    {
      return *this;
    }
    Tensor res(shape_, dtype_, -1, storage_->managed());
    res.copy_(*this);
    return res;
  }

  template <typename T>
  void copy_(const TensorView<T> &tensor, int device)
  {
    Tensor src = from_blob(tensor, device);
    return copy_(src);
  }

protected:
  DType dtype_;
  std::shared_ptr<detail::TensorStorage<uint8_t>> storage_;
  Shape shape_;
};

inline Tensor from_blob(void *ptr, Shape shape, DType dtype, int device)
{
  return Tensor(ptr, shape, dtype, device);
}

template <typename T>
Tensor from_blob(TensorView<T> tensor, int device)
{
  return Tensor(tensor.data(), tensor.shape, type_v<T>, device);
}

template <class... Ts, typename F>
void dispatch(DType t, F &&f)
{
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  spconv::mp_for_each<spconv::mp_list<Ts...>>([=, &notFound, &f](auto I) {
    if (type_v<decltype(I)> == t)
    {
      std::forward<F>(f)(decltype(I)());
      notFound = false;
    }
  });
  if (notFound)
  {
    std::stringstream ss;
    spconv::mp_for_each<spconv::mp_list<Ts...>>([=, &ss](auto I) {
      ss << detail::TypeToString<decltype(I)>::value << " ";
    });
    TV_THROW_RT_ERR("unknown type", detail::typeString(t),
                    ", available: ", ss.str());
  }
}

} // namespace tv