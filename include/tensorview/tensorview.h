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
#include "common.h"
#include "prettyprint.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>
#ifdef TV_CUDA
#include <cuda_runtime_api.h>
#endif
namespace tv {

#if (defined(__clang__) && defined(__CUDA__)) || defined(__NVCC__)

#define TV_HOST_DEVICE_INLINE __forceinline__ __device__ __host__
#define TV_DEVICE_INLINE __forceinline__ __device__
#define TV_HOST_DEVICE __device__ __host__
#define TV_ASSERT(expr) assert(expr)
#elif defined(__CUDACC_RTC__)
#define TV_ASSERT(expr) assert(expr)
#define TV_HOST_DEVICE_INLINE __forceinline__ __device__
#define TV_DEVICE_INLINE __forceinline__ __device__
#define TV_HOST_DEVICE __device__ __host__
#else
#define TV_ASSERT(x) assert(x)
#define TV_HOST_DEVICE_INLINE inline
#define TV_HOST_DEVICE
#endif

#define TV_REQUIRE(expr, ...)                                                  \
  {                                                                            \
    if (!(expr)) {                                                             \
      printf(__VA_ARGS__);                                                     \
      assert(expr);                                                            \
    }                                                                          \
  }

#define TV_CHECK_CUDA_ERR()                                                    \
  {                                                                            \
    auto __macro_err = cudaGetLastError();                                     \
    if (__macro_err != cudaSuccess) {                                          \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << " " << __LINE__ << "\n";                        \
      __macro_s << "cuda execution failed with error " << __macro_err;         \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  }

#define TV_CHECK_CUDA_ERR_V2(...)                                              \
  {                                                                            \
    auto __macro_err = cudaGetLastError();                                     \
    if (__macro_err != cudaSuccess) {                                          \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << " " << __LINE__ << "\n";                        \
      __macro_s << "cuda execution failed with error " << __macro_err;         \
      __macro_s << " " << cudaGetErrorString(__macro_err) << "\n";             \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  }

#ifdef TV_CUDA
struct GPU {
  GPU(cudaStream_t s = 0) : mStream(s) {}
  virtual cudaStream_t getStream() const { return mStream; }
  cudaStream_t mStream = 0;
};
#endif
struct CPU {};

#ifndef TV_MAX_DIM
#define TV_MAX_DIM 6
#endif

template <typename T> struct DefaultPtrTraits { typedef T *type; };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T> struct RestrictPtrTraits {
  typedef T *__restrict__ type;
};
#endif

/*
template <typename T>
constexpr size_t calc_align(size_t ndim)
{
  if (ndim * sizeof(T) == 1)
    return 1;
  else if (ndim * sizeof(T) == 2)
    return 2;
  else if (ndim * sizeof(T) <= 4 && ndim * sizeof(T) > 2)
    return 4;
  else if (ndim * sizeof(T) <= 8 && ndim * sizeof(T) > 4)
    return 8;
  else if (ndim * sizeof(T) <= 16 && ndim * sizeof(T) > 8)
    return 16;
  else if (ndim * sizeof(T) <= 32 && ndim * sizeof(T) > 16)
    return 32;
  else
    return 64;
}
*/

namespace detail {
template <typename _InIter>
using _RequireInputIter = typename std::enable_if<std::is_convertible<
    typename std::iterator_traits<_InIter>::iterator_category,
    std::input_iterator_tag>::value>::type;

}

template <typename T, size_t MaxDim = TV_MAX_DIM>
struct /*alignas(calc_align<T>(MaxDim))*/ SimpleVector {
public:
  TV_HOST_DEVICE_INLINE SimpleVector(){};
  TV_HOST_DEVICE_INLINE SimpleVector(size_t count, T init = T())
      : size_(count) {
    for (size_t i = 0; i < count; ++i) {
      array_[i] = init;
    }
  };
  template <typename Iterator, typename = detail::_RequireInputIter<Iterator>>
  SimpleVector(Iterator first, Iterator last) {
    size_ = 0;
    for (; first != last; ++first) {
      if (size_ >= MaxDim) {
        TV_THROW_INVALID_ARG("iterator too long");
      }
      array_[size_++] = *first;
    }
  };
  TV_HOST_DEVICE_INLINE SimpleVector(std::initializer_list<T> q) {
    TV_ASSERT(q.size() <= MaxDim);
    size_ = 0;
    for (T s : q) {
      array_[size_++] = s;
    }
    size_ = q.size();
  }
  SimpleVector(const std::vector<T> &arr) {
    TV_ASSERT(arr.size() <= MaxDim);
    for (size_t i = 0; i < arr.size(); ++i) {
      array_[i] = arr[i];
    }
    size_ = arr.size();
  }
  TV_HOST_DEVICE_INLINE SimpleVector(const SimpleVector<T, MaxDim> &arr) {
    TV_ASSERT(arr.size() <= MaxDim);
    for (size_t i = 0; i < arr.size(); ++i) {
      array_[i] = arr[i];
    }
    size_ = arr.size();
  }
  TV_HOST_DEVICE_INLINE T &operator[](int idx) {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < size_);
#endif
    return array_[idx];
  }
  TV_HOST_DEVICE_INLINE const T &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < size_);
#endif
    return array_[idx];
  }
  TV_HOST_DEVICE_INLINE void push_back(T s) {
#ifdef TV_DEBUG
    TV_ASSERT(size_ < MaxDim);
#endif
    array_[size_] = s;
    size_++;
  }
  TV_HOST_DEVICE_INLINE void pop_back() {
#ifdef TV_DEBUG
    TV_ASSERT(size_ > 0);
#endif
    size_--;
  }

  TV_HOST_DEVICE_INLINE size_t size() const { return size_; }
  TV_HOST_DEVICE_INLINE const T *data() const { return array_; }
  TV_HOST_DEVICE_INLINE T *data() { return array_; }
  TV_HOST_DEVICE_INLINE size_t empty() const { return size_ == 0; }

  typedef size_t size_type;

  class iterator {
  public:
    typedef iterator self_type;
    typedef T value_type;
    typedef T &reference;
    typedef T *pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::ptrdiff_t difference_type;
    TV_HOST_DEVICE_INLINE iterator(pointer ptr) : ptr_(ptr) {}
    TV_HOST_DEVICE_INLINE self_type operator++(int junk) {
      self_type i = *this;
      ptr_++;
      return i;
    }
    TV_HOST_DEVICE_INLINE self_type operator++() {
      ptr_++;
      return *this;
    }
    TV_HOST_DEVICE_INLINE reference operator*() { return *ptr_; }
    TV_HOST_DEVICE_INLINE pointer operator->() { return ptr_; }
    TV_HOST_DEVICE_INLINE bool operator==(const self_type &rhs) const {
      return ptr_ == rhs.ptr_;
    }
    TV_HOST_DEVICE_INLINE bool operator!=(const self_type &rhs) const {
      return ptr_ != rhs.ptr_;
    }

  private:
    pointer ptr_;
  };

  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef T value_type;
    typedef const T &reference;
    typedef const T *pointer;
    typedef std::ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;
    TV_HOST_DEVICE_INLINE const_iterator(pointer ptr) : ptr_(ptr) {}
    TV_HOST_DEVICE_INLINE self_type operator++(int junk) {
      self_type i = *this;
      ptr_++;
      return i;
    }
    TV_HOST_DEVICE_INLINE self_type operator++() {
      ptr_++;
      return *this;
    }
    TV_HOST_DEVICE_INLINE reference operator*() { return *ptr_; }
    TV_HOST_DEVICE_INLINE pointer operator->() { return ptr_; }
    TV_HOST_DEVICE_INLINE bool operator==(const self_type &rhs) const {
      return ptr_ == rhs.ptr_;
    }
    TV_HOST_DEVICE_INLINE bool operator!=(const self_type &rhs) const {
      return ptr_ != rhs.ptr_;
    }

  private:
    pointer ptr_;
  };

  TV_HOST_DEVICE_INLINE iterator begin() { return iterator(array_); }

  TV_HOST_DEVICE_INLINE iterator end() { return iterator(array_ + size_); }

  TV_HOST_DEVICE_INLINE const_iterator begin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE const_iterator end() const {
    return const_iterator(array_ + size_);
  }
  TV_HOST_DEVICE_INLINE const_iterator cbegin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE const_iterator cend() const {
    return const_iterator(array_ + size_);
  }

protected:
  T array_[MaxDim];
  size_t size_ = 0;
};

template <typename T, size_t MaxDim>
bool operator==(const SimpleVector<T, MaxDim> &lfs,
                const SimpleVector<T, MaxDim> &rfs) {
  if (lfs.size() != rfs.size())
    return false;
  for (size_t i = 0; i < lfs.size(); ++i) {
    if (lfs[i] != rfs[i])
      return false;
  }
  return true;
}

template <typename T, size_t MaxDim>
bool operator!=(const SimpleVector<T, MaxDim> &lfs,
                const SimpleVector<T, MaxDim> &rfs) {

  return !(lfs == rfs);
}

struct Slice {
  template <class... Integers> TV_HOST_DEVICE_INLINE Slice(Integers... ints) {
    static_assert(sizeof...(ints) <= 3, "slice init must smaller than 3");
    SimpleVector<int, 3> slices{int(ints)...};
    slices_[0] = -1;
    slices_[1] = -1;
    slices_[2] = -1;
    for (size_t i = 0; i < slices.size(); ++i) {
      slices_[i] = slices[i];
    }
  }

  TV_HOST_DEVICE_INLINE Slice() {
    slices_[0] = -1;
    slices_[1] = -1;
    slices_[2] = -1;
  }
  template <typename T>
  TV_HOST_DEVICE_INLINE Slice(std::initializer_list<T> slice) {
    slices_[0] = -1;
    slices_[1] = -1;
    slices_[2] = -1;
    TV_ASSERT(slice.size() <= 3);
    int idx = 0;
    for (T s : slice) {
      slices_[idx] = int(s);
      ++idx;
    }
  }
  TV_HOST_DEVICE_INLINE int &operator[](int idx) {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < 3);
#endif
    return slices_[idx];
  }
  TV_HOST_DEVICE_INLINE const int &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_ASSERT(idx >= 0 && idx < 3);
#endif
    return slices_[idx];
  }

protected:
  int slices_[3];
};

template <size_t MaxDim = TV_MAX_DIM, typename Tindex = int>
struct ShapeBase : public SimpleVector<Tindex, MaxDim> {
  TV_HOST_DEVICE_INLINE ShapeBase() : SimpleVector<Tindex, MaxDim>(){};
  TV_HOST_DEVICE_INLINE ShapeBase(std::initializer_list<Tindex> shape)
      : SimpleVector<Tindex, MaxDim>(shape) {}
  TV_HOST_DEVICE_INLINE ShapeBase(SimpleVector<Tindex, MaxDim> vec)
      : SimpleVector<Tindex, MaxDim>(vec) {}
  template <typename T, template <class...> class Container>
  ShapeBase(Container<T> shape) : SimpleVector<Tindex, MaxDim>(shape) {}
  TV_HOST_DEVICE_INLINE ShapeBase(const ShapeBase<MaxDim> &shape)
      : SimpleVector<Tindex, MaxDim>(shape) {}
  ShapeBase(const std::vector<Tindex> &arr)
      : SimpleVector<Tindex, MaxDim>(arr) {}

  ShapeBase<MaxDim, Tindex> &
  operator=(const ShapeBase<MaxDim, Tindex> &shape) = default;
  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> subshape(Tindex start,
                                                    Tindex end) const {
#ifdef TV_DEBUG
    TV_ASSERT(start >= 0 && end <= this->size_ && end > start);
#endif
    ShapeBase<MaxDim, Tindex> shape;
    for (Tindex i = start; i < end; ++i) {
      shape.push_back(this->array_[i]);
    }
    return shape;
  }
  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> subshape(Tindex start) const {
#ifdef TV_DEBUG
    TV_ASSERT(start >= 0 && start <= this->size_);
#endif
    ShapeBase<MaxDim, Tindex> shape;
    for (size_t i = start; i < this->size_; ++i) {
      shape.push_back(this->array_[i]);
    }
    return shape;
  }

  TV_HOST_DEVICE size_t size() const {
    if (this->size_ == 0)
      return 0;
    size_t s = 1;
    for (int i = 0; i < int(this->size_); ++i) {
      s *= this->array_[i];
    }
    return s;
  }
  TV_HOST_DEVICE_INLINE size_t ndim() const { return this->size_; }

  TV_HOST_DEVICE ShapeBase<MaxDim, Tindex> squeeze() const {
    ShapeBase<MaxDim, Tindex> shape;
    for (size_t i = 0; i < this->size_; ++i) {
      if (this->array_[i] != 1)
        shape.push_back(this->array_[i]);
    }
    if (shape.empty()) {
      // dont support empty shape for now
      shape.push_back(1);
    }
    return shape;
  }
  template <size_t MaxDim2 = MaxDim>
  TV_HOST_DEVICE ShapeBase<MaxDim2, Tindex> squeeze(int dim) const {
    static_assert(MaxDim2 >= MaxDim - 1, "error");

    ShapeBase<MaxDim2, Tindex> shape;
    for (size_t i = 0; i < this->size_; ++i) {
      if (i != size_t(dim) || this->array_[i] != 1)
        shape.push_back(this->array_[i]);
    }
    return shape;
  }
  template <size_t MaxDim2 = MaxDim>
  TV_HOST_DEVICE ShapeBase<MaxDim2, Tindex> unsqueeze(int dim) const {
    static_assert(MaxDim2 >= MaxDim - 1, "error");
    ShapeBase<MaxDim2, Tindex> shape;
    for (size_t i = 0; i < this->size_; ++i) {
      if (i == size_t(dim))
        shape.push_back(1);
      shape.push_back(this->array_[i]);
    }
    return shape;
  }

  TV_HOST_DEVICE size_t prod(Tindex start = 0) const {
    size_t res = 1;
    for (size_t i = start; i < this->size_; ++i) {
      res *= this->array_[i];
    }
    return res;
  }
  template <size_t MaxDim2 = MaxDim>
  TV_HOST_DEVICE ShapeBase<MaxDim2, Tindex> stride_rowmajor() {
    static_assert(MaxDim2 >= MaxDim, "error");
    Tindex p = Tindex(1);
    ShapeBase<MaxDim2, Tindex> res(this->size_);
    for (Tindex i = this->size_ - 1; i >= 0; --i) {
      res[i] = p;
      p *= this->array_[i];
    }
    return res;
  }
};

using Shape = ShapeBase<TV_MAX_DIM, int>;

template <class... Inds>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(std::vector<int> &shape,
                                           Inds... indexes) {
  unsigned offset = 0;
  unsigned m = 1;
  int indexes_vec[sizeof...(indexes)] = {indexes...};
#ifdef TV_DEBUG
  TV_ASSERT(sizeof...(indexes) == shape.size());
#endif
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int i = sizeof...(indexes) - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(std::vector<int> &shape,
                                           std::vector<int> &indexes_vec) {
  unsigned offset = 0;
  unsigned m = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

template <class... Inds>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Shape &shape,
                                           Inds... indexes) {
  unsigned offset = 0;
  unsigned m = 1;
  int indexes_vec[sizeof...(indexes)] = {indexes...};
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int i = sizeof...(indexes) - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Shape &shape,
                                           const Shape &indexes_vec) {
  unsigned offset = 0;
  unsigned m = 1;
  for (int i = indexes_vec.ndim() - 1; i >= 0; --i) {
    offset += m * indexes_vec[i];
    m *= shape[i];
  }
  return offset;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE unsigned rowArrayIdx(const Index *indexes,
                                           const Index *shape) {
  unsigned offset = 0;
  unsigned m = 1;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int i = NDim - 1; i >= 0; --i) {
    offset += m * indexes[i];
    m *= shape[i];
  }
  return offset;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE_INLINE Index rowArrayIdxInv(Index index, Index *output,
                                           const Index *shape) {
#pragma unroll
  for (int i = NDim - 1; i >= 0; --i) {
    output[i] = index % shape[i];
    index -= output[i];
    index /= shape[i];
  }
  return index;
}

template <typename Index>
TV_HOST_DEVICE Index rowArrayIdxInv(Index index, Index *output,
                                    const Index *shape, int ndim) {
  for (int i = ndim - 1; i >= 0; --i) {
    output[i] = index % shape[i];
    index -= output[i];
    index /= shape[i];
  }
  return index;
}

template <int N> struct ArrayIndexRowMajorReverse {
  template <typename TShape, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, T index,
                                            Ts... inds) {
    return index +
           shape[N - 1] * ArrayIndexRowMajorReverse<N - 1>::run(shape, inds...);
  }
  template <typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape, T index,
                                                 Ts... inds) {
    return index +
           shape[N - 1] * ArrayIndexRowMajorReverse<N - 1>::run(shape, inds...);
  }
};

template <> struct ArrayIndexRowMajorReverse<1> {
  template <typename TShape, typename T>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, T idx) {
    return idx;
  }
  template <typename T>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape, T idx) {
    return idx;
  }
};

template <int N, int Ndim> struct ArrayIndexRowMajor {
  // this array index provide almost same compiled code. compile it in
  // https://godbolt.org/ for more details.
  template <typename TShape, typename Tinit, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, Tinit start,
                                            T index, Ts... inds) {
    return ArrayIndexRowMajor<N - 1, Ndim>::run(
        shape, (index + start) * shape[Ndim - N + 1], inds...);
  }
  template <typename Tinit, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned
  runShape(const Shape &shape, Tinit start, T index, Ts... inds) {
    return ArrayIndexRowMajor<N - 1, Ndim>::runShape(
        shape, (index + start) * shape[Ndim - N + 1], inds...);
  }
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned
  runPtrs(const TShape *indexes, const TShape *shape, Tinit start) {
    return ArrayIndexRowMajor<N - 1, Ndim>::runPtrs(
        indexes, shape, (indexes[Ndim - N] + start) * shape[Ndim - N + 1]);
  }
};

template <int Ndim> struct ArrayIndexRowMajor<1, Ndim> {
  template <typename TShape, typename Tinit, typename T>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, Tinit start,
                                            T idx) {
    return start + idx;
  }
  template <typename Tinit, typename T>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape,
                                                 Tinit start, T idx) {
    return start + idx;
  }
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned
  runPtrs(const TShape *indexes, const TShape *shape, Tinit start) {
    return start + indexes[Ndim - 1];
  }
};

template <> struct ArrayIndexRowMajor<0, 0> {
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *shape, Tinit start) {
    return 0;
  }
  template <typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned runShape(const Shape &shape,
                                                 Tinit start) {
    return 0;
  }
  template <typename TShape, typename Tinit>
  TV_HOST_DEVICE_INLINE static unsigned
  runPtrs(const TShape *indexes, const TShape *shape, Tinit start) {
    return 0;
  }
};

template <int N, int Ndim> struct ArrayIndexStride {
  // this array index provide almost same compiled code. compile it in
  // https://godbolt.org/ for more details.
  template <typename TShape, typename Tinit, typename T, class... Ts>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *stride, Tinit start,
                                            T index, Ts... inds) {
    return ArrayIndexStride<N - 1, Ndim>::run(
        stride, start + index * stride[Ndim - N + 1], inds...);
  }
};

template <int Ndim> struct ArrayIndexStride<1, Ndim> {
  template <typename TShape, typename Tinit, typename T>
  TV_HOST_DEVICE_INLINE static unsigned run(const TShape *stride, Tinit start,
                                            T idx) {
    return start + idx * stride[Ndim - 1];
  }
};

#if __cplusplus >= 201703L
template <size_t... N, class T, class... Ts>
TV_HOST_DEVICE_INLINE T array_index_stride(const T *stride, Ts... ids) {
  return ((stride[N] * std::get<N>(std::forward_as_tuple(ids...))) + ...);
}
#endif

namespace detail {
template <typename T> struct TypeToString;
template <> struct TypeToString<bool> {
  static constexpr const char *value = "bool";
};
template <> struct TypeToString<const bool> {
  static constexpr const char *value = "bool";
};
template <> struct TypeToString<int32_t> {
  static constexpr const char *value = "int32";
};
template <> struct TypeToString<float> {
  static constexpr const char *value = "float";
};
template <> struct TypeToString<double> {
  static constexpr const char *value = "double";
};
template <> struct TypeToString<int16_t> {
  static constexpr const char *value = "int16";
};
template <> struct TypeToString<int8_t> {
  static constexpr const char *value = "int8";
};
template <> struct TypeToString<int64_t> {
  static constexpr const char *value = "int64";
};
template <> struct TypeToString<uint8_t> {
  static constexpr const char *value = "uint8";
};
template <> struct TypeToString<uint16_t> {
  static constexpr const char *value = "uint16";
};
template <> struct TypeToString<uint32_t> {
  static constexpr const char *value = "uint32";
};
template <> struct TypeToString<uint64_t> {
  static constexpr const char *value = "uint64";
};
template <> struct TypeToString<const int32_t> {
  static constexpr const char *value = "int32";
};
template <> struct TypeToString<const float> {
  static constexpr const char *value = "float";
};
template <> struct TypeToString<const double> {
  static constexpr const char *value = "double";
};
template <> struct TypeToString<const int16_t> {
  static constexpr const char *value = "int16";
};
template <> struct TypeToString<const int8_t> {
  static constexpr const char *value = "int8";
};
template <> struct TypeToString<const int64_t> {
  static constexpr const char *value = "int64";
};
template <> struct TypeToString<const uint8_t> {
  static constexpr const char *value = "uint8";
};
template <> struct TypeToString<const uint16_t> {
  static constexpr const char *value = "uint16";
};
template <> struct TypeToString<const uint32_t> {
  static constexpr const char *value = "uint32";
};
template <> struct TypeToString<const uint64_t> {
  static constexpr const char *value = "uint64";
};
} // namespace detail

template <typename T>
constexpr const char *type_s = detail::TypeToString<T>::value;

namespace detail {

template <typename T, int Rank,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = int>
struct TensorAccesserBase {
  static constexpr int rank_value = Rank;
  using ptr_t = typename PtrTraits<T>::type;

  static_assert(Rank > 0, "error");

  explicit TV_HOST_DEVICE_INLINE TensorAccesserBase(ptr_t ptr,
                                                    const Tindex *stride_ptr)
      : ptr_(ptr), stride_ptr_(stride_ptr) {}

  TV_HOST_DEVICE_INLINE ptr_t data() { return ptr_; }
  TV_HOST_DEVICE_INLINE const ptr_t data() const { return ptr_; }

  template <class... Inds> TV_HOST_DEVICE_INLINE T &operator()(Inds... inds) {
    static_assert(sizeof...(inds) == Rank, "error");
    return ptr_[ArrayIndexStride<Rank, Rank>::run(stride_ptr_, 0, inds...)];
  }

  template <class... Inds>
  TV_HOST_DEVICE_INLINE const T &operator()(Inds... inds) const {
    static_assert(sizeof...(inds) == Rank, "error");
    return ptr_[ArrayIndexStride<Rank, Rank>::run(stride_ptr_, 0, inds...)];
  }

protected:
  ptr_t ptr_;
  const Tindex *stride_ptr_;
};
} // namespace detail

template <typename T, int Rank,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = int>
struct TensorAccesser
    : public detail::TensorAccesserBase<T, Rank, PtrTraits, Tindex> {
  using ptr_t = typename PtrTraits<T>::type;
  static_assert(Rank > 0, "error");
  explicit TV_HOST_DEVICE_INLINE TensorAccesser(ptr_t ptr,
                                                const Tindex *stride_ptr)
      : detail::TensorAccesserBase<T, Rank, PtrTraits, Tindex>(ptr,
                                                               stride_ptr) {}

  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  operator[](int i) {
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        this->ptr_ + this->stride_ptr_[0] * i, this->stride_ptr_ + 1);
  }
  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  operator[](int i) const {
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        this->ptr_ + this->stride_ptr_[0] * i, this->stride_ptr_ + 1);
  }
};

template <typename T, template <class> class PtrTraits, typename Tindex>
struct TensorAccesser<T, 1, PtrTraits, Tindex>
    : public detail::TensorAccesserBase<T, 1, PtrTraits, Tindex> {
  using ptr_t = typename PtrTraits<T>::type;

  explicit TV_HOST_DEVICE_INLINE TensorAccesser(ptr_t ptr,
                                                const Tindex *stride_ptr)
      : detail::TensorAccesserBase<T, 1, PtrTraits, Tindex>(ptr, stride_ptr) {}

  TV_HOST_DEVICE_INLINE T &operator[](int i) {
    return this->ptr_[this->stride_ptr_[0] * i];
  }
  TV_HOST_DEVICE_INLINE T &operator[](int i) const {
    return this->ptr_[this->stride_ptr_[0] * i];
  }
};

template <typename T, int Rank = -1,
          template <class> class PtrTraits = DefaultPtrTraits,
          typename Tindex = int>
struct TensorView {
  static constexpr int rank_value = Rank;
  using ptr_t = typename PtrTraits<T>::type;
  using tv_shape_t = ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex>;
  using no_cv_type = typename std::remove_cv<T>::type;
  static_assert(Rank == -1 || Rank > 0, "error");

  TV_HOST_DEVICE_INLINE TensorView() {}
  explicit TV_HOST_DEVICE_INLINE TensorView(ptr_t ptr, tv_shape_t shape)
      : ptr_(ptr), shape_(shape), stride_(shape.stride_rowmajor()) {}

  explicit TV_HOST_DEVICE_INLINE TensorView(ptr_t ptr, tv_shape_t shape,
                                            tv_shape_t stride)
      : ptr_(ptr), shape_(shape), stride_(stride) {}

  operator TensorView<const no_cv_type, Rank, PtrTraits, Tindex>() {
    return TensorView<const no_cv_type, Rank, PtrTraits, Tindex>(ptr_, shape_);
  } // conversion function

  template <class... Inds> TV_HOST_DEVICE_INLINE T &operator()(Inds... inds) {
    static_assert(Rank == -1 || sizeof...(inds) == Rank, "error");
#if defined TV_DEBUG
    int idxes[sizeof...(Inds)]{int(inds)...};
    TV_REQUIRE(sizeof...(inds) == shape_.ndim(),
               "you provide %d indexes, but dim is %d\n", sizeof...(inds),
               shape_.ndim());
    for (int i = 0; i < sizeof...(inds); ++i) {
      TV_REQUIRE(idxes[i] >= 0 && idxes[i] < shape_[i],
                 "index-%d(%d) out-of-range: [0, %d)\n", i, idxes[i],
                 shape_[i]);
    }
#endif
    constexpr int Ndim = sizeof...(Inds);
    return ptr_[ArrayIndexRowMajor<Ndim, Ndim>::runShape(shape_, 0, inds...)];
  }
  template <class... Inds>
  TV_HOST_DEVICE_INLINE const T &operator()(Inds... inds) const {
    static_assert(Rank == -1 || sizeof...(inds) == Rank, "error");
#if defined TV_DEBUG
    int idxes[sizeof...(Inds)]{int(inds)...};
    TV_REQUIRE(sizeof...(inds) == shape_.ndim(),
               "you provide %d indexes, but dim is %d\n", sizeof...(inds),
               shape_.ndim());
    for (int i = 0; i < sizeof...(inds); ++i) {
      TV_REQUIRE(idxes[i] >= 0 && idxes[i] < shape_[i],
                 "index-%d(%d) out-of-range: [0, %d)\n", i, idxes[i],
                 shape_[i]);
    }
#endif
    constexpr int Ndim = sizeof...(Inds);
    return ptr_[ArrayIndexRowMajor<Ndim, Ndim>::runShape(shape_, 0, inds...)];
  }
  TV_HOST_DEVICE_INLINE T &operator()() {
    static_assert(Rank == -1 || 0 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(ptr_ != nullptr, "you want get value but the view is empty.%s",
               "\n");
    TV_REQUIRE(shape_.ndim() == 0, "you provide 0 indexes, but dim is %ld\n",
               shape_.ndim());
#endif
    return ptr_[0];
  }
  TV_HOST_DEVICE_INLINE const T &operator()() const {
    static_assert(Rank == -1 || 0 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(ptr_ != nullptr, "you want get value but the view is empty.%s",
               "\n");
    TV_REQUIRE(shape_.ndim() == 0, "you provide 0 indexes, but dim is %ld\n",
               shape_.ndim());
#endif
    return ptr_[0];
  }
  template <class T1> TV_HOST_DEVICE_INLINE T &operator()(T1 i1) {
    static_assert(Rank == -1 || 1 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 1, "you provide 1 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, i1, shape_[0]);
#endif
    return ptr_[i1];
  }
  template <class T1, class T2>
  TV_HOST_DEVICE_INLINE T &operator()(T1 i1, T2 i2) {
    static_assert(Rank == -1 || 2 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 2, "you provide 2 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
#endif
    return ptr_[i1 * shape_[1] + i2];
  }
  template <class T1, class T2, class T3>
  TV_HOST_DEVICE_INLINE T &operator()(T1 i1, T2 i2, T3 i3) {
    static_assert(Rank == -1 || 3 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 3, "you provide 3 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
    TV_REQUIRE(i3 >= 0 && i3 < shape_[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), shape_[2]);
#endif
    return ptr_[(i1 * shape_[1] + i2) * shape_[2] + i3];
  }
  template <class T1, class T2, class T3, class T4>
  TV_HOST_DEVICE_INLINE T &operator()(T1 i1, T2 i2, T3 i3, T4 i4) {
    static_assert(Rank == -1 || 4 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 4, "you provide 4 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
    TV_REQUIRE(i3 >= 0 && i3 < shape_[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), shape_[2]);
    TV_REQUIRE(i4 >= 0 && i4 < shape_[3],
               "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4), shape_[3]);
#endif
    return ptr_[((i1 * shape_[1] + i2) * shape_[2] + i3) * shape_[3] + i4];
  }

  template <class T1> TV_HOST_DEVICE_INLINE const T &operator()(T1 i1) const {
    static_assert(Rank == -1 || 1 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 1, "you provide 1 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
#endif
    return ptr_[i1];
  }
  template <class T1, class T2>
  TV_HOST_DEVICE_INLINE const T &operator()(T1 i1, T2 i2) const {
    static_assert(Rank == -1 || 2 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 2, "you provide 2 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
#endif
    return ptr_[i1 * shape_[1] + i2];
  }
  template <class T1, class T2, class T3>
  TV_HOST_DEVICE_INLINE const T &operator()(T1 i1, T2 i2, T3 i3) const {
    static_assert(Rank == -1 || 3 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 3, "you provide 3 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
    TV_REQUIRE(i3 >= 0 && i3 < shape_[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), shape_[2]);
#endif
    return ptr_[(i1 * shape_[1] + i2) * shape_[2] + i3];
  }
  template <class T1, class T2, class T3, class T4>
  TV_HOST_DEVICE_INLINE const T &operator()(T1 i1, T2 i2, T3 i3, T4 i4) const {
    static_assert(Rank == -1 || 4 == Rank, "error");
#if defined TV_DEBUG
    TV_REQUIRE(shape_.ndim() == 4, "you provide 4 indexes, but dim is %ld\n",
               shape_.ndim());
    TV_REQUIRE(i1 >= 0 && i1 < shape_[0],
               "index-%d(%d) out-of-range: [0, %d)\n", 0, int(i1), shape_[0]);
    TV_REQUIRE(i2 >= 0 && i2 < shape_[1],
               "index-%d(%d) out-of-range: [0, %d)\n", 1, int(i2), shape_[1]);
    TV_REQUIRE(i3 >= 0 && i3 < shape_[2],
               "index-%d(%d) out-of-range: [0, %d)\n", 2, int(i3), shape_[2]);
    TV_REQUIRE(i4 >= 0 && i4 < shape_[3],
               "index-%d(%d) out-of-range: [0, %d)\n", 3, int(i4), shape_[3]);
#endif
    return ptr_[((i1 * shape_[1] + i2) * shape_[2] + i3) * shape_[3] + i4];
  }

  TV_HOST_DEVICE_INLINE T &operator[](int idx) {
#ifdef TV_DEBUG
    TV_REQUIRE(idx >= 0 && idx < size(), "index(%d) out-of-range: [0, %ld)\n",
               int(idx), size());
#endif
    return ptr_[idx];
  }

  TV_HOST_DEVICE_INLINE const T &operator[](int idx) const {
#ifdef TV_DEBUG
    TV_REQUIRE(idx >= 0 && idx < size(), "index(%d) out-of-range: [0, %ld)\n",
               int(idx), size());
#endif
    return ptr_[idx];
  }

  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank - 1, PtrTraits, Tindex>
  accessor(Tindex idx) {
    static_assert(Rank > 1, "for Rank == 1, use accessor() or just use []");
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        ptr_ + stride_[0] * idx, stride_.data() + 1);
  }
  TV_HOST_DEVICE_INLINE TensorAccesser<T, Rank, PtrTraits, Tindex> accessor() {
    static_assert(Rank > 0, "rank must higher than zero");
    return TensorAccesser<T, Rank, PtrTraits, Tindex>(ptr_, stride_.data());
  }
  TV_HOST_DEVICE_INLINE
  TensorAccesser<T, Rank - 1, PtrTraits, Tindex> accessor(Tindex idx) const {
    static_assert(Rank > 1, "for Rank == 1, use accessor() or just use []");
    return TensorAccesser<T, Rank - 1, PtrTraits, Tindex>(
        ptr_ + stride_[0] * idx, stride_.data() + 1);
  }
  TV_HOST_DEVICE_INLINE
  TensorAccesser<T, Rank, PtrTraits, Tindex> accessor() const {
    static_assert(Rank > 0, "error");
    return TensorAccesser<T, Rank, PtrTraits, Tindex>(
        ptr_, stride_.data(), "rank must higher than zero");
  }

  TV_HOST_DEVICE_INLINE bool empty() const { return ptr_ == nullptr; }
  TV_HOST_DEVICE_INLINE ptr_t data() { return ptr_; }
  TV_HOST_DEVICE_INLINE const ptr_t data() const { return ptr_; }
  TV_HOST_DEVICE_INLINE const tv_shape_t &shape() const { return shape_; }
  TV_HOST_DEVICE_INLINE const tv_shape_t &stride() const { return stride_; }

  TV_HOST_DEVICE_INLINE int dim(int idx) const { return shape_[idx]; }
  TV_HOST_DEVICE_INLINE int ndim() const { return shape_.ndim(); }
  template <class... Inds>
  TV_HOST_DEVICE_INLINE
      TensorView<T, Rank == -1 ? -1 : sizeof...(Inds), PtrTraits, Tindex>
      view(Inds... newShapes) const {
    ShapeBase<Rank == -1 ? TV_MAX_DIM : sizeof...(Inds), Tindex> shapes{
        int(newShapes)...};
    for (size_t i = 0; i < sizeof...(newShapes); ++i) {
      if (shapes[i] == -1) {
        shapes[i] = 1;
        shapes[i] = size() / shapes.size();
        break;
      }
    }
    TV_ASSERT(shapes.size() == size());
    return TensorView < T, Rank == -1 ? -1 : sizeof...(Inds), PtrTraits,
           Tindex > (ptr_, shapes);
  }
  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex>
  view(Shape shapes) const {
    TV_ASSERT(shapes.size() == size());
    return TensorView<T, -1, PtrTraits, Tindex>(ptr_, shapes);
  }
  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex> squeeze() const {
    return TensorView<T, -1, PtrTraits, Tindex>(ptr_, shape_.squeeze());
  }
  TV_HOST_DEVICE_INLINE
  TensorView<T, Rank == -1 ? -1 : Rank - 1, PtrTraits, Tindex>
  squeeze(int dim) const {
    return TensorView < T, Rank == -1 ? -1 : Rank - 1, PtrTraits,
           Tindex > (ptr_, shape_.squeeze < Rank == -1 ? TV_MAX_DIM
                                                       : Rank - 1 > (dim));
  }
  TV_HOST_DEVICE_INLINE size_t size() const { return shape_.size(); }

  template <class... Integers>
  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex>
  subview(int id, Integers... ints) {
    tv_shape_t start = {id, ints...};
    for (int i = 1 + sizeof...(ints); i < ndim(); ++i) {
      start.push_back(0);
    }
    return TensorView<T, Rank, PtrTraits, Tindex>(
        ptr_ + rowArrayIdx(shape_, start),
        shape_.subshape(sizeof...(ints) + 1));
  }

  template <class... Integers>
  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex>
  subview(int id, Integers... ints) const {
    tv_shape_t start = {id, ints...};
    for (int i = 1 + sizeof...(ints); i < ndim(); ++i) {
      start.push_back(0);
    }
    return TensorView<T, Rank, PtrTraits, Tindex>(
        ptr_ + rowArrayIdx(shape_, start),
        shape_.subshape(sizeof...(ints) + 1));
  }

  TV_HOST_DEVICE_INLINE TensorView<T, -1, PtrTraits, Tindex>
  subview(SimpleVector<int> ids) const {
    Shape start = ids;
    for (int i = ids.size(); i < ndim(); ++i) {
      start.push_back(0);
    }
    return TensorView<T, Rank, PtrTraits, Tindex>(
        ptr_ + rowArrayIdx(shape_, start), shape_.subshape(ids.size()));
  }
  template <typename Os> std::string repr(Os &ss) const {
    if (empty())
      return "";
    if (shape_.ndim() == 0) {
      ss << "Tensor[" << type_s<T> << "]" << std::endl;
      ss << *ptr_;
      return ss.str();
    }

    SimpleVector<int64_t, TV_MAX_DIM> prev(ndim(), -1);
    SimpleVector<int64_t, TV_MAX_DIM> nd_index(ndim());
    SimpleVector<int64_t, TV_MAX_DIM> _shape;
    for (auto s : shape()) {
      _shape.push_back(s);
    }
    ss << "Tensor[" << type_s<T> << "]: shape=" << shape()
       << ", stride=" << stride() << std::endl;
    auto ndimValue = ndim();
    for (int64_t i = 0; i < int64_t(size()); ++i) {
      rowArrayIdxInv(i, nd_index.data(), _shape.data(), ndimValue);
      bool newline = false;
      int end_count = 0;
      for (int j = 0; j < ndimValue; ++j) {
        if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0 &&
            prev[j] != -1) {
          ss << "]";
          ++end_count;
          newline = true;
        }
      }
      if (prev[0] == -1) {
        end_count = ndimValue;
      }
      if (newline) {
        ss << "\n";
      }
      int starts_count = 0;
      for (int j = 0; j < ndimValue; ++j) {
        if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0) {
          ++starts_count;
        }
      }
      if (starts_count > 0) {
        for (int j = 0; j < ndimValue - end_count; ++j) {
          ss << " ";
        }
        for (int j = 0; j < starts_count; ++j) {
          ss << "[";
        }
      }
      if (std::is_same<T, uint8_t>::value ||
          std::is_same<T, const uint8_t>::value) {
        ss << unsigned((*this)[i]);
      } else {
        ss << (*this)[i];
      }
      if (nd_index[ndimValue - 1] != _shape[ndimValue - 1] - 1) {
        ss << ",";
      }
      for (int j = 0; j < ndimValue; ++j) {
        prev[j] = nd_index[j];
      }
    }
    for (int j = 0; j < ndimValue; ++j) {
      ss << "]";
    }
    return ss.str();
  }
  std::string repr() const {
    std::ostringstream ss;
    return repr(ss);
  }

protected:
  template <typename T1> TV_HOST_DEVICE_INLINE Slice to_slice(T1 s) const {
    return Slice{int(s), -1, -1};
  }

  TV_HOST_DEVICE_INLINE Slice to_slice(Slice s) const { return Slice(s); }

  ptr_t ptr_ = nullptr;
  tv_shape_t shape_;
  tv_shape_t stride_;
};

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

template <typename Os, typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
Os &operator<<(Os &os, const TensorView<T, Rank, PtrTraits, Tindex> &dt) {
  os << dt.repr();
  return os;
}

template <typename Os, typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
Os &operator<<(Os &os, const TensorView<const T, Rank, PtrTraits, Tindex> &dt) {
  os << dt.repr();
  return os;
}

namespace detail {
template <typename T> struct TypePrintfFormat;
template <> struct TypePrintfFormat<float> {
  static constexpr const char *value = "%.2f";
};
template <> struct TypePrintfFormat<double> {
  static constexpr const char *value = "%.2f";
};
template <> struct TypePrintfFormat<int8_t> {
  static constexpr const char *value = "%d";
};
template <> struct TypePrintfFormat<int16_t> {
  static constexpr const char *value = "%d";
};
template <> struct TypePrintfFormat<int32_t> {
  static constexpr const char *value = "%d";
};
template <> struct TypePrintfFormat<uint8_t> {
  static constexpr const char *value = "%u";
};
template <> struct TypePrintfFormat<uint16_t> {
  static constexpr const char *value = "%u";
};
template <> struct TypePrintfFormat<uint32_t> {
  static constexpr const char *value = "%u";
};
template <> struct TypePrintfFormat<int64_t> {
  static constexpr const char *value = "%ld";
};
template <> struct TypePrintfFormat<uint64_t> {
  static constexpr const char *value = "%lu";
};
template <> struct TypePrintfFormat<bool> {
  static constexpr const char *value = "%d";
};

template <typename T>
constexpr const char *type_printf_format_v = TypePrintfFormat<T>::value;

}; // namespace detail

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
TV_HOST_DEVICE void
printTensorView(const TensorView<T, Rank, PtrTraits, Tindex> &tensor,
                const char *format) {
  // used to print tensor in cuda kernel.
  if (tensor.empty())
    return;
  if (tensor.ndim() == 0) {
    printf(format, tensor());
    printf("\n");
    return;
  }
  SimpleVector<int64_t, TV_MAX_DIM> prev(tensor.ndim(), -1);
  SimpleVector<int64_t, TV_MAX_DIM> nd_index(tensor.ndim());
  SimpleVector<int64_t, TV_MAX_DIM> shape(tensor.shape());

  auto ndim = tensor.ndim();
  for (int64_t i = 0; i < tensor.size(); ++i) {
    rowArrayIdxInv(i, nd_index.data(), shape.data(), ndim);
    bool newline = false;
    int end_count = 0;
    for (int j = 0; j < ndim; ++j) {
      if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0 &&
          prev[j] != -1) {
        printf("]");
        ++end_count;
        newline = true;
      }
    }
    if (prev[0] == -1) {
      end_count = ndim;
    }
    if (newline) {
      printf("\n");
    }
    int starts_count = 0;
    for (int j = 0; j < ndim; ++j) {
      if (nd_index[j] != prev[j] && nd_index[j] == 0 && prev[j] != 0) {
        ++starts_count;
      }
    }
    if (starts_count > 0) {
      for (int j = 0; j < ndim - end_count; ++j) {
        printf(" ");
      }
      for (int j = 0; j < starts_count; ++j) {
        printf("]");
      }
    }
    printf(format, tensor[i]);
    if (nd_index[ndim - 1] != shape[ndim - 1] - 1) {
      printf(",");
    }
    for (int j = 0; j < ndim; ++j) {
      prev[j] = nd_index[j];
    }
  }
  for (int j = 0; j < ndim; ++j) {
    printf("]");
  }
  printf("\n");
}

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
TV_HOST_DEVICE void
printTensorView(TensorView<T, Rank, PtrTraits, Tindex> tensor) {
  using Traw = typename std::remove_const<T>::type;
  return printTensorView(tensor, detail::type_printf_format_v<Traw>);
}
template <typename T>
TV_HOST_DEVICE void printTensorView(const T *ptr, Shape shape) {
  using Traw = typename std::remove_const<T>::type;
  return printTensorView(TensorView<const T>(ptr, shape),
                         detail::type_printf_format_v<Traw>);
}
template <typename T>
TV_HOST_DEVICE void printTensorView(const T *ptr, Shape shape,
                                    const char *format) {
  return printTensorView(TensorView<const T>(ptr, shape), format);
}

#ifdef TV_CUDA

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) tv::check((val), #val, __FILE__, __LINE__)

template <typename T>
void host2dev(T *dst, const T *src, size_t size, cudaStream_t s = 0) {
  checkCudaErrors(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyHostToDevice, s));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s = 0) {
  host2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s = 0) {
  host2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T> void host2dev_sync(T *dst, const T *src, size_t size) {
  checkCudaErrors(
      cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev_sync(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
                   const TensorView<const T, Rank, PtrTraits2, Tindex2> src) {
  host2dev_sync(dst.data(), src.data(), std::min(dst.size(), src.size()));
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2dev_sync(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
                   const TensorView<T, Rank, PtrTraits2, Tindex2> src) {
  host2dev_sync(dst.data(), src.data(), std::min(dst.size(), src.size()));
}

template <typename T>
void dev2host(T *dst, const T *src, size_t size, cudaStream_t s = 0) {
  checkCudaErrors(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost, s));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s = 0) {
  dev2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
              const TensorView<T, Rank, PtrTraits2, Tindex2> src,
              cudaStream_t s = 0) {
  dev2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T>
void dev2dev(T *dst, const T *src, size_t size, cudaStream_t s = 0) {
  checkCudaErrors(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice, s));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
             const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
             cudaStream_t s = 0) {
  dev2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void dev2dev(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
             const TensorView<T, Rank, PtrTraits2, Tindex2> src,
             cudaStream_t s = 0) {
  dev2dev(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T>
void host2host(T *dst, const T *src, size_t size, cudaStream_t s = 0) {
  checkCudaErrors(
      cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyHostToHost, s));
}

template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
               const TensorView<const T, Rank, PtrTraits2, Tindex2> src,
               cudaStream_t s = 0) {
  host2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}
template <typename T, int Rank, template <class> class PtrTraits1,
          template <class> class PtrTraits2, typename Tindex1, typename Tindex2>
void host2host(TensorView<T, Rank, PtrTraits1, Tindex1> dst,
               const TensorView<T, Rank, PtrTraits2, Tindex2> src,
               cudaStream_t s = 0) {
  host2host(dst.data(), src.data(), std::min(dst.size(), src.size()), s);
}

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
void zero_dev(TensorView<T, Rank, PtrTraits, Tindex> tensor) {
  checkCudaErrors(cudaMemset(tensor.data(), 0, tensor.size() * sizeof(T)));
}

template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
void zero_dev(TensorView<T, Rank, PtrTraits, Tindex> tensor, cudaStream_t s) {
  checkCudaErrors(
      cudaMemsetAsync(tensor.data(), 0, tensor.size() * sizeof(T), s));
}
template <typename T, int Rank, template <class> class PtrTraits,
          typename Tindex>
void zero_host(TensorView<T, Rank, PtrTraits, Tindex> tensor) {
  std::fill(tensor.data(), tensor.data() + tensor.size(), 0);
}

#endif

} // namespace tv