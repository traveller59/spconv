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

#ifndef SPCONV_GEOMETRY_H_
#define SPCONV_GEOMETRY_H_

#include <iostream>
#include <limits>
#include <tensorview/tensorview.h>
#include <tsl/robin_map.h>
#include <unordered_map>
namespace spconv {

namespace detail {

template <typename T> struct ToUnsigned;

template <> struct ToUnsigned<int> { using type = uint32_t; };

template <> struct ToUnsigned<long> { using type = uint64_t; };

template <typename T> struct FNVInternal;
template <> struct FNVInternal<uint32_t> {
  constexpr static uint32_t defaultOffsetBasis = 0x811C9DC5;
  constexpr static uint32_t prime = 0x01000193;
};

template <> struct FNVInternal<uint64_t> {
  constexpr static uint64_t defaultOffsetBasis = 0xcbf29ce484222325;
  constexpr static uint64_t prime = 0x100000001b3;
};

} // namespace detail
template <typename T>
using to_unsigned_t = typename detail::ToUnsigned<std::remove_const_t<T>>::type;

template <typename T> struct FNV1a : detail::FNVInternal<T> {
  std::size_t operator()(const T *data, std::size_t size) {
    to_unsigned_t<T> hash = detail::FNVInternal<T>::defaultOffsetBasis;
    for (std::size_t i = 0; i < size; ++i) {
      hash *= detail::FNVInternal<T>::prime;
      hash ^= static_cast<to_unsigned_t<T>>(data[i]);
    }
    return hash;
  }
};

template <typename Index, unsigned NDim>
TV_HOST_DEVICE Index getValidOutPos(const Index *input_pos,
                                    const Index *kernelSize,
                                    const Index *stride, const Index *padding,
                                    const Index *dilation,
                                    const Index *outSpatialShape, Index *out) {
  Index lowers[NDim];
  Index uppers[NDim];
  Index counter[NDim];
  Index counterSize[NDim];
  Index pointCounter = 0;
  Index val;
  Index numPoints = 1;
  Index m, offset;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    lowers[i] = (input_pos[i] - (kernelSize[i] - 1) * dilation[i] - 1 +
                 stride[i] + padding[i]) /
                stride[i];
    uppers[i] = (input_pos[i] + padding[i]) / stride[i];
  }

#pragma unroll
  for (unsigned i = 0; i < NDim; ++i) {
    counterSize[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
    numPoints *= counterSize[i];
  }

#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
  }
  for (int i = 0; i < numPoints; ++i) {
    valid = true;
    m = 1;
    offset = 0;
#pragma unroll
    for (int j = NDim - 1; j >= 0; --j) {
      val = uppers[j] - counter[j] * dilation[j];
      out[pointCounter * (NDim + 1) + j] = val;
      if (val < 0 || (val > outSpatialShape[j] - 1)) {
        valid = false;
        // break;
      }
      offset += m * (input_pos[j] - val * stride[j] + padding[j]) / dilation[j];
      m *= kernelSize[j];
    }

    out[pointCounter * (NDim + 1) + NDim] = offset;
    if (valid)
      ++pointCounter;
    counter[NDim - 1] += 1;
#pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == counterSize[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }
  return pointCounter;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE Index getValidOutPosTranspose(
    const Index *input_pos, const Index *kernelSize, const Index *stride,
    const Index *padding, const Index *dilation, const Index *outSpatialShape,
    Index *out) {
  Index lowers[NDim];
  Index uppers[NDim];
  Index counter[NDim];
  Index counterSize[NDim];
  Index pointCounter = 0;
  Index val;
  Index numPoints = 1;
  Index m, offset;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    lowers[i] = input_pos[i] * stride[i] - padding[i];
    uppers[i] = lowers[i] + (kernelSize[i] - 1) * dilation[i];
  }
#pragma unroll
  for (unsigned i = 0; i < NDim; ++i) {
    counterSize[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
    numPoints *= counterSize[i];
  }
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
  }
  for (int i = 0; i < numPoints; ++i) {
    valid = true;
    m = 1;
    offset = 0;
#pragma unroll
    for (int j = NDim - 1; j >= 0; --j) {
      val = uppers[j] - counter[j] * dilation[j];
      out[pointCounter * (NDim + 1) + j] = val;
      if (val < 0 || (val > outSpatialShape[j] - 1)) {
        valid = false;
        // break;
      }
      offset += m * (val - lowers[j]) / dilation[j];
      m *= kernelSize[j];
    }
    out[pointCounter * (NDim + 1) + NDim] = offset;
    if (valid)
      ++pointCounter;
    counter[NDim - 1] += 1;
#pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == counterSize[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }
  return pointCounter;
}

} // namespace spconv

#endif