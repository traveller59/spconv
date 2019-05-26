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

#include <spconv/geometry.h>
#include <spconv/indice.h>
#include <spconv/spconv_ops.h>
#include <torch/script.h>
#include <ATen/Parallel.h>

namespace spconv {

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsConv(tv::TensorView<const Index> indicesIn,
                         tv::TensorView<Index> indicesOut,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *kernelSize, const Index *stride,
                         const Index *padding, const Index *dilation,
                         const Index *outSpatialShape) {
  // indicesOut: num_active * kernelVolume * (NDim + 1)
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
  Index* validPoints = validPoints_.data();
  Index *pointPtr = nullptr;
  Index hashval;
  tsl::robin_map<Index, Index> hash;
  for (int j = 0; j < numActIn; ++j) {
    batchIdx = indicesIn(j, 0);
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
        dilation, outSpatialShape, validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                   spatialVolume * batchIdx;
      auto iter = hash.find(index);
      if (iter == hash.end()) {
        for (unsigned k = 1; k < NDim + 1; ++k) {
          indicesOut(numAct, k) = pointPtr[k - 1];
        }
        indicesOut(numAct, 0) = batchIdx;
        hashval = numAct++;
        hash[index] = hashval;
      }else{
        hashval = iter->second;
      }
      // indicePairs: [K, 2, L]
      indicePairs(offset, 0, indiceNum[offset]) = j;
      indicePairs(offset, 1, indiceNum[offset]++) = hashval;
    }
  }
  return numAct;
}

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsDeConv(tv::TensorView<const Index> indicesIn,
                           tv::TensorView<Index> indicesOut,
                           tv::TensorView<IndexGrid> gridsOut,
                           tv::TensorView<Index> indicePairs,
                           tv::TensorView<Index> indiceNum,
                           const Index *kernelSize, const Index *stride,
                           const Index *padding, const Index *dilation,
                           const Index *outSpatialShape) {
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
  Index* validPoints = validPoints_.data();
  Index *pointPtr = nullptr;
  Index hashval;
  tsl::robin_map<Index, Index> hash;
  for (int j = 0; j < numActIn; ++j) {
    batchIdx = indicesIn(j, 0);
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
        dilation, outSpatialShape, validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                   spatialVolume * batchIdx;

      auto iter = hash.find(index);
      if (iter == hash.end()) {
        for (unsigned k = 1; k < NDim + 1; ++k) {
          indicesOut(numAct, k) = pointPtr[k - 1];
        }
        indicesOut(numAct, 0) = batchIdx;
        hashval = numAct++;
        hash[index] = hashval;
      }else{
        hashval = iter->second;
      }
      // indicePairs: [K, 2, L]
      indicePairs(offset, 0, indiceNum[offset]) = j;
      indicePairs(offset, 1, indiceNum[offset]++) = hashval;
    }
  }
  return numAct;
}


#ifndef TV_WINDOWS
template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsSubM(tv::TensorView<const Index> indicesIn,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *const kernelSize,
                         const Index *const stride, const Index *const padding,
                         const Index *dilation, const Index *const outSpatialShape) {
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  tsl::robin_map<Index, Index> hash;
  for (int j = 0; j < numActIn; ++j) {
    Index index = 0;
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + j * (NDim + 1) + 1,
                                         outSpatialShape) +
            spatialVolume * indicesIn(j, 0);
    hash[index] = j;
  }
  
  at::parallel_for(0, numActIn, 0, [&](int64_t begin, int64_t end){
    Index index = 0;
    Index numValidPoints = 0;
    std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
    Index* validPoints = validPoints_.data();
    Index *pointPtr = nullptr;
    Index oldOffset = 0;
    for (int j = begin; j < end; ++j) {
      numValidPoints = getValidOutPos<Index, NDim>(
          indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
          dilation, outSpatialShape, validPoints);
      for (Index i = 0; i < numValidPoints; ++i) {
        pointPtr = validPoints + i * (NDim + 1);
        auto offset = pointPtr[NDim];
        index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                spatialVolume * indicesIn(j, 0);
        auto iter = hash.find(index);
        if (iter != hash.end()) {
          #pragma omp atomic capture
          oldOffset = indiceNum[offset]++;
          indicePairs(offset, 0, oldOffset) = j;
          indicePairs(offset, 1, oldOffset) = iter->second;
        }
      }
    }
  });
  return numActIn;
}
#else 
template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsSubM(tv::TensorView<const Index> indicesIn,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *const kernelSize,
                         const Index *const stride, const Index *const padding,
                         const Index *dilation, const Index *const outSpatialShape) {
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  // Index validPoints[kernelVolume * (NDim + 1)];
  std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
  Index* validPoints = validPoints_.data();
  Index *pointPtr = nullptr;
  tsl::robin_map<Index, Index> hash;
  for (int j = 0; j < numActIn; ++j) {
    Index index = 0;
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + j * (NDim + 1) + 1,
                                         outSpatialShape) +
            spatialVolume * indicesIn(j, 0);
    hash[index] = j;
  }
  Index index = 0;
  for (int j = 0; j < numActIn; ++j) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
        dilation, outSpatialShape, validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
              spatialVolume * indicesIn(j, 0);
      auto iter = hash.find(index);
      if (iter != hash.end()) {
        indicePairs(offset, 0, indiceNum[offset]) = j;
        indicePairs(offset, 1, indiceNum[offset]++) = iter->second;
      }
    }
  }
  return numActIn;
}
#endif

namespace functor {
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctor<tv::CPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn,
                     tv::TensorView<Index> indicesOut,
                     tv::TensorView<IndexGrid> gridsOut,
                     tv::TensorView<Index> indicePairs,
                     tv::TensorView<Index> indiceNum,
                     const tv::SimpleVector<Index, NDim> kernelSize,
                     const tv::SimpleVector<Index, NDim> stride,
                     const tv::SimpleVector<Index, NDim> padding,
                     const tv::SimpleVector<Index, NDim> dilation,
                     const tv::SimpleVector<Index, NDim> outSpatialShape,
                     bool transpose, bool resetGrid, bool useHash) {
    if (transpose)
      return getIndicePairsDeConv<Index, IndexGrid, NDim>(
          indicesIn, indicesOut,
          gridsOut, indicePairs, indiceNum,
          kernelSize.data(), stride.data(), padding.data(), dilation.data(),
          outSpatialShape.data());
    else
      return getIndicePairsConv<Index, IndexGrid, NDim>(
          indicesIn, indicesOut,
          gridsOut, indicePairs, indiceNum,
          kernelSize.data(), stride.data(), padding.data(), dilation.data(),
          outSpatialShape.data());

  }
};
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor<tv::CPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn,
                     tv::TensorView<IndexGrid> gridsOut,
                     tv::TensorView<Index> indicePairs,
                     tv::TensorView<Index> indiceNum,
                     const tv::SimpleVector<Index, NDim> kernelSize,
                     const tv::SimpleVector<Index, NDim> stride,
                     const tv::SimpleVector<Index, NDim> padding,
                     const tv::SimpleVector<Index, NDim> dilation,
                     const tv::SimpleVector<Index, NDim> outSpatialShape,
                     bool transpose, bool resetGrid, bool useHash) {
    return getIndicePairsSubM<Index, IndexGrid, NDim>(
        indicesIn,
        gridsOut, indicePairs, indiceNum,
        kernelSize.data(), stride.data(), padding.data(), dilation.data(), outSpatialShape.data());
  }
};
} // namespace functor

#define DECLARE_CPU_SPECS_INDEX_NDIM(Index, NDIM)                              \
  template struct functor::CreateConvIndicePairFunctor<tv::CPU, Index, int, NDIM>;      \
  template struct functor::CreateSubMIndicePairFunctor<tv::CPU, Index, int,  \
                                                         NDIM>;


#define DECLARE_CPU_INDEX(Index)                                               \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 1);                                      \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 2);                                      \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 3);                                      \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 4);

DECLARE_CPU_INDEX(int);
DECLARE_CPU_INDEX(long);

#undef DECLARE_CPU_INDEX
#undef DECLARE_CPU_SPECS_INDEX_NDIM

} // namespace spconv

