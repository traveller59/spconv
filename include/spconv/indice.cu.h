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

#ifndef INDICE_CU_H_
#define INDICE_CU_H_
#include <cuhash/hash_table.cuh>
#include <spconv/geometry.h>
#include <tensorview/kernel_utils.h>
#include <tensorview/tensorview.h>

namespace spconv {
template <typename Index, unsigned NDim, int KernelMaxVolume = 256,
          typename Index1D = int>
__global__ void prepareIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index1D> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
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
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      Index oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(0, offset, oldNum) = ix;
      index = tv::ArrayIndexRowMajor<NDim, NDim>::runPtrs(
                  pointPtr, outSpatialShape.data(), 0) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(1, offset, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, unsigned NDim, int KernelMaxVolume = 256>
__global__ void prepareDeConvIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
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
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      Index oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(0, offset, oldNum) = ix;
      index = tv::ArrayIndexRowMajor<NDim, NDim>::runPtrs(
                  pointPtr, outSpatialShape.data(), 0) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(1, offset, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void assignGridAndIndiceOutKernel(
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numAct, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int batchSize) {

  Index index;
  auto indicesOutPtr = indicesOut.data();
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    index = indicePairUnique[ix];
    gridsOut[index] = ix;
    index = tv::rowArrayIdxInv<Index, NDim>(
        index, indicesOutPtr + ix * (NDim + 1) + 1, outSpatialShape.data());
    indicesOut[ix * (NDim + 1)] = index % batchSize;
  }
}

template <typename Index, unsigned NDim, unsigned kNumHashFunctions = 4>
__global__ void
assignIndiceOutKernel(tv::TensorView<Index> indicesOut, int numAct,
                      tv::TensorView<Index> indicePairUnique,
                      const tv::SimpleVector<Index, NDim> outSpatialShape,
                      int batchSize) {

  Index index;
  auto indicesOutPtr = indicesOut.data();
  for (unsigned ix : tv::KernelLoopX<unsigned>(numAct)) {
    index = indicePairUnique[ix];
    index = tv::rowArrayIdxInv<Index, NDim>(
        index, indicesOutPtr + ix * (NDim + 1) + 1, outSpatialShape.data());
    indicesOut[ix * (NDim + 1)] = index % batchSize;
  }
}

template <typename Index, unsigned NDim, unsigned kNumHashFunctions = 4>
__global__ void
assignIndicePairsHashKernel(tv::TensorView<Index> indicesOut, int numActIn,
                            tv::TensorView<Index> indicePairs,
                            tv::TensorView<Index> indicePairUnique,
                            unsigned table_size, const cuhash::Entry *table,
                            cuhash::Functions<kNumHashFunctions> constants,
                            uint2 stash_constants, unsigned stash_count) {

  Index index;
  int kernelVolume = indicePairs.dim(1);
  auto indicePairsOut = indicePairs.subview(1);
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    for (int i = 0; i < kernelVolume; ++i) {
      index = indicePairsOut(i, ix);
      if (index > -1) {
        auto val = cuhash::retrieve((unsigned)(index), table_size, table,
                                    constants, stash_constants, stash_count);
        assert(val != cuhash::kNotFound);
        indicePairsOut(i, ix) = (unsigned)val;
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void
assignIndicePairsKernel(tv::TensorView<Index> indicesOut,
                        tv::TensorView<IndexGrid> gridsOut, int numActIn,
                        tv::TensorView<Index> indicePairs,
                        tv::TensorView<Index> indicePairUnique,
                        const tv::SimpleVector<Index, NDim> outSpatialShape) {

  Index index;
  int kernelVolume = indicePairs.dim(1);
  auto indicePairsOut = indicePairs.subview(1);

  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    for (int i = 0; i < kernelVolume; ++i) {
      index = indicePairsOut(i, ix);
      if (index > -1) {
        indicePairsOut(i, ix) = gridsOut[index];
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void prepareSubMGridKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape, Index spatialVolume) {
  auto numActIn = indicesIn.dim(0);
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    index =
        tv::ArrayIndexRowMajor<NDim, NDim>::runPtrs(
            indicesIn.data() + ix * (NDim + 1) + 1, outSpatialShape.data(), 0) +
        spatialVolume * indicesIn(ix, 0);
    gridsOut[index] = ix;
  }
}

template <typename Index, unsigned NDim>
__global__ void
prepareSubMHashKernel(tv::TensorView<const Index> indicesIn, unsigned *keys,
                      unsigned *values,
                      const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + ix * (NDim + 1) + 1,
                                         outSpatialShape.data()) +
            spatialVolume * indicesIn(ix, 0);
    keys[ix] = index;
    values[ix] = ix;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void getSubMIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (int i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::ArrayIndexRowMajor<NDim, NDim>::runPtrs(
                  pointPtr, outSpatialShape.data(), 0) +
              spatialVolume * indicesIn(ix, 0);
      if (gridsOut[index] > -1) {
        Index oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
        indicePairs(1, offset, oldNum) = gridsOut[index];
        indicePairs(0, offset, oldNum) = ix;
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned K0, unsigned K1,
          unsigned K2>
__global__ void getSubMIndicePairsKernel3(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, 3> outSpatialShape, Index spatialVolume) {
  auto numActIn = indicesIn.dim(0);

  Index point[3];
  Index index = 0;
  Index offset;
  constexpr unsigned KV = K0 * K1 * K2;
  constexpr unsigned center = KV / 2;
  *(indiceNum.data() + center) = numActIn;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    const Index *indice_data = indicesIn.data() + ix * (3 + 1);
#pragma unroll
    for (int i = 0; i < K0; ++i) {
#pragma unroll
      for (int j = 0; j < K1; ++j) {
#pragma unroll
        for (int k = 0; k < K2; ++k) {
          offset = i * K1 * K2 + j * K2 + k;
          if (offset > center){
            continue;
          }
          if (center == offset){
              // center of subm indice pairs dont need atomicadd
              indicePairs(1, offset, ix) = ix;
              indicePairs(0, offset, ix) = ix;
          }else{
            point[2] = indice_data[3] - k + K2 / 2;
            point[1] = indice_data[2] - j + K1 / 2;
            point[0] = indice_data[1] - i + K0 / 2;
            if (point[1] >= 0 && point[1] < outSpatialShape[1] && point[2] >= 0 &&
                point[2] < outSpatialShape[2] && point[0] >= 0 &&
                point[0] < outSpatialShape[0]) {
              index = tv::ArrayIndexRowMajor<3, 3>::runPtrs(
                          point, outSpatialShape.data(), 0) +
                      spatialVolume * indice_data[0];
              if (gridsOut[index] != -1) {
                // for subm: indicePairs[0, i] = indicePairs[1, kernelVolume - i - 1]
                Index oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
                atomicAdd(indiceNum.data() + KV - offset - 1, Index(1));
                indicePairs(1, offset, oldNum) = gridsOut[index];
                indicePairs(0, offset, oldNum) = ix;
                indicePairs(1, KV - offset - 1, oldNum) = ix;
                indicePairs(0, KV - offset - 1, oldNum) = gridsOut[index];
              }
            }
          }
        }
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned K0, unsigned K1>
__global__ void getSubMIndicePairsKernel2(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, 2> outSpatialShape, Index spatialVolume) {
  auto numActIn = indicesIn.dim(0);
  Index point[2];
  Index index = 0;
  Index offset;
  constexpr unsigned KV = K0 * K1;
  constexpr unsigned center = KV / 2;
  *(indiceNum.data() + center) = numActIn;

  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    const Index *indice_data = indicesIn.data() + ix * (2 + 1);
#pragma unroll
    for (int i = 0; i < K0; ++i) {
#pragma unroll
      for (int j = 0; j < K1; ++j) {
        offset = i * K1 + j;
        if (offset > center){
          continue;
        }
        if (center == offset){
            // center of subm indice pairs dont need atomicadd
            indicePairs(1, offset, ix) = ix;
            indicePairs(0, offset, ix) = ix;
        }else{
          point[1] = indice_data[2] - j + K1 / 2;
          point[0] = indice_data[1] - i + K0 / 2;
          if (point[1] >= 0 && point[1] < outSpatialShape[1] && point[0] >= 0 &&
              point[0] < outSpatialShape[0]) {
            index = tv::ArrayIndexRowMajor<2, 2>::runPtrs(
                        point, outSpatialShape.data(), 0) +
                    spatialVolume * indice_data[0];
            if (gridsOut[index] > -1) {
              Index oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
              atomicAdd(indiceNum.data() + KV - offset - 1, Index(1));
              indicePairs(1, offset, oldNum) = gridsOut[index];
              indicePairs(0, offset, oldNum) = ix;
              indicePairs(1, KV - offset - 1, oldNum) = ix;
              indicePairs(0, KV - offset - 1, oldNum) = gridsOut[index];
            }
          }
        }
      }
    }
  }
}

template <typename Index, unsigned NDim, int KernelMaxVolume = 256,
          unsigned kNumHashFunctions = 4>
__global__ void getSubMIndicePairsHashKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape, unsigned table_size,
    const cuhash::Entry *table, cuhash::Functions<kNumHashFunctions> constants,
    uint2 stash_constants, unsigned stash_count) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (int i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::ArrayIndexRowMajor<NDim, NDim>::runPtrs(
                  pointPtr, outSpatialShape.data(), 0) +
              spatialVolume * indicesIn(ix, 0);
      auto val = cuhash::retrieve((unsigned)(index), table_size, table,
                                  constants, stash_constants, stash_count);
      if (val != cuhash::kNotFound) {
        Index oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
        indicePairs(1, offset, oldNum) = val;
        indicePairs(0, offset, oldNum) = ix;
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void resetGridKernel(const Index *indicePairUnique,
                                tv::TensorView<IndexGrid> gridsOut,
                                int numAct) {
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    gridsOut[indicePairUnique[ix]] = -1;
  }
}

template <typename T> __global__ void arangeKernel(T *data, int size) {
  for (int ix : tv::KernelLoopX<int>(size)) {
    data[ix] = ix;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void
resetGridSubMKernel(const Index *indices, tv::TensorView<IndexGrid> gridsOut,
                    const tv::SimpleVector<Index, NDim> outSpatialShape,
                    int numAct) {
  Index outSpatialShapeReg[NDim];
  for (int i = 0; i < NDim; ++i) {
    outSpatialShapeReg[i] = outSpatialShape[i];
  }
  Index spatialVolume = 1;
  auto indsPtr = indices;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index;
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    indsPtr = indices + ix * (NDim + 1);
    index = tv::ArrayIndexRowMajor<NDim, NDim>::runPtrs(indsPtr + 1,
                                                        outSpatialShapeReg, 0);
    gridsOut[index + spatialVolume * indsPtr[0]] = -1;
  }
}

} // namespace spconv

#undef atomicAdd

#endif