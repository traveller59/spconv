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

#ifndef REORDERING_CU_H_
#define REORDERING_CU_H_
#include <THC/THCAtomics.cuh>
#include <tensorview/kernel_utils.h>
// see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
namespace spconv {

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void gatherGenericKernel(T *buffer, const T *features,
                                    const Index *indices, int size,
                                    int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              features[inds[ilp] + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void gatherVecKernel(T *buffer, const T *features,
                                const Index *indices, int size, int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          reinterpret_cast<VecType *>(
              buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              reinterpret_cast<const VecType *>(features)[inds[ilp] + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void gatherVecBlockKernel(T *buffer, const T *features,
                                     const Index *indices, int size,
                                     int numPlanes) {
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  features += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;

  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      reinterpret_cast<VecType *>(
          buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x] =
          reinterpret_cast<const VecType *>(
              features)[indices[iy + ILPStrideY[ilp]] * numPlanes +
                        threadIdx.x];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void batchGatherGenericKernel(T *buffer, const T *features,
                                         const Index *indices, int size,
                                         int numPlanes, int indice_batch_stride,
                                         int feature_batch_stride) {
  // size: max indice num * kernel volume
  // inds: [volume, num_elems]
  int ILPStrideX[NumILP];
  Index inds[NumILP];
  Index inds_elem;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size) {
        inds_elem = ix + ILPStrideX[ilp];
        inds[ilp] =
            indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                    inds_elem % feature_batch_stride];
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size && inds[ilp] != -1)
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              features[inds[ilp] * numPlanes + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void batchGatherVecKernel(T *buffer, const T *features,
                                     const Index *indices, int size,
                                     int feature_offset,
                                     int numPlanes, int indice_batch_stride,
                                     int feature_batch_stride) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
  Index inds_elem;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size) {
        inds_elem = ix + ILPStrideX[ilp] + feature_offset;
        inds[ilp] =
            indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                    inds_elem % feature_batch_stride];
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size && inds[ilp] != -1)
          reinterpret_cast<VecType *>(
              buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              reinterpret_cast<const VecType *>(
                  features)[inds[ilp] * numPlanes + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void
batchGatherVecBlockKernel(T *buffer, const T *features, const Index *indices,
                          int size, int numPlanes, int indice_batch_stride,
                          int feature_batch_stride) {
  int ILPStrideY[NumILP];
  Index inds;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  features += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;

  Index inds_elem;

  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {

#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      inds_elem = iy + ILPStrideY[ilp];
      inds = indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                     inds_elem % feature_batch_stride];

      if (inds != -1) {
        reinterpret_cast<VecType *>(
            buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x] =
            reinterpret_cast<const VecType *>(
                features)[inds * numPlanes + threadIdx.x];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void scatterAddGenericKernel(T *outFeatures, const T *buffer,
                                        const Index *indices, int size,
                                        int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size) {
          outFeatures[inds[ilp] + iy] +=
              buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void scatterAddVecBlockKernel(T *outFeatures, const T *buffer,
                                         const Index *indices, int size,
                                         int numPlanes) {
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  outFeatures += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;
  T buf[vecloadFactor];
  T buf2[vecloadFactor];
  Index idx;
  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idx = indices[iy + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(buf)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idx];
      reinterpret_cast<VecType *>(buf2)[0] = reinterpret_cast<const VecType *>(
          buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        buf[i] += buf2[i];
      }
      reinterpret_cast<VecType *>(outFeatures)[idx] =
          reinterpret_cast<VecType *>(buf)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void batchScatterAddGenericKernel(T *outFeatures, const T *buffer,
                                             const Index *indices, int size,
                                             int feature_offset, int numPlanes,
                                             int indice_batch_stride,
                                             int feature_batch_stride) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
  Index inds_elem;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size) {
        inds_elem = ix + ILPStrideX[ilp] + feature_offset;
        inds[ilp] =
            indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                    inds_elem % feature_batch_stride];
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size && inds[ilp] != -1) {
          gpuAtomicAdd(outFeatures + inds[ilp] * numPlanes + iy,
                       buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy]);
          // outFeatures[inds[ilp] * numPlanes + iy] +=
          //     buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
batchScatterAddBlockKernel(T *outFeatures, const T *buffer,
                           const Index *indices, int size, int numPlanes,
                           int indice_batch_stride, int feature_batch_stride) {
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  outFeatures += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;
  Index inds, inds_elem;
  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      inds_elem = iy + ILPStrideY[ilp];
      inds = indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                     inds_elem % feature_batch_stride];
      if (inds != -1) {
        gpuAtomicAdd(outFeatures + inds * numPlanes + threadIdx.x,
                     buffer[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x]);
      }
    }
  }
}

} // namespace spconv

#endif