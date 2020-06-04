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
#include <THC/THCNumerics.cuh>
#include <cuda_fp16.h>
#include <tensorview/kernel_utils.h>

#if PYTORCH_VERSION < 10500
#define TH_ATOMIC_ADD atomicAdd
#else
#define TH_ATOMIC_ADD gpuAtomicAdd
#endif

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
  int ILPStrideX[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  features += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      reinterpret_cast<VecType *>(
          buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y] =
          reinterpret_cast<const VecType *>(
              features)[indices[ix + ILPStrideX[ilp]] * numPlanes +
                        threadIdx.y];
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
        if (ix + ILPStrideX[ilp] < size) {
          if (inds[ilp] != -1) {
            buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
                features[inds[ilp] * numPlanes + iy];

          } else {
            buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] = T(0);
          }
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void
batchGatherVecKernel(T *buffer, const T *features, const Index *indices,
                     int size, int feature_offset, int numPlanes,
                     int indice_batch_stride, int feature_batch_stride) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
  Index zero[sizeof(VecType) / sizeof(T)];
#pragma unroll
  for (int i = 0; i < sizeof(VecType) / sizeof(T); ++i) {
    zero[i] = T(0);
  }

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
        if (ix + ILPStrideX[ilp] < size) {
          if (inds[ilp] != -1) {
            reinterpret_cast<VecType *>(
                buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
                reinterpret_cast<const VecType *>(
                    features)[inds[ilp] * numPlanes + iy];

          } else {
            reinterpret_cast<VecType *>(
                buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
                reinterpret_cast<const VecType *>(&zero)[0];
          }
        }
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
  int ILPStrideX[NumILP];
  Index inds;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  features += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;

  Index inds_elem;
  Index zero[sizeof(VecType) / sizeof(T)];
#pragma unroll
  for (int i = 0; i < sizeof(VecType) / sizeof(T); ++i) {
    zero[i] = T(0);
  }

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {

#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      inds_elem = ix + ILPStrideX[ilp];
      inds = indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                     inds_elem % feature_batch_stride];

      if (inds != -1) {
        reinterpret_cast<VecType *>(
            buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y] =
            reinterpret_cast<const VecType *>(
                features)[inds * numPlanes + threadIdx.y];
      } else {
        reinterpret_cast<VecType *>(
            buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y] =
            reinterpret_cast<const VecType *>(&zero)[0];
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
  int ILPStrideX[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  constexpr int vecloadHalf2Factor = sizeof(VecType) / sizeof(__half2);

#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  outFeatures += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;
  T buf[vecloadFactor];
  T buf2[vecloadFactor];
  Index idx;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idx = indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y;
      reinterpret_cast<VecType *>(buf)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idx];
      reinterpret_cast<VecType *>(buf2)[0] = reinterpret_cast<const VecType *>(
          buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y];
      if (std::is_same<T, at::Half>::value) {
#if __CUDA_ARCH__ >= 530
#pragma unroll
        for (int i = 0; i < vecloadHalf2Factor; i++) {
          reinterpret_cast<__half2 *>(buf)[i] =
              __hadd2(reinterpret_cast<__half2 *>(buf)[i],
                      reinterpret_cast<__half2 *>(buf2)[i]);
        }
#else
#pragma unroll
        for (int i = 0; i < vecloadFactor; i++) {
          buf[i] += buf2[i];
        }
#endif
      } else {
#pragma unroll
        for (int i = 0; i < vecloadFactor; i++) {
          buf[i] += buf2[i];
        }
      }
      reinterpret_cast<VecType *>(outFeatures)[idx] =
          reinterpret_cast<VecType *>(buf)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void scatterAddBlockKernel(T *outFeatures, const T *buffer,
                                      const Index *indices, int size,
                                      int numPlanes) {
  int ILPStrideX[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  outFeatures += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      outFeatures[indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y] +=
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y];
    }
  }
}

#if __CUDA_ARCH__ >= 530
template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void scatterAddHalfBlockKernel(T *outFeatures, const T *buffer,
                                          const Index *indices, int size,
                                          int numPlanes) {
  int ILPStrideX[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  outFeatures += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;
  Index idx;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idx = indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y;
      reinterpret_cast<__half2 *>(outFeatures)[idx] = __hadd2(
          reinterpret_cast<__half2 *>(outFeatures)[idx],
          reinterpret_cast<__half2 *>(
              buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y]);
    }
  }
}
#endif

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void batchScatterAddGenericKernel(T *outFeatures, const T *buffer,
                                             const Index *indices, int size,
                                             int feature_offset, int numPlanes,
                                             int indice_batch_stride,
                                             int feature_batch_stride) {
  // batch scatter add is greatly slower than native scatter when the number of
  // points is large. this may due to atomicAdd?
  // batch scatter add is greatly faster than native when the number of points
  // is small.
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
          TH_ATOMIC_ADD(outFeatures + inds[ilp] * numPlanes + iy,
                        buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy]);
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
  int ILPStrideX[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  outFeatures += blockIdx.y * NumTLP;
  buffer += blockIdx.y * NumTLP;
  Index inds, inds_elem;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      inds_elem = ix + ILPStrideX[ilp];
      inds = indices[(inds_elem / feature_batch_stride) * indice_batch_stride +
                     inds_elem % feature_batch_stride];
      if (inds != -1) {
        TH_ATOMIC_ADD(outFeatures + inds * numPlanes + threadIdx.y,
                      buffer[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y]);
      }
    }
  }
}

} // namespace spconv

#undef TH_ATOMIC_ADD

#endif