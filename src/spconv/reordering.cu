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

#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <spconv/reordering.cu.h>
#include <spconv/reordering.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/kernel_utils.h>
#include <tensorview/mp_helper.h>
#include <tensorview/tensor.h>
#include <tensorview/tensorview.h>
#include <tensorview/torch_utils.h>
#include <type_traits>
#include <utility/timer.h>
namespace spconv {

using float_types_t = tv::mp_list<float, double, at::Half>;
using int_types_t = tv::mp_list<int32_t, int64_t>;


template <typename T>
struct half_vec{
  using type = typename std::conditional_t<std::is_same<T, at::Half>::value, int4, int4>;
};

template <typename T>
struct half_vec_sadd{
  using type = typename std::conditional_t<std::is_same<T, at::Half>::value, int4, int4>;
};
using kernel_block_t = tv::mp_list_c<int, 64, 32, 16>;

void sparse_gather_cuda(torch::Tensor buffer, torch::Tensor features,
                        torch::Tensor indices, int size) {
  if (size <= 0)
    return;
  int numPlanes = features.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dtype = features.scalar_type();
  auto inds_dtype = indices.scalar_type();
  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    using vecload_type_t = typename half_vec_sadd<T>::type;
    tv::DispatchTorch<int_types_t>()(inds_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      bool notFound = true;
      constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);

      tv::mp_for_each<kernel_block_t>([=, &buffer, &features, &indices,
                                       &notFound](auto NumTLP) {
        constexpr int NumILP = NumTLP / 4;
        // constexpr int NumILP = NumTLP / (64 / (NumTLP / vecloadFactor));
        int nHotBlock = (size / NumTLP) * NumTLP;
        if (notFound) {
          if (numPlanes % NumTLP == 0) {
            if (nHotBlock >= NumTLP) {
              gatherVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                   vecload_type_t>
                  <<<dim3(size / NumTLP, numPlanes / NumTLP),
                     dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                     stream>>>(buffer.data_ptr<T>(), features.data_ptr<T>(),
                               indices.data_ptr<Index>(), nHotBlock,
                               numPlanes / vecloadFactor);
#ifdef TV_LOG_KERNEL_INFO
              cudaFuncAttributes attr;
              checkCudaErrors(cudaFuncGetAttributes(
                  &attr, gatherVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                              vecload_type_t>));
              tv::ssprint("gatherVecBlockKernel<", tv::type_s<T>,
                          tv::type_s<Index>, int(NumTLP), NumILP, ">",
                          attr.numRegs);
#endif
              TV_CHECK_CUDA_ERR();
            }
            if (size - nHotBlock > 0) {
              gatherVecKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                  <<<dim3(1, numPlanes / NumTLP),
                     dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                     stream>>>(buffer.data_ptr<T>() + nHotBlock * numPlanes,
                               features.data_ptr<T>(),
                               indices.data_ptr<Index>() + nHotBlock,
                               size - nHotBlock, numPlanes / vecloadFactor);
#ifdef TV_LOG_KERNEL_INFO
              cudaFuncAttributes attr;
              checkCudaErrors(cudaFuncGetAttributes(
                  &attr, gatherVecKernel<T, Index, int(NumTLP), NumILP,
                                         vecload_type_t>));
              tv::ssprint("gatherVecKernel<", tv::type_s<T>, tv::type_s<Index>,
                          int(NumTLP), NumILP, ">", attr.numRegs);
#endif

              TV_CHECK_CUDA_ERR();
            }
            notFound = false;
          }
        }
      });
      if (notFound) {
        constexpr int NumTLP = 64;
        constexpr int NumILP = NumTLP / 4;
        gatherGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(tv::cuda::DivUp(size, NumTLP),
                    tv::cuda::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, stream>>>(
                buffer.data_ptr<T>(), features.data_ptr<T>(),
                indices.data_ptr<Index>(), size, numPlanes);
#ifdef TV_LOG_KERNEL_INFO
        cudaFuncAttributes attr;
        checkCudaErrors(cudaFuncGetAttributes(
            &attr, gatherGenericKernel<T, Index, NumTLP, NumILP>));
        tv::ssprint("gatherGenericKernel<", tv::type_s<T>, tv::type_s<Index>,
                    int(NumTLP), NumILP, ">", attr.numRegs);
#endif

        TV_CHECK_CUDA_ERR();
      }
    });
  });
}

void sparse_scatter_add_cuda(torch::Tensor buffer, torch::Tensor outFeatures,
                             torch::Tensor indices, int size) {
  if (size <= 0)
    return;
  int numPlanes = outFeatures.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dtype = outFeatures.scalar_type();
  auto inds_dtype = indices.scalar_type();

  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    using vecload_type_t = typename half_vec_sadd<T>::type;
    tv::DispatchTorch<int_types_t>()(inds_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      bool notFound = true;
      constexpr int vecloadFactor =
          sizeof(vecload_type_t) / sizeof(T); // important for half.

      tv::mp_for_each<kernel_block_t>([=, &outFeatures, &buffer, &indices,
                                       &notFound](auto NumTLP) {
        // constexpr int NumILP = NumTLP / (64 / (NumTLP /
        // vecloadFactor));
        constexpr int NumILP = NumTLP / 4;
        int nHotBlock = (size / NumTLP) * NumTLP;
        if (notFound) {
          if (numPlanes % NumTLP == 0) {
            if (nHotBlock >= NumTLP) {
              scatterAddVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                       vecload_type_t>
                  <<<dim3(size / NumTLP, numPlanes / NumTLP),
                     dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                     stream>>>(outFeatures.data_ptr<T>(), buffer.data_ptr<T>(),
                               indices.data_ptr<Index>(), nHotBlock,
                               numPlanes / vecloadFactor);
#ifdef TV_LOG_KERNEL_INFO
              cudaFuncAttributes attr;
              checkCudaErrors(cudaFuncGetAttributes(
                  &attr, scatterAddVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                                  vecload_type_t>));
              tv::ssprint("scatterAddVecBlockKernel<", tv::type_s<T>,
                          tv::type_s<Index>, int(NumTLP), NumILP, ">",
                          attr.numRegs);
#endif

              TV_CHECK_CUDA_ERR();
            }
            if (size - nHotBlock > 0) {
              scatterAddGenericKernel<T, Index, int(NumTLP), NumILP>
                  <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                     0, stream>>>(outFeatures.data_ptr<T>(),
                                  buffer.data_ptr<T>() + nHotBlock * numPlanes,
                                  indices.data_ptr<Index>() + nHotBlock,
                                  size - nHotBlock, numPlanes);
#ifdef TV_LOG_KERNEL_INFO
              cudaFuncAttributes attr;
              checkCudaErrors(cudaFuncGetAttributes(
                  &attr,
                  scatterAddGenericKernel<T, Index, int(NumTLP), NumILP>));
              tv::ssprint("scatterAddGenericKernel<", tv::type_s<T>,
                          tv::type_s<Index>, int(NumTLP), NumILP, ">",
                          attr.numRegs);
#endif
              TV_CHECK_CUDA_ERR();
            }
            notFound = false;
          }
        }
      });
      if (notFound) {
        constexpr int NumTLP = 64;
        constexpr int NumILP = NumTLP / 4;
        scatterAddGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(tv::cuda::DivUp(size, NumTLP),
                    tv::cuda::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, stream>>>(
                outFeatures.data_ptr<T>(), buffer.data_ptr<T>(),
                indices.data_ptr<Index>(), size, numPlanes);
#ifdef TV_LOG_KERNEL_INFO
        cudaFuncAttributes attr;
        checkCudaErrors(cudaFuncGetAttributes(
            &attr, scatterAddGenericKernel<T, Index, int(NumTLP), NumILP>));
        tv::ssprint("notfound scatterAddGenericKernel<", tv::type_s<T>,
                    tv::type_s<Index>, int(NumTLP), NumILP, ">", attr.numRegs);
#endif

        TV_CHECK_CUDA_ERR();
      }
    });
  });
}

void batch_sparse_gather_cuda(torch::Tensor buffer, torch::Tensor features,
                              torch::Tensor indices, int size) {
  // indices: [volume, inds_stride]
  // buffer: [volume, num_points, num_features]
  // size == volume * num_points
  if (size <= 0)
    return;
  int numPlanes = features.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dtype = features.scalar_type();
  auto inds_dtype = indices.scalar_type();
  int inds_stride = indices.size(1);
  int feature_stride = buffer.size(1);
  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    using vecload_type_t = typename half_vec<T>::type;
    tv::DispatchTorch<int_types_t>()(inds_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      bool notFound = true;
      constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
      tv::mp_for_each<kernel_block_t>(
          [=, &buffer, &features, &indices, &notFound](auto NumTLP) {
            constexpr int NumILP = NumTLP / 4;
            // constexpr int NumILP = NumTLP / (64 / (NumTLP / vecloadFactor));
            int nHotBlock = (size / NumTLP) * NumTLP;
            if (notFound) {
              if (numPlanes % NumTLP == 0) {
                if (nHotBlock >= NumTLP) {
                  batchGatherVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                            vecload_type_t>
                      <<<dim3(size / NumTLP, numPlanes / NumTLP),
                         dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                         stream>>>(buffer.data_ptr<T>(), features.data_ptr<T>(),
                                   indices.data_ptr<Index>(), nHotBlock,
                                   numPlanes / vecloadFactor, inds_stride,
                                   feature_stride);
                  TV_CHECK_CUDA_ERR_V2("batchGatherVecBlockKernel");
                }
                if (size - nHotBlock > 0) {
                  batchGatherVecKernel<T, Index, int(NumTLP), NumILP,
                                       vecload_type_t>
                      <<<dim3(1, numPlanes / NumTLP),
                         dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                         stream>>>(buffer.data_ptr<T>() + nHotBlock * numPlanes,
                                   features.data_ptr<T>(),
                                   indices.data_ptr<Index>(), size - nHotBlock,
                                   nHotBlock, numPlanes / vecloadFactor,
                                   inds_stride, feature_stride);
                  TV_CHECK_CUDA_ERR_V2("batchGatherVecKernel");
                }
                notFound = false;
              }
            }
          });

      if (notFound) {
        constexpr int NumTLP = 64;
        constexpr int NumILP = NumTLP / 4;
        batchGatherGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(tv::cuda::DivUp(size, NumTLP),
                    tv::cuda::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, stream>>>(
                buffer.data_ptr<T>(), features.data_ptr<T>(),
                indices.data_ptr<Index>(), size, numPlanes, inds_stride,
                feature_stride);
        TV_CHECK_CUDA_ERR();
      }
    });
  });
}

void batch_sparse_scatter_add_cuda(torch::Tensor buffer,
                                   torch::Tensor outFeatures,
                                   torch::Tensor indices, int size) {
  // indices: [volume, inds_stride]
  // buffer: [volume, num_points, num_features]
  // size == volume * num_points
  if (size <= 0)
    return;
  int numPlanes = outFeatures.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dtype = outFeatures.scalar_type();
  auto inds_dtype = indices.scalar_type();
  int inds_stride = indices.size(1);
  int feature_stride = buffer.size(1);

  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    using vecload_type_t = typename half_vec_sadd<T>::type;
    tv::DispatchTorch<int_types_t>()(inds_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      bool notFound = true;
      constexpr int vecloadFactor = 1; // important for half.

      tv::mp_for_each<kernel_block_t>([=, &outFeatures, &buffer, &indices,
                                       &notFound](auto NumTLP) {
        // constexpr int NumILP = NumTLP / (64 / (NumTLP /
        // vecloadFactor));
        constexpr int NumILP = NumTLP / 4;
        int nHotBlock = (size / NumTLP) * NumTLP;
        if (notFound) {
          if (numPlanes % NumTLP == 0) {
            if (nHotBlock >= NumTLP) {
              batchScatterAddBlockKernel<T, Index, int(NumTLP), NumILP>
                  <<<dim3(size / NumTLP, numPlanes / NumTLP),
                     dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                     stream>>>(outFeatures.data_ptr<T>(), buffer.data_ptr<T>(),
                               indices.data_ptr<Index>(), nHotBlock,
                               numPlanes / vecloadFactor, inds_stride,
                               feature_stride);
              TV_CHECK_CUDA_ERR();
            }
            if (size - nHotBlock > 0) {
              batchScatterAddGenericKernel<T, Index, int(NumTLP), NumILP>
                  <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                     0, stream>>>(outFeatures.data_ptr<T>(),
                                  buffer.data_ptr<T>() + nHotBlock * numPlanes,
                                  indices.data_ptr<Index>(), size - nHotBlock,
                                  nHotBlock, numPlanes, inds_stride,
                                  feature_stride);
              TV_CHECK_CUDA_ERR();
            }
            notFound = false;
          }
        }
      });

      if (notFound) {
        constexpr int NumTLP = 64;
        constexpr int NumILP = NumTLP / 4;
        batchScatterAddGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(tv::cuda::DivUp(size, NumTLP),
                    tv::cuda::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, stream>>>(
                outFeatures.data_ptr<T>(), buffer.data_ptr<T>(),
                indices.data_ptr<Index>(), size, 0, numPlanes, inds_stride,
                feature_stride);
        TV_CHECK_CUDA_ERR();
      }
    });
  });
}

} // namespace spconv
