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
#include <cuhash/hash_table.h>
#include <limits>
#include <spconv/indice.cu.h>
#include <spconv/indice.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/mp_helper.h>
#include <tensorview/tensor.h>
#include <tensorview/tensorview.h>
#include <tensorview/torch_utils.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility/timer.h>

namespace spconv {

using max_kernel_vol_t = tv::mp_list_c<int, 9, 16, 27, 32, 128, 256, 4096>;

int create_conv_indice_pair_p1_cuda(
    torch::Tensor indicesIn, torch::Tensor indicePairs, torch::Tensor indiceNum,
    torch::Tensor indicePairUnique, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
    bool transpose) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ndim = kernelSize.size();
  auto numActIn = indicesIn.size(0);
  auto kernelVolume = indiceNum.size(0);
  if (numActIn == 0)
    return 0;
  tv::dispatch_torch<int32_t>(indicesIn.scalar_type(), [&](auto IndexValue) {
    using Index = TV_DECLTYPE(IndexValue);
    using IndexGrid = int32_t;
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = TV_DECLTYPE(I)::value;
      tv::SimpleVector<Index, NDim> ks(kernelSize.begin(), kernelSize.end());
      tv::SimpleVector<Index, NDim> st(stride.begin(), stride.end());
      tv::SimpleVector<Index, NDim> pa(padding.begin(), padding.end());
      tv::SimpleVector<Index, NDim> di(dilation.begin(), dilation.end());
      tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                       outSpatialShape.end());
      tv::DispatchInt<max_kernel_vol_t>()(
          kernelVolume, std::less_equal<int>(), [&](auto I2) {
            constexpr int MaxKernelVolume = TV_DECLTYPE(I2)::value;
            if (transpose) {
              prepareDeConvIndicePairsKernel<Index, NDim, MaxKernelVolume>
                  <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS,
                     0, stream>>>(tv::torch2tv<Index>(indicesIn),
                                  tv::torch2tv<Index>(indicePairs),
                                  tv::torch2tv<Index>(indiceNum),
                                  tv::torch2tv<Index>(indicePairUnique), ks, st,
                                  pa, di, ou);
              TV_CHECK_CUDA_ERR_V2("prepareDeConvIndicePairsKernel failed");
            } else {
              prepareIndicePairsKernel<Index, NDim, MaxKernelVolume>
                  <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS,
                     0, stream>>>(tv::torch2tv<Index>(indicesIn),
                                  tv::torch2tv<Index>(indicePairs),
                                  tv::torch2tv<Index>(indiceNum),
                                  tv::torch2tv<Index>(indicePairUnique), ks, st,
                                  pa, di, ou);
              TV_CHECK_CUDA_ERR_V2("prepareIndicePairsKernel failed");
            }
#ifdef TV_LOG_KERNEL_INFO
            cudaFuncAttributes attr;
            checkCudaErrors(cudaFuncGetAttributes(
                &attr,
                prepareDeConvIndicePairsKernel<Index, NDim, MaxKernelVolume>));
            tv::ssprint("prepareIndicePairsKernel<", tv::type_s<Index>, NDim,
                        MaxKernelVolume, ">", attr.numRegs);
#endif
          });
    });
  });
  return 1;
}

int create_conv_indice_pair_p2_cuda(
    torch::Tensor indicesIn, torch::Tensor indicesOut, torch::Tensor gridsOut,
    torch::Tensor indicePairs, torch::Tensor indiceNum,
    torch::Tensor indicePairUnique, std::vector<int64_t> outSpatialShape,
    bool transpose, bool resetGrid, bool useHash) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ndim = outSpatialShape.size();
  auto numActIn = indicesIn.size(0);
  int batchSize = gridsOut.size(0);
  int numAct = indicePairUnique.size(0) - 1;

  auto kernelVolume = indiceNum.size(0);
  if (numActIn == 0)
    return 0;
  bool failed = false;
  tv::dispatch_torch<int32_t>(indicesIn.scalar_type(), [&](auto IndexValue) {
    using Index = TV_DECLTYPE(IndexValue);
    using IndexGrid = int32_t;
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = TV_DECLTYPE(I)::value;
      using IndexGrid = int32_t;
      tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                       outSpatialShape.end());
      if (useHash) {
        auto table = cuhash::HashTable();
        // std::cout << "create " << numAct << " size table..." << std::endl;
        table.Initialize(numAct, 2.0, 4);
        unsigned *d_values = nullptr;
        cudaMalloc((void **)&d_values, sizeof(unsigned) * numAct);
        TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");
        arangeKernel<unsigned>
            <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(d_values, numAct);
        TV_CHECK_CUDA_ERR_V2("arangeKernel failed");
        bool res = table.Build(
            numAct,
            reinterpret_cast<unsigned *>(indicePairUnique.data_ptr<Index>()),
            d_values);
        cudaFree(d_values);
        TV_CHECK_CUDA_ERR_V2("cudaFree failed");
        if (!res) {
          failed = true;
          return;
        }
        assignIndiceOutKernel<Index, NDim>
            <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesOut), numAct,
                         tv::torch2tv<Index>(indicePairUnique), ou, batchSize);
        TV_CHECK_CUDA_ERR_V2("assignIndiceOutKernel failed");

        auto tableSize = table.get_table_size();
        auto tableData = table.data();
        auto constants = table.get_constants_4();
        auto stash_constants = table.get_stash_constants();
        auto stash_count = table.get_stash_count();
        assignIndicePairsHashKernel<Index, NDim>
            <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesOut), numActIn,
                         tv::torch2tv<Index>(indicePairs),
                         tv::torch2tv<Index>(indicePairUnique), tableSize,
                         tableData, constants, stash_constants, stash_count);
        TV_CHECK_CUDA_ERR_V2("assignIndicePairsHashKernel failed");

      } else {
        assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>
            <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesOut),
                         tv::torch2tv<IndexGrid>(gridsOut), numAct,
                         tv::torch2tv<Index>(indicePairs),
                         tv::torch2tv<Index>(indicePairUnique), ou, batchSize);
        TV_CHECK_CUDA_ERR_V2("assignGridAndIndiceOutKernel failed");
        assignIndicePairsKernel<Index, IndexGrid, NDim>
            <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesOut),
                         tv::torch2tv<IndexGrid>(gridsOut), numActIn,
                         tv::torch2tv<Index>(indicePairs),
                         tv::torch2tv<Index>(indicePairUnique), ou);
        TV_CHECK_CUDA_ERR_V2("assignIndicePairsKernel failed");
#ifdef TV_LOG_KERNEL_INFO
        cudaFuncAttributes attr;
        checkCudaErrors(cudaFuncGetAttributes(
            &attr, assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>));
        tv::ssprint("assignGridAndIndiceOutKernel<", tv::type_s<Index>, NDim,
                    ">", attr.numRegs);
        cudaFuncAttributes attr2;
        checkCudaErrors(cudaFuncGetAttributes(
            &attr2, assignIndicePairsKernel<Index, IndexGrid, NDim>));
        tv::ssprint("assignIndicePairsKernel<", tv::type_s<Index>, NDim, ">",
                    attr2.numRegs);
#endif
      }

      if (resetGrid && (!useHash)) {
        resetGridKernel<Index, IndexGrid, NDim>
            <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(indicePairUnique.data_ptr<Index>(),
                         tv::torch2tv<IndexGrid>(gridsOut), numAct);
        TV_CHECK_CUDA_ERR_V2("resetGridKernel failed");
      }
    });
  });
  if (failed){
    return -1;
  }
  return numAct;
}

template <typename T> struct is_valid {
  __device__ __forceinline__ bool operator()(const T x) { return x != -1; }
};

int create_submconv_indice_pair_cuda(
    torch::Tensor indicesIn, torch::Tensor gridsOut, torch::Tensor indicePairs,
    torch::Tensor indiceNum, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
    bool transpose, bool resetGrid, bool useHash) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ndim = outSpatialShape.size();
  auto numActIn = indicesIn.size(0);
  int batchSize = gridsOut.size(0);

  auto kernelVolume = indiceNum.size(0);
  if (numActIn == 0)
    return 0;
  bool failed = false;

  tv::dispatch_torch<int32_t>(indicesIn.scalar_type(), [&](auto IndexValue) {
    using Index = TV_DECLTYPE(IndexValue);
    using IndexGrid = int32_t;
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = TV_DECLTYPE(I)::value;
      tv::SimpleVector<Index, NDim> ks(kernelSize.begin(), kernelSize.end());
      tv::SimpleVector<Index, NDim> st(stride.begin(), stride.end());
      tv::SimpleVector<Index, NDim> pa(padding.begin(), padding.end());
      tv::SimpleVector<Index, NDim> di(dilation.begin(), dilation.end());
      tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                       outSpatialShape.end());
      Index spatialVolume = 1;
      for (int i = 0; i < NDim; ++i) {
        spatialVolume *= outSpatialShape[i];
      }
      if (useHash) {
        auto table = cuhash::HashTable();
        // std::cout << "create " << numAct << " size table..." << std::endl;
        table.Initialize(numActIn, 2.0, 4);
        unsigned *d_keyvalues = nullptr;
        cudaMalloc((void **)&d_keyvalues, sizeof(unsigned) * numActIn * 2);
        unsigned *d_values = d_keyvalues + numActIn;
        TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");
        prepareSubMHashKernel<Index, NDim>
            <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesIn), d_keyvalues, d_values,
                         ou);
        TV_CHECK_CUDA_ERR_V2("prepareSubMHashKernel failed");
        bool res =
            table.Build(numActIn, reinterpret_cast<unsigned *>(d_keyvalues),
                        reinterpret_cast<unsigned *>(d_values));
        cudaFree(d_keyvalues);
        TV_CHECK_CUDA_ERR_V2("cudaFree failed");
        if (!res) {
          failed = true;
          return;
        }
        auto tableSize = table.get_table_size();
        auto tableData = table.data();
        auto constants = table.get_constants_4();
        auto stash_constants = table.get_stash_constants();
        auto stash_count = table.get_stash_count();
        tv::DispatchInt<max_kernel_vol_t>()(
            kernelVolume, std::less_equal<int>(), [&](auto I2) {
              constexpr int MaxKernelVolume = TV_DECLTYPE(I2)::value;
              getSubMIndicePairsHashKernel<Index, NDim, MaxKernelVolume>
                  <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS,
                     0, stream>>>(tv::torch2tv<Index>(indicesIn),
                                  tv::torch2tv<Index>(indicePairs),
                                  tv::torch2tv<Index>(indiceNum), ks, st, pa,
                                  di, ou, tableSize, tableData, constants,
                                  stash_constants, stash_count);
              TV_CHECK_CUDA_ERR_V2("getSubMIndicePairsHashKernel failed");
            });
      } else {
        // auto timer = spconv::CudaContextTimer<>();
        prepareSubMGridKernel<Index, IndexGrid, NDim>
            <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesIn),
                         tv::torch2tv<IndexGrid>(gridsOut), ou, spatialVolume);
        // tv::ssprint("prepareSubMGridKernel", timer.report() / 1000.0);
        TV_CHECK_CUDA_ERR_V2("prepareSubMGridKernel failed");
        // when dilation all one, we use a simple kernel to calc result
        bool dilation_one = true;
        for (int i = 0; i < NDim; ++i) {
          dilation_one &= di[i] == 1;
        }
        auto found = false;
        if (dilation_one && (NDim == 2 || NDim == 3)) {
          auto indiceNumCpu = indiceNum.cpu();
          if (NDim == 2) {
            tv::SimpleVector<Index, 2> ou_(outSpatialShape.begin(),
                                           outSpatialShape.end());

            tv::dispatch_int_noexcept<1, 3, 5>(kernelSize[0], [&](auto K0C) {
              tv::dispatch_int_noexcept<1, 3, 5>(kernelSize[1], [&](auto K1C) {
                constexpr int K0 = TV_DECLTYPE(K0C)::value;
                constexpr int K1 = TV_DECLTYPE(K1C)::value;
                found = true;
                getSubMIndicePairsKernel2<Index, IndexGrid, K0, K1>
                    <<<tv::cuda::getBlocks(numActIn),
                       tv::cuda::CUDA_NUM_THREADS, 0, stream>>>(
                        tv::torch2tv<Index>(indicesIn),
                        tv::torch2tv<IndexGrid>(gridsOut),
                        tv::torch2tv<Index>(indicePairs),
                        tv::torch2tv<Index>(indiceNum), ou_, spatialVolume);
              });
            });
          } else if (NDim == 3) {
            tv::SimpleVector<Index, 3> ou_(outSpatialShape.begin(),
                                           outSpatialShape.end());

            tv::dispatch_int_noexcept<1, 3, 5>(kernelSize[0], [&](auto K0C) {
              tv::dispatch_int_noexcept<1, 3, 5>(kernelSize[1], [&](auto K1C) {
                tv::dispatch_int_noexcept<1, 3, 5>(
                    kernelSize[2], [&](auto K2C) {
                      constexpr int K0 = TV_DECLTYPE(K0C)::value;
                      constexpr int K1 = TV_DECLTYPE(K1C)::value;
                      constexpr int K2 = TV_DECLTYPE(K2C)::value;
                      found = true;
                      getSubMIndicePairsKernel3<Index, IndexGrid, K0, K1, K2>
                          <<<tv::cuda::getBlocks(numActIn),
                             tv::cuda::CUDA_NUM_THREADS, 0, stream>>>(
                              tv::torch2tv<Index>(indicesIn),
                              tv::torch2tv<IndexGrid>(gridsOut),
                              tv::torch2tv<Index>(indicePairs),
                              tv::torch2tv<Index>(indiceNum), ou_,
                              spatialVolume);
                    });
              });
            });
          }
        }
        if (!found) {
          tv::DispatchInt<
              max_kernel_vol_t>()(ndim, std::less_equal<int>(), [&](auto I2) {
            constexpr int MaxKernelVolume = TV_DECLTYPE(I2)::value;
            getSubMIndicePairsKernel<Index, IndexGrid, NDim, MaxKernelVolume>
                <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
                   stream>>>(tv::torch2tv<Index>(indicesIn),
                             tv::torch2tv<IndexGrid>(gridsOut),
                             tv::torch2tv<Index>(indicePairs),
                             tv::torch2tv<Index>(indiceNum), ks, st, pa, di,
                             ou);
            TV_CHECK_CUDA_ERR_V2("getSubMIndicePairsKernel failed");
          });
        }
        // tv::ssprint("getSubMIndicePairsKernel", timer.report() / 1000.0);
      }

      if (resetGrid && (!useHash)) {
        resetGridSubMKernel<Index, IndexGrid, NDim>
            <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(indicesIn.data_ptr<Index>(),
                         tv::torch2tv<IndexGrid>(gridsOut), ou, numActIn);
        TV_CHECK_CUDA_ERR_V2("resetGridKernel failed");
      }
    });
  });
  if (failed){
    return -1;
  }

  return numActIn;
}

} // namespace spconv