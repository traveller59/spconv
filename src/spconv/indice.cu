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
#include <tensorview/torch_utils.h>

#include <tensorview/tensor.h>
#include <tensorview/tensorview.h>
#include <type_traits>
#include <utility/timer.h>

namespace spconv {

int create_conv_indice_pair_p1_cuda(
    torch::Tensor indicesIn, torch::Tensor indicePairs, torch::Tensor indiceNum,
    torch::Tensor indicePairUnique, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
    bool transpose) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ndim = kernelSize.size();
  auto numActIn = indicesIn.size(0);
  auto kernelVolume = indicePairs.size(0);
  if (numActIn == 0)
    return 0;
  tv::dispatch_torch<int32_t>(indicesIn.scalar_type(), [&](auto IndexValue) {
    using Index = decltype(IndexValue);
    using IndexGrid = int32_t;
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = decltype(I)::value;
      tv::SimpleVector<Index, NDim> ks(kernelSize.begin(), kernelSize.end());
      tv::SimpleVector<Index, NDim> st(stride.begin(), stride.end());
      tv::SimpleVector<Index, NDim> pa(padding.begin(), padding.end());
      tv::SimpleVector<Index, NDim> di(dilation.begin(), dilation.end());
      tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                       outSpatialShape.end());
      tv::dispatch_int<16, 32, 256, 4096>(
          kernelVolume, std::less_equal<int>(), [&](auto I2) {
            constexpr int MaxKernelVolume = decltype(I2)::value;
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

  auto kernelVolume = indicePairs.size(0);
  if (numActIn == 0)
    return 0;
  tv::dispatch_torch<int32_t>(indicesIn.scalar_type(), [&](auto IndexValue) {
    using Index = decltype(IndexValue);
    using IndexGrid = int32_t;
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = decltype(I)::value;
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
        bool res =
            table.Build(numAct,
                        reinterpret_cast<unsigned *>(
                            tv::torch2tv<Index>(indicePairUnique).data()),
                        d_values);
        cudaFree(d_values);
        TV_CHECK_CUDA_ERR_V2("cudaFree failed");
        if (!res) {
          return -1; // use -1 to tell outside use CPU implementation
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
  return numAct;
}

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

  auto kernelVolume = indicePairs.size(0);
  if (numActIn == 0)
    return 0;
  tv::dispatch_torch<int32_t>(indicesIn.scalar_type(), [&](auto IndexValue) {
    using Index = decltype(IndexValue);
    using IndexGrid = int32_t;
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = decltype(I)::value;
      tv::SimpleVector<Index, NDim> ks(kernelSize.begin(), kernelSize.end());
      tv::SimpleVector<Index, NDim> st(stride.begin(), stride.end());
      tv::SimpleVector<Index, NDim> pa(padding.begin(), padding.end());
      tv::SimpleVector<Index, NDim> di(dilation.begin(), dilation.end());
      tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                       outSpatialShape.end());
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
        cudaFree(d_values);
        TV_CHECK_CUDA_ERR_V2("cudaFree failed");
        if (!res) {
          return -1; // use -1 to tell outside use CPU implementation
        }
        auto tableSize = table.get_table_size();
        auto tableData = table.data();
        auto constants = table.get_constants_4();
        auto stash_constants = table.get_stash_constants();
        auto stash_count = table.get_stash_count();
        tv::dispatch_int<16, 32, 256, 4096>(
            kernelVolume, std::less_equal<int>(), [&](auto I2) {
              constexpr int MaxKernelVolume = decltype(I2)::value;
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
        prepareSubMGridKernel<Index, IndexGrid, NDim>
            <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
               stream>>>(tv::torch2tv<Index>(indicesIn),
                         tv::torch2tv<IndexGrid>(gridsOut), ou);
        TV_CHECK_CUDA_ERR_V2("prepareSubMGridKernel failed");
        tv::dispatch_int<16, 32, 256, 4096>(
            ndim, std::less_equal<int>(), [&](auto I2) {
              constexpr int MaxKernelVolume = decltype(I2)::value;
              getSubMIndicePairsKernel<Index, IndexGrid, NDim, MaxKernelVolume>
                  <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS,
                     0, stream>>>(tv::torch2tv<Index>(indicesIn),
                                  tv::torch2tv<IndexGrid>(gridsOut),
                                  tv::torch2tv<Index>(indicePairs),
                                  tv::torch2tv<Index>(indiceNum), ks, st, pa,
                                  di, ou);
              TV_CHECK_CUDA_ERR_V2("assignIndicePairsKernel failed");
            });
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
  return numActIn;
}

namespace functor {
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP1<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose) {
    Index batchSize = gridsOut.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    if (transpose)
      prepareDeConvIndicePairsKernel<Index, NDim, 4096>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicePairs, indiceNum,
                              indicePairUnique, kernelSize, stride, padding,
                              dilation, outSpatialShape);
    else
      prepareIndicePairsKernel<Index, NDim, 4096>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicePairs, indiceNum,
                              indicePairUnique, kernelSize, stride, padding,
                              dilation, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    // std::cout << "p1 gene time " << timer.report() / 1000.0 << std::endl;
    return 1;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP2<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid, bool useHash) {
    Index batchSize = gridsOut.dim(0);
    auto kernelVolume = indicePairs.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // after unique, there is a std::numeric_limits<int>::max() in the end of
    // indicePairUnique
    Index numAct = indicePairUnique.dim(0) - 1;
    if (useHash) {
      auto table = cuhash::HashTable();
      // std::cout << "create " << numAct << " size table..." << std::endl;
      table.Initialize(numAct, 2.0, 4);
      unsigned *d_values = nullptr;
      cudaMalloc((void **)&d_values, sizeof(unsigned) * numAct);
      TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");
      arangeKernel<unsigned>
          <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(d_values, numAct);
      bool res = table.Build(
          numAct, reinterpret_cast<unsigned *>(indicePairUnique.data()),
          d_values);
      cudaFree(d_values);
      if (!res) {
        return -1; // use -1 to tell outside use CPU implementation
      }
      assignIndiceOutKernel<Index, NDim>
          <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesOut, numAct, indicePairUnique,
                              outSpatialShape, batchSize);
      TV_CHECK_CUDA_ERR_V2("assignGridAndIndiceOutKernel failed");
      auto tableSize = table.get_table_size();
      auto tableData = table.data();
      auto constants = table.get_constants_4();
      auto stash_constants = table.get_stash_constants();
      auto stash_count = table.get_stash_count();
      assignIndicePairsHashKernel<Index, NDim>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesOut, numActIn, indicePairs,
                              indicePairUnique, tableSize, tableData, constants,
                              stash_constants, stash_count);
      TV_CHECK_CUDA_ERR_V2("assignIndicePairsKernel failed");
    } else {
      assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>
          <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesOut, gridsOut, numAct, indicePairs,
                              indicePairUnique, outSpatialShape, batchSize);
      TV_CHECK_CUDA_ERR_V2("assignGridAndIndiceOutKernel failed");
      assignIndicePairsKernel<Index, IndexGrid, NDim>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesOut, gridsOut, numActIn, indicePairs,
                              indicePairUnique, outSpatialShape);
      TV_CHECK_CUDA_ERR_V2("assignIndicePairsKernel failed");
    }

    if (resetGrid && (!useHash)) {
      resetGridKernel<Index, IndexGrid, NDim>
          <<<tv::cuda::getBlocks(numAct), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicePairUnique.data(), gridsOut, numAct);
      TV_CHECK_CUDA_ERR_V2("resetGridKernel failed");
    }
    return numAct;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid, bool useHash) {
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    if (useHash) {
      auto table = cuhash::HashTable();
      // std::cout << "subm create " << numActIn << " size table..." <<
      // std::endl;
      table.Initialize(numActIn, 2.0, 4);
      unsigned *d_keyvalues = nullptr;
      cudaMalloc((void **)&d_keyvalues, sizeof(unsigned) * numActIn * 2);
      unsigned *d_values = d_keyvalues + numActIn;
      prepareSubMHashKernel<Index, NDim>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, d_keyvalues, d_values,
                              outSpatialShape);
      TV_CHECK_CUDA_ERR_V2("prepareSubMHashKernel failed");
      bool res =
          table.Build(numActIn, reinterpret_cast<unsigned *>(d_keyvalues),
                      reinterpret_cast<unsigned *>(d_values));
      cudaFree(d_keyvalues);
      if (!res) {
        return -1; // use -1 to tell outside use CPU implementation
      }
      auto tableSize = table.get_table_size();
      auto tableData = table.data();
      auto constants = table.get_constants_4();
      auto stash_constants = table.get_stash_constants();
      auto stash_count = table.get_stash_count();
      getSubMIndicePairsHashKernel<Index, NDim, 4096>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicePairs, indiceNum, kernelSize,
                              stride, padding, dilation, outSpatialShape,
                              tableSize, tableData, constants, stash_constants,
                              stash_count);
      TV_CHECK_CUDA_ERR_V2("getSubMIndicePairsHashKernel failed");
    } else {
      prepareSubMGridKernel<Index, IndexGrid, NDim>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, gridsOut, outSpatialShape);
      TV_CHECK_CUDA_ERR();
      getSubMIndicePairsKernel<Index, IndexGrid, NDim, 4096>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, gridsOut, indicePairs, indiceNum,
                              kernelSize, stride, padding, dilation,
                              outSpatialShape);
      TV_CHECK_CUDA_ERR();
    }
    // std::cout << "subm gene time " << timer.report() / 1000.0 << std::endl;
    if (resetGrid && (!useHash)) {
      resetGridSubMKernel<Index, IndexGrid, NDim>
          <<<tv::cuda::getBlocks(numActIn), tv::cuda::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn.data(), gridsOut, outSpatialShape,
                              numActIn);
      TV_CHECK_CUDA_ERR();
    }
    return numActIn;
  }
};
} // namespace functor

#define DECLARE_GPU_SPECS_INDEX_NDIM(Index, NDIM)                              \
  template struct functor::CreateConvIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;                  \
  template struct functor::CreateConvIndicePairFunctorP1<tv::GPU, Index, int,  \
                                                         NDIM>;                \
  template struct functor::CreateConvIndicePairFunctorP2<tv::GPU, Index, int,  \
                                                         NDIM>;                \
  template struct functor::CreateSubMIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;

#define DECLARE_GPU_INDEX(Index)                                               \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 1);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 2);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 3);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 4);

DECLARE_GPU_INDEX(int);

#undef DECLARE_GPU_INDEX
#undef DECLARE_GPU_SPECS_INDEX_NDIM
} // namespace spconv