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
#include <spconv/mp_helper.h>
#include <spconv/indice.h>
#include <spconv/indice.cu.h>
#include <tensorview/helper_launch.h>
#include <tensorview/tensorview.h>
#include <type_traits>
#include <utility/timer.h>
#include <cuhash/hash_table.h>

namespace spconv {
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
      prepareDeConvIndicePairsKernel<Index, IndexGrid, NDim, 4096>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicesOut, gridsOut, indicePairs,
                           indiceNum, indicePairUnique, kernelSize, stride,
                           padding, dilation, outSpatialShape);
    else
      prepareIndicePairsKernel<Index, IndexGrid, NDim, 4096>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicesOut, gridsOut, indicePairs,
                           indiceNum, indicePairUnique, kernelSize, stride,
                           padding, dilation, outSpatialShape);
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
    // after unique, there is a std::numeric_limits<int>::max() in the end of indicePairUnique
    Index numAct = indicePairUnique.dim(0) - 1; 
    if (useHash){
      auto table = cuhash::HashTable();
      // std::cout << "create " << numAct << " size table..." << std::endl;
      table.Initialize(numAct, 2.0, 4);
      unsigned *d_values = nullptr;
      cudaMalloc((void**)&d_values, sizeof(unsigned) * numAct);
      TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");
      arangeKernel<unsigned><<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(d_values, numAct);
      bool res = table.Build(numAct, reinterpret_cast<unsigned*>(indicePairUnique.data()), 
                d_values);
      cudaFree(d_values);
      if (!res){
        return -1; //use -1 to tell outside use CPU implementation
      }
      assignIndiceOutKernel<Index, NDim>
          <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesOut, numAct,
                          indicePairUnique, outSpatialShape, batchSize);
      TV_CHECK_CUDA_ERR_V2("assignGridAndIndiceOutKernel failed");
      auto tableSize = table.get_table_size();
      auto tableData = table.data();
      auto constants = table.get_constants_4();
      auto stash_constants = table.get_stash_constants();
      auto stash_count = table.get_stash_count();
      assignIndicePairsHashKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesOut, numActIn, indicePairs,
                          indicePairUnique,
                          tableSize, tableData, constants, stash_constants,
                          stash_count);
      TV_CHECK_CUDA_ERR_V2("assignIndicePairsKernel failed");
    }else{
      assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesOut, gridsOut, numAct, indicePairs,
                          indicePairUnique, outSpatialShape, batchSize);
      TV_CHECK_CUDA_ERR_V2("assignGridAndIndiceOutKernel failed");
      assignIndicePairsKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesOut, gridsOut, numActIn, indicePairs,
                          indicePairUnique, outSpatialShape);
      TV_CHECK_CUDA_ERR_V2("assignIndicePairsKernel failed");

    }

    if (resetGrid && (!useHash)) {
      resetGridKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
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
    if (useHash){
      auto table = cuhash::HashTable();
      // std::cout << "subm create " << numActIn << " size table..." << std::endl;
      table.Initialize(numActIn, 2.0, 4);
      unsigned *d_keyvalues = nullptr;
      cudaMalloc((void**)&d_keyvalues, sizeof(unsigned) * numActIn * 2);
      unsigned *d_values = d_keyvalues + numActIn;
      prepareSubMHashKernel<Index, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesIn, d_keyvalues, d_values, outSpatialShape);
      TV_CHECK_CUDA_ERR_V2("prepareSubMHashKernel failed");
      bool res = table.Build(numActIn, reinterpret_cast<unsigned*>(d_keyvalues), 
                reinterpret_cast<unsigned*>(d_values));
      cudaFree(d_keyvalues);
      if (!res){
        return -1; //use -1 to tell outside use CPU implementation
      }
      auto tableSize = table.get_table_size();
      auto tableData = table.data();
      auto constants = table.get_constants_4();
      auto stash_constants = table.get_stash_constants();
      auto stash_count = table.get_stash_count();
      getSubMIndicePairsHashKernel<Index, NDim, 4096>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesIn, indicePairs, indiceNum,
                          kernelSize, stride, padding, dilation, outSpatialShape,
                          tableSize, tableData, constants, stash_constants,
                          stash_count);
      TV_CHECK_CUDA_ERR_V2("getSubMIndicePairsHashKernel failed");
    }else{
      prepareSubMGridKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesIn, gridsOut, outSpatialShape);
      TV_CHECK_CUDA_ERR();
      getSubMIndicePairsKernel<Index, IndexGrid, NDim, 4096>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
            d.getStream()>>>(indicesIn, gridsOut, indicePairs, indiceNum,
                          kernelSize, stride, padding, dilation, outSpatialShape);
      TV_CHECK_CUDA_ERR();
    }
    // std::cout << "subm gene time " << timer.report() / 1000.0 << std::endl;
    if (resetGrid && (!useHash)) {
      resetGridSubMKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn.data(), gridsOut, outSpatialShape, numActIn);
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