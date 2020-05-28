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

#ifndef SPARSE_CONV_OP_H_
#define SPARSE_CONV_OP_H_

#include <spconv/indice.h>
#include <spconv/reordering.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>
#include <utility/timer.h>

namespace spconv {

enum ConvAlgo { kNative = 0, kBatch = 1, kBatchGemmGather = 2 };

// torch.jit's doc says only support int64, so we need to convert to int32.
template <unsigned NDim>
std::vector<torch::Tensor>
getIndicePair(torch::Tensor indices, int64_t batchSize,
              std::vector<int64_t> outSpatialShape,
              std::vector<int64_t> spatialShape,
              std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
              std::vector<int64_t> padding, std::vector<int64_t> dilation,
              std::vector<int64_t> outPadding, int64_t _subM,
              int64_t _transpose, int64_t _useHash) {
  // auto timer = spconv::CudaContextTimer<>();
  bool subM = _subM != 0;
  bool transpose = _transpose != 0;
  // CPU always use hash (tsl::robin_map).
  bool useHash = _useHash != 0 || indices.device().type() == torch::kCPU;
  auto numAct = indices.size(0);
  auto coorDim = indices.size(1) - 1; // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  std::string msg = "due to limits of cuda hash, the volume of dense space "
                    "include batch size ";
  msg += "must less than std::numeric_limits<int>::max() = 2e9";
  TV_ASSERT_RT_ERR(batchSize * outputVolume < std::numeric_limits<int>::max(),
                   msg);
  torch::Tensor indicePairs =
      torch::full({2, kernelVolume, numAct}, -1,
                  torch::dtype(torch::kInt32).device(indices.device()));
  torch::Tensor indiceNum = torch::zeros(
      {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
  auto gridSize = batchSize * outputVolume;
  if (useHash) {
    gridSize = batchSize;
  }
  torch::Tensor gridOut = torch::full(
      {gridSize}, -1, torch::dtype(torch::kInt32).device(indices.device()));
  gridOut = gridOut.view({batchSize, -1});
  int64_t numActOut = -1;
  tv::SimpleVector<int, NDim> outSpatialShape32;
  tv::SimpleVector<int, NDim> kernelSize32;
  tv::SimpleVector<int, NDim> stride32;
  tv::SimpleVector<int, NDim> padding32;
  tv::SimpleVector<int, NDim> dilation32;
  for (int i = 0; i < NDim; ++i) {
    outSpatialShape32.push_back(outSpatialShape[i]);
    kernelSize32.push_back(kernelSize[i]);
    if (subM) {
      stride32.push_back(1);
      padding32.push_back(kernelSize[i] / 2);
      dilation32.push_back(dilation[i]);
    } else {
      stride32.push_back(stride[i]);
      padding32.push_back(padding[i]);
      dilation32.push_back(dilation[i]);
    }
  }
  if (subM) {
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose, false, useHash);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose, false, useHash);
      if (numActOut == -1) {
        // build hash failed. use CPU algorithm
        auto device = indices.device();
        indicePairs = indicePairs.to({torch::kCPU});
        indiceNum = indiceNum.to({torch::kCPU});
        indices = indices.to({torch::kCPU});
        auto getIndicePairFtor =
            functor::CreateSubMIndicePairFunctor<tv::CPU, int, int, NDim>();
        numActOut = getIndicePairFtor(
            tv::CPU(), tv::torch2tv<const int>(indices),
            tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
            tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
            dilation32, outSpatialShape32, transpose, false, useHash);
        return {indices.to(device), indicePairs.to(device),
                indiceNum.to(device)};
      }
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    return {indices, indicePairs, indiceNum};
  } else {
    auto indicePairUnique = torch::full(
        {indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
        torch::dtype(torch::kInt32).device(indices.device()));
    torch::Tensor outInds =
        torch::zeros({numAct * kernelVolume, coorDim + 1},
                     torch::dtype(torch::kInt32).device(indices.device()));
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateConvIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          kernelSize32, stride32, padding32, dilation32, outSpatialShape32,
          transpose);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      auto getIndicePairFtorP1 =
          functor::CreateConvIndicePairFunctorP1<tv::GPU, int, int, NDim>();
      auto getIndicePairFtorP2 =
          functor::CreateConvIndicePairFunctorP2<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtorP1(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          tv::torch2tv<int>(indicePairUnique), kernelSize32, stride32,
          padding32, dilation32, outSpatialShape32, transpose);
      if (numActOut > 0) {
        auto res = torch::_unique(indicePairUnique);
        indicePairUnique = std::get<0>(res);
        numActOut = getIndicePairFtorP2(
            tv::TorchGPU(), tv::torch2tv<const int>(indices),
            tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
            tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
            tv::torch2tv<int>(indicePairUnique), outSpatialShape32, transpose,
            false, useHash);
        if (numActOut == -1) {
          // build hash failed. use CPU algorithm
          auto getIndicePairFtor =
              functor::CreateConvIndicePairFunctor<tv::CPU, int, int, NDim>();
          auto device = indices.device();
          outInds = outInds.to({torch::kCPU});
          indicePairs = indicePairs.to({torch::kCPU});
          indiceNum = indiceNum.to({torch::kCPU});
          indices = indices.to({torch::kCPU});
          numActOut = getIndicePairFtor(
              tv::CPU(), tv::torch2tv<const int>(indices),
              tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
              tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
              kernelSize32, stride32, padding32, dilation32, outSpatialShape32,
              transpose);
          return {outInds.to(device).slice(0, 0, numActOut),
                  indicePairs.to(device), indiceNum.to(device)};
        }
      }
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}

std::vector<torch::Tensor>
getIndicePairV2(torch::Tensor indices, int64_t batchSize,
                std::vector<int64_t> outSpatialShape,
                std::vector<int64_t> spatialShape,
                std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                std::vector<int64_t> padding, std::vector<int64_t> dilation,
                std::vector<int64_t> outPadding, int64_t _subM,
                int64_t _transpose, int64_t _useHash);

template <unsigned NDim>
std::vector<torch::Tensor> getIndicePairPreGrid(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose,
    int64_t _useHash) {
  // auto timer = spconv::CudaContextTimer<>();
  bool subM = _subM != 0;
  bool transpose = _transpose != 0;
  bool useHash = _useHash != 0;
  TV_ASSERT_RT_ERR(!useHash, "error");
  auto numAct = indices.size(0);
  auto coorDim = indices.size(1) - 1; // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  TV_ASSERT_INVALID_ARG(gridOut.numel() >= outputVolume * batchSize, "error");
  torch::Tensor indicePairs =
      torch::full({kernelVolume, 2, numAct}, -1,
                  torch::dtype(torch::kInt32).device(indices.device()));
  torch::Tensor indiceNum = torch::zeros(
      {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
  // std::cout << "full time " << timer.report() / 1000.0 << std::endl;
  int64_t numActOut = -1;
  tv::SimpleVector<int, NDim> outSpatialShape32;
  tv::SimpleVector<int, NDim> kernelSize32;
  tv::SimpleVector<int, NDim> stride32;
  tv::SimpleVector<int, NDim> padding32;
  tv::SimpleVector<int, NDim> dilation32;
  auto indicePairUnique = torch::full(
      {indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
      torch::dtype(torch::kInt32).device(indices.device()));
  for (int i = 0; i < NDim; ++i) {
    outSpatialShape32.push_back(outSpatialShape[i]);
    kernelSize32.push_back(kernelSize[i]);
    if (subM) {
      stride32.push_back(1);
      padding32.push_back(kernelSize[i] / 2);
      dilation32.push_back(dilation[i]);
    } else {
      stride32.push_back(stride[i]);
      padding32.push_back(padding[i]);
      dilation32.push_back(dilation[i]);
    }
  }
  if (subM) {
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose);
      gridOut.fill_(-1);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose, true);
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    return {indices, indicePairs, indiceNum};
  } else {
    torch::Tensor outInds =
        torch::zeros({numAct * kernelVolume, coorDim + 1},
                     torch::dtype(torch::kInt32).device(indices.device()));
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateConvIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          kernelSize32, stride32, padding32, dilation32, outSpatialShape32,
          transpose, true);
      gridOut.fill_(-1);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      auto getIndicePairFtorP1 =
          functor::CreateConvIndicePairFunctorP1<tv::GPU, int, int, NDim>();
      auto getIndicePairFtorP2 =
          functor::CreateConvIndicePairFunctorP2<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtorP1(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          tv::torch2tv<int>(indicePairUnique), kernelSize32, stride32,
          padding32, dilation32, outSpatialShape32, transpose);
      if (numActOut > 0) {
        auto res = torch::_unique(indicePairUnique);
        indicePairUnique = std::get<0>(res);
        numActOut = getIndicePairFtorP2(
            tv::TorchGPU(), tv::torch2tv<const int>(indices),
            tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
            tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
            tv::torch2tv<int>(indicePairUnique), outSpatialShape32, transpose,
            true);
      }
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}
torch::Tensor indiceConvBatch(torch::Tensor features, torch::Tensor filters,
                              torch::Tensor indicePairs,
                              torch::Tensor indiceNum, int64_t numActOut,
                              int64_t _inverse, int64_t _subM,
                              bool batchScatter);

torch::Tensor indiceConv(torch::Tensor features, torch::Tensor filters,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM,
                         int64_t algo);
std::vector<torch::Tensor>
indiceConvBackward(torch::Tensor features, torch::Tensor filters,
                   torch::Tensor outGrad, torch::Tensor indicePairs,
                   torch::Tensor indiceNum, int64_t _inverse, int64_t _subM,
                   int64_t algo);

std::vector<torch::Tensor>
indiceConvBackwardBatch(torch::Tensor features, torch::Tensor filters,
                        torch::Tensor outGrad, torch::Tensor indicePairs,
                        torch::Tensor indiceNum, int64_t _inverse,
                        int64_t _subM, bool batchScatter);
} // namespace spconv

#endif