#include <spconv/spconv_ops.h>
namespace spconv {

std::vector<torch::Tensor>
getIndicePairV2(torch::Tensor indices, int64_t batchSize,
                std::vector<int64_t> outSpatialShape,
                std::vector<int64_t> spatialShape,
                std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                std::vector<int64_t> padding, std::vector<int64_t> dilation,
                std::vector<int64_t> outPadding, int64_t _subM,
                int64_t _transpose, int64_t _useHash) {
  // auto timer = spconv::CudaContextTimer<>();
  bool subM = _subM != 0;
  bool transpose = _transpose != 0;
  auto NDim = kernelSize.size();
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
      torch::full({kernelVolume, 2, numAct}, -1,
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
  for (int i = 0; i < NDim; ++i) {
    if (subM) {
      padding[i] = kernelSize[i] / 2;
      stride[i] = 1;
    }
  }
  if (subM) {
    if (indices.device().type() == torch::kCPU) {
      numActOut = create_submconv_indice_pair_cpu(
          indices, gridOut, indicePairs, indiceNum, kernelSize, stride, padding,
          dilation, outSpatialShape, transpose, false, useHash);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      numActOut = create_submconv_indice_pair_cuda(
          indices, gridOut, indicePairs, indiceNum, kernelSize, stride, padding,
          dilation, outSpatialShape, transpose, false, useHash);
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
      numActOut = create_conv_indice_pair_cpu(
          indices, outInds, gridOut, indicePairs, indiceNum, kernelSize, stride,
          padding, dilation, outSpatialShape, transpose, false, useHash);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      numActOut = create_conv_indice_pair_p1_cuda(
          indices, indicePairs, indiceNum, indicePairUnique, kernelSize, stride,
          padding, dilation, outSpatialShape, transpose);
      if (numActOut > 0) {
        auto res = torch::_unique(indicePairUnique);
        indicePairUnique = std::get<0>(res);
        numActOut = create_conv_indice_pair_p2_cuda(
            indices, outInds, gridOut, indicePairs, indiceNum, indicePairUnique,
            outSpatialShape, transpose, false, useHash);
      }
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}

torch::Tensor indiceConv(torch::Tensor features, torch::Tensor filters,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM) {
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  int indicePairMaxOffset =
      indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
  int indicePairMaxSize = *indicePairMaxSizeIter;

  /*if (_subM){
  std::vector<int> indicePairNumVec(indicePairNumCpu.data_ptr<int>(),
  indicePairNumCpu.data_ptr<int>() + kernelVolume);
  indicePairNumVec.erase(indicePairNumVec.begin() + indicePairMaxOffset);

  auto indicePairVecMaxSizeIter = std::max_element(
      indicePairNumVec.begin(), indicePairNumVec.end());
  indicePairMaxSize = *indicePairVecMaxSizeIter;
  }*/

  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  // auto indicePairOptions =
  //     torch::TensorOptions().dtype(torch::kInt64).device(indicePairs.device());

  torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
  torch::Tensor inputBuffer =
      torch::zeros({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::zeros({indicePairMaxSize, numOutPlanes}, options);
  filters = filters.view({-1, numInPlanes, numOutPlanes});
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    torch::mm_out(output, features, filters[indicePairMaxOffset]);
  }
  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    auto outputBufferBlob = torch::from_blob(
        outputBuffer.data_ptr(), {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr(),
                                            {nHot, numInPlanes}, options);

    if (device == torch::kCPU) {
      sparse_gather_cpu(inputBuffer, features, indicePairs[i][inverse], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[i][inverse], nHot);
      /* slower than SparseGatherFunctor, may due to int->long conversion
      auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
      auto indicePairBlob = torch::from_blob(indicePairLong.data<long>(),
      {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
      features, 0, indicePairBlob);*/
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }

    // totalGatherTime += timer.report() / 1000.0;
    torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
    // totalGEMMTime += timer.report() / 1000.0;

    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(outputBuffer, output, indicePairs[i][!inverse], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(outputBuffer, output, indicePairs[i][!inverse], nHot);
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    // totalSAddTime += timer.report() / 1000.0;
  }

  // std::cout << "gather time " << totalGatherTime << std::endl;
  // std::cout << "gemm time " << totalGEMMTime << std::endl;
  // std::cout << "scatteradd time " << totalSAddTime << std::endl;
  return output;
}


std::vector<torch::Tensor>
indiceConvBackward(torch::Tensor features, torch::Tensor filters,
                   torch::Tensor outGrad, torch::Tensor indicePairs,
                   torch::Tensor indiceNum, int64_t _inverse, int64_t _subM) {
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;

  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  int indicePairMaxOffset =
      indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
  int indicePairMaxSize = *indicePairMaxSizeIter;
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  auto filterShape = filters.sizes();
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  torch::Tensor filtersGrad = torch::zeros(filterShape, options);
  torch::Tensor inputBuffer =
      torch::zeros({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::zeros({indicePairMaxSize, numOutPlanes}, options);

  filters = filters.view({-1, numInPlanes, numOutPlanes});
  filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});
  if (subM) {
    auto filterGradSub = filtersGrad[indicePairMaxOffset];
    torch::mm_out(filterGradSub, features.t(), outGrad);
    torch::mm_out(inputGrad, outGrad, filters[indicePairMaxOffset].t());
  }
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    if (device == torch::kCPU) {
      sparse_gather_cpu(inputBuffer, features, indicePairs[i][inverse], nHot);
      sparse_gather_cpu(outputBuffer, outGrad, indicePairs[i][!inverse], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[i][inverse], nHot);
      sparse_gather_cuda(outputBuffer, outGrad, indicePairs[i][!inverse], nHot);
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }

    auto filterGradSub = filtersGrad[i];
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                              {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr(),
                                            {nHot, numInPlanes}, options);

    torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
    torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());
    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(inputBuffer, inputGrad, indicePairs[i][inverse], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(inputBuffer, inputGrad, indicePairs[i][inverse], nHot);
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
  }
  return {inputGrad, filtersGrad.view(filterShape)};
}


} // namespace spconv