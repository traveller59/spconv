#include <spconv/spconv_ops.h>
namespace spconv {

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
  tv::torch_dispatch<float, double, at::Half>(
      features.scalar_type(), [&](auto I) {
        using T = decltype(I);
        for (int i = 0; i < kernelVolume; ++i) {
          auto nHot = indicePairNumCpu.data_ptr<int>()[i];
          if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
            continue;
          }
          // auto timer = spconv::CudaContextTimer<>();
          auto outputBufferBlob = torch::from_blob(
              outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
          auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(),
                                                  {nHot, numInPlanes}, options);

          if (device == torch::kCPU) {
            functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
            gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                       tv::torch2tv<const T>(features),
                       tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                       nHot);
          }
#ifdef SPCONV_CUDA
          else if (device == torch::kCUDA) {
            functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
            gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBuffer),
                       tv::torch2tv<const T>(features),
                       tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                       nHot);
            TV_CHECK_CUDA_ERR();
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
            functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
            scatterFtor(
                tv::CPU(), tv::torch2tv<T>(output),
                tv::torch2tv<const T>(outputBuffer),
                tv::torch2tv<const int>(indicePairs).subview(i, !inverse), nHot,
                true);
          }
#ifdef SPCONV_CUDA
          else if (device == torch::kCUDA) {
            functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
            scatterFtor(
                tv::TorchGPU(), tv::torch2tv<T>(output),
                tv::torch2tv<const T>(outputBuffer),
                tv::torch2tv<const int>(indicePairs).subview(i, !inverse), nHot,
                true);
            TV_CHECK_CUDA_ERR();
          }
#endif
          else {
            TV_ASSERT_INVALID_ARG(false, "unknown device type");
          }
          // totalSAddTime += timer.report() / 1000.0;
        }
      });
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
  tv::torch_dispatch<float, double,
                     at::Half>(features.scalar_type(), [&](auto I) {
    using T = decltype(I);
    for (int i = 0; i < kernelVolume; ++i) {
      auto nHot = indicePairNumCpu.data_ptr<int>()[i];
      if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
        continue;
      }
      if (device == torch::kCPU) {
        functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
        functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtorOut;
        gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                   nHot);
        gatherFtorOut(tv::CPU(), tv::torch2tv<T>(outputBuffer),
                      tv::torch2tv<const T>(outGrad),
                      tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                      nHot);
      }
#ifdef SPCONV_CUDA
      else if (device == torch::kCUDA) {
        functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
        functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtorOut;
        gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBuffer),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                   nHot);
        TV_CHECK_CUDA_ERR();
        gatherFtorOut(tv::TorchGPU(), tv::torch2tv<T>(outputBuffer),
                      tv::torch2tv<const T>(outGrad),
                      tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                      nHot);
        TV_CHECK_CUDA_ERR();
      }
#endif
      else {
        TV_ASSERT_INVALID_ARG(false, "unknown device type");
      }

      auto filterGradSub = filtersGrad[i];
      auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(),
                                               {nHot, numOutPlanes}, options);
      auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(),
                                              {nHot, numInPlanes}, options);

      torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
      torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());
      if (device == torch::kCPU) {
        functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
        scatterFtor(tv::CPU(), tv::torch2tv<T>(inputGrad),
                    tv::torch2tv<const T>(inputBuffer),
                    tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                    nHot);
      }
#ifdef SPCONV_CUDA
      else if (device == torch::kCUDA) {
        functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
        scatterFtor(tv::TorchGPU(), tv::torch2tv<T>(inputGrad),
                    tv::torch2tv<const T>(inputBuffer),
                    tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                    nHot);
        TV_CHECK_CUDA_ERR();
      }
#endif
      else {
        TV_ASSERT_INVALID_ARG(false, "unknown device type");
      }
    }
  });
  return {inputGrad, filtersGrad.view(filterShape)};
}

} // namespace spconv