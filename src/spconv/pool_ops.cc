#include <spconv/pool_ops.h>

namespace spconv {
torch::Tensor indiceMaxPool(torch::Tensor features, torch::Tensor indicePairs,
                            torch::Tensor indiceNum, int64_t numAct) {
  auto device = features.device().type();
  auto kernelVolume = indiceNum.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      maxpool_fwd_cpu(output, features, indicePairs[0][i], indicePairs[1][i],
                      nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      maxpool_fwd_cuda(output, features, indicePairs[0][i], indicePairs[1][i],
                       nHot);
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

torch::Tensor indiceMaxPoolBackward(torch::Tensor features,
                                    torch::Tensor outFeatures,
                                    torch::Tensor outGrad,
                                    torch::Tensor indicePairs,
                                    torch::Tensor indiceNum) {
  auto device = features.device().type();
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  auto kernelVolume = indiceNum.size(0);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      maxpool_bwd_cpu(outFeatures, features, outGrad, inputGrad,
                      indicePairs[0][i], indicePairs[1][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      maxpool_bwd_cuda(outFeatures, features, outGrad, inputGrad,
                       indicePairs[0][i], indicePairs[1][i], nHot);
    }
#endif
    else {
      TV_ASSERT_INVALID_ARG(false, "unknown device type");
    }
  }
  return inputGrad;
}

} // namespace spconv