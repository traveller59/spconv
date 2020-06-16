#include <ATen/ATen.h>
#include <spconv/fused_conv.cu.h>
#include <spconv/fused_conv.h>
#include <tensorview/torch_utils.h>
namespace spconv {
void fused_conv_cuda(torch::Tensor output, torch::Tensor features,
                     torch::Tensor filters, torch::Tensor indicesIn,
                     torch::Tensor indicesOut, int nHot) {
  auto dtype = output.scalar_type();
  auto input_nPlanes = features.size(1);
  auto output_nPlanes = output.size(1);

  auto stream = at::cuda::getCurrentCUDAStream();

  tv::dispatch_torch<float, at::Half>(dtype, [&](auto I) {
    using T = decltype(I);
    dConvolution_forward2(stream, features.data_ptr<T>(), output.data_ptr<T>(),
                          filters.data_ptr<T>(), indicesIn.data_ptr<int32_t>(),
                          indicesOut.data_ptr<int32_t>(), nHot, input_nPlanes,
                          input_nPlanes, output_nPlanes, output_nPlanes, 1);
  });
}

void fused_conv_backward_cuda(torch::Tensor features, torch::Tensor din,
                              torch::Tensor dout, torch::Tensor filters,
                              torch::Tensor dfilters, torch::Tensor indicesIn,
                              torch::Tensor indicesOut, int nHot) {
  auto dtype = features.scalar_type();
  auto input_nPlanes = features.size(1);
  auto output_nPlanes = dout.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  tv::dispatch_torch<float>(dtype, [&](auto I) {
    using T = decltype(I);
    dConvolution_backward_dW2(
        stream, features.data_ptr<T>(), din.data_ptr<T>(), dout.data_ptr<T>(),
        filters.data_ptr<T>(), dfilters.data_ptr<T>(),
        indicesIn.data_ptr<int32_t>(), indicesOut.data_ptr<int32_t>(), nHot,
        input_nPlanes, input_nPlanes, output_nPlanes, output_nPlanes, 1);
  });
}

} // namespace spconv