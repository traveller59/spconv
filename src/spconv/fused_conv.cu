// Copyright 2019-2020 Yan Yan
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
#include <spconv/fused_conv.cu.h>
#include <spconv/fused_conv.h>
#include <tensorview/torch_utils.h>
#include <spconv/minkowski.cu.h>

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

void fused_conv_cuda_minkowski(torch::Tensor output, torch::Tensor features,
                     torch::Tensor filters, torch::Tensor indicesIn,
                     torch::Tensor indicesOut, int nHot) {
  auto dtype = output.scalar_type();
  auto in_nchannel = features.size(1);
  auto out_nchannel = output.size(1);
  int shared_mem_size = -1;
  if ((in_nchannel > 16 && out_nchannel > 16 &&
       in_nchannel * out_nchannel >= 512) ||
      (in_nchannel > 24 && out_nchannel > 24))
    shared_mem_size = 32;
  else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    shared_mem_size = 24;
  else if ((in_nchannel > 8 && out_nchannel > 8) ||
           (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
    shared_mem_size = 16;
  else
    shared_mem_size = 8;
  constexpr int MAX_GRID = 65535;
  auto stream = at::cuda::getCurrentCUDAStream();
  using shmem_sizes_t = tv::mp_list_c<int, 32, 24, 16, 8>;
  int num_grid = (nHot + shared_mem_size - 1) / shared_mem_size;
  int num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
  int step = (nHot + num_div - 1) / num_div;
  dim3 threads(shared_mem_size, shared_mem_size);


  tv::dispatch_torch<float>(dtype, [&](auto I) {
    using T = decltype(I);
    tv::DispatchInt<shmem_sizes_t>()(shared_mem_size, [&](auto ShSizeValue){
      constexpr int ShmemSize = decltype(ShSizeValue)::value;
      for (int s = 0; s < num_div; s++) {
        int remainder = nHot - step * s;
        int curr_num_active = remainder < step ? remainder : step;
        dim3 grid((out_nchannel + threads.x - 1) / threads.x,
                  (curr_num_active + threads.y - 1) / threads.y);
        matmul<T, int32_t, ShmemSize><<<grid, threads, 0, stream>>>(
            features.data_ptr<T>(), in_nchannel, curr_num_active,
            filters.data_ptr<T>(), out_nchannel,
            in_nchannel, output.data_ptr<T>(), indicesIn.data_ptr<int32_t>(),
                          indicesOut.data_ptr<int32_t>());
      }
    });
  });
}
void fused_conv_backward_cuda_minkowski(torch::Tensor features, torch::Tensor din,
                              torch::Tensor dout, torch::Tensor filters,
                              torch::Tensor dfilters, torch::Tensor indicesIn,
                              torch::Tensor indicesOut, int nHot) {
  auto dtype = features.scalar_type();
  auto in_nchannel = features.size(1);
  auto out_nchannel = dout.size(1);
  int shared_mem_size = -1;
  if ((in_nchannel > 16 && out_nchannel > 16 &&
       in_nchannel * out_nchannel >= 512) ||
      (in_nchannel % 32 == 0 && out_nchannel % 32 == 0))
    shared_mem_size = 32;
  else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    shared_mem_size = 24;
  else if ((in_nchannel > 8 && out_nchannel > 8) ||
           (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
    shared_mem_size = 16;
  else
    shared_mem_size = 8;
  dim3 threads(shared_mem_size, shared_mem_size);

  constexpr int MAX_GRID = 65535;
  auto stream = at::cuda::getCurrentCUDAStream();
  using shmem_sizes_t = tv::mp_list_c<int, 32, 24, 16, 8>;

  int num_grid = (nHot + shared_mem_size - 1) / shared_mem_size;
  int num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
  int step = (nHot + num_div - 1) / num_div;

  tv::dispatch_torch<float>(dtype, [&](auto I) {
    using T = decltype(I);
    tv::DispatchInt<shmem_sizes_t>()(shared_mem_size, [&](auto ShSizeValue){
      constexpr int ShmemSize = decltype(ShSizeValue)::value;
      for (int s = 0; s < num_div; s++) {
        int remainder = nHot - step * s;
        int curr_num_active = remainder < step ? remainder : step;
        dim3 grid((in_nchannel + threads.x - 1) / threads.x,
                  (curr_num_active + threads.y - 1) / threads.y);
        matmul2<T, int32_t, ShmemSize><<<grid, threads, 0, stream>>>(
            dout.data_ptr<T>(), out_nchannel, curr_num_active, // A
            filters.data_ptr<T>(), out_nchannel,
            in_nchannel,                                    // B
            features.data_ptr<T>(), in_nchannel, curr_num_active,        // D
            din.data_ptr<T>(),                                 // C
            dfilters.data_ptr<T>(), // E
            indicesIn.data_ptr<int32_t>(), indicesOut.data_ptr<int32_t>());
      }
    });
  });
}

} // namespace spconv