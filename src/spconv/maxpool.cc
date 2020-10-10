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

#include <spconv/maxpool.h>
#include <torch/script.h>

namespace spconv {

using float_types_t = tv::mp_list<float, double, at::Half>;
using int_types_t = tv::mp_list<int32_t, int64_t>;

void maxpool_fwd_cpu(torch::Tensor outFeatures, torch::Tensor inFeatures,
                     torch::Tensor indicesIn, torch::Tensor indicesOut,
                     int size) {
  if (size <= 0)
    return;
  int stride = inFeatures.size(1);
  auto dtype = inFeatures.scalar_type();
  auto int_dtype = indicesIn.scalar_type();
  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    tv::DispatchTorch<int_types_t>()(int_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      auto outFeaturesData = outFeatures.data_ptr<T>();
      auto inFeaturesData = inFeatures.data_ptr<T>();
      auto indicesInData = indicesIn.data_ptr<Index>();
      auto indicesOutData = indicesOut.data_ptr<Index>();
      Index idxi, idxo;
      for (int row = 0; row < size; row++) {
        idxi = indicesInData[row] * stride;
        idxo = indicesOutData[row] * stride;
        for (int plane = 0; plane < stride; ++plane)
          if (outFeaturesData[idxo + plane] < inFeaturesData[idxi + plane])
            outFeaturesData[idxo + plane] = inFeaturesData[idxi + plane];
      }
    });
  });
}

void maxpool_bwd_cpu(torch::Tensor outFeatures, torch::Tensor inFeatures,
                     torch::Tensor dout, torch::Tensor din,
                     torch::Tensor indicesIn, torch::Tensor indicesOut,
                     int size) {
  if (size <= 0)
    return;
  int stride = inFeatures.size(1);
  auto dtype = inFeatures.scalar_type();
  auto int_dtype = indicesIn.scalar_type();
  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    tv::DispatchTorch<int_types_t>()(int_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      auto outFeaturesData = outFeatures.data_ptr<T>();
      auto inFeaturesData = inFeatures.data_ptr<T>();
      auto doutData = dout.data_ptr<T>();
      auto dinData = din.data_ptr<T>();
      auto indicesInData = indicesIn.data_ptr<Index>();
      auto indicesOutData = indicesOut.data_ptr<Index>();
      Index idxi, idxo;
      for (int row = 0; row < size; row++) {
        idxi = indicesInData[row] * stride;
        idxo = indicesOutData[row] * stride;
        for (int plane = 0; plane < stride; ++plane)
          if (outFeaturesData[idxo + plane] == inFeaturesData[idxi + plane])
            dinData[idxi + plane] += doutData[idxo + plane];
      }
    });
  });
}

} // namespace spconv
