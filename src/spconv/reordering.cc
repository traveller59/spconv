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

#include <ATen/Parallel.h>
#include <spconv/reordering.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>

namespace spconv {
using float_types_t = tv::mp_list<float, double, at::Half>;
using int_types_t = tv::mp_list<int32_t, int64_t>;
void sparse_gather_cpu(torch::Tensor buffer, torch::Tensor features,
                       torch::Tensor indices, int size) {
  int numPlanes = features.size(1);
  auto dtype = features.scalar_type();
  auto int_dtype = indices.scalar_type();
  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    tv::DispatchTorch<int_types_t>()(int_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      Index *indices_data = indices.data_ptr<Index>();
      T *buffer_data = buffer.data_ptr<T>();
      const T *features_data = features.data_ptr<T>();
      at::parallel_for(0, size, 0, [&](int64_t begin, int64_t end) {
        for (int i = begin; i < end; ++i) {
          std::memcpy(buffer_data + i * numPlanes,
                      features_data + indices_data[i] * numPlanes,
                      sizeof(T) * numPlanes);
        }
      });
    });
  });
}

void sparse_scatter_add_cpu(torch::Tensor buffer, torch::Tensor outFeatures,
                            torch::Tensor indices, int size) {
  int numPlanes = outFeatures.size(1);
  auto dtype = outFeatures.scalar_type();
  auto int_dtype = indices.scalar_type();

  tv::DispatchTorch<float_types_t>()(dtype, [&](auto TValue) {
    using T = TV_DECLTYPE(TValue);
    tv::DispatchTorch<int_types_t>()(int_dtype, [&](auto IndexValue) {
      using Index = TV_DECLTYPE(IndexValue);
      Index *indices_data = indices.data_ptr<Index>();
      const T *buffer_data = buffer.data_ptr<T>();
      T *features_data = outFeatures.data_ptr<T>();
      at::parallel_for(0, size, 0, [&](int64_t begin, int64_t end) {
        const T *buf = buffer.data_ptr<T>();
        T *out = outFeatures.data_ptr<T>();
        for (int i = begin; i < end; ++i) {
          buf = buffer_data + i * numPlanes;
          out = features_data + indices_data[i] * numPlanes;
          for (int j = 0; j < numPlanes; ++j) {
            out[j] += buf[j];
          }
        }
      });
    });
  });
}

} // namespace spconv
