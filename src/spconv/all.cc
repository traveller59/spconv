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

#include <spconv/fused_spconv_ops.h>
#include <spconv/nms_ops.h>
#include <spconv/pillar_scatter_ops.h>
#include <spconv/pool_ops.h>
#include <spconv/spconv_ops.h>
#include <torch/script.h>

static auto registry =
    torch::RegisterOperators()
        .op("spconv::get_indice_pairs", &spconv::getIndicePairs)
        .op("spconv::indice_conv", &spconv::indiceConv)
        .op("spconv::indice_conv_batch", &spconv::indiceConvBatch)
        .op("spconv::indice_conv_backward", &spconv::indiceConvBackward)
        .op("spconv::fused_indice_conv_bn", &spconv::fusedIndiceConvBatchNorm)
        .op("spconv::indice_maxpool", &spconv::indiceMaxPool)
        .op("spconv::indice_maxpool_backward", &spconv::indiceMaxPoolBackward)
        .op("spconv::nms", &spconv::nonMaxSuppression<float>)
        .op("spconv::pillar_scatter_float", &spconv::pointPillarScatter<float>)
        .op("spconv::pillar_scatter_half",
            &spconv::pointPillarScatter<at::Half>);
