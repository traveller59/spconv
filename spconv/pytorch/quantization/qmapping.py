# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union, Dict, Set, Callable, Any
import spconv.pytorch.quantization.quantized as nnq

from spconv.pytorch.conv import DEFAULT_SPARSE_CONV_TYPES
import spconv.pytorch.quantization.intrinsic.qat as snniqat
import spconv.pytorch.quantization.intrinsic as snni
import spconv.pytorch.quantization.intrinsic.quantized as snniq
import spconv.pytorch.quantization.quantized as snnq


STATIC_SPCONV_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {}

for x in DEFAULT_SPARSE_CONV_TYPES:
    STATIC_SPCONV_QUANT_MODULE_MAPPINGS[x] = nnq.SparseConv

STATIC_SPCONV_QUANT_MODULE_MAPPINGS.update({
    snni.SpconvReLUNd: snniq.SparseConvReLU,
    snniqat.SparseConvBn: snnq.SparseConv,
    snniqat.SparseConvBnReLU: snniq.SparseConvReLU,
    snniqat.SparseConvReLU: snniq.SparseConvReLU,
})

SPCONV_QAT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    # nn.Conv2d: nnqat.Conv2d,
    # Intrinsic modules:
    snni.SpconvBnNd: snniqat.SparseConvBn,
    snni.SpconvBnReLUNd: snniqat.SparseConvBnReLU,
    snni.SpconvBnAddReLUNd: snniqat.SparseConvBnAddReLU,
}


def get_spconv_qat_to_static_mapping():
    return STATIC_SPCONV_QUANT_MODULE_MAPPINGS

def get_spconv_fmod_to_qat_mapping():
    return SPCONV_QAT_MODULE_MAPPINGS

