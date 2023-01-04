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

from typing import Optional
from spconv.pytorch.core import SparseConvTensor
import spconv.pytorch.quantization.quantized as nnq
from spconv.pytorch.quantization.intrinsic import SpconvReLUNd, SpconvAddReLUNd
from cumm import tensorview as tv 
from spconv.pytorch.quantization.utils import fuse_spconv_bn_weights

import spconv.pytorch.quantization.intrinsic.qat as snniqat
import spconv.pytorch.quantization.intrinsic as snni
import torch

__all__ = ["SparseConvReLU", "SparseConvAddReLU"]

class SparseConvReLU(nnq.SparseConv):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """
    _FLOAT_MODULE = SpconvReLUNd  # type: ignore[assignment]

    def forward(self, input):
        inp_scale = input.q_scale()
        w_scales = self.weight().q_per_channel_scales().to(torch.float32)
        out_scale = self.scale 
        channel_scale = (inp_scale * w_scales) / out_scale
        scaled_bias = self.bias() / out_scale
        # print(bias.dtype, input.features.dtype, channel_scale.dtype, w_scales.dtype)
        res = self._conv_forward(False, input, 
            self.weight(), scaled_bias, channel_scale=channel_scale, output_scale=out_scale,
            act_type=tv.gemm.Activation.ReLU)
        return res 

    def _get_name(self):
        return 'QuantizedSparseConvReLU'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == snniqat.SparseConvBnReLU:
            mod.weight, mod.bias = fuse_spconv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        return super(SparseConvReLU, cls).from_float(mod)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) != snni.SpconvBnReLUNd, \
            "BatchNorm1d should be fused into Conv1d before converting to reference module"
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class SparseConvAddReLU(nnq.SparseConv):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """
    _FLOAT_MODULE = SpconvAddReLUNd  # type: ignore[assignment]

    def forward(self, input, add_input: Optional[SparseConvTensor] = None):
        inp_scale = input.q_scale()
        w_scales = self.weight().q_per_channel_scales().to(torch.float32)
        out_scale = self.scale 
        channel_scale = (inp_scale * w_scales) / out_scale
        scaled_bias = self.bias() / out_scale
        # print(bias.dtype, input.features.dtype, channel_scale.dtype, w_scales.dtype)
        res = self._conv_forward(False, input, 
            self.weight(), scaled_bias, channel_scale=channel_scale, output_scale=out_scale,
            act_type=tv.gemm.Activation.ReLU, add_input=add_input)
        return res 

    def _get_name(self):
        return 'QuantizedSparseConvReLU'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == snniqat.SparseConvBnAddReLU:
            mod.weight, mod.bias = fuse_spconv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        return super(SparseConvAddReLU, cls).from_float(mod)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) != snni.SpconvBnReLUNd, \
            "BatchNorm1d should be fused into Conv1d before converting to reference module"
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
