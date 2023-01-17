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
from spconv.pytorch.cppcore import get_current_stream, torch_tensor_to_tv
import spconv.pytorch.quantization.quantized as nnq
from spconv.pytorch.quantization.intrinsic import SpconvReLUNd, SpconvAddReLUNd
from cumm import tensorview as tv 
from spconv.pytorch.quantization.utils import fuse_spconv_bn_weights
import torch.ao.nn.intrinsic as nni

import spconv.pytorch.quantization.intrinsic.qat as snniqat
import spconv.pytorch.quantization.intrinsic as snni
import torch

__all__ = ["SparseConvReLU", "SparseConvAddReLU", "LinearPerChannelWeightReLU"]

class SparseConvReLU(nnq.SparseConv):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """
    _FLOAT_MODULE = SpconvReLUNd  # type: ignore[assignment]

    def forward(self, input):
        msg = f"{input.features.shape[0]}, {input.features.shape[1]}, {self.weight().shape[0]}"

        with tv.measure_and_print(f"QuantizedSparseConvReLU|{msg}", get_current_stream(), enable=False):

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
        msg = f"{input.features.shape[0]}, {input.features.shape[1]}, {self.weight().shape[0]}"
        with tv.measure_and_print(f"QuantizedSparseConvAddReLU|{msg}", get_current_stream(), enable=False):
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
        return 'QuantizedSparseConvAddReLU'

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


class LinearPerChannelWeightReLU(nnq.LinearPerChannelWeight):
    r"""
    A LinearPerChannelWeight module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, nvrtc_params = self._linear_fwd(x, self.weight(), self.bias(), self.scale, tv.gemm.Activation.ReLU, self._nvrtc_params)
        if self._nvrtc_params is None:
            self._nvrtc_params = nvrtc_params
        return out

    def _get_name(self):
        return 'QuantizedLinearPerChannelWeightReLU'

    @classmethod
    def from_float(cls, mod):
        return super(LinearPerChannelWeightReLU, cls).from_float(mod)

    @classmethod
    def from_reference(cls, ref_linear_relu, output_scale, output_zero_point):
        return super().from_reference(ref_linear_relu[0], output_scale, output_zero_point)
