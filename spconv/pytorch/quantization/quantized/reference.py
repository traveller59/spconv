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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from torch.nn.common_types import _size_1_t
from torch.ao.nn.quantized.reference.modules.utils import ReferenceQuantizedModule
from typing import List, Optional, Tuple, Union
from spconv.core import ConvAlgo
from cumm import tensorview as tv
from spconv.pytorch.core import SparseConvTensor

import spconv.pytorch.conv as sconvmod

class _SpConvNd(sconvmod.SparseConvolution, ReferenceQuantizedModule):
    """ A reference version of nn.quantized.Conv2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    __annotations__ = {"bias": Optional[torch.Tensor]}
    _IS_REFERENCE = True

    # @staticmethod
    # def from_float(cls, float_conv, weight_qparams):
    #     qref_conv = cls(
    #         float_conv.in_channels,
    #         float_conv.out_channels,
    #         float_conv.kernel_size,  # type: ignore[arg-type]
    #         float_conv.stride,  # type: ignore[arg-type]
    #         float_conv.padding,  # type: ignore[arg-type]
    #         float_conv.dilation,  # type: ignore[arg-type]
    #         float_conv.groups,
    #         float_conv.bias is not None,  # type: ignore[arg-type]
    #         float_conv.padding_mode,
    #         device=float_conv.weight.device,
    #         dtype=float_conv.weight.dtype,
    #         weight_qparams=weight_qparams)
    #     qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
    #     if float_conv.bias is not None:
    #         qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
    #     return qref_conv

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        conv: sconvmod.SparseConvolution = float_conv
        qref_conv = cls(conv.ndim, conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, 
                         conv.bias is not None,
                         subm=conv.subm,
                         output_padding=conv.output_padding,
                         transposed=conv.transposed,
                         inverse=conv.inverse,
                         indice_key=conv.indice_key,
                         algo=conv.algo,
                         fp32_accum=conv.fp32_accum,
                         record_voxel_count=conv.record_voxel_count,
                         act_type=conv.act_type,
                         act_alpha=conv.act_alpha,
                         act_beta=conv.act_beta,
                         name=conv.name,
                        device=float_conv.weight.device,
                        dtype=float_conv.weight.dtype,
                        weight_qparams=weight_qparams)
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        if conv.get_max_num_voxels() is not None:
            qref_conv.get_max_num_voxels()[:] = conv.get_max_num_voxels()
        return qref_conv


class SpConv(_SpConvNd, sconvmod.SparseConvolution):
    def __init__(self,
                 ndim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 subm: bool = False,
                 output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 transposed: bool = False,
                 inverse: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                 act_alpha: float = 0,
                 act_beta: float = 0,
                 name=None,
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        sconvmod.SparseConvolution.__init__(self, ndim, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
            bias=False,
            subm=subm,
            output_padding=output_padding,
            transposed=transposed,
            inverse=inverse,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            record_voxel_count=record_voxel_count,
            act_type=act_type,
            act_alpha=act_alpha,
            act_beta=act_beta,
            name=name,
            dtype=dtype,
            device=device)
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: SparseConvTensor, add_input: Optional[SparseConvTensor] = None) -> SparseConvTensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- SparseConvolution ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *SparseConvolution --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized SparseConvolution
        """
        weight_quant_dequant = self.get_weight()
        result = self._conv_forward(self.training, x, weight_quant_dequant, self.bias, add_input=add_input)
        return result

    def _get_name(self):
        return "QuantizedSparseConv(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return _SpConvNd.from_float(cls, float_conv, weight_qparams)
