# coding=utf-8
r"""Quantized convolution modules."""

from typing import Optional, List, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spconv.pytorch.conv import SparseConvolution, SparseConvolutionBase
from typing import List, Optional, Tuple, Union
from spconv.core import ConvAlgo
from cumm import tensorview as tv
from spconv.pytorch.core import SparseConvTensor

from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single, _pair, _triple

from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule, _quantize_weight
import spconv.pytorch.quantization.intrinsic.qat.modules as snniqat 
import spconv.pytorch.quantization.intrinsic.modules as snni 
from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval, fuse_spconv_bn_weights

class _SparseConv(SparseConvolutionBase, WeightedQuantizedModule):
    _FLOAT_MODULE = SparseConvolution
    _NNIQAT_CONV_BN_MODULE = snniqat.SparseConvBn
    _NNI_CONV_RELU_MODULE = snni.SpconvReLUNd

    def __init__(self, ndim: int,
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
                 act_beta: float = 0, device=None, dtype=None):
        SparseConvolutionBase.__init__(self, ndim, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
            bias=bias,
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
            act_beta=act_beta)
        WeightedQuantizedModule.__init__(self)
        factory_kwargs = {'device': device, 'dtype': dtype}
        qweight = torch._empty_affine_quantized(
            self.weight_shape,
            scale=1, zero_point=0, dtype=torch.qint8,
            **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})
        bias_float = (
            torch.zeros(out_channels, dtype=torch.float,
                        **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}) if bias else None)

        self.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0

    def set_weight_bias(self, qweight, bias_float):
        self._weight: torch.Tensor = qweight 
        self._bias: torch.Tensor = bias_float

    def bias(self):
        return self._bias

    def _weight_bias(self):
        return (self._weight, self._bias)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias() is None:
            s += ', bias=False'
        s += f', wqscheme={self._weight_bias()[0].qscheme()}'
        return s.format(**self.__dict__)

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into
    # their regular QTensor form for serialization. Packed weights should not
    # live outside the process in which they were created, rather they should be
    # derived from the QTensor weight.
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # TODO: maybe change to this when https://github.com/pytorch/pytorch/pull/32958 is landed
    #   self
    #   |--- _packed_params : Conv2dPackedParamsBase or Conv3dPackedParamsBase
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_SparseConv, self)._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + 'weight'] = w
        destination[prefix + 'bias'] = b
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    # @torch.jit.export
    # def __getstate__(self):
    #     (w, b) = self._weight_bias()
    #     return (
    #         self.in_channels,
    #         self.out_channels,
    #         self.kernel_size,
    #         self.stride,
    #         self.padding,
    #         self.dilation,
    #         self.transposed,
    #         self.output_padding,
    #         self.groups,
    #         self.padding_mode,
    #         w,
    #         b,
    #         self.scale,
    #         self.zero_point,
    #         self.training
    #     )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.set_weight_bias(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')
        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')
        super(_SparseConv, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys,
            unexpected_keys, error_msgs)

    # @torch.jit.export
    # def __setstate__(self, state):
    #     self.in_channels = state[0]
    #     self.out_channels = state[1]
    #     self.kernel_size = state[2]
    #     self.stride = state[3]
    #     self.padding = state[4]
    #     self.dilation = state[5]
    #     self.transposed = state[6]
    #     self.output_padding = state[7]
    #     self.groups = state[8]
    #     self.padding_mode = state[9]
    #     self.set_weight_bias(state[10], state[11])
    #     self.scale = state[12]
    #     self.zero_point = state[13]
    #     self.training = state[14]

    # def __deepcopy__(self, memo):
    #     new_instance = type(self).__new__(type(self))
    #     torch.nn.Module.__init__(new_instance)
    #     state = self.__getstate__()
    #     new_instance.__setstate__(state)
    #     return new_instance

    # def __copy__(self):
    #     return self.__deepcopy__({})

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        r"""Creates a qconv object and returns it.
        """
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        assert weight_post_process.dtype == torch.qint8, \
            'Weight observer must have a dtype of qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvNd
        qconv = cls(mod.ndim, mod.in_channels, mod.out_channels, mod.kernel_size,
                         mod.stride, mod.padding, mod.dilation,
                         mod.groups, 
                         mod.bias is not None,
                         subm=mod.subm,
                         output_padding=mod.output_padding,
                         transposed=mod.transposed,
                         inverse=mod.inverse,
                         indice_key=mod.indice_key,
                         algo=mod.algo,
                         fp32_accum=mod.fp32_accum,
                         record_voxel_count=mod.record_voxel_count,
                         act_type=mod.act_type,
                         act_alpha=mod.act_alpha,
                         act_beta=mod.act_beta)
        qconv.set_weight_bias(qweight, mod.bias)
        if activation_post_process is None or activation_post_process.dtype == torch.float:
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = activation_post_process.calculate_qparams()
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_float(cls, mod):
        if hasattr(mod, "weight_fake_quant"):
            # assert type(mod) == cls.__QAT_MODULE, " nnq." + cls.__name__ + \
            # ".from_float only works for " + cls.__QAT_MODULE.__name__
            if type(mod) == cls._NNIQAT_CONV_BN_MODULE:
                mod.weight, mod.bias = fuse_spconv_bn_weights(
                    mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                    mod.bn.eps, mod.bn.weight, mod.bn.bias)
            assert hasattr(mod, "activation_post_process"), \
                "Input QAT module must have observer attached"
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == cls._FLOAT_MODULE, \
                " nnq." + cls.__name__ + ".from_float only works for " + \
                cls._FLOAT_MODULE.__name__ + " but got:" + str(type(mod))
            assert hasattr(mod, "qconfig"), \
                "Input float module must have qconfig defined."
            activation_post_process = None if not hasattr(
                mod, "activation_post_process") else mod.activation_post_process
            if type(mod) == cls._NNI_CONV_RELU_MODULE:
                mod = mod[0]
            weight_post_process = mod.qconfig.weight()
        return cls.get_qconv(mod, activation_post_process, weight_post_process)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconv (Module): a reference quantized  module, either produced by torch.ao.quantization
                                utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconv.ndim, ref_qconv.in_channels, ref_qconv.out_channels, ref_qconv.kernel_size,
                         ref_qconv.stride, ref_qconv.padding, ref_qconv.dilation,
                         ref_qconv.groups, 
                         ref_qconv.bias is not None,
                         subm=ref_qconv.subm,
                         output_padding=ref_qconv.output_padding,
                         transposed=ref_qconv.transposed,
                         inverse=ref_qconv.inverse,
                         indice_key=ref_qconv.indice_key,
                         algo=ref_qconv.algo,
                         fp32_accum=ref_qconv.fp32_accum,
                         record_voxel_count=ref_qconv.record_voxel_count,
                         act_type=ref_qconv.act_type,
                         act_alpha=ref_qconv.act_alpha,
                         act_beta=ref_qconv.act_beta,
                        device=ref_qconv.weight.device,
                        dtype=ref_qconv.weight.dtype)
        qweight = ref_qconv.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconv.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
        return qconv


class SparseConv(_SparseConv):
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
        ...                                     dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE = SparseConvolution
    _NNIQAT_CONV_BN_MODULE = snniqat.SparseConvBn
    _NNI_CONV_RELU_MODULE = snni.SpconvReLUNd

    def _get_name(self):
        return 'QuantizedSparseConvolution'

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._weight = w 
        if b is None:
            # currently bias tensor must exists.
            self._bias = torch.zeros((w.shape[0],), dtype=torch.float32, device=w.device)
        else:
            self._bias = b

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input: SparseConvTensor):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        inp_scale = input.q_scale()
        w_scales = self.weight().q_per_channel_scales().to(torch.float32)
        out_scale = self.scale 
        channel_scale = (inp_scale * w_scales) / out_scale
        bias = self.bias() / out_scale
        return self._conv_forward(False, input, 
            self.weight(), bias, channel_scale=channel_scale, output_scale=out_scale)
        return ops.quantized.conv1d(input, self._packed_params, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _SparseConv.from_float(cls, mod)


