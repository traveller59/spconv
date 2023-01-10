
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.ao.quantization import get_default_qat_qconfig, get_default_qconfig
from torch.ao.quantization.fake_quantize import (
    FakeQuantize, FixedQParamsFakeQuantize, FusedMovingAvgObsFakeQuantize,
    default_fused_per_channel_wt_fake_quant,
    default_per_channel_weight_fake_quant, default_weight_fake_quant)
from torch.ao.quantization.observer import (
    MinMaxObserver,
    HistogramObserver, MovingAverageMinMaxObserver,
    default_per_channel_weight_observer, default_placeholder_observer,
    default_weight_observer)
from torch.ao.quantization.qconfig import (QConfig, QConfigAny,
                                           default_reuse_input_qconfig)
from torch.ao.quantization.qconfig_mapping import (
    _FIXED_QPARAMS_OP_TO_OBSERVER, QConfigMapping)

from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.modules import PrintTensorMeta, PrintCurrentTime

__all__ = ["get_default_spconv_trt_ptq_qconfig", "get_default_spconv_trt_qat_qconfig"]

class SparseFusedMovingAvgObsFakeQuantize(FusedMovingAvgObsFakeQuantize):
    def forward(self, input:Union[SparseConvTensor, torch.Tensor]):
        if isinstance(input, SparseConvTensor):
            # add lines to support spconv
            x = input.features
            res_features = super().forward(x)
            return input.replace_feature(res_features)
        else:
            return super().forward(input)

class SparseMovingAvgObsFakeQuantize(FakeQuantize):
    def forward(self, input:Union[SparseConvTensor, torch.Tensor]):
        if isinstance(input, SparseConvTensor):
            # add lines to support spconv
            x = input.features
            res_features = super().forward(x)
            return input.replace_feature(res_features)
        else:
            return super().forward(input)

# class SparseMovingAvgObsFakeQuantize(FusedMovingAvgObsFakeQuantize):
#     def forward(self, input:Union[SparseConvTensor, torch.Tensor]):
#         if isinstance(input, SparseConvTensor):
#             # add lines to support spconv
#             x = input.features
#             res_features = super().forward(x)
#             return input.replace_feature(res_features)
#         else:
#             return super().forward(input)

class SparseHistogramObserver(HistogramObserver):
    def forward(self, input:Union[SparseConvTensor, torch.Tensor]):
        if isinstance(input, SparseConvTensor):
            # add lines to support spconv
            x = input.features
            res_features = super().forward(x)
            return input.replace_feature(res_features)
        else:
            return super().forward(input)

class SparseMinMaxObserver(MinMaxObserver):
    def forward(self, input:Union[SparseConvTensor, torch.Tensor]):
        if isinstance(input, SparseConvTensor):
            # add lines to support spconv
            x = input.features
            res_features = super().forward(x)
            return input.replace_feature(res_features)
        else:
            return super().forward(input)

default_symmetric_spconv_ptq_qconfig = QConfig(
    activation=SparseHistogramObserver.with_args(quant_min=-128,
                                            quant_max=127,
                                            dtype=torch.qint8,
                                            reduce_range=False,
                                            qscheme=torch.per_tensor_symmetric,
                                            eps=2 ** -12),
    weight=default_per_channel_weight_observer)

# default_symmetric_ptq_qconfig = QConfig(
#     activation=HistogramObserver.with_args(quant_min=-128,
#                                             quant_max=127,
#                                             dtype=torch.qint8,
#                                             reduce_range=False,
#                                             eps=2 ** -12),
#     weight=default_per_channel_weight_observer)

default_symmetric_spconv_qat_qconfig = QConfig(
    activation=SparseFusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       reduce_range=False,
                                                       qscheme=torch.per_tensor_symmetric,
                                                       eps=2 ** -12),
    weight=default_fused_per_channel_wt_fake_quant)


def get_default_spconv_trt_ptq_qconfig(backend, version):
    return default_symmetric_spconv_ptq_qconfig

def get_default_spconv_trt_qat_qconfig(backend, version):
    return default_symmetric_spconv_qat_qconfig

def get_default_spconv_qconfig_mapping(is_qat: bool, backend: str = "fbgemm", version: int = 0) -> QConfigMapping:
    """
    From torch.ao.quantization.qconfig_mapping
    Return the default QConfigMapping for the given quantization type and backend.
    """
    # get_default_qconfig(backend, version)
    if is_qat:
        # qconfig = get_default_qat_qconfig(backend, version)
        qconfig = get_default_spconv_trt_qat_qconfig(backend, version)
    else:
        # qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False, dtype=torch.qint8),
        #                       weight=default_per_channel_weight_observer)
        qconfig = get_default_spconv_trt_ptq_qconfig(backend, version)
    default_weight = default_weight_fake_quant if is_qat else default_weight_observer

    # default_per_channel_weight_observer is not currently compatible with fbgemm backend
    # so we have to modify the weight observer to default_weight_observer or another
    # per tensor supported observer.
    # see https://github.com/pytorch/pytorch/issues/47535
    if backend in ("fbgemm", "x86"):
        qconfig_transpose = QConfig(activation=qconfig.activation, weight=default_weight)
    else:
        qconfig_transpose = qconfig

    # currently layernorm only supports float weights
    # we have to add this because otherwise there will be a extra quantize-dequantize pair
    qconfig_layernorm = QConfig(activation=qconfig.activation, weight=default_placeholder_observer)

    qconfig_mapping = QConfigMapping() \
        .set_global(qconfig) \
        .set_object_type("reshape", default_reuse_input_qconfig) \
        .set_object_type(torch.nn.ConvTranspose1d, qconfig_transpose) \
        .set_object_type(torch.nn.ConvTranspose2d, qconfig_transpose) \
        .set_object_type(torch.nn.ConvTranspose3d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose1d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose2d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose3d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.layer_norm, qconfig_layernorm) \
        .set_object_type(torch.nn.LayerNorm, qconfig_layernorm) \

    # Use special observers for ops with fixed qparams
    fixed_qparams_observer_to_qconfig: Dict[Any, QConfigAny] = {}
    for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        if observer in fixed_qparams_observer_to_qconfig:
            fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[observer]
        else:
            if is_qat:
                activation = FixedQParamsFakeQuantize.with_args(observer=observer)
            else:
                activation = observer
            fixed_qparams_qconfig = QConfig(activation=activation, weight=default_weight)
            fixed_qparams_observer_to_qconfig[observer] = fixed_qparams_qconfig
        qconfig_mapping.set_object_type(fixed_qparams_op, fixed_qparams_qconfig)

    # QConfig for fused ops for onednn backend
    # Separate ops are required to have the same qconfig as fused ops
    # TODO: we should be able to configure qconfig for patterns
    if backend == 'onednn':
        qconfig_mapping.set_object_type(torch.nn.Linear, qconfig) \
                       .set_object_type(torch.nn.LeakyReLU, qconfig) \
                       .set_object_type(torch.nn.functional.leaky_relu, qconfig) \
                       .set_object_type(torch.nn.Tanh, qconfig) \
                       .set_object_type(torch.nn.functional.tanh, qconfig)
    qconfig_mapping.set_object_type(PrintTensorMeta, None)
    qconfig_mapping.set_object_type(PrintCurrentTime, None)

    return qconfig_mapping

