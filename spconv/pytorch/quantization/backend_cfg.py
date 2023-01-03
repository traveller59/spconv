from collections import namedtuple
from typing import List, Dict, Union, Type, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization.backend_config import (BackendConfig,
                                                  BackendPatternConfig,
                                                  DTypeConfig, ObservationType,
                                                  get_tensorrt_backend_config)
from torch.ao.quantization.fx.custom_config import (ConvertCustomConfig,
                                                    FuseCustomConfig,
                                                    PrepareCustomConfig)
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule

import spconv.pytorch.conv as sconvmod
import spconv.pytorch.quantization.intrinsic as snni
import spconv.pytorch.quantization.intrinsic.qat as snniqat
import spconv.pytorch.quantization.intrinsic.quantized as snniq
import spconv.pytorch.quantization.quantized as snnq
import spconv.pytorch.quantization.quantized.reference as snnqr
from spconv.pytorch.constants import PYTORCH_VERSION
from spconv.pytorch.quantization.fuse_mapping import (fuse_conv_bn,
                                                      fuse_conv_bn_relu)
from spconv.pytorch import ToDense

_SpConvMetadataDef = namedtuple(
    "_ConvMetadata",
    ["root", "bn", "reference",
     "fused_conv_relu", "fused_conv_bn", "fused_conv_bn_relu",
     "qat", "relu_qat", "bn_qat", "bn_relu_qat"])

_SpConvMetadatas: List[_SpConvMetadataDef] = []

for t in sconvmod.DEFAULT_SPARSE_CONV_TYPES:
    _SpConvMetadatas.append(_SpConvMetadataDef(t, nn.BatchNorm1d, 
        snnqr.SpConv, 
        snni.SpconvReLUNd, snni.SpconvBnNd, snni.SpconvBnReLUNd,
        snniqat.SparseConv, snniqat.SparseConvReLU, snniqat.SparseConvBn, snniqat.SparseConvBnReLU))

_SpConvMetadatas.append(_SpConvMetadataDef(
    sconvmod.SparseConvolution, nn.BatchNorm1d, 
    snnqr.SpConv, 
    snni.SpconvReLUNd, snni.SpconvBnNd, snni.SpconvBnReLUNd,
    snniqat.SparseConv, snniqat.SparseConvReLU, snniqat.SparseConvBn, snniqat.SparseConvBnReLU))

def _sequential_wrapper2(sequential):
    """ Given a sequential class for two modules, return a function that takes
    is_qat, and then two modules as argument, that ignores the is_qat flag
    and always returns the sequential that combines the two input modules
    """
    def fuser_method(is_qat, m1, m2):
        return sequential(m1, m2)
    return fuser_method

    # new cfg remove reverse pattern.
def _get_spconv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    if PYTORCH_VERSION <= [1, 13, 1]:
        from torch.ao.quantization.fuser_method_mappings import (
            reverse2, reverse3, reverse_sequential_wrapper2)
        for convs in _SpConvMetadatas:
            # (1) Single conv modules/functions
            # -----------------------------------
            # conv module
            conv_configs.append(
                BackendPatternConfig(convs.root)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference)
                    .set_qat_module(convs.qat))
            # conv qat module
            conv_configs.append(
                BackendPatternConfig(convs.qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))
            # (2) Conv + relu
            # -----------------
            # 2.1 conv module + relu fusion configs
            # conv relu fusion, conv module + relu module
            conv_configs.append(
                BackendPatternConfig((torch.nn.ReLU, convs.root))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_relu))
                    .set_fused_module(convs.fused_conv_relu))
            # conv relu fusion, conv module + functional relu
            conv_configs.append(
                BackendPatternConfig((F.relu, convs.root))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_relu))
                    .set_fused_module(convs.fused_conv_relu))
            # 2.2 conv module + relu fused module configs
            # conv relu, fused module
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_relu)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference)
                    .set_qat_module(convs.relu_qat))
            # conv relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.relu_qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))
            # 2.3 functional conv + relu configs
            # fused conv relu
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_relu)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_qat_module(convs.relu_qat))

            conv_configs.append(
                BackendPatternConfig(convs.relu_qat)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))

            # (3) Conv + batchnorm (+ relu)
            # -------------------------------
            # 3.1 conv bn fusion configs
            # conv + bn fusion
            conv_configs.append(
                BackendPatternConfig((convs.bn, convs.root))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(reverse2(fuse_conv_bn))
                    .set_fused_module(convs.fused_conv_bn))
            # conv + bn + relu module fusion
            conv_configs.append(
                BackendPatternConfig((nn.ReLU, (convs.bn, convs.root)))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(reverse3(fuse_conv_bn_relu))
                    .set_fused_module(convs.fused_conv_bn_relu))
            # conv + bn + relu functional fusion
            conv_configs.append(
                BackendPatternConfig((F.relu, (convs.bn, convs.root)))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_root_module(convs.root)
                    .set_fuser_method(reverse3(fuse_conv_bn_relu))
                    .set_fused_module(convs.fused_conv_bn_relu))
            # TODO: we can add fusion for torch.relu as well

            # 3.2 conv + bn (+ relu) fused module configs
            # fused conv bn
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_bn)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_qat_module(convs.bn_qat))

            # fused conv bn relu
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_bn_relu)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_qat_module(convs.bn_relu_qat))

            # conv bn, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))
            # conv bn relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_relu_qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))

            # (4) conv transpose and its fusion
            # 4.1 conv transpose config
            # conv_configs.append(
            #     BackendPatternConfig(convs.transpose)
            #         .set_dtype_configs(dtype_configs)  # noqa: E131
            #         .set_root_module(convs.transpose)
            #         .set_reference_quantized_module(convs.transpose_reference))

            # # 4.2 conv transpose + bn fusion
            # conv_configs.append(
            #     BackendPatternConfig((convs.bn, convs.transpose))
            #         .set_dtype_configs(dtype_configs)  # noqa: E131
            #         .set_fuser_method(reverse2(fuse_conv_bn))
            #         .set_root_module(convs.transpose)
            #         .set_reference_quantized_module(convs.transpose_reference))
        return conv_configs
    else:
        for convs in _SpConvMetadatas:
            # (1) Single conv modules/functions
            # -----------------------------------
            # conv module
            conv_configs.append(
                BackendPatternConfig(convs.root)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference)
                    .set_qat_module(convs.qat))
            # conv qat module
            conv_configs.append(
                BackendPatternConfig(convs.qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))

            # (2) Conv + relu
            # -----------------
            # 2.1 conv module + relu fusion configs
            # conv relu fusion, conv module + relu module
            conv_configs.append(
                BackendPatternConfig((convs.root, torch.nn.ReLU))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
                    .set_fused_module(convs.fused_conv_relu))
            # conv relu fusion, conv module + functional relu
            conv_configs.append(
                BackendPatternConfig((convs.root, F.relu))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
                    .set_fused_module(convs.fused_conv_relu))
            # 2.2 conv module + relu fused module configs
            # conv relu, fused module
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_relu)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference)
                    .set_qat_module(convs.relu_qat))
            # conv relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.relu_qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))
            # fused conv relu
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_relu)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_qat_module(convs.relu_qat))

            conv_configs.append(
                BackendPatternConfig(convs.relu_qat)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))

            # (3) Conv + batchnorm (+ relu)
            # -------------------------------
            # 3.1 conv bn fusion configs
            # conv + bn fusion
            conv_configs.append(
                BackendPatternConfig((convs.root, convs.bn))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(fuse_conv_bn)
                    .set_fused_module(convs.fused_conv_bn))
            # conv + bn + relu module fusion
            conv_configs.append(
                BackendPatternConfig((convs.root, convs.bn, nn.ReLU))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_fuser_method(fuse_conv_bn_relu)
                    .set_fused_module(convs.fused_conv_bn_relu))
            # conv + bn + relu functional fusion
            conv_configs.append(
                BackendPatternConfig((convs.root, convs.bn, F.relu))
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_root_module(convs.root)
                    .set_fuser_method(fuse_conv_bn_relu)
                    .set_fused_module(convs.fused_conv_bn_relu))
            # TODO: we can add fusion for torch.relu as well

            # 3.2 conv + bn (+ relu) fused module configs
            # fused conv bn
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_bn)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_qat_module(convs.bn_qat))

            # fused conv bn relu
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_bn_relu)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    .set_qat_module(convs.bn_relu_qat))

            # conv bn, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))
            # conv bn relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_relu_qat)
                    .set_observation_type(observation_type)  # noqa: E131
                    .set_dtype_configs(dtype_configs)
                    .set_root_module(convs.root)
                    .set_reference_quantized_module(convs.reference))

            # # (4) conv transpose and its fusion
            # # 4.1 conv transpose config
            # conv_configs.append(
            #     BackendPatternConfig(convs.transpose)
            #         .set_dtype_configs(dtype_configs)  # noqa: E131
            #         .set_root_module(convs.transpose)
            #         .set_reference_quantized_module(convs.transpose_reference))

            # # 4.2 conv transpose + bn fusion
            # conv_configs.append(
            #     BackendPatternConfig((convs.transpose, convs.bn))
            #         .set_dtype_configs(dtype_configs)  # noqa: E131
            #         .set_fuser_method(fuse_conv_bn)
            #         .set_root_module(convs.transpose)
            #         .set_reference_quantized_module(convs.transpose_reference))
        return conv_configs


weighted_op_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)
conv_dtype_configs = [
    weighted_op_qint8_dtype_config,
]
_to_dense_cfg = (BackendPatternConfig(ToDense)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT))
backend_config = get_tensorrt_backend_config() \
    .set_backend_pattern_configs(_get_spconv_configs(conv_dtype_configs) + [_to_dense_cfg])

SPCONV_STATIC_LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[WeightedQuantizedModule]]] = {
    snni.SpconvReLUNd: (snnqr.SpConv, snniq.SparseConvReLU),
}

def get_spconv_backend_config():
    return backend_config

def get_spconv_prepare_custom_config():
    cfg = PrepareCustomConfig()
    cfg.non_traceable_module_classes = [*sconvmod.DEFAULT_SPARSE_CONV_TYPES]
    return cfg 

def get_spconv_convert_custom_config():
    cfg = ConvertCustomConfig()
    cfg.set_observed_to_quantized_mapping(snni.SpconvReLUNd, snniq.SparseConvReLU)
    # cfg.set_observed_to_quantized_mapping(snni., snniq.SparseConvReLU)

    return cfg 