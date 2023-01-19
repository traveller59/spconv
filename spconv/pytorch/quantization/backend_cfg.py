import operator
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.backend_config import (BackendConfig,
                                                  BackendPatternConfig,
                                                  DTypeConfig, ObservationType,
                                                  get_tensorrt_backend_config)
from torch.ao.quantization.fx.custom_config import (ConvertCustomConfig,
                                                    FuseCustomConfig,
                                                    PrepareCustomConfig)
from torch.ao.quantization.fx.match_utils import MatchAllNode
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized._reference as nnqr

import spconv.pytorch.conv as sconvmod
import spconv.pytorch.quantization.intrinsic as snni
import spconv.pytorch.quantization.intrinsic.qat as snniqat
import spconv.pytorch.quantization.intrinsic.quantized as snniq
import spconv.pytorch.quantization.quantized as snnq
import spconv.pytorch.quantization.quantized.reference as snnqr
from spconv.pytorch import ToDense
from spconv.pytorch.constants import PYTORCH_VERSION
from spconv.pytorch.modules import (PrintTensorMeta, SparseBatchNorm,
                                    SparseIdentity, SparseReLU,
                                    SparseSyncBatchNorm, PrintCurrentTime)
from spconv.pytorch.pool import ALL_POOL_LAYERS
from spconv.pytorch.quantization.fuse_mapping import (fuse_conv_bn,
                                                      fuse_conv_bn_add_relu,
                                                      fuse_conv_bn_relu)



_SpConvMetadataDef = namedtuple("_ConvMetadata", [
    "root", "bn", "reference", "fused_conv_relu", "fused_conv_bn",
    "fused_conv_bn_relu", "fused_conv_add_relu", "fused_conv_bn_add_relu",
    "qat", "relu_qat", "bn_qat", "bn_relu_qat", "add_relu_qat",
    "bn_add_relu_qat"
])

_SpConvMetadatas: List[_SpConvMetadataDef] = []

for t in sconvmod.DEFAULT_SPARSE_CONV_TYPES:
    _SpConvMetadatas.append(
        _SpConvMetadataDef(t, nn.BatchNorm1d, snnqr.SpConv, snni.SpconvReLUNd,
                           snni.SpconvBnNd, snni.SpconvBnReLUNd,
                           snni.SpconvAddReLUNd, snni.SpconvBnAddReLUNd,
                           snniqat.SparseConv, snniqat.SparseConvReLU,
                           snniqat.SparseConvBn, snniqat.SparseConvBnReLU,
                           snniqat.SparseConvAddReLU,
                           snniqat.SparseConvBnAddReLU))

_SpConvMetadatas.append(
    _SpConvMetadataDef(sconvmod.SparseConvolution, nn.BatchNorm1d,
                       snnqr.SpConv, snni.SpconvReLUNd, snni.SpconvBnNd,
                       snni.SpconvBnReLUNd, snni.SpconvAddReLUNd,
                       snni.SpconvBnAddReLUNd, snniqat.SparseConv,
                       snniqat.SparseConvReLU, snniqat.SparseConvBn,
                       snniqat.SparseConvBnReLU, snniqat.SparseConvAddReLU,
                       snniqat.SparseConvBnAddReLU))


def _sequential_wrapper2(sequential):
    """ Given a sequential class for two modules, return a function that takes
    is_qat, and then two modules as argument, that ignores the is_qat flag
    and always returns the sequential that combines the two input modules
    """

    def fuser_method(is_qat, m1, m2):
        return sequential(m1, m2)

    return fuser_method

    # new cfg remove reverse pattern.


def _conv_bn_res_relu_root_node_getter(pattern):
    relu, add_pattern = pattern
    _, bn_pattern, _ = add_pattern
    bn, conv = bn_pattern
    return conv


def _conv_bn_res_relu_extra_inputs_getter(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, bn_pattern, extra_input = add_pattern
    bn, conv = bn_pattern
    return [extra_input]


def _conv_res_relu_root_node_getter(pattern):
    relu, add_pattern = pattern
    _, conv, _ = add_pattern
    return conv


def _conv_res_relu_extra_inputs_getter(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, conv, extra_input = add_pattern
    return [extra_input]


# def _get_custom_bn_linear_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
#     """
#     Return all configs related to linear modules and ops.
#     """
#     observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
#     linear_configs: List[BackendPatternConfig] = []
#     # (3) Linear + batchnorm
#     # ------------------------
#     # 3.1 linear bn fusion
#     if PYTORCH_VERSION[:2] <= [1, 13]:
#         linear_configs.append(
#             BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
#                 .set_dtype_configs(dtype_configs)  # noqa: E131
#                 .set_fuser_method(fuse_linear_bn)
#                 .set_fused_module(nni.LinearBn1d))
#     else:
#         linear_configs.append(
#             BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
#                 .set_dtype_configs(dtype_configs)  # noqa: E131
#                 .set_fuser_method(fuse_linear_bn)
#                 .set_fused_module(nni.LinearBn1d))

#     return linear_configs


def _get_bn_spconv_configs(bn_cls, dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    if PYTORCH_VERSION[:2] <= [1, 13]:
        from torch.ao.quantization.fuser_method_mappings import (
            reverse2, reverse3, reverse_sequential_wrapper2)
        for convs in _SpConvMetadatas:
            # (3) Conv + batchnorm (+ relu)
            # -------------------------------
            # 3.1 conv bn fusion configs
            # conv + bn fusion
            conv_configs.append(
                BackendPatternConfig((bn_cls, convs.root)).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_fuser_method(reverse2(fuse_conv_bn)).set_fused_module(
                    convs.fused_conv_bn))
            # conv + bn + relu module fusion
            for relu_type in [torch.nn.ReLU, F.relu, SparseReLU]:
                conv_configs.append(
                    BackendPatternConfig(
                        (relu_type, (bn_cls, convs.root))).set_dtype_configs(
                            dtype_configs)  # noqa: E131
                    .set_fuser_method(
                        reverse3(fuse_conv_bn_relu)).set_fused_module(
                            convs.fused_conv_bn_relu))

            # 5.1 fuse conv + bn + add + relu to one op
            for add_op in [torch.add, operator.add]:
                for relu_op in [SparseReLU]:
                    conv_configs.append(
                        BackendPatternConfig((relu_op, (add_op, (bn_cls, convs.root), MatchAllNode)))
                            .set_dtype_configs(dtype_configs)
                            # .set_root_module(convs.root)
                            .set_fuser_method(fuse_conv_bn_add_relu) \
                            ._set_root_node_getter(_conv_bn_res_relu_root_node_getter) \
                            ._set_extra_inputs_getter(_conv_bn_res_relu_extra_inputs_getter)
                            .set_fused_module(convs.fused_conv_bn_add_relu))

        return conv_configs
    else:
        for convs in _SpConvMetadatas:
            # (3) Conv + batchnorm (+ relu)
            # -------------------------------
            # 3.1 conv bn fusion configs
            # conv + bn fusion
            conv_configs.append(
                BackendPatternConfig(
                    (convs.root,
                     bn_cls)).set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(fuse_conv_bn).set_fused_module(
                    convs.fused_conv_bn))
            # conv + bn + relu module fusion
            for relu_type in [torch.nn.ReLU, F.relu, SparseReLU]:
                conv_configs.append(
                    BackendPatternConfig(
                        (convs.root, bn_cls, relu_type)).set_dtype_configs(
                            dtype_configs)  # noqa: E131
                    .set_fuser_method(fuse_conv_bn_relu).set_fused_module(
                        convs.fused_conv_bn_relu))
            # (5) conv add and its fusion
            # 5.1 fuse conv + bn + add + relu to one op
            for add_op in [torch.add, operator.add]:
                for relu_op in [SparseReLU]:
                    conv_configs.append(
                        BackendPatternConfig() \
                            ._set_pattern_complex_format((relu_op, (add_op, (bn_cls, convs.root), MatchAllNode)))
                            .set_dtype_configs(dtype_configs)
                            # .set_root_module(convs.root)
                            .set_fuser_method(fuse_conv_bn_add_relu) \
                            ._set_root_node_getter(_conv_bn_res_relu_root_node_getter) \
                            ._set_extra_inputs_getter(_conv_bn_res_relu_extra_inputs_getter)
                            .set_fused_module(convs.fused_conv_bn_add_relu))

        return conv_configs


def _get_spconv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    conv_configs = (_get_bn_spconv_configs(SparseBatchNorm, dtype_configs) +
                    _get_bn_spconv_configs(nn.BatchNorm1d, dtype_configs) +
                    _get_bn_spconv_configs(SparseSyncBatchNorm, dtype_configs))
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    if PYTORCH_VERSION[:2] <= [1, 13]:
        from torch.ao.quantization.fuser_method_mappings import (
            reverse2, reverse3, reverse_sequential_wrapper2)
        for convs in _SpConvMetadatas:
            # (1) Single conv modules/functions
            # -----------------------------------
            # conv module
            conv_configs.append(
                BackendPatternConfig(convs.root).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference).set_qat_module(convs.qat))
            # conv qat module
            conv_configs.append(
                BackendPatternConfig(convs.qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            # (2) Conv + relu
            # -----------------
            # 2.1 conv module + relu fusion configs
            # conv relu fusion, conv module + relu module
            for relu_type in [torch.nn.ReLU, F.relu, SparseReLU]:
                conv_configs.append(
                    BackendPatternConfig(
                        (relu_type, convs.root)).set_dtype_configs(
                            dtype_configs)  # noqa: E131
                    .set_fuser_method(
                        reverse_sequential_wrapper2(
                            convs.fused_conv_relu)).set_fused_module(
                                convs.fused_conv_relu))
            # 2.2 conv module + relu fused module configs
            # conv relu, fused module
            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_relu).set_observation_type(
                        observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference).set_qat_module(convs.relu_qat))
            # conv relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.relu_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            # 2.3 functional conv + relu configs
            # fused conv relu
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_relu).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_qat_module(convs.relu_qat))

            conv_configs.append(
                BackendPatternConfig(convs.relu_qat).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_root_module(convs.root).set_reference_quantized_module(
                    convs.reference))

            # TODO: we can add fusion for torch.relu as well

            # 3.2 conv + bn (+ relu) fused module configs
            # fused conv bn
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_bn).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_qat))

            # fused conv bn relu
            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_bn_relu).set_dtype_configs(
                        dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_relu_qat))

            # conv bn, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            # conv bn relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_relu_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))

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
            # (5) conv add and its fusion
            # 5.1 fuse conv + bn + add + relu to one op
            for add_op in [torch.add, operator.add]:
                for relu_op in [SparseReLU]:
                    conv_configs.append(
                        BackendPatternConfig((relu_op, (add_op, convs.root, MatchAllNode)))
                            .set_dtype_configs(dtype_configs)
                            # .set_root_module(convs.root)
                            .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_add_relu)) \
                            ._set_root_node_getter(_conv_res_relu_root_node_getter) \
                            ._set_extra_inputs_getter(_conv_res_relu_extra_inputs_getter)
                            .set_fused_module(convs.fused_conv_add_relu))

            # 5.2 fused add
            # fused conv bn relu
            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_add_relu).set_dtype_configs(
                        dtype_configs)  # noqa: E131
                .set_qat_module(convs.add_relu_qat))

            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_bn_add_relu).set_dtype_configs(
                        dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_add_relu_qat))

            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_add_relu).set_observation_type(
                        observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference).set_qat_module(convs.add_relu_qat))

            # conv bn, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.add_relu_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            conv_configs.append(
                BackendPatternConfig(
                    convs.bn_add_relu_qat).set_observation_type(
                        observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))

        return conv_configs
    else:
        for convs in _SpConvMetadatas:
            # (1) Single conv modules/functions
            # -----------------------------------
            # conv module
            conv_configs.append(
                BackendPatternConfig(convs.root).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference).set_qat_module(convs.qat))
            # conv qat module
            conv_configs.append(
                BackendPatternConfig(convs.qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))

            # (2) Conv + relu
            # -----------------
            # 2.1 conv module + relu fusion configs
            # conv relu fusion, conv module + relu module
            for relu_type in [torch.nn.ReLU, F.relu, SparseReLU]:
                conv_configs.append(
                    BackendPatternConfig(
                        (convs.root, relu_type)).set_dtype_configs(
                            dtype_configs)  # noqa: E131
                    .set_fuser_method(
                        _sequential_wrapper2(
                            convs.fused_conv_relu)).set_fused_module(
                                convs.fused_conv_relu))
            # 2.2 conv module + relu fused module configs
            # conv relu, fused module
            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_relu).set_observation_type(
                        observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference).set_qat_module(convs.relu_qat))
            # conv relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.relu_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            # fused conv relu
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_relu).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_qat_module(convs.relu_qat))

            conv_configs.append(
                BackendPatternConfig(convs.relu_qat).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_root_module(convs.root).set_reference_quantized_module(
                    convs.reference))

            # (3) Conv + batchnorm (+ relu)
            # -------------------------------
            # 3.1 conv bn fusion configs
            # conv + bn fusion

            # # conv + bn + relu functional fusion
            # conv_configs.append(
            #     BackendPatternConfig((convs.root, convs.bn, F.relu))
            #         .set_dtype_configs(dtype_configs)  # noqa: E131
            #         .set_root_module(convs.root)
            #         .set_fuser_method(fuse_conv_bn_relu)
            #         .set_fused_module(convs.fused_conv_bn_relu))
            # TODO: we can add fusion for torch.relu as well

            # 3.2 conv + bn (+ relu) fused module configs
            # fused conv bn
            conv_configs.append(
                BackendPatternConfig(convs.fused_conv_bn).set_dtype_configs(
                    dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_qat))

            # fused conv bn relu
            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_bn_relu).set_dtype_configs(
                        dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_relu_qat))

            # conv bn, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            # conv bn relu, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.bn_relu_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))

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
            # (5) conv add and its fusion
            # 5.1 fuse conv + bn + add + relu to one op
            for add_op in [torch.add, operator.add]:
                for relu_op in [SparseReLU]:
                    conv_configs.append(
                        BackendPatternConfig() \
                            ._set_pattern_complex_format((relu_op, (add_op, convs.root, MatchAllNode)))
                            .set_dtype_configs(dtype_configs)
                            # .set_root_module(convs.root)
                            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_add_relu)) \
                            ._set_root_node_getter(_conv_res_relu_root_node_getter) \
                            ._set_extra_inputs_getter(_conv_res_relu_extra_inputs_getter)
                            .set_fused_module(convs.fused_conv_add_relu))

            # 5.2 fused add
            # fused conv bn relu
            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_add_relu).set_dtype_configs(
                        dtype_configs)  # noqa: E131
                .set_qat_module(convs.add_relu_qat))

            conv_configs.append(
                BackendPatternConfig(
                    convs.fused_conv_bn_add_relu).set_dtype_configs(
                        dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_add_relu_qat))

            # conv bn, qat fused module
            conv_configs.append(
                BackendPatternConfig(convs.add_relu_qat).set_observation_type(
                    observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))
            conv_configs.append(
                BackendPatternConfig(
                    convs.bn_add_relu_qat).set_observation_type(
                        observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs).set_root_module(
                    convs.root).set_reference_quantized_module(
                        convs.reference))

        return conv_configs


def _get_share_observer_ops(dtype_configs):
    res: List[BackendPatternConfig] = []
    _to_dense_cfg = (BackendPatternConfig(ToDense).set_observation_type(
        ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(
            dtype_configs))
    iden_cfg = (BackendPatternConfig(SparseIdentity).set_observation_type(
        ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(
            dtype_configs))

    res.append(_to_dense_cfg)
    res.append(iden_cfg)
    res.append(BackendPatternConfig(PrintCurrentTime).set_observation_type(
        ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(
            dtype_configs))

    for p in ALL_POOL_LAYERS:
        _pool_cfg = (BackendPatternConfig(p).set_observation_type(
            ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).
                     set_dtype_configs(dtype_configs))
        res.append(_pool_cfg)
    return res


weighted_op_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

non_weighted_op_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
)

conv_dtype_configs = [
    weighted_op_qint8_dtype_config,
]


SPCONV_STATIC_LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[
    Type[nn.Module], Type[WeightedQuantizedModule]]] = {
        snni.SpconvReLUNd: (snnqr.SpConv, snniq.SparseConvReLU),
        snni.SpconvAddReLUNd: (snnqr.SpConv, snniq.SparseConvAddReLU),
        # use simple cumm i8 conv to implement linear
        # nni.LinearReLU: (nnqr.Linear, snniq.LinearPerChannelWeightReLU),

    }

SPCONV_STATIC_LOWER_MODULE_MAP: Dict[Type[nn.Module],
                                     Type[WeightedQuantizedModule]] = {
                                         snnqr.SpConv: snnq.SparseConv,
                                        #  nnqr.Linear: snnq.LinearPerChannelWeight,
                                     }


def get_spconv_backend_config(additional_bns: Optional[List[Type[nn.Module]]] = None):
    backend_config = get_tensorrt_backend_config() \
        .set_backend_pattern_configs(_get_spconv_configs(conv_dtype_configs) + _get_share_observer_ops([non_weighted_op_qint8_dtype_config]))
    if additional_bns is not None:
        for bn_type in additional_bns:
            backend_config.set_backend_pattern_configs(_get_bn_spconv_configs(bn_type, conv_dtype_configs))
    return backend_config


def get_spconv_prepare_custom_config(additional_bns: Optional[List[Type[nn.Module]]] = None):
    cfg = PrepareCustomConfig()
    cfg.non_traceable_module_classes = [*sconvmod.DEFAULT_SPARSE_CONV_TYPES]
    cfg.non_traceable_module_classes.extend(
        [SparseReLU, SparseBatchNorm, SparseSyncBatchNorm, PrintTensorMeta,
        PrintCurrentTime])
    if additional_bns is not None:
        cfg.non_traceable_module_classes.extend(additional_bns)
    return cfg


def get_spconv_convert_custom_config():
    cfg = ConvertCustomConfig()
    cfg.set_observed_to_quantized_mapping(snni.SpconvReLUNd,
                                          snniq.SparseConvReLU)
    cfg.set_observed_to_quantized_mapping(snni.SpconvAddReLUNd,
                                          snniq.SparseConvReLU)

    # cfg.set_observed_to_quantized_mapping(snni., snniq.SparseConvReLU)

    return cfg

def prepare_spconv_torch_inference(with_linear: bool):
    from torch.ao.quantization.fx._lower_to_native_backend import \
        STATIC_LOWER_FUSED_MODULE_MAP, STATIC_LOWER_MODULE_MAP
    fmap = SPCONV_STATIC_LOWER_FUSED_MODULE_MAP.copy()
    lmap = SPCONV_STATIC_LOWER_MODULE_MAP.copy()
    if with_linear:
        fmap.update({
            nni.LinearReLU: (nnqr.Linear, snniq.LinearPerChannelWeightReLU),
        })
        lmap.update({
            nnqr.Linear: snnq.LinearPerChannelWeight
        })
    STATIC_LOWER_FUSED_MODULE_MAP.update(fmap)
    STATIC_LOWER_MODULE_MAP.update(lmap)
