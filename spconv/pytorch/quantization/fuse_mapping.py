from functools import partial
from typing import Union, Callable, Tuple, Dict, Optional, Type, Any
import torch.nn as nn
import spconv.pytorch as spconv
from .utils import fuse_spconv_bn_eval
from . import intrinsic as snni
from .intrinsic.qat.modules import SparseConvBn, SparseConvBnReLU, SparseConvBnAddReLU
from spconv.pytorch.conv import DEFAULT_SPARSE_CONV_TYPES

def fuse_conv_bn(is_qat, conv, bn, is_add_fuse: bool = False):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    fuse_cls = snni.SpconvAddReLUNd if is_add_fuse else snni.SpconvBnNd
    fused_module_class_map = {
        k: fuse_cls for k in DEFAULT_SPARSE_CONV_TYPES
    }
    if is_qat:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn)))
    else:
        return fuse_spconv_bn_eval(conv, bn)

def fuse_conv_bn_relu(is_qat, conv, bn, relu, is_add_fuse: bool = False):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    fused_module : Optional[Type[spconv.SparseSequential]] = None
    if is_qat:
        fuse_cls = snni.SpconvBnAddReLUNd if is_add_fuse else snni.SpconvBnReLUNd
        map_to_fused_module_train = {
            k: fuse_cls for k in DEFAULT_SPARSE_CONV_TYPES
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn, relu)))
    else:
        fuse_cls = snni.SpconvAddReLUNd if is_add_fuse else snni.SpconvReLUNd
        map_to_fused_module_eval = {
            k: fuse_cls for k in DEFAULT_SPARSE_CONV_TYPES
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = fuse_spconv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError("Cannot fuse eval modules: {}".format((conv, bn, relu)))


def fuse_conv_bn_add_relu(is_qat, relu, add_pattern):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    _, bn_pattern, _ = add_pattern
    bn, conv = bn_pattern
    return fuse_conv_bn_relu(is_qat, conv, bn, relu, True)

