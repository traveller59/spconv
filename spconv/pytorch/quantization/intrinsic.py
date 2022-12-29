import torch
from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, Linear, BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.utils.parametrize import type_before_parametrizations
import torch.ao.nn.intrinsic as nni

from spconv.pytorch.conv import SparseConvolution


class SpconvReLUNd(nni._FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert isinstance(conv, SparseConvolution) and isinstance(relu, ReLU), \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)

class SpconvBnNd(nni._FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert isinstance(conv, SparseConvolution) and isinstance(bn, BatchNorm1d), \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super().__init__(conv, bn)


class SpconvBnReLUNd(nni._FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert isinstance(conv, SparseConvolution) and isinstance(bn, BatchNorm1d) and \
            isinstance(relu, ReLU), 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)
