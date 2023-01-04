import torch
from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, Linear, BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.utils.parametrize import type_before_parametrizations
import torch.ao.nn.intrinsic as nni

from spconv.pytorch.conv import SparseConvolution
from spconv.pytorch.modules import is_spconv_module
from spconv.pytorch.core import SparseConvTensor

class _FusedSparseModule(nni._FusedModule):
    def forward(self, input):
        for k, module in self._modules.items():
            if is_spconv_module(module):  # use SpConvTensor as input
                if isinstance(input, list):
                    input = module(input)
                else:
                    # assert isinstance(input, spconv.SparseConvTensor)
                    # self._sparity_dict[k] = input.sparity
                    input = module(input)
            else:
                if isinstance(input, SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input

class SpconvReLUNd(_FusedSparseModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert isinstance(conv, SparseConvolution) and isinstance(relu, ReLU), \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)

class SpconvBnNd(_FusedSparseModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert isinstance(conv, SparseConvolution) and isinstance(bn, BatchNorm1d), \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super().__init__(conv, bn)

class SpconvBnReLUNd(_FusedSparseModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert isinstance(conv, SparseConvolution) and isinstance(bn, BatchNorm1d) and \
            isinstance(relu, ReLU), 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)

class SpconvBnAddReLUNd(_FusedSparseModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert isinstance(conv, SparseConvolution) and isinstance(bn, BatchNorm1d) and \
            isinstance(relu, ReLU), 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)

    def forward(self, input, add_input):
        conv = self[0]
        bn = self[1]
        relu = self[2]
        conv_res = conv(input)
        conv_res = conv_res.replace_feature(bn(conv_res.features))
        return conv_res.replace_feature(relu(conv_res.features + add_input.features))
        
class SpconvAddReLUNd(_FusedSparseModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert isinstance(conv, SparseConvolution) and isinstance(relu, ReLU), \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)

    def forward(self, input, add_input):
        conv = self[0]
        relu = self[1]
        conv_res = conv(input)
        return conv_res.replace_feature(relu(conv_res.features + add_input.features))
        
