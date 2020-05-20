import torch
from torch.autograd import Function

import spconv
#from torch.nn import Module
from spconv.modules import SparseModule


class JoinTable(SparseModule):  # Module):
    def forward(self, input):
        output = spconv.SparseConvTensor(
            torch.cat([i.features for i in input], 1), input[1].indices,
            input[1].spatial_shape, input[0].batch_size)
        output.indice_dict = input[1].indice_dict
        output.grid = input[1].grid
        return output

    def input_spatial_size(self, out_size):
        return out_size


class AddTable(SparseModule):  # Module):
    def forward(self, input):
        output = spconv.SparseConvTensor(sum([i.features for i in input]),
                                         input[1].indices,
                                         input[1].spatial_shape,
                                         input[1].batch_size)
        output.indice_dict = input[1].indice_dict
        output.grid = input[1].grid

        return output

    def input_spatial_size(self, out_size):
        return out_size


class ConcatTable(SparseModule):  # Module):
    def forward(self, input):
        return [module(input) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self

    def input_spatial_size(self, out_size):
        return self._modules['0'].input_spatial_size(out_size)
