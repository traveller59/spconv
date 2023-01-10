# Copyright 2021 Yan Yan
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

import sys
import time
from collections import OrderedDict
from typing import Union

import torch
from torch import nn

from spconv import pytorch as spconv


def is_spconv_module(module):
    spconv_modules = (SparseModule, SparseBatchNorm, SparseReLU)
    return isinstance(module, spconv_modules)


def is_sparse_conv(module):
    from spconv.pytorch.conv import SparseConvolution
    return isinstance(module, SparseConvolution)


def _mean_update(vals, m_vals, t):
    outputs = []
    if not isinstance(vals, list):
        vals = [vals]
    if not isinstance(m_vals, list):
        m_vals = [m_vals]
    for val, m_val in zip(vals, m_vals):
        output = t / float(t + 1) * m_val + 1 / float(t + 1) * val
        outputs.append(output)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


class SparseModule(nn.Module):
    """ place holder, all module subclass from this will take sptensor in SparseSequential.
    """
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self._sparse_unique_name = ""


class SparseSequential(SparseModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = SparseSequential(
                  SparseConv2d(1,20,5),
                  nn.ReLU(),
                  SparseConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = SparseSequential(OrderedDict([
                  ('conv1', SparseConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', SparseConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = SparseSequential(
                  conv1=SparseConv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=SparseConv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """
    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        # self._sparity_dict = {}

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    # @property
    # def sparity_dict(self):
    #     return self._sparity_dict

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

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
                if isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


def assign_name_for_sparse_modules(module: nn.Module):
    for k, n in module.named_modules():
        if isinstance(n, SparseModule):
            n._sparse_unique_name = k


class SparseBatchNorm(nn.BatchNorm1d):
    """this module is exists only for torch.fx transformation for quantization.
    """
    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)

class SparseSyncBatchNorm(nn.SyncBatchNorm):
    """this module is exists only for torch.fx transformation for quantization.
    """
    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)

class SparseReLU(nn.ReLU):
    """this module is exists only for torch.fx transformation for quantization.
    """
    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)


class SparseIdentity(nn.Identity):
    """this module is exists only for torch.fx transformation for quantization.
    """
    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)

class PrintTensorMeta(nn.Module):
    def forward(self, x: Union[spconv.SparseConvTensor, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            print(x.min(), x.max(), x.mean())
        elif isinstance(x, spconv.SparseConvTensor):
            ft = x.features 
            print(ft.min(), ft.max(), ft.mean())
        return x

class PrintCurrentTime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_time = time.time()

    def forward(self, x, msg="", reset: bool = False):
        if reset:
            self.first_time = time.time()
        torch.cuda.synchronize()
        print(msg, time.time() - self.first_time)
        return x
