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

import torch
from torch.autograd import Function

#from torch.nn import Module
from spconv.pytorch.modules import SparseModule
from spconv.pytorch.core import SparseConvTensor
from typing import List
from spconv.pytorch import functional as F


class JoinTable(SparseModule):
    def forward(self, input: List[SparseConvTensor]):
        msg = "you can't use JoinTable in two sptensor with different indices."

        for ten in input:
            assert ten.spatial_shape == input[0].spatial_shape, msg
            assert ten.batch_size == input[0].batch_size, msg
            assert ten.features.shape[1] == input[0].features.shape[1], msg
            assert ten.indices.shape[0] == input[0].indices.shape[0], msg
        output = SparseConvTensor(torch.cat([i.features for i in input], 1),
                                  input[0].indices, input[0].spatial_shape,
                                  input[0].batch_size, input[0].grid,
                                  input[0].voxel_num, input[0].indice_dict)
        output.benchmark_record = input[1].benchmark_record
        output.thrust_allocator = input[1].thrust_allocator
        output._timer = input[1]._timer

        return output

    def input_spatial_size(self, out_size):
        return out_size


class AddTable(SparseModule): 
    def forward(self, input: List[SparseConvTensor]):
        msg = "you can't use AddTable in two sptensor with different indices. use AddTableMisaligned instead."
        for ten in input:
            assert ten.spatial_shape == input[0].spatial_shape, msg
            assert ten.batch_size == input[0].batch_size, msg
            assert ten.features.shape[1] == input[0].features.shape[1], msg
            assert ten.indices.shape[0] == input[0].indices.shape[0], msg

        output = SparseConvTensor(sum([i.features for i in input]),
                                  input[0].indices, input[0].spatial_shape,
                                  input[0].batch_size, input[0].grid,
                                  input[0].voxel_num, input[0].indice_dict)
        output.benchmark_record = input[1].benchmark_record
        output.thrust_allocator = input[1].thrust_allocator
        output._timer = input[1]._timer

        return output

    def input_spatial_size(self, out_size):
        return out_size

class AddTableMisaligned(SparseModule):
    """add sptensors with same shape but different indices.
    slower than AddTable.
    WARNING: you shouldn't use this in segmentation network such as U-Net
    because add misaligned tensors will clear downsample indices and make 
    SparseInverseConvXd not working.
    """
    def forward(self, input: List[SparseConvTensor]):
        return F.sparse_add_hash_based(*input)

    def input_spatial_size(self, out_size):
        return out_size

class ConcatTable(SparseModule):
    def forward(self, input):
        return [module(input) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self

    def input_spatial_size(self, out_size):
        return self._modules['0'].input_spatial_size(out_size)
