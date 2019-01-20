# Copyright 2019 Yan Yan
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

from pathlib import Path

import numpy as np
import torch
from spconv import utils
from spconv.conv import SparseConv2d, SparseConv3d, SubMConv2d, SubMConv3d
from spconv.conv import SparseConvTranspose2d, SparseConvTranspose3d
from spconv.conv import SparseInverseConv2d, SparseInverseConv3d
from spconv.modules import SparseModule, SparseSequential
from spconv.pool import SparseMaxPool2d, SparseMaxPool3d

_LIB_PATH = str(Path(__file__).parent / "libspconv.so")
torch.ops.load_library(_LIB_PATH)

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

class SparseConvTensor(object):
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
        """
        Args:
            grid: pre-allocated grid tensor. should be used when the volume of spatial shape
                is very large.
        """
        self.features = features
        self.indices = indices 
        if self.indices.dtype != torch.int32:
            self.indices.int()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key):
        if key is None:
            return None 
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first=True):
        output_shape = [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(self.indices.long(), self.features, output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor.
    """
    def forward(self, x: SparseConvTensor):
        return x.dense()

class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer.
    """
    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x