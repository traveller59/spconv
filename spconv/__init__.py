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

import platform
from pathlib import Path

import numpy as np
import torch

from spconv import ops, utils
from spconv.conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                         SparseConvTranspose3d, SparseInverseConv2d,
                         SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.identity import Identity
from spconv.modules import SparseModule, SparseSequential
from spconv.ops import ConvAlgo
from spconv.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.tables import AddTable, ConcatTable, JoinTable

_LIB_FILE_NAME = "libspconv.so"
if platform.system() == "Windows":
    _LIB_FILE_NAME = "spconv.dll"
_LIB_PATH = str(Path(__file__).parent / _LIB_FILE_NAME)
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
    def __init__(self, features, indices, spatial_shape, batch_size,
                 grid=None):
        """
        Args:
            features: [num_points, num_features] feature tensor
            indices: [num_points, ndim + 1] indice tensor. batch index saved in indices[:, 0]
            spatial_shape: spatial shape of your sparse data
            batch_size: batch size of your sparse data
            grid: pre-allocated grid tensor. should be used when the volume of spatial shape
                is very large.
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid

    @classmethod
    def from_dense(cls, x: torch.Tensor):
        """create sparse tensor fron channel last dense tensor by to_sparse
        x must be NHWC tensor, channel last
        """
        x = x.to_sparse(x.ndim - 1)
        spatial_shape = x.shape[1:-1]
        batch_size = x.shape[0]
        indices_th = x.indices().permute(1, 0).contiguous().int()
        features_th = x.values()
        return cls(features_th, indices_th, spatial_shape, batch_size)

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
        output_shape = [self.batch_size] + list(
            self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return self.indices.shape[0] / np.prod(
            self.spatial_shape) / self.batch_size


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
