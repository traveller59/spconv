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

from typing import List, Optional, Union

import numpy as np
import torch
from spconv.core import ConvAlgo
from spconv.pytorch.constants import PYTORCH_VERSION
from spconv.tools import CUDAKernelTimer

if PYTORCH_VERSION >= [1, 8, 0]:
    try:
        import torch.fx
        if PYTORCH_VERSION >= [1, 10, 0]:
            from torch.fx import ProxyableClassMeta
        else:
            from torch.fx.symbolic_trace import ProxyableClassMeta
        SpConvTensorMeta = ProxyableClassMeta
    except:

        class SpConvTensorMeta(type):
            pass
else:

    class SpConvTensorMeta(type):
        pass


class ThrustSortAllocator:
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.alloced_objs = {}

        self.device = device

    def alloc(self, n: int):
        if n in self.alloced_objs:
            return self.alloced_objs[n].data_ptr()
        for n_cur, ten in self.alloced_objs.items():
            if n < n_cur:
                return ten.data_ptr()
        ten = torch.empty([n], dtype=torch.uint8, device=self.device)
        self.alloced_objs[n] = ten
        return ten.data_ptr()


class IndiceData(object):
    def __init__(self, out_indices, indices, indice_pairs, indice_pair_num,
                 spatial_shape, out_spatial_shape, is_subm: bool, algo: ConvAlgo):
        self.out_indices = out_indices
        self.indices = indices
        self.indice_pairs = indice_pairs
        self.indice_pair_num = indice_pair_num
        self.spatial_shape = spatial_shape
        self.out_spatial_shape = out_spatial_shape
        self.is_subm = is_subm
        self.algo = algo


class ImplicitGemmIndiceData(object):
    def __init__(self, out_indices: torch.Tensor, indices: torch.Tensor,
                 pair_fwd: torch.Tensor, pair_bwd: torch.Tensor,
                 pair_mask_fwd_splits: List[torch.Tensor],
                 pair_mask_bwd_splits: List[torch.Tensor],
                 mask_argsort_fwd_splits: List[torch.Tensor],
                 mask_argsort_bwd_splits: List[torch.Tensor],
                 masks: List[np.ndarray], spatial_shape, 
                 out_spatial_shape, is_subm: bool, algo: ConvAlgo):
        self.out_indices = out_indices
        self.indices = indices
        self.pair_fwd = pair_fwd
        self.pair_bwd = pair_bwd
        self.pair_mask_fwd_splits = pair_mask_fwd_splits
        self.pair_mask_bwd_splits = pair_mask_bwd_splits
        self.mask_argsort_fwd_splits = mask_argsort_fwd_splits
        self.mask_argsort_bwd_splits = mask_argsort_bwd_splits
        self.masks = masks
        self.spatial_shape = spatial_shape
        self.out_spatial_shape = out_spatial_shape
        self.is_subm = is_subm
        self.algo = algo


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


# ProxyableClassMeta is used for TensorRT conversion in future.
class SparseConvTensor(metaclass=SpConvTensorMeta):
    def __init__(self,
                 features: torch.Tensor,
                 indices: torch.Tensor,
                 spatial_shape: List[int],
                 batch_size: int,
                 grid: Optional[torch.Tensor] = None,
                 voxel_num: Optional[torch.Tensor] = None,
                 indice_dict: Optional[dict] = None,
                 benchmark: bool = False,
                 permanent_thrust_allocator: bool = False,
                 enable_timer: bool = False):
        """
        Args:
            features: [num_points, num_features] feature tensor
            indices: [num_points, ndim + 1] indice tensor. batch index saved in indices[:, 0]
            spatial_shape: spatial shape of your sparse data
            batch_size: batch size of your sparse data
            grid: pre-allocated grid tensor. should be used when the volume of spatial shape
                is very large.
            benchmark: whether to enable benchmark. if enabled, all sparse operators will be record to
                SparseConvTensor.
        """
        ndim = indices.shape[1] - 1
        assert features.ndim == 2
        assert indices.ndim == 2
        assert len(spatial_shape) == ndim, "spatial shape must equal to ndim"
        assert indices.dtype == torch.int32, "only support int32"
        assert batch_size > 0
        self._features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        if indice_dict is None:
            indice_dict = {}
        self.indice_dict = indice_dict
        if grid is None:
            grid = torch.Tensor()  # empty tensor
        self.grid = grid
        self.voxel_num = voxel_num  # for tensorrt
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator: Optional[ThrustSortAllocator] = None
        if permanent_thrust_allocator:
            self.thrust_allocator = ThrustSortAllocator(features.device)
        self._timer = CUDAKernelTimer(enable_timer)

    def replace_feature(self, feature: torch.Tensor):
        """we need to replace x.features = F.relu(x.features) with x = x.replace_feature(F.relu(x.features))
        due to limit of torch.fx
        """
        # assert feature.shape[0] == self.indices.shape[0], "replaced num of features not equal to indices"
        new_spt = SparseConvTensor(feature, self.indices, self.spatial_shape,
                                   self.batch_size, self.grid, self.voxel_num,
                                   self.indice_dict)
        new_spt.benchmark = self.benchmark
        new_spt.benchmark_record = self.benchmark_record
        new_spt.thrust_allocator = self.thrust_allocator
        new_spt._timer = self._timer
        return new_spt

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, val):
        msg = (
            "you can't set feature directly, use 'x = x.replace_feature(your_new_feature)'"
            " to generate new SparseConvTensor instead.")
        raise ValueError(msg)

    @classmethod
    def from_dense(cls, x: torch.Tensor):
        """create sparse tensor fron channel last dense tensor by to_sparse
        x must be NHWC tensor, channel last
        """
        x_sp = x.to_sparse(x.ndim - 1)
        spatial_shape = list(x_sp.shape[1:-1])
        batch_size = x_sp.shape[0]
        indices_th = x_sp.indices().permute(1, 0).contiguous().int()
        features_th = x_sp.values()
        return cls(features_th, indices_th, spatial_shape, batch_size)

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(
            self, key) -> Optional[Union[IndiceData, ImplicitGemmIndiceData]]:
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first: bool = True):
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

    # remove this due to limit of torch.fx
    # @property
    # def sparity(self):
    #     return self.indices.shape[0] / np.prod(
    #         self.spatial_shape) / self.batch_size

    def shadow_copy(self) -> "SparseConvTensor":
        """create a new spconv tensor with all member unchanged"""
        tensor = SparseConvTensor(self.features, self.indices,
                                  self.spatial_shape, self.batch_size,
                                  self.grid, self.voxel_num, self.indice_dict,
                                  self.benchmark)
        tensor.benchmark_record = self.benchmark_record
        tensor.thrust_allocator = self.thrust_allocator
        tensor._timer = self._timer
        return tensor
