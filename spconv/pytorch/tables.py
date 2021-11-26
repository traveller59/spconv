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


class JoinTable(SparseModule):  # Module):
    def forward(self, input: List[SparseConvTensor]):
        output = SparseConvTensor(torch.cat([i.features for i in input], 1),
                                  input[0].indices, input[0].spatial_shape,
                                  input[0].batch_size, input[0].grid,
                                  input[0].voxel_num, input[0].indice_dict)
        output.benchmark_record = input[1].benchmark_record
        output.thrust_allocator = input[1].thrust_allocator
        return output

    def input_spatial_size(self, out_size):
        return out_size


class AddTable(SparseModule):  # Module):
    def forward(self, input: List[SparseConvTensor]):
        output = SparseConvTensor(sum([i.features for i in input]),
                                  input[0].indices, input[0].spatial_shape,
                                  input[0].batch_size, input[0].grid,
                                  input[0].voxel_num, input[0].indice_dict)
        output.benchmark_record = input[1].benchmark_record
        output.thrust_allocator = input[1].thrust_allocator
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


class AddSparseTensor2D(SparseModule):
    '''
    Add two SparseConvTensor 2D with same SHAPE but different number of FEATURES.
    - Dont handle zeros created by the additions.
    - Dont work for more than 2 sparse tensors yet.
    '''
    def forward(self, a, b):
        assert isinstance(a, SparseConvTensor), 'a must be a SparseConvTensor'
        assert isinstance(b, SparseConvTensor), 'b must be a SparseConvTensor'
        assert a.spatial_shape == b.spatial_shape, 'tensor a and b must have the same spatial shape'

        device = a.features.device

        size_a = len(a.indices)
        size_b = len(b.indices)
        list_a = a.indices.tolist()
        list_b = b.indices.tolist()

        # sorting indices and record index of features to sum.
        sorting_a = []
        sorting_b = []
        idx_to_sum_a = []
        idx_to_sum_b = []
        sorting_idx_to_sum = []
        order, i, j = 0, 0, 0
        while i < size_a and j < size_b:
            if list_a[i] == list_b[j]:
                idx_to_sum_a.append(i)
                idx_to_sum_b.append(j)
                sorting_idx_to_sum.append(order)
                i += 1
                j += 1
            elif list_a[i] < list_b[j]:
                sorting_a.append(order)
                i += 1
            else:
                sorting_b.append(order)
                j += 1
            order += 1
        for _ in range(i, size_a):
            sorting_a.append(order)
            order += 1
        for _ in range(j, size_b):
            sorting_b.append(order)
            order += 1

        sorting_a_torch = torch.tensor(sorting_a)
        sorting_b_torch = torch.tensor(sorting_b)
        sorting_idx_to_sum_torch = torch.tensor(sorting_idx_to_sum)

        # do additions
        features_added = (a.features[idx_to_sum_a] + b.features[idx_to_sum_b])
        # record indices of the corresponding added features (same for a and b)
        indices_added = a.indices[idx_to_sum_a]

        # keep features & indices that are not used for additions
        mask_a = torch.ones(a.features.size(0), dtype=bool)
        mask_a[idx_to_sum_a] = False
        a_feat_kept = a.features[mask_a]
        a_indx_kept = a.indices[mask_a]

        mask_b = torch.ones(b.features.size(0), dtype=bool)
        mask_b[idx_to_sum_b] = False
        b_feat_kept = b.features[mask_b]
        b_indx_kept = b.indices[mask_b.squeeze()]

        # concat and sort "sorting"
        sorting_cat = torch.cat((sorting_a_torch, sorting_b_torch, sorting_idx_to_sum_torch), dim=0)
        _, sort_ind = torch.sort(sorting_cat, dim=0)

        # concat and sort "features", using "sorting"
        feat_concat = torch.cat((a_feat_kept, b_feat_kept, features_added), dim=0)
        feat_concat_sort = feat_concat[sort_ind]

        indx_concat = torch.cat((a_indx_kept, b_indx_kept, indices_added), dim=0)
        indx_concat_sort = indx_concat[sort_ind]

        output = SparseConvTensor(features=feat_concat_sort,
                                  indices=indx_concat_sort,
                                  spatial_shape=a.spatial_shape,
                                  batch_size=a.batch_size,
                                  grid=a.grid,
                                  voxel_num=a.voxel_num,
                                  indice_dict=a.indice_dict)
        return output

    def input_spatial_size(self, out_size):
        return out_size
