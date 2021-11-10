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

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from spconv import pytorch as spconv
from spconv.pytorch.modules import SparseModule


class RemoveDuplicate(SparseModule):
    def forward(self, x: spconv.SparseConvTensor):
        inds = x.indices
        spatial_shape = [x.batch_size, *x.spatial_shape]
        spatial_stride = [0] * len(spatial_shape)
        val = 1
        for i in range(inds.shape[1] - 1, -1, -1):
            spatial_stride[i] = val
            val *= spatial_shape[i]
        indices_index = inds[:, -1]
        for i in range(len(spatial_shape) - 1):
            indices_index += spatial_stride[i] * inds[:, i]
        _, unique_inds = torch.unique(indices_index)
        new_inds = inds[unique_inds]
        new_features = x.features[unique_inds]
        res = spconv.SparseConvTensor(new_features, new_inds, x.spatial_shape,
                                      x.batch_size, x.grid)
        return res
