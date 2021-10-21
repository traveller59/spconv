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
from spconv.algo import ConvAlgo
import spconv.pytorch.functional as Fsp
from spconv.pytorch import ops
from spconv.pytorch.core import IndiceData
from spconv.pytorch.modules import SparseModule


class SparseMaxPool(SparseModule):
    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 subm=False,
                 name=None):
        super(SparseMaxPool, self).__init__(name=name)
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if stride is None:
            stride = kernel_size.copy()
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation
        self.indice_key = indice_key

    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding,
                self.dilation)
        else:
            out_spatial_shape = spatial_shape
        out_tensor = input.shadow_copy()
        if input.benchmark:
            if self.name is None:
                raise ValueError(
                    "you need to assign name to spmodules before benchmark (spconv.utils.bench.assign_name_to_spmod)"
                )
            if self.name not in input.benchmark_record:
                input.benchmark_record[self.name] = {
                    "type": "SparseMaxPool",
                    "indice_gen_time": [],
                    "time": [],
                    "num_points": [],
                    "num_out_points": [],
                    "params": {
                        "kernel_size": self.kernel_size,
                        "stride": self.stride,
                        "padding": self.padding,
                        "dilation": self.dilation,
                        "channels": features.shape[1],
                    }
                }

        if input.benchmark:
            torch.cuda.synchronize()
            t = time.time()

        outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(
            indices,
            batch_size,
            spatial_shape,
            ConvAlgo.Native,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            0,
            False)
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["indice_gen_time"].append(
                interval)
            t = time.time()

        if self.indice_key is not None:
            datas = input.find_indice_pair(self.indice_key)
            if datas is None:
                indice_data = IndiceData(outids, indices, indice_pairs,
                                         indice_pairs_num, spatial_shape)
                input.indice_dict[self.indice_key] = indice_data
            else:
                raise ValueError("indice data exists")

        out_features = Fsp.indice_maxpool(features, indice_pairs.to(device),
                                          indice_pairs_num.to(device),
                                          outids.shape[0])
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["time"].append(interval)
            out_tensor.benchmark_record[self.name]["num_points"].append(
                features.shape[0])
            out_tensor.benchmark_record[self.name]["num_out_points"].append(
                out_features.shape[0])
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.spatial_shape = out_spatial_shape
        return out_tensor


class SparseMaxPool1d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 name=None):
        super(SparseMaxPool1d, self).__init__(1,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              name=name)

class SparseMaxPool2d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 name=None):
        super(SparseMaxPool2d, self).__init__(2,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              name=name)


class SparseMaxPool3d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 name=None):
        super(SparseMaxPool3d, self).__init__(3,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              name=name)

class SparseMaxPool4d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 name=None):
        super(SparseMaxPool4d, self).__init__(4,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              name=name)
