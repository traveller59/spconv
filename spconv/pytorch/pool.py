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
from typing import List, Optional, Tuple, Union

from spconv import pytorch as spconv
from spconv.core import ConvAlgo
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops
from spconv.pytorch.core import IndiceData, ImplicitGemmIndiceData, expand_nd
from spconv.pytorch.modules import SparseModule
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.utils import nullcontext


class SparseMaxPool(SparseModule):
    def __init__(self,
                 ndim,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Optional[Union[int, List[int], Tuple[int, ...]]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 indice_key: Optional[str] = None,
                 subm: bool = False,
                 algo: Optional[ConvAlgo] = None,
                 name=None):
        super(SparseMaxPool, self).__init__(name=name)
        self.ndim = ndim
        self.kernel_size = expand_nd(ndim, kernel_size)
        if stride is None:
            self.stride = self.kernel_size.copy()
        else:
            self.stride = expand_nd(ndim, stride)
        self.padding = expand_nd(ndim, padding)
        self.subm = subm
        self.dilation = expand_nd(ndim, dilation)
        self.indice_key = indice_key
        kv = int(np.prod(kernel_size))
        if algo is None:
            # keep in mind that this algorithm is set for Inverse Sparse Conv
            # maxpool itself don't need mask.
            if kv <= 32 and not CPU_ONLY_BUILD:
                if kv < 8:
                    algo = ConvAlgo.MaskImplicitGemm
                else:
                    algo = ConvAlgo.MaskImplicitGemm
            else:
                algo = ConvAlgo.Native
        if kv > 32:
            assert algo == ConvAlgo.Native, "implicit gemm don't support kv >= 32 for now"
        if CPU_ONLY_BUILD:
            assert algo == ConvAlgo.Native, "cpu only build only support native algorithm"

        self.algo = algo

    def extra_repr(self):
        s = ('kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.algo is not None:
            s += f', algo={self.algo}'
        return s.format(**self.__dict__)

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
        out_padding = [0] * self.ndim
        indice_dict = input.indice_dict.copy()
        profile_ctx = nullcontext()
        if input._timer is not None and self._sparse_unique_name:
            profile_ctx = input._timer.namespace(self._sparse_unique_name)
        with profile_ctx:
            if self.algo == ConvAlgo.Native:
                outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(
                    indices, batch_size, spatial_shape, ConvAlgo.Native,
                    self.kernel_size, self.stride, self.padding, self.dilation,
                    out_padding, False)
                if input.benchmark:
                    torch.cuda.synchronize()
                    interval = time.time() - t
                    out_tensor.benchmark_record[
                        self.name]["indice_gen_time"].append(interval)
                    t = time.time()

                if self.indice_key is not None:
                    datas = input.find_indice_pair(self.indice_key)
                    if datas is None:
                        indice_data = IndiceData(outids,
                                                 indices,
                                                 indice_pairs,
                                                 indice_pairs_num,
                                                 spatial_shape,
                                                 out_spatial_shape,
                                                 is_subm=False,
                                                 algo=self.algo,
                                                 ksize=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)
                        indice_dict[self.indice_key] = indice_data
                    else:
                        raise ValueError(
                            f"indice key {self.indice_key} exists")

                out_features = Fsp.indice_maxpool(features,
                                                  indice_pairs.to(device),
                                                  indice_pairs_num.to(device),
                                                  outids.shape[0])
            else:
                with input._timer.namespace("gen_pairs"):
                    res = ops.get_indice_pairs_implicit_gemm(
                        indices,
                        batch_size,
                        spatial_shape,
                        self.algo,
                        ksize=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        out_padding=out_padding,
                        subm=self.subm,
                        is_train=(not self.subm) or self.training,
                        alloc=input.thrust_allocator,
                        timer=input._timer)
                outids = res[0]
                num_inds_per_loc = res[1]
                pair_fwd = res[2]
                pair_bwd = res[3]
                pair_mask_fwd_splits = res[4]
                pair_mask_bwd_splits = res[5]
                mask_argsort_fwd_splits = res[6]
                mask_argsort_bwd_splits = res[7]
                masks = res[8]
                if self.indice_key is not None:
                    indice_data = ImplicitGemmIndiceData(
                        outids,
                        indices,
                        pair_fwd,
                        pair_bwd,
                        pair_mask_fwd_splits=pair_mask_fwd_splits,
                        pair_mask_bwd_splits=pair_mask_bwd_splits,
                        mask_argsort_fwd_splits=mask_argsort_fwd_splits,
                        mask_argsort_bwd_splits=mask_argsort_bwd_splits,
                        masks=masks,
                        is_subm=self.subm,
                        spatial_shape=spatial_shape,
                        out_spatial_shape=out_spatial_shape,
                        algo=self.algo,
                        ksize=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation)
                    msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                    assert self.indice_key not in indice_dict, msg
                    indice_dict[self.indice_key] = indice_data
                out_features = Fsp.indice_maxpool_implicit_gemm(
                    features, pair_fwd, pair_bwd, outids.shape[0])

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
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        return out_tensor


class SparseMaxPool1d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 name=None):
        super(SparseMaxPool1d, self).__init__(1,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              indice_key=indice_key,
                                              algo=algo,
                                              name=name)


class SparseMaxPool2d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 name=None):
        super(SparseMaxPool2d, self).__init__(2,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              indice_key=indice_key,
                                              algo=algo,
                                              name=name)


class SparseMaxPool3d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 name=None):
        super(SparseMaxPool3d, self).__init__(3,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              indice_key=indice_key,
                                              algo=algo,
                                              name=name)


class SparseMaxPool4d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 name=None):
        super(SparseMaxPool4d, self).__init__(4,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              indice_key=indice_key,
                                              algo=algo,
                                              name=name)
