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
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from spconv import pytorch as spconv
from spconv import SPCONV_VERSION_NUMBERS
from spconv.core import ConvAlgo
from spconv.debug_utils import spconv_save_debug_data
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.pytorch.core import IndiceData, SparseConvTensor, ImplicitGemmIndiceData, expand_nd
from spconv.pytorch.modules import SparseModule
from spconv.constants import SAVED_WEIGHT_LAYOUT, ALL_WEIGHT_IS_KRSC, SPCONV_DEBUG_WEIGHT
from spconv.utils import nullcontext
from torch.nn.init import calculate_gain
from cumm import tensorview as tv

FILTER_HWIO = False

_MAX_NUM_VOXELS_DURING_TRAINING = "max_num_voxels_during_training"


class SparseConvolution(SparseModule):
    __constants__ = [
        'stride', 'padding', 'dilation', 'groups', 'bias', 'subm', 'inverse',
        'transposed', 'output_padding'
    ]

    def __init__(self,
                 ndim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: Union[int, List[int], Tuple[int, ...]] = 1,
                 bias: bool = True,
                 subm: bool = False,
                 output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 transposed: bool = False,
                 inverse: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                 act_alpha: float = 0,
                 act_beta: float = 0,
                 name=None):
        super(SparseConvolution, self).__init__(name=name)
        assert groups == 1, "don't support groups for now"
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = expand_nd(ndim, kernel_size)

        self.stride = expand_nd(ndim, stride)
        kv = int(np.prod(self.kernel_size))
        kv_stride = int(np.prod(self.stride))
        self.dilation = expand_nd(ndim, dilation)
        self.padding = expand_nd(ndim, padding)

        self.conv1x1 = kv == 1
        # TODO we should deprecate support for ksize == 1 but stride != 1.
        if not subm:
            self.conv1x1 &= kv_stride == 1
            if self.conv1x1:
                assert self.padding == [
                    0
                ] * ndim, "padding must be zero for 1x1 conv (k=1,s=1)"
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = expand_nd(ndim, output_padding)
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        if record_voxel_count and not self.subm and not self.inverse:
            # we record maximum voxel num in both inference and training if
            # record_voxel_count flag setting.
            self.register_buffer(_MAX_NUM_VOXELS_DURING_TRAINING,
                                 torch.zeros(1, dtype=torch.int32))
        self.record_voxel_count = record_voxel_count
        if algo is None:
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
        self.fp32_accum = fp32_accum
        # self.algo = ConvAlgo.Native
        if self.algo == ConvAlgo.Native and not ALL_WEIGHT_IS_KRSC:
            if FILTER_HWIO:
                # RSCK
                self.weight = Parameter(
                    torch.Tensor(*self.kernel_size, in_channels, out_channels))
            else:
                # RSKC
                self.weight = Parameter(
                    torch.Tensor(*self.kernel_size, out_channels, in_channels))
        else:
            # KRSC
            self.weight = Parameter(
                torch.Tensor(out_channels, *self.kernel_size, in_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.act_type = act_type
        self.act_alpha = act_alpha
        self.act_beta = act_beta
        if self.conv1x1:
            assert act_type == tv.gemm.Activation.None_, "conv1x1 don't support fused act"
        self.reset_parameters()
        if hasattr(self, "_register_load_state_dict_pre_hook"):
            self._register_load_state_dict_pre_hook(
                self._load_weight_different_layout)

    def get_max_num_voxels(self) -> Optional[torch.Tensor]:
        if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
            return getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING)
        return None

    def _load_weight_different_layout(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs):
        if self.record_voxel_count and not self.subm and not self.inverse and _MAX_NUM_VOXELS_DURING_TRAINING not in state_dict:
            state_dict[prefix + _MAX_NUM_VOXELS_DURING_TRAINING] = torch.zeros(
                1, dtype=torch.int32)
        if not SAVED_WEIGHT_LAYOUT:
            return
        key = prefix + "weight"
        assert key in state_dict
        ndim = self.ndim
        if SAVED_WEIGHT_LAYOUT == "RSKC":
            state_dict[key] = state_dict[key].permute(ndim, *range(ndim),
                                                      ndim + 1).contiguous()
        elif SAVED_WEIGHT_LAYOUT == "RSCK":
            state_dict[key] = state_dict[key].permute(ndim + 1, *range(ndim),
                                                      ndim).contiguous()

        if ALL_WEIGHT_IS_KRSC or self.algo != ConvAlgo.Native:
            # in spconv 2.2, we only support KRSC layout.
            if SAVED_WEIGHT_LAYOUT == "RSKC":
                state_dict[key] = state_dict[key].permute(
                    ndim, *range(ndim), ndim + 1).contiguous()
            elif SAVED_WEIGHT_LAYOUT == "RSCK":
                state_dict[key] = state_dict[key].permute(
                    ndim + 1, *range(ndim), ndim).contiguous()

        else:
            if self.algo == ConvAlgo.Native:
                # to RSCK
                if SAVED_WEIGHT_LAYOUT == "RSKC":
                    state_dict[key] = state_dict[key].permute(
                        *range(ndim), ndim + 1, ndim).contiguous()
                elif SAVED_WEIGHT_LAYOUT == "KRSC":
                    state_dict[key] = state_dict[key].permute(
                        *range(1, ndim + 1), 0, ndim + 1).contiguous()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.algo is not None:
            s += f', algo={self.algo}'
        return s.format(**self.__dict__)

    def _calculate_fan_in_and_fan_out(self):
        receptive_field_size = 1
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in self.kernel_size:
            receptive_field_size *= s
        fan_in = self.in_channels * receptive_field_size
        fan_out = self.out_channels * receptive_field_size
        return fan_in, fan_out

    def _calculate_correct_fan(self, mode):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError(
                "Mode {} not supported, please use one of {}".format(
                    mode, valid_modes))

        fan_in, fan_out = self._calculate_fan_in_and_fan_out()
        return fan_in if mode == 'fan_in' else fan_out

    def _custom_kaiming_uniform_(self,
                                 tensor,
                                 a=0,
                                 mode='fan_in',
                                 nonlinearity='leaky_relu'):
        r"""same as torch.init.kaiming_uniform_, with KRSC layout support
        """
        fan = self._calculate_correct_fan(mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(
            3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def reset_parameters(self):
        if SPCONV_DEBUG_WEIGHT:
            self._custom_kaiming_uniform_(self.weight, a=math.sqrt(0.005))
        else:
            self._custom_kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def is_inverseable(self):
        return self.indice_key is not None and not self.subm

    def forward(self, input: SparseConvTensor):
        assert isinstance(input, SparseConvTensor)
        assert input.features.shape[
            1] == self.in_channels, "channel size mismatch"
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        bias_for_training = self.bias if self.training else None
        bias_for_infer = self.bias if not self.training else None

        if self.training:
            msg =  "act don't support backward, only used in inference"
            assert self.act_type == tv.gemm.Activation.None_, msg
        
        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)
        else:
            out_spatial_shape = spatial_shape
        # print(self._sparse_unique_name, spatial_shape, out_spatial_shape)
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        out_tensor = input.shadow_copy()
        if input.benchmark:
            if self.name is None:
                raise ValueError(
                    "you need to assign name to spmodules before benchmark (spconv.utils.bench.assign_name_to_spmod)"
                )
            if self.name not in input.benchmark_record:
                input.benchmark_record[self.name] = {
                    "type": "SparseConvolution",
                    "indice_gen_time": [],
                    "time": [],
                    "num_points": [],
                    "num_out_points": [],
                    "params": {
                        "kernel_size": self.kernel_size,
                        "stride": self.stride,
                        "padding": self.padding,
                        "dilation": self.dilation,
                        "output_padding": self.output_padding,
                        "subm": self.subm,
                        "transposed": self.transposed,
                        "input_channels": self.in_channels,
                        "out_channels": self.out_channels,
                    }
                }
        if self.conv1x1:
            if FILTER_HWIO:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.out_channels, self.in_channels).T)
            else:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.in_channels, self.out_channels))

            if self.bias is not None:
                features += self.bias
            out_tensor = out_tensor.replace_feature(features)
            # padding may change spatial shape of conv 1x1.
            out_tensor.spatial_shape = out_spatial_shape
            return out_tensor
        indice_dict = input.indice_dict.copy()

        algo = self.algo
        if self.indice_key is not None:
            datas = input.find_indice_pair(self.indice_key)
            if datas is not None:
                msg = "due to limitation of pytorch, you must provide same algo to layers share same indice key."
                assert algo == datas.algo, msg
                # algo = datas.algo
        profile_ctx = nullcontext()
        if input._timer is not None and self._sparse_unique_name:
            profile_ctx = input._timer.namespace(self._sparse_unique_name)
        with profile_ctx:
            if algo == ConvAlgo.Native:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, IndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."

                    outids = datas.indices
                    indice_pairs = datas.indice_pairs
                    indice_pair_num = datas.indice_pair_num
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        indice_pairs = datas.indice_pairs
                        indice_pair_num = datas.indice_pair_num
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:
                        if input.benchmark:
                            torch.cuda.synchronize()
                            t = time.time()
                        try:
                            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                                indices, batch_size, spatial_shape, algo,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, self.output_padding, self.subm,
                                self.transposed)
                        except Exception as e:
                            msg = "[Exception|native_pair]"
                            msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                            msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                            msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                            msg += f"transpose={self.transposed}"
                            print(msg, file=sys.stderr)
                            spconv_save_debug_data(indices)
                            raise e
                        if input.benchmark:
                            torch.cuda.synchronize()
                            interval = time.time() - t
                            out_tensor.benchmark_record[
                                self.name]["indice_gen_time"].append(interval)

                        indice_data = IndiceData(outids,
                                                 indices,
                                                 indice_pairs,
                                                 indice_pair_num,
                                                 spatial_shape,
                                                 out_spatial_shape,
                                                 is_subm=self.subm,
                                                 algo=algo,
                                                 ksize=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)
                        if self.indice_key is not None:
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                indice_pairs_calc = indice_pairs
                if indice_pairs.device != features.device:
                    indice_pairs_calc = indice_pairs.to(features.device)
                if self.subm:
                    out_features = Fsp.indice_subm_conv(
                        features,
                        self.weight,
                        indice_pairs_calc,
                        indice_pair_num,
                        outids.shape[0],
                        algo,
                        input._timer,
                        bias_for_infer,
                        self.act_alpha,
                        self.act_beta,
                        self.act_type)
                else:
                    if self.inverse:
                        out_features = Fsp.indice_inverse_conv(
                            features,
                            self.weight,
                            indice_pairs_calc,
                            indice_pair_num,
                            outids.shape[0],
                            algo,
                            input._timer,
                            bias_for_infer,
                            self.act_alpha,
                            self.act_beta,
                            self.act_type)
                    else:
                        out_features = Fsp.indice_conv(
                            features,
                            self.weight,
                            indice_pairs_calc,
                            indice_pair_num,
                            outids.shape[0],
                            algo,
                            input._timer,
                            bias_for_infer,
                            self.act_alpha,
                            self.act_beta,
                            self.act_type)

            else:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, ImplicitGemmIndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."
                    outids = datas.indices
                    pair_fwd = datas.pair_bwd
                    pair_bwd = datas.pair_fwd
                    pair_mask_fwd_splits = datas.pair_mask_bwd_splits
                    pair_mask_bwd_splits = datas.pair_mask_fwd_splits
                    mask_argsort_fwd_splits = datas.mask_argsort_bwd_splits
                    mask_argsort_bwd_splits = datas.mask_argsort_fwd_splits
                    masks = datas.masks
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        pair_fwd = datas.pair_fwd
                        pair_bwd = datas.pair_bwd
                        pair_mask_fwd_splits = datas.pair_mask_fwd_splits
                        pair_mask_bwd_splits = datas.pair_mask_bwd_splits
                        mask_argsort_fwd_splits = datas.mask_argsort_fwd_splits
                        mask_argsort_bwd_splits = datas.mask_argsort_bwd_splits
                        masks = datas.masks
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:
                        with input._timer.namespace("gen_pairs"):
                            # we need to gen bwd indices for regular conv
                            # because it may be inversed.
                            try:
                                res = ops.get_indice_pairs_implicit_gemm(
                                    indices,
                                    batch_size,
                                    spatial_shape,
                                    algo,
                                    ksize=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    out_padding=self.output_padding,
                                    subm=self.subm,
                                    transpose=self.transposed,
                                    is_train=(not self.subm) or self.training,
                                    alloc=input.thrust_allocator,
                                    timer=input._timer)
                            except Exception as e:
                                msg = "[Exception|implicit_gemm_pair]"
                                msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                                msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                                msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                                msg += f"transpose={self.transposed}"
                                print(msg, file=sys.stderr)
                                spconv_save_debug_data(indices)
                                raise e

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
                                algo=algo,
                                ksize=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                num_activate_out = outids.shape[0]
                out_features = Fsp.implicit_gemm(
                    features, self.weight, pair_fwd, pair_bwd,
                    pair_mask_fwd_splits, pair_mask_bwd_splits,
                    mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                    num_activate_out, masks, self.training, self.subm,
                    input._timer, self.fp32_accum,
                    bias_for_infer,
                    self.act_alpha,
                    self.act_beta,
                    self.act_type)
        if bias_for_training is not None:
            out_features += bias_for_training
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["time"].append(interval)
            out_tensor.benchmark_record[self.name]["num_points"].append(
                features.shape[0])
            out_tensor.benchmark_record[self.name]["num_out_points"].append(
                out_features.shape[0])
        if not self.subm and not self.inverse and self.record_voxel_count:
            if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
                ops.maximum_value_int_(
                    getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING),
                    outids.shape[0])
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        # print(outids.shape, spatial_shape, self.kernel_size, self.stride, self.padding,
        #             self.dilation, self.output_padding, out_spatial_shape)

        return out_tensor

    def _check_subm_reuse_valid(self, inp: SparseConvTensor,
                                spatial_shape: List[int],
                                datas: Union[ImplicitGemmIndiceData,
                                             IndiceData]):
        assert datas.is_subm, "only support reuse subm indices"
        if self.kernel_size != datas.ksize:
            raise ValueError(
                f"subm with same indice_key must have same kernel"
                f" size, expect {datas.ksize}, this layer {self.kernel_size}")
        if self.dilation != datas.dilation:
            raise ValueError(
                f"subm with same indice_key must have same dilation"
                f", expect {datas.dilation}, this layer {self.dilation}")
        if inp.spatial_shape != datas.spatial_shape:
            raise ValueError(
                f"subm with same indice_key must have same spatial structure"
                f", expect {datas.spatial_shape}, input {spatial_shape}")
        if inp.indices.shape[0] != datas.indices.shape[0]:
            raise ValueError(
                f"subm with same indice_key must have same num of indices"
                f", expect {datas.indices.shape[0]}, input {inp.indices.shape[0]}"
            )


class SparseConv1d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConv1d,
              self).__init__(1,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConv2d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConv2d,
              self).__init__(2,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConv3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConv3d,
              self).__init__(3,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConv4d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConv4d,
              self).__init__(4,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConvTranspose1d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConvTranspose1d,
              self).__init__(1,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             transposed=True,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConvTranspose2d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConvTranspose2d,
              self).__init__(2,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             transposed=True,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConvTranspose3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConvTranspose3d,
              self).__init__(3,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             transposed=True,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseConvTranspose4d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 name=None):
        super(SparseConvTranspose4d,
              self).__init__(4,
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             bias,
                             transposed=True,
                             indice_key=indice_key,
                             algo=algo,
                             fp32_accum=fp32_accum,
                             record_voxel_count=record_voxel_count,
                             name=name)


class SparseInverseConv1d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key,
                 bias=True,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SparseInverseConv1d, self).__init__(1,
                                                  in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  bias=bias,
                                                  inverse=True,
                                                  indice_key=indice_key,
                                                  algo=algo,
                                                  fp32_accum=fp32_accum,
                                                  name=name)


class SparseInverseConv2d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key,
                 bias=True,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SparseInverseConv2d, self).__init__(2,
                                                  in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  bias=bias,
                                                  inverse=True,
                                                  indice_key=indice_key,
                                                  algo=algo,
                                                  fp32_accum=fp32_accum,
                                                  name=name)


class SparseInverseConv3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key,
                 bias=True,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SparseInverseConv3d, self).__init__(3,
                                                  in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  bias=bias,
                                                  inverse=True,
                                                  indice_key=indice_key,
                                                  algo=algo,
                                                  fp32_accum=fp32_accum,
                                                  name=name)


class SparseInverseConv4d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key,
                 bias=True,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SparseInverseConv4d, self).__init__(4,
                                                  in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  bias=bias,
                                                  inverse=True,
                                                  indice_key=indice_key,
                                                  algo=algo,
                                                  fp32_accum=fp32_accum,
                                                  name=name)


class SubMConv1d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SubMConv1d, self).__init__(1,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         True,
                                         indice_key=indice_key,
                                         algo=algo,
                                         fp32_accum=fp32_accum,
                                         name=name)


class SubMConv2d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SubMConv2d, self).__init__(2,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         True,
                                         indice_key=indice_key,
                                         algo=algo,
                                         fp32_accum=fp32_accum,
                                         name=name)


class SubMConv3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SubMConv3d, self).__init__(3,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         True,
                                         indice_key=indice_key,
                                         algo=algo,
                                         fp32_accum=fp32_accum,
                                         name=name)


class SubMConv4d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SubMConv4d, self).__init__(4,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         True,
                                         indice_key=indice_key,
                                         algo=algo,
                                         fp32_accum=fp32_accum,
                                         name=name)
