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
import functools
from enum import Enum
from cumm import tensorview as tv
from cumm.gemm.algospec.core import ShuffleStrideType

import torch
import numpy as np
import spconv
from spconv.algo import AlgoHint, ConvAlgo
from typing import List, Union
from spconv.pytorch.cppcore import torch_tensor_to_tv, get_current_stream
from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.algo import GEMM  # , GATHER, SCATTER
import time
from spconv.constants import FILTER_HWIO


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                           output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[
            i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(indices: torch.Tensor,
                     batch_size: int,
                     spatial_shape: List[int],
                     algo: ConvAlgo,
                     ksize: List[int],
                     stride: List[int],
                     padding: List[int],
                     dilation: List[int],
                     out_padding: List[int],
                     subm: bool = False,
                     transpose: bool = False):
    # torch.cuda.synchronize()
    # t = time.time()
    ndim = indices.shape[1] - 1
    kv: int = functools.reduce(lambda x, y: x * y, ksize, 1)
    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                               padding, dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation)
    else:
        out_shape = spatial_shape
    assert algo == ConvAlgo.Native, "TODO"
    stream = get_current_stream()

    pair = torch.full((2, kv, indices.shape[0]),
                      -1,
                      dtype=indices.dtype,
                      device=indices.device)
    indice_num_per_loc = torch.zeros((kv, ),
                                     dtype=indices.dtype,
                                     device=indices.device)

    inds_tv = torch_tensor_to_tv(indices)
    pair_tv = torch_tensor_to_tv(pair)
    indice_num_per_loc_tv = torch_tensor_to_tv(indice_num_per_loc)

    if subm:
        out_inds = indices

        hashdata = torch.empty((out_inds.shape[0] * 2, ),
                               dtype=torch.int64,
                               device=indices.device)
        out_inds_tv = torch_tensor_to_tv(out_inds)
        hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)

        SpconvOps.generate_subm_conv_inds(inds_tv,
                                          hashdata_tv,
                                          pair_tv,
                                          out_inds_tv,
                                          indice_num_per_loc_tv,
                                          batch_size=batch_size,
                                          input_dims=spatial_shape,
                                          ksize=ksize,
                                          dilation=dilation,
                                          stream_int=stream)
        # torch.cuda.synchronize()
        # print("SUBM", time.time() - t)

    else:
        indice_pairs_uniq = torch.empty((pair.numel() // 2 + 1, ),
                                        dtype=indices.dtype,
                                        device=indices.device)
        indice_pairs_uniq_tv = torch_tensor_to_tv(indice_pairs_uniq)

        SpconvOps.generate_conv_inds_stage1(inds_tv,
                                            pair_tv,
                                            indice_pairs_uniq_tv,
                                            indice_num_per_loc_tv,
                                            batch_size=batch_size,
                                            output_dims=out_shape,
                                            input_dims=spatial_shape,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            transposed=transpose,
                                            stream_int=stream)
        uniq_res = indice_pairs_uniq.unique()
        num_act_out = uniq_res.shape[0] - 1
        uniq_res_tv = torch_tensor_to_tv(uniq_res)
        # num_act_out = SpconvOps.generate_conv_inds_stage1_5(
        #     indice_pairs_uniq_tv,
        #     ndim,
        #     uniq_size=indice_pairs_uniq_tv.size,
        #     stream_int=stream)
        # uniq_res_tv = indice_pairs_uniq_tv.slice_first_axis(0, num_act_out)
        out_inds = torch.empty((num_act_out, indices.shape[1]),
                               dtype=indices.dtype,
                               device=indices.device)
        hashdata = torch.empty((out_inds.shape[0] * 2, ),
                               dtype=torch.int64,
                               device=indices.device)
        out_inds_tv = torch_tensor_to_tv(out_inds)
        hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
        SpconvOps.generate_conv_inds_stage2(inds_tv,
                                            hashdata_tv,
                                            pair_tv,
                                            uniq_res_tv,
                                            out_inds_tv,
                                            num_out_act=num_act_out,
                                            batch_size=batch_size,
                                            output_dims=out_shape,
                                            input_dims=spatial_shape,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            transposed=transpose,
                                            stream_int=stream)
        # torch.cuda.synchronize()
        # print("REGU", time.time() - t)
    return out_inds, pair, indice_num_per_loc


def indice_conv(features: torch.Tensor,
                filters: torch.Tensor,
                indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: int,
                inverse: bool = False,
                subm: bool = False,
                algo: ConvAlgo = ConvAlgo.Native):
    # filters: RSKC
    # torch.cuda.synchronize()
    # t = time.time()
    if features.dtype == torch.int8 or features.dtype == torch.qint8:
        raise NotImplementedError("work in progress")
    if FILTER_HWIO:
        out_channel = filters.shape[-1]
    else:
        out_channel = filters.shape[-2]
    filters = filters.reshape(-1, *filters.shape[-2:])
    kv = filters.shape[0]
    kv_center = kv // 2
    if subm:
        if FILTER_HWIO:
            out_features = torch.mm(features, filters[kv_center])
        else:
            out_features = torch.mm(features, filters[kv_center].T)
    else:
        out_features = torch.zeros((num_activate_out, out_channel),
                                   dtype=features.dtype,
                                   device=features.device)
    if kv == 1 and subm:
        return out_features

    stream = get_current_stream()
    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    arch = torch.cuda.get_device_capability()
    inited: bool = subm
    a = torch_tensor_to_tv(features)
    c = torch_tensor_to_tv(out_features)
    profile_idx = kv_center
    if subm:
        profile_idx = kv_center - 1
    # profile_idx = first_n
    nhot_profile = indice_pair_num_cpu[profile_idx]

    # print(nhot_profile, indice_pair_num_cpu)
    profile_res = GEMM.get_profiled_algo(
        a.shape,
        filters.shape[-2:],
        c.shape,
        False,
        False if FILTER_HWIO else True,
        False,
        arch=arch,
        shuffle_type=ShuffleStrideType.ShuffleAC,
        a_inds_shape=[nhot_profile],
        c_inds_shape=[nhot_profile],
        hint=AlgoHint.Fowrard.value)


    maxnhot = max(indice_pair_num_cpu)
    if profile_res is None:
        # run profile on center
        inp_indices_th = indice_pairs[int(inverse)][profile_idx, :nhot_profile]
        out_indices_th = indice_pairs[int(not inverse)][
            profile_idx, :nhot_profile]
        inp_indices = torch_tensor_to_tv(inp_indices_th)
        out_indices = torch_tensor_to_tv(out_indices_th)
        filter_tv = torch_tensor_to_tv(filters)[profile_idx]

        profile_res, min_time = GEMM.profile_and_cache(
            a,
            filter_tv,
            c,
            False,
            False if FILTER_HWIO else True,
            False,
            arch=arch,
            shuffle_type=ShuffleStrideType.ShuffleAC,
            a_inds=inp_indices,
            c_inds=out_indices,
            alpha=1.0,
            beta=0.0,
            hint=AlgoHint.Fowrard.value,
            stream=stream)

    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    pair_in = indice_pairs_tv[int(inverse)]
    pair_out = indice_pairs_tv[int(not inverse)]
    filters_tv = torch_tensor_to_tv(filters)
    for i, nhot in enumerate(indice_pair_num_cpu):
        if subm and i == kv_center:
            continue
        if subm and i > kv_center:
            nhot = indice_pair_num_cpu[kv - i - 1]
        if nhot <= 0:
            continue
        inp_indices = pair_in[i].slice_first_axis(0, nhot)
        out_indices = pair_out[i].slice_first_axis(0, nhot)
        b = filters_tv[i]
        # inp @ filter.T, NC @ KC
        beta = 1.0 if inited else 0.0
        algo_desp = GEMM.run_profile(profile_res,
                                     a,
                                     b,
                                     c,
                                     False,
                                     False if FILTER_HWIO else True,
                                     False,
                                     arch=arch,
                                     stream=stream,
                                     shuffle_type=ShuffleStrideType.ShuffleAC,
                                     a_inds=inp_indices,
                                     c_inds=out_indices,
                                     hint=AlgoHint.Fowrard.value,
                                     alpha=1.0,
                                     beta=beta)

        # gather_times += gather_time
        inited = True
    # torch.cuda.synchronize()
    # # print(stream, valid_count, maxnhot, features.shape[0], features.shape[1], out_channel, time.time() - t, total_times, txt)
    # # print(algo_desp, profile_res.external_gather, profile_res.splitk, features.shape[0], features.shape[1], out_channel, time.time() - t)

    # # print(indice_pair_num_cpu)
    # print("G", time.time() - t)
    return out_features


def fused_indice_conv(features, filters, bias, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm):
    raise NotImplementedError


def indice_conv_backward(features: torch.Tensor,
                         filters: torch.Tensor,
                         out_bp: torch.Tensor,
                         indice_pairs: torch.Tensor,
                         indice_pair_num: torch.Tensor,
                         inverse: bool = False,
                         subm: bool = False,
                         algo: ConvAlgo = ConvAlgo.Native):
    # torch.cuda.synchronize()
    # t = time.time()

    num_activate_out = out_bp.shape[0]
    out_channel = out_bp.shape[-1]
    filters_shape = filters.shape
    filters = filters.reshape(-1, *filters.shape[-2:])
    kv = filters.shape[0]
    kv_center = kv // 2
    assert out_bp.is_contiguous()
    assert filters.is_contiguous()
    assert features.is_contiguous()

    if subm:
        dfilters = torch.zeros_like(filters)
        if FILTER_HWIO:
            torch.mm(features.T, out_bp, out=dfilters[kv_center])
            # TODO can we use torch mm for f16 backward weight?
            din = torch.mm(out_bp, filters[kv_center].T)
        else:
            torch.mm(out_bp.T, features, out=dfilters[kv_center])
            # TODO can we use torch mm for f16 backward weight?
            din = torch.mm(out_bp, filters[kv_center])
    else:
        dfilters = torch.zeros_like(filters)
        din = torch.zeros_like(features)
    if kv == 1 and subm:
        return (din, dfilters.reshape(filters_shape))

    inited: bool = subm
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    # torch slice (a_th[x]) is very slow, so we need to use tv.Tensor earlier.
    pair_in = indice_pairs_tv[int(inverse)]
    pair_out = indice_pairs_tv[int(not inverse)]

    stream = get_current_stream()
    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    arch = torch.cuda.get_device_capability()
    filters_tv = torch_tensor_to_tv(filters)

    dfilters_tv = torch_tensor_to_tv(dfilters)
    out_bp_tv = torch_tensor_to_tv(out_bp)
    features_tv = torch_tensor_to_tv(features)

    din_tv = torch_tensor_to_tv(din)

    profile_idx = kv_center
    if subm:
        profile_idx = kv_center - 1
    # profile_idx = first_n
    nhot_profile = indice_pair_num_cpu[profile_idx]

    # print(nhot_profile, indice_pair_num_cpu)
    profile_res_dgrad = GEMM.get_profiled_algo(
        out_bp_tv.shape,
        filters.shape[-2:],
        din_tv.shape,
        False,
        True if FILTER_HWIO else False,
        False,
        arch=arch,
        shuffle_type=ShuffleStrideType.ShuffleAC,
        a_inds_shape=[nhot_profile],
        c_inds_shape=[nhot_profile],
        hint=AlgoHint.BackwardInput.value)
    if profile_res_dgrad is None:
        inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile)
        out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile)
        filter_tv = filters_tv[profile_idx]
        profile_res_dgrad, min_time = GEMM.profile_and_cache(
            out_bp_tv,
            filter_tv,
            din_tv,
            False,
            True if FILTER_HWIO else False,
            False,
            arch=arch,
            shuffle_type=ShuffleStrideType.ShuffleAC,
            a_inds=inp_indices,
            c_inds=out_indices,
            alpha=1.0,
            beta=0.0,
            hint=AlgoHint.BackwardInput.value,
            stream=stream)
    if not FILTER_HWIO:
        a_wgrad = out_bp_tv
        b_wgrad = features_tv
    else:
        a_wgrad = features_tv
        b_wgrad = out_bp_tv
    profile_res_wgrad = GEMM.get_profiled_algo(
        a_wgrad.shape,
        b_wgrad.shape,
        filters.shape[-2:],
        True,
        False,
        False,
        arch=arch,
        shuffle_type=ShuffleStrideType.ShuffleAB,
        a_inds_shape=[nhot_profile],
        b_inds_shape=[nhot_profile],
        hint=AlgoHint.BackwardWeight.value)

    if profile_res_wgrad is None:
        inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile)
        out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile)
        dfilter_tv = dfilters_tv[profile_idx]
        if not FILTER_HWIO:
            a_inds_wgrad = out_indices
            b_inds_wgrad = inp_indices
        else:
            a_inds_wgrad = inp_indices
            b_inds_wgrad = out_indices
        profile_res_wgrad, min_time = GEMM.profile_and_cache(
            a_wgrad,
            b_wgrad,
            dfilter_tv,
            True,
            False,
            False,
            arch=arch,
            shuffle_type=ShuffleStrideType.ShuffleAB,
            a_inds=a_inds_wgrad,
            b_inds=b_inds_wgrad,
            alpha=1.0,
            beta=0.0,
            hint=AlgoHint.BackwardWeight.value,
            stream=stream)
        # print(profile_res_wgrad.algo_desp, profile_res_wgrad.splitk, min_time)
    maxnhot = max(indice_pair_num_cpu)
    # get workspace size for wgrad
    if not FILTER_HWIO:
        a_shape = [maxnhot, out_bp_tv.dim(1)]
        b_shape = [maxnhot, features_tv.dim(1)]
    else:
        b_shape = [maxnhot, out_bp_tv.dim(1)]
        a_shape = [maxnhot, features_tv.dim(1)]
    m, n, k = GEMM.extract_mnk(a_shape,
                               b_shape,
                               profile_res_wgrad.algo_desp.trans_a,
                               profile_res_wgrad.algo_desp.trans_b,
                               profile_res_wgrad.algo_desp.trans_c,
                               arch=arch,
                               shuffle_type=ShuffleStrideType.ShuffleAB,
                               a_inds_shape=[maxnhot],
                               b_inds_shape=[maxnhot],
                               hint=AlgoHint.BackwardWeight.value)
    workspace_size = profile_res_wgrad.algo_desp.query_workspace_size(
        m, n, k, profile_res_wgrad.splitk)
    workspace = torch.Tensor()

    workspace_tv = tv.Tensor()
    if workspace_size > 0:
        workspace = torch.empty((workspace_size, ),
                                dtype=torch.int8,
                                device=features.device)
        workspace_tv = torch_tensor_to_tv(workspace)
    # print(workspace_size, m, n, k, profile_res_wgrad.splitk)
    # torch.cuda.synchronize()
    # di_time = time.time() - t
    # t = time.time()
    inited = subm
    for i, nhot in enumerate(indice_pair_num_cpu):
        if subm and i == kv_center:
            continue
        if subm and i > kv_center:
            nhot = indice_pair_num_cpu[kv - i - 1]
        if nhot <= 0:
            continue
        beta = 1.0 if inited else 0.0
        inp_indices = pair_in[i].slice_first_axis(0, nhot)
        out_indices = pair_out[i].slice_first_axis(0, nhot)
        # out.T @ inp, NK @ NC
        # print(features_tv.shape, out_bp_tv.shape)
        GEMM.run_profile(profile_res_dgrad,
                         out_bp_tv,
                         filters_tv[i],
                         din_tv,
                         False,
                         True if FILTER_HWIO else False,
                         False,
                         arch=arch,
                         stream=stream,
                         shuffle_type=ShuffleStrideType.ShuffleAC,
                         a_inds=out_indices,
                         c_inds=inp_indices,
                         hint=AlgoHint.BackwardInput.value,
                         alpha=1.0,
                         beta=beta)

        if not FILTER_HWIO:
            a = out_bp_tv
            b = features_tv
            a_inds = out_indices
            b_inds = inp_indices
        else:
            a = features_tv
            b = out_bp_tv
            a_inds = inp_indices
            b_inds = out_indices
        GEMM.run_profile(profile_res_wgrad,
                         a,
                         b,
                         dfilters_tv[i],
                         True,
                         False,
                         False,
                         arch=arch,
                         stream=stream,
                         shuffle_type=ShuffleStrideType.ShuffleAB,
                         a_inds=a_inds,
                         b_inds=b_inds,
                         hint=AlgoHint.BackwardWeight.value,
                         alpha=1.0,
                         beta=beta,
                         workspace=workspace_tv)
        inited = True

    # torch.cuda.synchronize()
    # dw_time = time.time() - t
    # # print(dw_time + di_time, di_time, dw_time, profile_res_wgrad.splitk, profile_res_wgrad.algo_desp, dfilters.shape)
    # # print(dw_time + di_time)
    # print("BWG", time.time() - t)
    return (din, dfilters.reshape(filters_shape))


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    # torch.cuda.synchronize()
    # t = time.time()
    out_channel = features.shape[-1]
    out_features = torch.zeros((num_activate_out, out_channel),
                               dtype=features.dtype,
                               device=features.device)
    stream = get_current_stream()
    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    for i, nhot in enumerate(indice_pair_num_cpu):
        if nhot <= 0:
            continue
        inp_indices = torch_tensor_to_tv(indice_pairs[0][i, :nhot])
        out_indices = torch_tensor_to_tv(indice_pairs[1][i, :nhot])
        SpconvOps.maxpool_forward(out_features_tv, features_tv, out_indices,
                                  inp_indices, stream)
    # torch.cuda.synchronize()
    # print("M", time.time() - t)

    return out_features


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs,
                            indice_pair_num):
    out_channel = features.shape[-1]
    din = torch.zeros_like(features)
    stream = get_current_stream()
    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    out_bp_tv = torch_tensor_to_tv(out_bp)
    din_tv = torch_tensor_to_tv(din)
    for i, nhot in enumerate(indice_pair_num_cpu):
        if nhot <= 0:
            continue
        inp_indices = torch_tensor_to_tv(indice_pairs[0][i, :nhot])
        out_indices = torch_tensor_to_tv(indice_pairs[1][i, :nhot])
        SpconvOps.maxpool_backward(out_features_tv, features_tv, out_bp_tv,
                                   din_tv, out_indices, inp_indices, stream)

    return din


def nms(boxes, scores, pre_max_size, post_max_size, thresh, eps):
    raise NotImplementedError


def pillar_scatter(features, coors, shape):
    raise NotImplementedError
