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
from cumm.conv.bases import KRSC, NHWC, ConvOpType
from cumm.gemm.algospec.core import ShuffleStrideType

import torch
import numpy as np
import spconv
from spconv.core import AlgoHint, ConvAlgo
from typing import List, Optional, Union
from spconv.pytorch.core import ThrustSortAllocator
from spconv.pytorch.cppcore import torch_tensor_to_tv, get_current_stream, get_arch
from spconv.core_cc.csrc.sparse.all import SpconvOps
import spconv.core_cc as _ext

from spconv.utils import nullcontext

if hasattr(_ext, "cumm"):
    CPU_ONLY_BUILD = False
    from spconv.algo import GEMM, CONV  # , GATHER, SCATTER
else:
    CPU_ONLY_BUILD = True
    GEMM = None
    CONV = None
import time
from spconv.constants import FILTER_HWIO
from cumm.gemm import codeops
from spconv.tools import CUDAKernelTimer

DEBUG = False


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
    # stream = get_current_stream()

    # CONV.stream_synchronize(stream)
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
    if any([x == 0 for x in out_shape]):
        raise ValueError(
            f"your out spatial shape {out_shape} reach zero!!! input shape: {spatial_shape}"
        )
    assert algo == ConvAlgo.Native, "TODO"
    # indices = indices.cpu()

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
        if indices.is_cuda:
            stream = get_current_stream()
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
        else:
            out_inds_tv = torch_tensor_to_tv(out_inds)
            SpconvOps.generate_subm_conv_inds_cpu(inds_tv,
                                                  pair_tv,
                                                  out_inds_tv,
                                                  indice_num_per_loc_tv,
                                                  batch_size=batch_size,
                                                  input_dims=spatial_shape,
                                                  ksize=ksize,
                                                  dilation=dilation)
        # CONV.stream_synchronize(stream)
        # print("SUBM", time.time() - t)

    else:
        if indices.is_cuda:
            stream = get_current_stream()
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
        else:
            out_inds = torch.empty((kv * indices.shape[0], indices.shape[1]),
                                   dtype=indices.dtype,
                                   device=indices.device)
            out_inds_tv = torch_tensor_to_tv(out_inds)

            num_act_out = SpconvOps.generate_conv_inds_cpu(
                inds_tv,
                pair_tv,
                out_inds_tv,
                indice_num_per_loc_tv,
                batch_size=batch_size,
                output_dims=out_shape,
                input_dims=spatial_shape,
                ksize=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                transposed=transpose)
            out_inds = out_inds[:num_act_out]
        # CONV.stream_synchronize(stream)
        # print("REGU", time.time() - t)
    return out_inds, pair, indice_num_per_loc


def get_indice_pairs_implicit_gemm(
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: List[int],
    algo: ConvAlgo,
    ksize: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    out_padding: List[int],
    subm: bool = False,
    transpose: bool = False,
    is_train: bool = True,
    alloc: Optional[ThrustSortAllocator] = None,
    timer: CUDAKernelTimer = CUDAKernelTimer(False)):
    """
    Why return tuple? because pytorch seems don't support custom object in autograd.
    return: (
        out_inds,
        num_inds_per_loc,
        pair_fwd,
        pair_bwd, # None if subm or inference mode
        pair_mask_fwd_splits,
        pair_mask_bwd_splits, # None if subm or inference mode
        mask_argsort_fwd_splits,
        mask_argsort_bwd_splits, # None if subm or inference mode
        masks,
    )
    """
    stream = get_current_stream()
    t = 0
    if DEBUG:
        CONV.stream_synchronize(stream)
        t = time.time()
    assert indices.is_cuda, "implicit gemm only support cuda"
    ndim = indices.shape[1] - 1
    kv: int = functools.reduce(lambda x, y: x * y, ksize, 1)
    # TODO in future we will support up to 128 kernel volume.
    assert kv <= 32, "currently only support kernel volume <= 32 to use implicit gemm"
    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                               padding, dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation)
    else:
        out_shape = spatial_shape
    if any([x == 0 for x in out_shape]):
        raise ValueError(
            f"your out spatial shape {out_shape} reach zero!!! input shape: {spatial_shape}"
        )
    assert algo == ConvAlgo.MaskImplicitGemm or algo == ConvAlgo.MaskSplitImplicitGemm, "TODO"
    is_mask_split = algo == ConvAlgo.MaskSplitImplicitGemm
    mask_split_count = 2 if is_mask_split else 1
    if subm:
        pair = torch.full((2, kv, indices.shape[0]),
                          -1,
                          dtype=indices.dtype,
                          device=indices.device)
    else:
        # for regular conv, pair-in not equal to pair-out
        pair = torch.full((kv, indices.shape[0]),
                          -1,
                          dtype=indices.dtype,
                          device=indices.device)

    indice_num_per_loc = torch.zeros((kv, ),
                                     dtype=indices.dtype,
                                     device=indices.device)

    inds_tv = torch_tensor_to_tv(indices)
    pair_tv = torch_tensor_to_tv(pair)
    indice_num_per_loc_tv = torch_tensor_to_tv(indice_num_per_loc)
    if is_mask_split:
        kv_div_2 = kv // 2
        remain = kv - kv_div_2
        mask_np_1 = np.array([1], dtype=np.uint64)
        first = ((mask_np_1 << (remain)) - 1)
        second = ((mask_np_1 << (kv_div_2)) - 1) << remain
        masks = [first.astype(np.uint32), second.astype(np.uint32)]
    else:
        masks = [np.array([0xffffffff], dtype=np.uint32)]
    # torch.cuda.synchronize()
    # print("SUBM0", time.time() - t)

    if subm:
        out_inds = indices
        hashdata = torch.empty((out_inds.shape[0] * 2, ),
                               dtype=torch.int64,
                               device=indices.device)
        pair_mask = torch.empty((mask_split_count, indices.shape[0]),
                                dtype=torch.int32,
                                device=indices.device)

        out_inds_tv = torch_tensor_to_tv(out_inds)
        hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
        pair_mask_tv = torch_tensor_to_tv(pair_mask, dtype=tv.uint32)
        with timer.record("gen_subm_inds", stream):
            SpconvOps.generate_subm_conv_inds(inds_tv,
                                              hashdata_tv,
                                              pair_tv,
                                              out_inds_tv,
                                              indice_num_per_loc_tv,
                                              batch_size=batch_size,
                                              input_dims=spatial_shape,
                                              ksize=ksize,
                                              dilation=dilation,
                                              indice_pair_mask=pair_mask_tv,
                                              stream_int=stream)
        # torch.cuda.synchronize()
        # print("SUBM0", time.time() - t)
        # CONV.stream_synchronize(stream)

        mask_argsort = torch.empty((mask_split_count, out_inds.shape[0]),
                                   dtype=torch.int32,
                                   device=indices.device)
        mask_argsort_tv = torch_tensor_to_tv(mask_argsort)
        if alloc is None:
            alloc = ThrustSortAllocator(indices.device)
        with timer.record("gen_subm_inds_sort", stream):
            for j in range(mask_split_count):
                # thrust don't provide two-step sort (first step return workspace size)
                # so I use this stupid hack to use torch allocator without touch
                # pytorch binary (c++).
                # f**k thrust
                SpconvOps.sort_1d_by_key_allocator(pair_mask_tv[j],
                                                   alloc.alloc,
                                                   mask_argsort_tv[j], stream)
        # CONV.stream_synchronize(stream)
        pair_mask_in_splits = [pair_mask[i] for i in range(mask_split_count)]
        mask_argsort_in_splits = [
            mask_argsort[i] for i in range(mask_split_count)
        ]
        if DEBUG:

            CONV.stream_synchronize(stream)
            print("SUBM", time.time() - t)

        return (out_inds, indice_num_per_loc, pair[0], pair[1],
                pair_mask_in_splits, [], mask_argsort_in_splits, [], masks)

    else:
        if DEBUG:

            CONV.stream_synchronize(stream)
            print("REGU_PREPARE", time.time() - t)
            t = time.time()

        pair_bwd = pair
        pair_bwd_tv = pair_tv
        indice_pairs_uniq = torch.empty((pair.numel() + 1, ),
                                        dtype=indices.dtype,
                                        device=indices.device)
        indice_pairs_uniq_tv = torch_tensor_to_tv(indice_pairs_uniq)
        with timer.record("gen_conv_inds_stage1", stream):
            SpconvOps.generate_conv_inds_mask_stage1(inds_tv,
                                                     pair_bwd_tv,
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
        if DEBUG:

            CONV.stream_synchronize(stream)
            print("REGU_S1", time.time() - t)
            t = time.time()

        uniq_res = indice_pairs_uniq.unique()
        num_act_out = uniq_res.shape[0] - 1
        if DEBUG:

            CONV.stream_synchronize(stream)
            print("REGU_UNIQ", time.time() - t)
            t = time.time()

        uniq_res_tv = torch_tensor_to_tv(uniq_res)
        out_inds = torch.empty((num_act_out, indices.shape[1]),
                               dtype=indices.dtype,
                               device=indices.device)

        pair_fwd = torch.full((kv, num_act_out),
                              -1,
                              dtype=indices.dtype,
                              device=indices.device)
        pair_mask_fwd = torch.zeros((mask_split_count, num_act_out),
                                    dtype=torch.int32,
                                    device=indices.device)
        pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
        pair_mask_fwd_tv = torch_tensor_to_tv(pair_mask_fwd, dtype=tv.uint32)
        pair_mask_bwd = torch.Tensor()
        pair_mask_bwd_tv = tv.Tensor()
        if is_train:
            pair_mask_bwd = torch.zeros((mask_split_count, indices.shape[0]),
                                        dtype=torch.int32,
                                        device=indices.device)
            pair_mask_bwd_tv = torch_tensor_to_tv(pair_mask_bwd,
                                                  dtype=tv.uint32)

        hashdata = torch.empty((out_inds.shape[0] * 2, ),
                               dtype=torch.int64,
                               device=indices.device)
        out_inds_tv = torch_tensor_to_tv(out_inds)
        hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
        if DEBUG:

            CONV.stream_synchronize(stream)
            print("REGU_S2_PREPARE", time.time() - t)
            t = time.time()
        with timer.record("gen_conv_inds_stage2", stream):
            SpconvOps.generate_conv_inds_mask_stage2(inds_tv,
                                                     hashdata_tv,
                                                     pair_fwd_tv,
                                                     pair_bwd_tv,
                                                     uniq_res_tv,
                                                     out_inds_tv,
                                                     pair_mask_fwd_tv,
                                                     pair_mask_bwd_tv,
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
        if DEBUG:

            CONV.stream_synchronize(stream)
            print("REGU_S2", time.time() - t)
            t = time.time()

        mask_argsort_fwd = torch.empty((mask_split_count, out_inds.shape[0]),
                                       dtype=torch.int32,
                                       device=indices.device)
        mask_argsort_fwd_tv = torch_tensor_to_tv(mask_argsort_fwd)
        mask_argsort_bwd_tv = tv.Tensor()
        mask_argsort_bwd = torch.Tensor()
        if is_train:
            mask_argsort_bwd = torch.empty(
                (mask_split_count, indices.shape[0]),
                dtype=torch.int32,
                device=indices.device)

            mask_argsort_bwd_tv = torch_tensor_to_tv(mask_argsort_bwd)
        if alloc is None:
            alloc = ThrustSortAllocator(indices.device)
        with timer.record("gen_conv_inds_sort", stream):
            if is_mask_split:
                for j in range(mask_split_count):
                    mask_tv = tv.from_numpy(masks[j])
                    # here we try to ensure only call allocator once.
                    if not is_train:
                        SpconvOps.sort_1d_by_key_split_allocator(
                            pair_mask_fwd_tv[j], alloc.alloc, mask_tv,
                            mask_argsort_fwd_tv[j], stream)
                    else:
                        if pair_mask_bwd_tv.dim(1) > pair_mask_fwd_tv.dim(1):
                            SpconvOps.sort_1d_by_key_split_allocator(
                                pair_mask_bwd_tv[j], alloc.alloc, mask_tv,
                                mask_argsort_bwd_tv[j], stream)
                            SpconvOps.sort_1d_by_key_split_allocator(
                                pair_mask_fwd_tv[j], alloc.alloc, mask_tv,
                                mask_argsort_fwd_tv[j], stream)
                        else:
                            SpconvOps.sort_1d_by_key_split_allocator(
                                pair_mask_fwd_tv[j], alloc.alloc, mask_tv,
                                mask_argsort_fwd_tv[j], stream)
                            SpconvOps.sort_1d_by_key_split_allocator(
                                pair_mask_bwd_tv[j], alloc.alloc, mask_tv,
                                mask_argsort_bwd_tv[j], stream)

                    # SpconvOps.sort_1d_by_key_split(pair_mask_fwd_tv[j], mask_tv,
                    #                                mask_argsort_fwd_tv[j], stream)
                    # if is_train:
                    #     SpconvOps.sort_1d_by_key_split(pair_mask_bwd_tv[j],
                    #                                    mask_tv,
                    #                                    mask_argsort_bwd_tv[j],
                    #                                    stream)

            else:
                # if pair_mask_bwd_tv.dim(1) > pair_mask_fwd_tv.dim(1):
                if not is_train:
                    SpconvOps.sort_1d_by_key_allocator(pair_mask_fwd_tv[0],
                                                       alloc.alloc,
                                                       mask_argsort_fwd_tv[0],
                                                       stream)
                else:
                    if pair_mask_bwd_tv.dim(1) > pair_mask_fwd_tv.dim(1):
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_bwd_tv[0], alloc.alloc,
                            mask_argsort_bwd_tv[0], stream)
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_fwd_tv[0], alloc.alloc,
                            mask_argsort_fwd_tv[0], stream)
                    else:
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_fwd_tv[0], alloc.alloc,
                            mask_argsort_fwd_tv[0], stream)
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_bwd_tv[0], alloc.alloc,
                            mask_argsort_bwd_tv[0], stream)
        if DEBUG:
            CONV.stream_synchronize(stream)
            print("REGU_S2_FINISH", time.time() - t)
            t = time.time()

        # CONV.stream_synchronize(stream)
        if not is_train:
            pair_bwd = torch.Tensor()
            pair_mask_bwd_splits: List[torch.Tensor] = []
            mask_argsort_bwd_splits: List[torch.Tensor] = []
        else:
            pair_mask_bwd_splits = [
                pair_mask_bwd[i] for i in range(mask_split_count)
            ]
            mask_argsort_bwd_splits = [
                mask_argsort_bwd[i] for i in range(mask_split_count)
            ]
        pair_mask_fwd_splits = [
            pair_mask_fwd[i] for i in range(mask_split_count)
        ]
        mask_argsort_fwd_splits = [
            mask_argsort_fwd[i] for i in range(mask_split_count)
        ]
        if DEBUG:
            CONV.stream_synchronize(stream)
            print("REGU", time.time() - t)

        return (out_inds, indice_num_per_loc, pair_fwd, pair_bwd,
                pair_mask_fwd_splits, pair_mask_bwd_splits,
                mask_argsort_fwd_splits, mask_argsort_bwd_splits, masks)


def indice_conv(features: torch.Tensor,
                filters: torch.Tensor,
                indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: int,
                inverse: bool = False,
                subm: bool = False,
                algo: ConvAlgo = ConvAlgo.Native,
                timer: CUDAKernelTimer = CUDAKernelTimer(False)):
    # filters: RSKC
    # stream = get_current_stream()
    # CONV.stream_synchronize(stream)
    # t = time.time()
    if not features.is_contiguous():
        features = features.contiguous()

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
        # out_features = torch.zeros((num_activate_out, out_channel),
        #                            dtype=features.dtype,
        #                            device=features.device)
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

    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    if subm and all(x == 0 for x in indice_pair_num_cpu):
        return out_features
    maxnhot = max(indice_pair_num_cpu)

    inited: bool = subm
    a = torch_tensor_to_tv(features)
    c = torch_tensor_to_tv(out_features)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    pair_in = indice_pairs_tv[int(inverse)]
    pair_out = indice_pairs_tv[int(not inverse)]
    filters_tv = torch_tensor_to_tv(filters)

    if not features.is_cuda:
        # perform gather-mm-scatter_add for cpu data
        assert not filters.is_cuda
        assert not indice_pairs.is_cuda
        inp_buffer = torch.empty([maxnhot, features.shape[1]],
                                 dtype=features.dtype)
        out_buffer = torch.empty([maxnhot, out_features.shape[1]],
                                 dtype=out_features.dtype)
        inp_buffer_tv = torch_tensor_to_tv(inp_buffer)
        out_buffer_tv = torch_tensor_to_tv(out_buffer)

        for i, nhot in enumerate(indice_pair_num_cpu):
            if subm and i == kv_center:
                continue
            if subm and i > kv_center:
                nhot = indice_pair_num_cpu[kv - i - 1]
            if nhot <= 0:
                continue
            inp_indices = pair_in[i].slice_first_axis(0, nhot)
            out_indices = pair_out[i].slice_first_axis(0, nhot)
            SpconvOps.gather_cpu(inp_buffer_tv, a, inp_indices)
            filters_cur = filters[i] if FILTER_HWIO else filters[i].T
            torch.mm(inp_buffer[:nhot], filters_cur, out=out_buffer[:nhot])
            SpconvOps.scatter_add_cpu(c, out_buffer_tv, out_indices)

        return out_features
    stream = get_current_stream()

    profile_idx = kv_center
    if subm:
        profile_idx = kv_center - 1
    # profile_idx = first_n
    nhot_profile = indice_pair_num_cpu[profile_idx]
    if nhot_profile == 0:
        # find a non-zero profile index
        profile_idx = 0
        for i, nhot in enumerate(indice_pair_num_cpu):
            if nhot > nhot_profile:
                nhot_profile = nhot
                profile_idx = i
    assert nhot_profile > 0, "this shouldn't happen"
    # print(nhot_profile, indice_pair_num_cpu)
    arch = get_arch()

    tuned_res = GEMM.get_tuned_algo(a.dtype,
                                    filters_tv.dtype,
                                    c.dtype,
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

    if tuned_res is None:
        # run profile on center
        inp_indices_th = indice_pairs[int(inverse)][profile_idx, :nhot_profile]
        out_indices_th = indice_pairs[int(not inverse)][
            profile_idx, :nhot_profile]
        inp_indices = torch_tensor_to_tv(inp_indices_th)
        out_indices = torch_tensor_to_tv(out_indices_th)
        filter_tv = torch_tensor_to_tv(filters)[profile_idx]

        tuned_res, min_time = GEMM.tune_and_cache(
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
    # CONV.stream_synchronize(stream)
    # t = time.time()
    with timer.record("forward", stream):
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
            algo_desp = GEMM.run_with_tuned_result(
                tuned_res,
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
    # CONV.stream_synchronize(stream)
    # print(out_features.mean(), out_features.max(), out_features.min())

    # # print(stream, valid_count, maxnhot, features.shape[0], features.shape[1], out_channel, time.time() - t, total_times, txt)
    # # print(algo_desp, tuned_res.external_gather, tuned_res.splitk, features.shape[0], features.shape[1], out_channel, time.time() - t)
    # print("F", time.time() - t)
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
                         algo: ConvAlgo = ConvAlgo.Native,
                         timer: CUDAKernelTimer = CUDAKernelTimer(False)):
    # print(out_bp.mean(), out_bp.max(), out_bp.min())

    num_activate_out = out_bp.shape[0]
    out_channel = out_bp.shape[-1]
    filters_shape = filters.shape
    filters = filters.reshape(-1, *filters.shape[-2:])
    kv = filters.shape[0]
    kv_center = kv // 2
    # TODO handle this in nn.Module to make sure features in backward is contiguous
    if not features.is_contiguous():
        features = features.contiguous()
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()
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
    if subm and all(x == 0 for x in indice_pair_num_cpu):
        return (din, dfilters.reshape(filters_shape))
    maxnhot = max(indice_pair_num_cpu)

    arch = get_arch()
    filters_tv = torch_tensor_to_tv(filters)

    dfilters_tv = torch_tensor_to_tv(dfilters)
    out_bp_tv = torch_tensor_to_tv(out_bp)
    features_tv = torch_tensor_to_tv(features)

    din_tv = torch_tensor_to_tv(din)

    if not features.is_cuda:
        # perform gather-mm-scatter_add for cpu data
        assert not filters.is_cuda
        assert not indice_pairs.is_cuda
        inp_buffer = torch.empty([maxnhot, features.shape[1]],
                                 dtype=features.dtype)
        out_buffer = torch.empty([maxnhot, out_bp.shape[1]],
                                 dtype=out_bp.dtype)
        inp_buffer_tv = torch_tensor_to_tv(inp_buffer)
        out_buffer_tv = torch_tensor_to_tv(out_buffer)

        for i, nhot in enumerate(indice_pair_num_cpu):
            if subm and i == kv_center:
                continue
            if subm and i > kv_center:
                nhot = indice_pair_num_cpu[kv - i - 1]
            if nhot <= 0:
                continue
            inp_indices = pair_in[i].slice_first_axis(0, nhot)
            out_indices = pair_out[i].slice_first_axis(0, nhot)
            SpconvOps.gather_cpu(inp_buffer_tv, features_tv, inp_indices)
            SpconvOps.gather_cpu(out_buffer_tv, out_bp_tv, out_indices)
            filters_T_cur = filters[i].T if FILTER_HWIO else filters[i]
            dfilters_cur = dfilters[i] if FILTER_HWIO else dfilters[i].T

            torch.mm(inp_buffer[:nhot].T, out_buffer[:nhot], out=dfilters_cur)
            torch.mm(out_buffer[:nhot], filters_T_cur, out=inp_buffer[:nhot])

            SpconvOps.scatter_add_cpu(din_tv, inp_buffer_tv, inp_indices)
        return (din, dfilters.reshape(filters_shape))

    profile_idx = kv_center
    if subm or indice_pair_num_cpu[profile_idx] == 0:
        profile_idx = kv_center - 1
    # profile_idx = first_n
    nhot_profile = indice_pair_num_cpu[profile_idx]
    if nhot_profile == 0:
        # find a non-zero profile index
        profile_idx = 0
        for i, nhot in enumerate(indice_pair_num_cpu):
            if nhot > nhot_profile:
                nhot_profile = nhot
                profile_idx = i
    assert nhot_profile > 0, "this shouldn't happen"

    # print(nhot_profile, indice_pair_num_cpu)
    tuned_res_dgrad = GEMM.get_tuned_algo(
        out_bp_tv.dtype,
        filters_tv.dtype,
        din_tv.dtype,
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
    if tuned_res_dgrad is None:
        inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile)
        out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile)
        filter_tv = filters_tv[profile_idx]
        tuned_res_dgrad, min_time = GEMM.tune_and_cache(
            out_bp_tv,
            filter_tv,
            din_tv,
            False,
            True if FILTER_HWIO else False,
            False,
            arch=arch,
            shuffle_type=ShuffleStrideType.ShuffleAC,
            a_inds=out_indices,
            c_inds=inp_indices,
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
    tuned_res_wgrad = GEMM.get_tuned_algo(
        a_wgrad.dtype,
        b_wgrad.dtype,
        filters_tv.dtype,
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

    if tuned_res_wgrad is None:
        inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile)
        out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile)
        dfilter_tv = dfilters_tv[profile_idx]
        if not FILTER_HWIO:
            a_inds_wgrad = out_indices
            b_inds_wgrad = inp_indices
        else:
            a_inds_wgrad = inp_indices
            b_inds_wgrad = out_indices
        tuned_res_wgrad, min_time = GEMM.tune_and_cache(
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
        # print(tuned_res_wgrad.algo_desp, tuned_res_wgrad.splitk, min_time)
    # get workspace size for wgrad
    if not FILTER_HWIO:
        a_shape = [maxnhot, out_bp_tv.dim(1)]
        b_shape = [maxnhot, features_tv.dim(1)]
    else:
        b_shape = [maxnhot, out_bp_tv.dim(1)]
        a_shape = [maxnhot, features_tv.dim(1)]
    m, n, k = GEMM.extract_mnk(a_shape,
                               b_shape,
                               tuned_res_wgrad.algo_desp.trans_a,
                               tuned_res_wgrad.algo_desp.trans_b,
                               tuned_res_wgrad.algo_desp.trans_c,
                               arch=arch,
                               shuffle_type=ShuffleStrideType.ShuffleAB,
                               a_inds_shape=[maxnhot],
                               b_inds_shape=[maxnhot],
                               hint=AlgoHint.BackwardWeight.value)
    workspace_size = tuned_res_wgrad.algo_desp.query_workspace_size(
        m, n, k, tuned_res_wgrad.splitk)
    workspace = torch.Tensor()

    workspace_tv = tv.Tensor()
    if workspace_size > 0:
        workspace = torch.empty((workspace_size, ),
                                dtype=torch.int8,
                                device=features.device)
        workspace_tv = torch_tensor_to_tv(workspace)
    # print(workspace_size, m, n, k, tuned_res_wgrad.splitk)
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
        GEMM.run_with_tuned_result(tuned_res_dgrad,
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
        GEMM.run_with_tuned_result(tuned_res_wgrad,
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
    # # print(dw_time + di_time, di_time, dw_time, tuned_res_wgrad.splitk, tuned_res_wgrad.algo_desp, dfilters.shape)
    # # print(dw_time + di_time)
    # print("BWG", time.time() - t)
    return (din, dfilters.reshape(filters_shape))


def implicit_gemm(features: torch.Tensor,
                  filters: torch.Tensor,
                  pair_fwd: torch.Tensor,
                  pair_mask_fwd_splits: List[torch.Tensor],
                  mask_argsort_fwd_splits: List[torch.Tensor],
                  num_activate_out: int,
                  masks: List[np.ndarray],
                  is_train: bool,
                  is_subm: bool,
                  timer: CUDAKernelTimer = CUDAKernelTimer(False),
                  fp32_accum: Optional[bool] = None):
    stream = get_current_stream()
    # if DEBUG:

    # CONV.stream_synchronize(stream)

    # t = time.time()
    if not features.is_contiguous():
        features = features.contiguous()
    assert features.is_contiguous()
    assert filters.is_contiguous()

    if features.dtype == torch.int8 or features.dtype == torch.qint8:
        raise NotImplementedError("work in progress")
    # here filters is KRSC
    masks_ints = [m.item() for m in masks]
    out_channel = filters.shape[0]
    in_channel = filters.shape[-1]
    num_split = len(pair_mask_fwd_splits)
    filters = filters.reshape(out_channel, -1, filters.shape[-1])
    kv = filters.shape[1]
    if is_subm:
        out_features = torch.empty((num_activate_out, out_channel),
                                   dtype=features.dtype,
                                   device=features.device)
    else:
        out_features = torch.zeros((num_activate_out, out_channel),
                                   dtype=features.dtype,
                                   device=features.device)

    pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
    features_tv = torch_tensor_to_tv(features)
    filters_tv = torch_tensor_to_tv(filters)
    out_features_tv = torch_tensor_to_tv(out_features)
    arch = get_arch()
    pair_mask_fwd_split_tvs = [
        torch_tensor_to_tv(x, dtype=tv.uint32) for x in pair_mask_fwd_splits
    ]
    mask_argsort_fwd_split_tvs = [
        torch_tensor_to_tv(x) for x in mask_argsort_fwd_splits
    ]
    # CONV.stream_synchronize(stream)
    # t = time.time()

    tune_res = CONV.get_tuned_algo(ConvOpType.kForward, features_tv.dtype,
                                   filters_tv.dtype, out_features_tv.dtype,
                                   out_channel, in_channel, arch)
    if tune_res is None:
        tune_res, _ = CONV.tune_and_cache(
            ConvOpType.kForward,
            features_tv,
            filters_tv,
            out_features_tv,
            NHWC,
            KRSC,
            NHWC,
            arch,
            mask=pair_mask_fwd_split_tvs[0],
            mask_argsort=mask_argsort_fwd_split_tvs[0],
            indices=pair_fwd_tv,
            reverse_mask=False,
            mask_filter=masks[0].item(),
            stream=stream,
            fp32_accum=fp32_accum)
    mask_width = tune_res.algo_desp.tile_shape[0]
    if is_train:
        mask_output_fwd = torch.empty(
            [num_split,
             codeops.div_up(num_activate_out, mask_width)],
            dtype=torch.int32,
            device=features.device)
        # pytorch don't support uint32.
        mask_output_fwd_tv = torch_tensor_to_tv(mask_output_fwd,
                                                dtype=tv.uint32)
        mask_output_fwd_tvs = [mask_output_fwd_tv[j] for j in range(num_split)]
    else:
        mask_output_fwd = None
        mask_output_fwd_tv = tv.Tensor()
        mask_output_fwd_tvs = [tv.Tensor() for _ in range(num_split)]
    # CONV.stream_synchronize(stream)
    # print("FPREPARE", time.time() - t)

    # # t = time.time()
    # CONV.stream_synchronize(stream)

    # t = time.time()
    with timer.record("implicit_gemm", stream):
        for j in range(num_split):
            beta = 0 if j == 0 else 1
            CONV.run_with_tuned_result(
                tune_res,
                ConvOpType.kForward,
                features_tv,
                filters_tv,
                out_features_tv,
                mask=pair_mask_fwd_split_tvs[j],
                mask_argsort=mask_argsort_fwd_split_tvs[j],
                mask_output=mask_output_fwd_tvs[j],
                indices=pair_fwd_tv,
                reverse_mask=False,
                mask_filter=masks_ints[j],
                mask_width=-1,
                beta=beta,
                stream=stream,
                verbose=False)

    # torch.cuda.synchronize()
    # if DEBUG:

    # CONV.stream_synchronize(stream)
    # dura = time.time() - t
    # print("F", tune_res.algo_desp, dura)
    # print(out_features.mean(), out_features.max(), out_features.min())

    return out_features, mask_output_fwd, mask_width


def implicit_gemm_backward(features: torch.Tensor,
                           filters: torch.Tensor,
                           out_bp: torch.Tensor,
                           pair_fwd: torch.Tensor,
                           pair_bwd: torch.Tensor,
                           pair_mask_fwd_splits: List[torch.Tensor],
                           pair_mask_bwd_splits: List[torch.Tensor],
                           mask_argsort_fwd_splits: List[torch.Tensor],
                           mask_argsort_bwd_splits: List[torch.Tensor],
                           mask_output_fwd: torch.Tensor,
                           masks: List[np.ndarray],
                           mask_width: int,
                           is_subm: bool,
                           timer: CUDAKernelTimer = CUDAKernelTimer(False),
                           fp32_accum: Optional[bool] = None):
    # print(out_bp.mean(), out_bp.max(), out_bp.min())
    if features.dtype == torch.int8 or features.dtype == torch.qint8:
        raise NotImplementedError("work in progress")
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()
    if not features.is_contiguous():
        features = features.contiguous()

    assert out_bp.is_contiguous()
    assert filters.is_contiguous()
    assert features.is_contiguous()
    # here filters is KRSC
    filters_shape = filters.shape
    out_channel = filters.shape[0]
    in_channel = filters.shape[-1]
    num_split = len(pair_mask_fwd_splits)
    if is_subm:
        din = torch.empty_like(features)
    else:
        din = torch.zeros_like(features)
    dfilters = torch.zeros_like(filters)

    filters = filters.reshape(out_channel, -1, filters.shape[-1])
    kv = filters.shape[1]

    stream = get_current_stream()
    pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
    pair_bwd_tv = torch_tensor_to_tv(pair_bwd)

    features_tv = torch_tensor_to_tv(features)
    filters_tv = torch_tensor_to_tv(filters)
    dfilters_tv = torch_tensor_to_tv(dfilters)

    dout_tv = torch_tensor_to_tv(out_bp)
    din_tv = torch_tensor_to_tv(din)
    mask_output_fwd_tv = torch_tensor_to_tv(mask_output_fwd, dtype=tv.uint32)
    arch = get_arch()
    pair_mask_fwd_split_tvs = [
        torch_tensor_to_tv(x, dtype=tv.uint32) for x in pair_mask_fwd_splits
    ]
    pair_mask_bwd_split_tvs = [
        torch_tensor_to_tv(x, dtype=tv.uint32) for x in pair_mask_bwd_splits
    ]

    mask_argsort_fwd_split_tvs = [
        torch_tensor_to_tv(x) for x in mask_argsort_fwd_splits
    ]
    mask_argsort_bwd_split_tvs = [
        torch_tensor_to_tv(x) for x in mask_argsort_bwd_splits
    ]

    dgrad_tune_res = CONV.get_tuned_algo(ConvOpType.kBackwardInput,
                                         din_tv.dtype, filters_tv.dtype,
                                         dout_tv.dtype, out_channel,
                                         in_channel, arch)
    wgrad_tune_res = CONV.get_tuned_algo(ConvOpType.kBackwardWeight,
                                         features_tv.dtype, dfilters_tv.dtype,
                                         dout_tv.dtype, out_channel,
                                         in_channel, arch, mask_width)

    if dgrad_tune_res is None:
        # TODO split mask maybe completely invalid
        if is_subm:
            mask = pair_mask_fwd_split_tvs[0]
            mask_argsort = mask_argsort_fwd_split_tvs[0]
        else:
            mask = pair_mask_bwd_split_tvs[0]
            mask_argsort = mask_argsort_bwd_split_tvs[0]

        dgrad_tune_res, _ = CONV.tune_and_cache(ConvOpType.kBackwardInput,
                                                din_tv,
                                                filters_tv,
                                                dout_tv,
                                                NHWC,
                                                KRSC,
                                                NHWC,
                                                arch,
                                                mask=mask,
                                                mask_argsort=mask_argsort,
                                                indices=pair_bwd_tv,
                                                reverse_mask=is_subm,
                                                mask_filter=masks[0].item(),
                                                stream=stream,
                                                fp32_accum=fp32_accum)
    if wgrad_tune_res is None:
        wgrad_tune_res, _ = CONV.tune_and_cache(
            ConvOpType.kBackwardWeight,
            features_tv,
            dfilters_tv,
            dout_tv,
            NHWC,
            KRSC,
            NHWC,
            arch,
            mask=pair_mask_fwd_split_tvs[0],
            mask_argsort=mask_argsort_fwd_split_tvs[0],
            indices=pair_fwd_tv,
            reverse_mask=False,
            mask_filter=masks[0].item(),
            mask_output=mask_output_fwd_tv[0],
            mask_width=mask_width,
            stream=stream)
    workspace_size = CONV.query_workspace_size(wgrad_tune_res.algo_desp,
                                               wgrad_tune_res.splitk,
                                               ConvOpType.kBackwardWeight,
                                               pair_fwd_tv.dim(1), in_channel,
                                               out_channel, kv)
    workspace = torch.Tensor()

    workspace_tv = tv.Tensor()
    if workspace_size > 0:
        workspace = torch.empty((workspace_size, ),
                                dtype=torch.int8,
                                device=features.device)
        workspace_tv = torch_tensor_to_tv(workspace)
    with timer.record("implicit_gemm_backward", stream):
        for j in range(num_split):
            beta = 0 if j == 0 else 1
            if is_subm:
                mask = pair_mask_fwd_split_tvs[j]
                mask_argsort = mask_argsort_fwd_split_tvs[j]
            else:
                mask = pair_mask_bwd_split_tvs[j]
                mask_argsort = mask_argsort_bwd_split_tvs[j]

            CONV.run_with_tuned_result(dgrad_tune_res,
                                       ConvOpType.kBackwardInput,
                                       din_tv,
                                       filters_tv,
                                       dout_tv,
                                       mask=mask,
                                       mask_argsort=mask_argsort,
                                       mask_output=tv.Tensor(),
                                       indices=pair_bwd_tv,
                                       reverse_mask=is_subm,
                                       mask_filter=masks[j].item(),
                                       mask_width=-1,
                                       beta=beta,
                                       stream=stream)
            CONV.run_with_tuned_result(
                wgrad_tune_res,
                ConvOpType.kBackwardWeight,
                features_tv,
                dfilters_tv,
                dout_tv,
                mask=mask_output_fwd_tv[j],
                mask_argsort=mask_argsort_fwd_split_tvs[j],
                mask_output=tv.Tensor(),
                indices=pair_fwd_tv,
                reverse_mask=False,
                mask_filter=masks[j].item(),
                mask_width=mask_width,
                beta=beta,
                workspace=workspace_tv,
                stream=stream)

    return (din, dfilters.reshape(filters_shape))


def indice_maxpool(features: torch.Tensor, indice_pairs: torch.Tensor,
                   indice_pair_num: torch.Tensor, num_activate_out):
    # torch.cuda.synchronize()
    # t = time.time()
    # stream = get_current_stream()
    # CONV.stream_synchronize(stream)
    # t = time.time()
    if not features.is_contiguous():
        features = features.contiguous()

    out_channel = features.shape[-1]
    out_features = torch.zeros((num_activate_out, out_channel),
                               dtype=features.dtype,
                               device=features.device)
    stream = 0
    is_cpu = not features.is_cuda
    if not is_cpu:
        stream = get_current_stream()
    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    for i, nhot in enumerate(indice_pair_num_cpu):
        if nhot <= 0:
            continue
        inp_indices = indice_pairs_tv[0][i].slice_first_axis(0, nhot)
        out_indices = indice_pairs_tv[1][i].slice_first_axis(0, nhot)
        if is_cpu:
            SpconvOps.maxpool_forward_cpu(out_features_tv, features_tv,
                                          out_indices, inp_indices)
        else:
            SpconvOps.maxpool_forward(out_features_tv, features_tv,
                                      out_indices, inp_indices, stream)

    # CONV.stream_synchronize(stream)
    # print("M", time.time() - t)

    return out_features


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs,
                            indice_pair_num):
    out_channel = features.shape[-1]
    din = torch.zeros_like(features)
    is_cpu = not features.is_cuda
    stream = 0
    if not is_cpu:
        stream = get_current_stream()
    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()
    if not features.is_contiguous():
        features = features.contiguous()

    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    out_bp_tv = torch_tensor_to_tv(out_bp)
    din_tv = torch_tensor_to_tv(din)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    for i, nhot in enumerate(indice_pair_num_cpu):
        if nhot <= 0:
            continue
        inp_indices = indice_pairs_tv[0][i].slice_first_axis(0, nhot)
        out_indices = indice_pairs_tv[1][i].slice_first_axis(0, nhot)
        if is_cpu:
            SpconvOps.maxpool_backward_cpu(out_features_tv, features_tv,
                                           out_bp_tv, din_tv, out_indices,
                                           inp_indices)
        else:
            SpconvOps.maxpool_backward(out_features_tv, features_tv, out_bp_tv,
                                       din_tv, out_indices, inp_indices,
                                       stream)

    return din


def indice_maxpool_implicit_gemm(features: torch.Tensor,
                                 indice_pairs: torch.Tensor, num_activate_out):
    # torch.cuda.synchronize()
    # t = time.time()
    stream = get_current_stream()
    # CONV.stream_synchronize(stream)
    # t = time.time()
    if not features.is_contiguous():
        features = features.contiguous()

    out_channel = features.shape[-1]
    out_features = torch.empty((num_activate_out, out_channel),
                               dtype=features.dtype,
                               device=features.device)
    assert features.is_cuda
    stream = get_current_stream()
    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    SpconvOps.maxpool_implicit_gemm_forward(out_features_tv, features_tv,
                                            indice_pairs_tv, stream)

    # CONV.stream_synchronize(stream)
    # print("M", time.time() - t)

    return out_features


def indice_maxpool_implicit_gemm_backward(features, out_features, out_bp,
                                          indice_pairs):
    # torch.cuda.synchronize()
    # t = time.time()
    out_channel = features.shape[-1]
    din = torch.zeros_like(features)
    assert features.is_cuda
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()
    if not features.is_contiguous():
        features = features.contiguous()

    stream = get_current_stream()
    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    out_bp_tv = torch_tensor_to_tv(out_bp)
    din_tv = torch_tensor_to_tv(din)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    SpconvOps.maxpool_implicit_gemm_backward(out_features_tv, features_tv,
                                             out_bp_tv, din_tv,
                                             indice_pairs_tv, stream)
    return din

