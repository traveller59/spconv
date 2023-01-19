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
from typing import Dict, List, Optional, Union
from spconv.pytorch.core import ThrustSortAllocator
from spconv.pytorch.cppcore import _TORCH_DTYPE_TO_TV, TorchAllocator, torch_tensor_to_tv, get_current_stream, get_arch, TorchSpconvMatmul
from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.core_cc.csrc.sparse.alloc import ExternalAllocator
from spconv.constants import SPCONV_CPP_INDICE_PAIRS, SPCONV_CPP_INDICE_PAIRS_IGEMM, SPCONV_CPP_GEMM, SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE, SPCONV_ALLOW_TF32, SPCONV_DO_SORT
import spconv.core_cc as _ext
from spconv.core_cc.csrc.sparse.convops.spops import ConvGemmOps
from spconv.core_cc.csrc.sparse.inference import InferenceOps
from spconv.cppconstants import CPU_ONLY_BUILD
from cumm.gemm.codeops import div_up
from spconv.utils import nullcontext

if not CPU_ONLY_BUILD:
    from spconv.algo import GEMM, CONV, GEMM_CPP, CONV_CPP
else:
    GEMM = None
    CONV = None
    GEMM_CPP = None
    CONV_CPP = None
import time
from spconv.constants import FILTER_HWIO, ALL_WEIGHT_IS_KRSC, AllocKeys, SPCONV_USE_DIRECT_TABLE
from cumm.gemm import codeops
from spconv.tools import CUDAKernelTimer
from spconv import constants

DEBUG = False
DEBUG_INT64_HASH_K = False
INT32_MAX = SpconvOps.get_int32_max()

_POINT_VANISH_MSG = """Your points vanished here, this usually because you provide 
conv params that may ignore some input points. Example: 
    spatial_shape=[8, 200, 200]
    ksize=3
    stride=2
    padding=[0, 1, 1]
    dilation=1
    Coordinates=[[0, 7, 153, 142]]
these params will cause ALL points in z == 7 dropped because of padding_z=0.
enlarge your spatial shape or change your conv param to make sure 
every input point has a corresponding output point.
Your Conv Params: 
    spatial_shape={}
    ksize={}
    stride={}
    padding={}
    dilation={}"""


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


class _HashData:

    def __init__(self,
                 num: int,
                 use_i64: bool,
                 device: torch.device,
                 rate: float = 2.0) -> None:
        if use_i64:
            self.hashdata_k = torch.empty((int(num * rate), ),
                                          dtype=torch.int64,
                                          device=device)
            self.hashdata_v = torch.empty((int(num * rate), ),
                                          dtype=torch.int32,
                                          device=device)
            self.hashdata_k_tv = torch_tensor_to_tv(self.hashdata_k)
            self.hashdata_v_tv = torch_tensor_to_tv(self.hashdata_v)

        else:
            self.hashdata = torch.empty((
                2,
                int(num * rate),
            ),
                                        dtype=torch.int32,
                                        device=device)
            hashdata_tv = torch_tensor_to_tv(self.hashdata)
            if num == 0:
                self.hashdata_k_tv = tv.Tensor()
                self.hashdata_v_tv = tv.Tensor()
            else:
                self.hashdata_k_tv = hashdata_tv[0]
                self.hashdata_v_tv = hashdata_tv[1]


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
                     transpose: bool = False,
                     num_out_act_bound: int = -1):
    # torch.cuda.synchronize()
    # t = time.time()
    # stream = get_current_stream()

    # CONV.stream_synchronize(stream)
    # t = time.time()
    if SPCONV_CPP_INDICE_PAIRS:
        alloc = TorchAllocator(indices.device)
        stream = 0
        if indices.is_cuda:
            stream = get_current_stream()

        num_act_out = SpconvOps.get_indice_pairs(alloc,
                                                 torch_tensor_to_tv(indices),
                                                 batch_size, spatial_shape,
                                                 algo.value, ksize, stride,
                                                 padding, dilation,
                                                 out_padding, subm, transpose,
                                                 stream)
        if subm:
            out_inds = indices
        else:
            out_inds = alloc.allocated[AllocKeys.OutIndices]
        pair = alloc.allocated[AllocKeys.PairFwd]
        indice_num_per_loc = alloc.allocated[AllocKeys.IndiceNumPerLoc]
        # print(subm, out_inds.shape, pair.shape, indice_num_per_loc.shape, num_act_out)
        return out_inds[:num_act_out], pair, indice_num_per_loc
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
    spatial_volume = functools.reduce(lambda x, y: x * y, out_shape, 1) * batch_size
    use_int64_hash_k = spatial_volume >= INT32_MAX or DEBUG_INT64_HASH_K
    indice_dtype = torch.int64 if use_int64_hash_k else indices.dtype
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
            hashdata = _HashData(out_inds.shape[0], use_int64_hash_k,
                                 indices.device)
            # hashdata = torch.empty((out_inds.shape[0] * 2, ),
            #                        dtype=torch.int64,
            #                        device=indices.device)
            out_inds_tv = torch_tensor_to_tv(out_inds)
            # hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
            SpconvOps.generate_subm_conv_inds(inds_tv,
                                              hashdata.hashdata_k_tv,
                                              hashdata.hashdata_v_tv,
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
                                            dtype=indice_dtype,
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
            if (num_act_out == 0):
                msg = _POINT_VANISH_MSG.format(spatial_shape, ksize, stride, padding, dilation)
                raise ValueError(msg)
            use_bound_algo = False
            if num_out_act_bound > 0 and num_act_out > num_out_act_bound:
                num_act_out = num_out_act_bound
                use_bound_algo = True
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
            # hashdata = torch.empty((out_inds.shape[0] * 2, ),
            #                        dtype=torch.int64,
            #                        device=indices.device)
            hashdata = _HashData(out_inds.shape[0], use_int64_hash_k,
                                 indices.device)

            out_inds_tv = torch_tensor_to_tv(out_inds)
            # hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
            SpconvOps.generate_conv_inds_stage2(inds_tv,
                                                hashdata.hashdata_k_tv,
                                                hashdata.hashdata_v_tv,
                                                pair_tv,
                                                uniq_res_tv,
                                                indice_pairs_uniq_tv,
                                                out_inds_tv,
                                                indice_num_per_loc_tv,
                                                num_out_act=num_act_out,
                                                batch_size=batch_size,
                                                output_dims=out_shape,
                                                input_dims=spatial_shape,
                                                ksize=ksize,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                transposed=transpose,
                                                stream_int=stream,
                                                use_bound_algo=use_bound_algo)
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
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        num_out_act_bound: int = -1,
        direct_table: bool = SPCONV_USE_DIRECT_TABLE,
        do_sort=SPCONV_DO_SORT):
    """
    Why return tuple? because pytorch seems don't support custom object in autograd.
    return: (
        out_inds,
        num_inds_per_loc,
        pair_fwd,
        pair_bwd, # torch.Tensor() if subm or inference mode
        pair_mask_fwd_splits,
        pair_mask_bwd_splits, # torch.Tensor() if subm or inference mode
        mask_argsort_fwd_splits,
        mask_argsort_bwd_splits, # torch.Tensor() if subm or inference mode
        masks,
    )
    direct_table: a hash-based regular conv pair gen algo to avoid unique operation.
    runs faster than pytorch unique with num_voxel < 1000k.
    """
    stream = get_current_stream()
    if SPCONV_CPP_INDICE_PAIRS_IGEMM:
        thalloc = TorchAllocator(indices.device)
        timer_cpp = tv.CUDAKernelTimer(False)
        if timer._timer is not None:
            timer_cpp = timer._timer
        mask_tensor, num_act_out = SpconvOps.get_indice_pairs_implicit_gemm(
            thalloc,
            torch_tensor_to_tv(indices),
            batch_size,
            spatial_shape,
            algo.value,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
            is_train,
            stream,
            num_out_act_bound,
            timer=timer_cpp,
            direct_table=direct_table,
            do_sort=do_sort)
        mask_split_count = mask_tensor.dim(0)
        masks = [mask_tensor[i:i + 1].numpy() for i in range(mask_split_count)]
        if subm:
            out_inds = indices
        else:
            out_inds = thalloc.allocated[AllocKeys.OutIndices]
        indice_num_per_loc = thalloc.allocated[AllocKeys.IndiceNumPerLoc]
        if subm:
            # for subm, if training, pair shape is [2, kv, ...]
            # if not training, pair is [1, kv, ...]
            pair = thalloc.allocated[AllocKeys.PairFwd]
            pair_mask = thalloc.allocated[AllocKeys.PairMask]
            mask_argsort = thalloc.allocated[AllocKeys.MaskArgSort]
            pair_mask_in_splits = [
                pair_mask[i] for i in range(mask_split_count)
            ]
            mask_argsort_in_splits = [
                mask_argsort[i] for i in range(mask_split_count)
            ]
            pair_bwd = torch.Tensor()
            pair_fwd = pair[0]
            if is_train:
                assert pair.shape[0] == 2
                pair_bwd = pair[1]
            return (out_inds, indice_num_per_loc, pair[0], pair_bwd,
                    pair_mask_in_splits, [], mask_argsort_in_splits, [], masks)
        else:
            pair_bwd = thalloc.allocated.get(AllocKeys.PairBwd, torch.Tensor())
            pair_fwd = thalloc.allocated[AllocKeys.PairFwd]
            pair_mask_fwd = thalloc.allocated[AllocKeys.PairMask]
            pair_mask_bwd = torch.Tensor()
            mask_argsort_bwd = torch.Tensor()
            if is_train:
                pair_mask_bwd = thalloc.allocated[AllocKeys.PairMaskBwd]
                mask_argsort_bwd = thalloc.allocated[AllocKeys.MaskArgSortBwd]
            mask_argsort_fwd = thalloc.allocated[AllocKeys.MaskArgSort]
            if not is_train:
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
            return (out_inds, indice_num_per_loc, pair_fwd, pair_bwd,
                    pair_mask_fwd_splits, pair_mask_bwd_splits,
                    mask_argsort_fwd_splits, mask_argsort_bwd_splits, masks)
    assert indices.is_cuda, "implicit gemm only support cuda"
    ndim = indices.shape[1] - 1
    kv: int = functools.reduce(lambda x, y: x * y, ksize, 1)
    # TODO in future we will support up to 128 kernel volume.
    # assert kv <= 32, "currently only support kernel volume <= 32 to use implicit gemm"
    mask_int_count = div_up(kv, 32)

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
    spatial_volume = functools.reduce(lambda x, y: x * y, spatial_shape, 1) * batch_size
    use_int64_hash_k = spatial_volume >= INT32_MAX or DEBUG_INT64_HASH_K
    indice_dtype = torch.int64 if use_int64_hash_k else indices.dtype
    assert algo == ConvAlgo.MaskImplicitGemm or algo == ConvAlgo.MaskSplitImplicitGemm, "TODO"
    is_mask_split = algo == ConvAlgo.MaskSplitImplicitGemm
    mask_split_count = 2 if is_mask_split else 1
    if subm:
        if is_train:
            pair = torch.full((2, kv, indices.shape[0]),
                              -1,
                              dtype=indices.dtype,
                              device=indices.device)
        else:
            pair = torch.full((1, kv, indices.shape[0]),
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
        assert mask_int_count == 1, "Not Implemented"
        kv_div_2 = kv // 2
        remain = kv - kv_div_2
        mask_np_1 = np.array([1], dtype=np.uint64)
        first = ((mask_np_1 << (remain)) - 1)
        second = ((mask_np_1 << (kv_div_2)) - 1) << remain
        masks = [first.astype(np.uint32), second.astype(np.uint32)]
    else:
        masks = [np.array([0xffffffff], dtype=np.uint32)]

    if subm:
        out_inds = indices
        # hashdata = torch.empty((out_inds.shape[0] * 2, ),
        #                        dtype=torch.int64,
        #                        device=indices.device)
        hashdata = _HashData(out_inds.shape[0], use_int64_hash_k,
                             indices.device)

        pair_mask = torch.empty((mask_split_count, indices.shape[0], mask_int_count),
                                dtype=torch.int32,
                                device=indices.device)

        out_inds_tv = torch_tensor_to_tv(out_inds)
        # hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
        pair_mask_tv = torch_tensor_to_tv(pair_mask, dtype=tv.uint32)
        with timer.record("gen_subm_inds", stream):
            SpconvOps.generate_subm_conv_inds(inds_tv,
                                              hashdata.hashdata_k_tv,
                                              hashdata.hashdata_v_tv,
                                              pair_tv,
                                              out_inds_tv,
                                              indice_num_per_loc_tv,
                                              batch_size=batch_size,
                                              input_dims=spatial_shape,
                                              ksize=ksize,
                                              dilation=dilation,
                                              indice_pair_mask=pair_mask_tv,
                                              backward=is_train,
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
                                                    mask_argsort_tv[j], stream,
                                                    mask_int_count,
                                                    do_sort=do_sort)
        # CONV.stream_synchronize(stream)
        pair_mask_in_splits = [pair_mask[i] for i in range(mask_split_count)]
        mask_argsort_in_splits = [
            mask_argsort[i] for i in range(mask_split_count)
        ]
        if is_train:
            return (out_inds, indice_num_per_loc, pair[0], pair[1],
                    pair_mask_in_splits, [], mask_argsort_in_splits, [], masks)
        else:
            return (out_inds, indice_num_per_loc, pair[0], torch.Tensor(),
                    pair_mask_in_splits, [], mask_argsort_in_splits, [], masks)
    else:
        max_num_act = SpconvOps.get_handcrafted_max_act_out(
            indices.shape[0], ksize, stride, padding, dilation)
        if transpose:
            max_num_act = kv * indices.shape[0]
        pair_bwd = pair
        pair_bwd_tv = pair_tv
        indice_pairs_uniq = torch.empty((pair.numel() + 1, ),
                                        dtype=indice_dtype,
                                        device=indices.device)
        indice_pairs_uniq_tv = torch_tensor_to_tv(indice_pairs_uniq)
        hashdata = _HashData(0, use_int64_hash_k, indices.device)
        indice_pairs_uniq_bkp_tv = tv.Tensor()
        if direct_table:
            # print("HASH SIZE", max_num_act * 2)
            hashdata = _HashData(max_num_act, use_int64_hash_k, indices.device,
                                 SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE)
            indice_pairs_uniq_bkp = torch.empty((pair.numel() + 1, ),
                                                dtype=indice_dtype,
                                                device=indices.device)
            indice_pairs_uniq_bkp_tv = torch_tensor_to_tv(
                indice_pairs_uniq_bkp)
            with timer.record("gen_conv_inds_stage1", stream):
                SpconvOps.generate_conv_inds_mask_stage1_direct_table(
                    inds_tv,
                    hashdata.hashdata_k_tv,
                    hashdata.hashdata_v_tv,
                    pair_bwd_tv,
                    indice_pairs_uniq_bkp_tv,
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
        else:
            with timer.record("gen_conv_inds_stage1", stream):
                SpconvOps.generate_conv_inds_mask_stage1(
                    inds_tv,
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
        uniq_out_indices_offset_tv = tv.Tensor()
        with timer.record(f"unique_{indice_pairs_uniq.shape[0]}", stream):
            if direct_table:
                uniq_cnt = torch.zeros([1],
                                       dtype=torch.int32,
                                       device=indices.device)
                uniq_cnt_tv = torch_tensor_to_tv(uniq_cnt)
                num_act_out = SpconvOps.unique_hash(hashdata.hashdata_k_tv,
                                                    hashdata.hashdata_v_tv,
                                                    uniq_cnt_tv,
                                                    indice_pairs_uniq_tv,
                                                    num_out_act_bound, stream)
                uniq_out_indices_offset_tv = indice_pairs_uniq_tv
                raw_out_indices_offset_tv = indice_pairs_uniq_bkp_tv
            else:
                uniq_res = indice_pairs_uniq.unique()
                num_act_out = uniq_res.shape[0] - 1
                uniq_out_indices_offset_tv = torch_tensor_to_tv(uniq_res)
                raw_out_indices_offset_tv = indice_pairs_uniq_tv
            if (num_act_out == 0):
                msg = _POINT_VANISH_MSG.format(spatial_shape, ksize, stride, padding, dilation)
                raise ValueError(msg)

        if num_out_act_bound > 0 and num_act_out > num_out_act_bound:
            num_act_out = num_out_act_bound
        with timer.record(f"alloc_stage2", stream):

            out_inds = torch.empty((num_act_out, indices.shape[1]),
                                   dtype=indices.dtype,
                                   device=indices.device)

            pair_fwd = torch.full((kv, num_act_out),
                                  -1,
                                  dtype=indices.dtype,
                                  device=indices.device)
            pair_mask_fwd = torch.zeros((mask_split_count, num_act_out, mask_int_count),
                                        dtype=torch.int32,
                                        device=indices.device)
            pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
            pair_mask_fwd_tv = torch_tensor_to_tv(pair_mask_fwd,
                                                  dtype=tv.uint32)
            pair_mask_bwd = torch.Tensor()
            pair_mask_bwd_tv = tv.Tensor()
            if is_train:
                pair_mask_bwd = torch.zeros(
                    (mask_split_count, indices.shape[0], mask_int_count),
                    dtype=torch.int32,
                    device=indices.device)
                pair_mask_bwd_tv = torch_tensor_to_tv(pair_mask_bwd,
                                                      dtype=tv.uint32)
        if not direct_table:
            hashdata = _HashData(out_inds.shape[0], use_int64_hash_k,
                                 indices.device)

        # hashdata = torch.empty((out_inds.shape[0] * 2, ),
        #                        dtype=torch.int64,
        #                        device=indices.device)
        out_inds_tv = torch_tensor_to_tv(out_inds)
        # hashdata_tv = torch_tensor_to_tv(hashdata, dtype=tv.custom64)
        with timer.record(f"gen_conv_inds_stage2_{num_act_out}", stream):
            stage2_fn = SpconvOps.generate_conv_inds_mask_stage2
            if direct_table:
                SpconvOps.assign_output_direct_hash(indice_pairs_uniq_tv,
                                                    out_inds_tv,
                                                    batch_size=batch_size,
                                                    output_dims=out_shape,
                                                    input_dims=spatial_shape,
                                                    ksize=ksize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    stream_int=stream)
                stage2_fn = SpconvOps.generate_conv_inds_stage2_mask_direct_table

            stage2_fn(inds_tv,
                      hashdata.hashdata_k_tv,
                      hashdata.hashdata_v_tv,
                      pair_fwd_tv,
                      pair_bwd_tv,
                      uniq_out_indices_offset_tv,
                      raw_out_indices_offset_tv,
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
                                                                 stream,
                                                                 mask_int_count, do_sort=do_sort)
                else:
                    if pair_mask_bwd_tv.dim(1) > pair_mask_fwd_tv.dim(1):
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_bwd_tv[0], alloc.alloc,
                            mask_argsort_bwd_tv[0], stream, mask_int_count, do_sort=do_sort)
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_fwd_tv[0], alloc.alloc,
                            mask_argsort_fwd_tv[0], stream, mask_int_count, do_sort=do_sort)
                    else:
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_fwd_tv[0], alloc.alloc,
                            mask_argsort_fwd_tv[0], stream, mask_int_count, do_sort=do_sort)
                        SpconvOps.sort_1d_by_key_allocator(
                            pair_mask_bwd_tv[0], alloc.alloc,
                            mask_argsort_bwd_tv[0], stream, mask_int_count, do_sort=do_sort)

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
                timer: CUDAKernelTimer = CUDAKernelTimer(False),
                bias: Optional[torch.Tensor] = None,
                act_alpha: float = 0.0,
                act_beta: float = 0.0,
                act_type: tv.gemm.Activation = tv.gemm.Activation.None_):
    # filters: RSKC
    # stream = get_current_stream()
    # CONV.stream_synchronize(stream)
    # t = time.time()
    if not features.is_contiguous():
        features = features.contiguous()
    if features.dtype == torch.int8 or features.dtype == torch.qint8:
        raise NotImplementedError("work in progress")
    bias_tv = tv.Tensor()
    if bias is not None:
        bias_tv = torch_tensor_to_tv(bias)

    if SPCONV_CPP_GEMM and GEMM_CPP is not None:
        # print("CPPPPPP!!!", features.device)
        alloc = TorchAllocator(features.device)
        ext_mm = TorchSpconvMatmul(alloc)

        # from spconv.core_cc.csrc.sparse.convops import SimpleExternalSpconvMatmul
        # if features.is_cuda:
        #     ext_mm = SimpleExternalSpconvMatmul(alloc)
        # else:
        #     ext_mm = TorchSpconvMatmul(alloc)

        alloc.allocated[AllocKeys.Features] = features
        alloc.allocated[AllocKeys.Filters] = filters

        features_tv = torch_tensor_to_tv(features)
        indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
        indice_pair_num_tv = torch_tensor_to_tv(indice_pair_num)
        filters_tv = torch_tensor_to_tv(filters)
        stream = 0
        arch = (0, 0)
        if features.is_cuda:
            # plain get_arch by cuda api is VERY SLOW.
            arch = get_arch()
            stream = get_current_stream()
        ConvGemmOps.indice_conv(alloc, ext_mm, GEMM_CPP, ALL_WEIGHT_IS_KRSC,
                                FILTER_HWIO, features_tv, filters_tv,
                                indice_pairs_tv, indice_pair_num_tv, arch,
                                num_activate_out, inverse, subm, algo.value,
                                stream, bias_tv, act_alpha, act_beta, act_type,
                                use_tf32=constants.SPCONV_ALLOW_TF32)
        out_features = alloc.allocated[AllocKeys.OutFeatures]
        return out_features
    if not features.is_cuda:
        stream = 0
    else:
        stream = get_current_stream()

    has_bias = bias is not None
    has_act = act_type != tv.gemm.Activation.None_
    if has_bias or has_act:
        assert features.is_cuda, "cpu don't support act and bias"
    if not ALL_WEIGHT_IS_KRSC:
        kv_dim = 0
        is_KC_not_CK = not FILTER_HWIO
        if FILTER_HWIO:
            out_channel = filters.shape[-1]
            filter_shape_per_kv = [filters.shape[-2], out_channel]
        else:
            out_channel = filters.shape[-2]
            filter_shape_per_kv = [out_channel, filters.shape[-1]]
        filters = filters.reshape(-1, *filters.shape[-2:])
        kv = filters.shape[0]
    else:
        kv_dim = 1
        out_channel = filters.shape[0]
        filters = filters.reshape(out_channel, -1, filters.shape[-1])
        is_KC_not_CK = True
        kv = filters.shape[1]
        filter_shape_per_kv = [out_channel, filters.shape[-1]]

    kv_center = kv // 2
    if subm:
        # out_features = torch.zeros((num_activate_out, out_channel),
        #                            dtype=features.dtype,
        #                            device=features.device)
        if not ALL_WEIGHT_IS_KRSC:
            if not is_KC_not_CK:
                out_features = torch.mm(features, filters[kv_center])
            else:
                out_features = torch.mm(features, filters[kv_center].T)
        else:
            if features.is_cuda or (features.dtype != torch.float16):
                out_features = torch.mm(features, filters[:, kv_center].T)
            else:
                # pytorch 1.12 don't support cpu half mm, f**k pytorch
                # we need cpu fp16 mm for test only.
                out_features = torch.empty((features.shape[0], out_channel),
                                           dtype=features.dtype,
                                           device=features.device)
                features_np = torch_tensor_to_tv(features).numpy_view()
                filters_np = torch_tensor_to_tv(filters).numpy_view()
                out_features_np = torch_tensor_to_tv(out_features).numpy_view()
                np.matmul(features_np,
                          filters_np[:, kv_center].T,
                          out=out_features_np)
                # out_features = torch.mm(features, filters[:, kv_center].T)
    else:
        out_features = torch.zeros((num_activate_out, out_channel),
                                   dtype=features.dtype,
                                   device=features.device)
    c = torch_tensor_to_tv(out_features)

    if kv == 1 and subm:
        if (has_act and has_bias):
            InferenceOps.bias_add_act_inplace(c, bias_tv, act_type, act_alpha, act_beta, stream)
        else:
            if has_act:
                InferenceOps.activation_inplace(c, act_type, act_alpha, act_beta, stream)
            if has_bias:
                InferenceOps.bias_add_inplace(c, bias_tv, stream)

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
            filters_i = filters.select(kv_dim, i)
            filters_cur = filters_i if not is_KC_not_CK else filters_i.T
            if features.dtype == torch.float16:
                inp_buffer_np = torch_tensor_to_tv(inp_buffer).numpy_view()
                filters_np = torch_tensor_to_tv(filters).numpy_view()
                filters_i_np = filters_np[
                    i] if not ALL_WEIGHT_IS_KRSC else filters_np[:, i]
                filters_cur_np = filters_i_np if not is_KC_not_CK else filters_i_np.T
                out_buffer_np = torch_tensor_to_tv(out_buffer).numpy_view()
                np.matmul(inp_buffer_np[:nhot],
                          filters_cur_np,
                          out=out_buffer_np[:nhot])
            else:
                torch.mm(inp_buffer[:nhot], filters_cur, out=out_buffer[:nhot])
            SpconvOps.scatter_add_cpu(c, out_buffer_tv, out_indices)

        return out_features

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
                                    filter_shape_per_kv,
                                    c.shape,
                                    False,
                                    is_KC_not_CK,
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
        # filter_tv = torch_tensor_to_tv(filters)[profile_idx]
        filter_tv = torch_tensor_to_tv(filters).select(kv_dim, profile_idx)

        tuned_res, min_time = GEMM.tune_and_cache(
            a,
            filter_tv,
            c,
            False,
            is_KC_not_CK,
            False,
            arch=arch,
            shuffle_type=ShuffleStrideType.ShuffleAC,
            a_inds=inp_indices,
            c_inds=out_indices,
            alpha=1.0,
            beta=0.0,
            hint=AlgoHint.Fowrard.value,
            stream=stream,
            use_tf32=constants.SPCONV_ALLOW_TF32)
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
            b = filters_tv.select(kv_dim, i)
            # inp @ filter.T, NC @ KC
            beta = 1.0 if inited else 0.0
            algo_desp = GEMM.run_with_tuned_result(
                tuned_res,
                a,
                b,
                c,
                False,
                is_KC_not_CK,
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
        if (has_act and has_bias):
            InferenceOps.bias_add_act_inplace(c, bias_tv, act_type, act_alpha, act_beta, stream)
        else:
            if has_act:
                InferenceOps.activation_inplace(c, act_type, act_alpha, act_beta, stream)
            if has_bias:
                InferenceOps.bias_add_inplace(c, bias_tv, stream)

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
    filters_shape = filters.shape
    # TODO handle this in nn.Module to make sure features in backward is contiguous
    if not features.is_contiguous():
        features = features.contiguous()
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()
    assert out_bp.is_contiguous()
    assert filters.is_contiguous()
    assert features.is_contiguous()

    if SPCONV_CPP_GEMM and GEMM_CPP is not None:
        alloc = TorchAllocator(features.device)
        ext_mm = TorchSpconvMatmul(alloc)
        alloc.allocated[AllocKeys.Features] = features
        alloc.allocated[AllocKeys.Filters] = filters
        alloc.allocated[AllocKeys.OutBp] = out_bp

        features_tv = torch_tensor_to_tv(features)
        out_bp_tv = torch_tensor_to_tv(out_bp)

        indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
        indice_pair_num_tv = torch_tensor_to_tv(indice_pair_num)
        filters_tv = torch_tensor_to_tv(filters)
        stream = 0
        arch = (0, 0)

        if features.is_cuda:
            stream = get_current_stream()
            arch = get_arch()
        ConvGemmOps.indice_conv_backward(alloc, ext_mm, GEMM_CPP,
                                         ALL_WEIGHT_IS_KRSC, FILTER_HWIO,
                                         features_tv, filters_tv, out_bp_tv,
                                         indice_pairs_tv, indice_pair_num_tv,
                                         arch, inverse, subm, algo.value,
                                         stream, use_tf32=constants.SPCONV_ALLOW_TF32)
        din = alloc.allocated[AllocKeys.DIn]
        df = alloc.allocated[AllocKeys.DFilters]
        return din, df

    if not ALL_WEIGHT_IS_KRSC:
        kv_dim = 0
        is_KC_not_CK = not FILTER_HWIO
        if FILTER_HWIO:
            out_channel = filters.shape[-1]
            filter_shape_per_kv = [filters.shape[-2], out_channel]
        else:
            out_channel = filters.shape[-2]
            filter_shape_per_kv = [out_channel, filters.shape[-1]]
        filters = filters.reshape(-1, *filters.shape[-2:])
        kv = filters.shape[0]

    else:
        kv_dim = 1
        out_channel = filters.shape[0]
        filters = filters.reshape(out_channel, -1, filters.shape[-1])
        is_KC_not_CK = True
        kv = filters.shape[1]
        filter_shape_per_kv = [out_channel, filters.shape[-1]]

    kv_center = kv // 2

    if subm:
        dfilters = torch.zeros_like(filters)
        if not ALL_WEIGHT_IS_KRSC:
            if not is_KC_not_CK:
                torch.mm(features.T, out_bp, out=dfilters[kv_center])
                din = torch.mm(out_bp, filters[kv_center].T)
            else:
                torch.mm(out_bp.T, features, out=dfilters[kv_center])
                din = torch.mm(out_bp, filters[kv_center])
        else:
            # KN @ NC
            torch.mm(out_bp.T, features, out=dfilters[:, kv_center])
            # NK @ KC
            din = torch.mm(out_bp, filters[:, kv_center])

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
            filters_i = filters.select(kv_dim, i)
            dfilters_i = dfilters.select(kv_dim, i)

            filters_KC = filters_i if is_KC_not_CK else filters_i.T
            if is_KC_not_CK:
                # KN @ NC
                torch.mm(out_buffer[:nhot].T,
                         inp_buffer[:nhot],
                         out=dfilters_i)
            else:
                # CN @ NK
                torch.mm(inp_buffer[:nhot].T,
                         out_buffer[:nhot],
                         out=dfilters_i)
            # NK @ KC
            torch.mm(out_buffer[:nhot], filters_KC, out=inp_buffer[:nhot])
            SpconvOps.scatter_add_cpu(din_tv, inp_buffer_tv, inp_indices)
        return (din, dfilters.reshape(filters_shape))
    arch = get_arch()
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
        filter_shape_per_kv,
        din_tv.shape,
        False,
        not is_KC_not_CK,
        False,
        arch=arch,
        shuffle_type=ShuffleStrideType.ShuffleAC,
        a_inds_shape=[nhot_profile],
        c_inds_shape=[nhot_profile],
        hint=AlgoHint.BackwardInput.value)
    if tuned_res_dgrad is None:
        inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile)
        out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile)
        filter_tv = filters_tv.select(kv_dim, profile_idx)
        tuned_res_dgrad, min_time = GEMM.tune_and_cache(
            out_bp_tv,
            filter_tv,
            din_tv,
            False,
            not is_KC_not_CK,
            False,
            arch=arch,
            shuffle_type=ShuffleStrideType.ShuffleAC,
            a_inds=out_indices,
            c_inds=inp_indices,
            alpha=1.0,
            beta=0.0,
            hint=AlgoHint.BackwardInput.value,
            stream=stream,
            use_tf32=constants.SPCONV_ALLOW_TF32)
    if is_KC_not_CK:
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
        filter_shape_per_kv,
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
        dfilter_tv = dfilters_tv.select(kv_dim, profile_idx)
        if is_KC_not_CK:
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
            stream=stream,
            use_tf32=constants.SPCONV_ALLOW_TF32)
        # print(tuned_res_wgrad.algo_desp, tuned_res_wgrad.splitk, min_time)
    # get workspace size for wgrad
    if is_KC_not_CK:
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
        filter_i_tv = filters_tv.select(kv_dim, i)
        GEMM.run_with_tuned_result(tuned_res_dgrad,
                                   out_bp_tv,
                                   filter_i_tv,
                                   din_tv,
                                   False,
                                   not is_KC_not_CK,
                                   False,
                                   arch=arch,
                                   stream=stream,
                                   shuffle_type=ShuffleStrideType.ShuffleAC,
                                   a_inds=out_indices,
                                   c_inds=inp_indices,
                                   hint=AlgoHint.BackwardInput.value,
                                   alpha=1.0,
                                   beta=beta)

        if is_KC_not_CK:
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
                                   dfilters_tv.select(kv_dim, i),
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
                  fp32_accum: Optional[bool] = None,
                  bias: Optional[torch.Tensor] = None,
                  act_alpha: float = 0.0,
                  act_beta: float = 0.0,
                  act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                  output_scale: float = 1.0,
                  scale: Optional[torch.Tensor] = None,
                  output_add: Optional[torch.Tensor] = None,
                  output_add_scale: float = 0.0,
                  output_dtype: Optional[torch.dtype] = None):
    stream = get_current_stream()
    bias_tv = tv.Tensor()
    scale_tv = tv.Tensor()
    output_add_tv = tv.Tensor()
    is_int8 = features.is_quantized and filters.is_quantized
    if output_add is not None:
        assert features.dtype == torch.qint8, "fused residual add only support int8"
    if bias is not None:
        bias_tv = torch_tensor_to_tv(bias)
    if scale is not None:
        scale_tv = torch_tensor_to_tv(scale)
    if output_add is not None:
        output_add_tv = torch_tensor_to_tv(output_add)
    
    if not features.is_contiguous():
        features = features.contiguous()
    assert features.is_contiguous()
    assert filters.is_contiguous()
    if output_dtype is None:
        output_dtype = features.dtype

    if SPCONV_CPP_GEMM and CONV_CPP is not None:
        alloc = TorchAllocator(features.device, features.dtype == torch.qint8)
        features_tv = torch_tensor_to_tv(features)
        pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
        pair_mask_fwd_splits_tv = [
            torch_tensor_to_tv(t, tv.uint32) for t in pair_mask_fwd_splits
        ]
        mask_argsort_fwd_splits_tv = [
            torch_tensor_to_tv(t) for t in mask_argsort_fwd_splits
        ]

        filters_tv = torch_tensor_to_tv(filters)
        mask = np.concatenate(masks)
        mask_tv = tv.from_numpy(mask).clone()
        timer_cpp = tv.CUDAKernelTimer(False)
        if timer._timer is not None:
            timer_cpp = timer._timer
        auto_fp32_accum = fp32_accum is None
        if fp32_accum is None:
            fp32_accum = False
        arch = get_arch()
        output_dtype_tv = _TORCH_DTYPE_TO_TV[output_dtype]
        mask_width, tune_res_cpp = ConvGemmOps.implicit_gemm(
            alloc, CONV_CPP, features_tv, filters_tv, pair_fwd_tv,
            pair_mask_fwd_splits_tv, mask_argsort_fwd_splits_tv,
            num_activate_out, mask_tv, arch, is_train, is_subm, stream,
            timer_cpp, auto_fp32_accum, fp32_accum, bias_tv, act_alpha, act_beta, act_type,
            use_tf32=constants.SPCONV_ALLOW_TF32, output_scale=output_scale,
            scale=scale_tv, output_add=output_add_tv, output_add_scale=output_add_scale,
            output_dtype=output_dtype_tv)
        out_features = alloc.allocated[AllocKeys.OutFeatures]
        mask_output_fwd = alloc.allocated.get(AllocKeys.MaskOutputFwd, None)
        if is_train:
            assert mask_output_fwd is not None
        return out_features, mask_output_fwd, mask_width
    # if DEBUG:

    # CONV.stream_synchronize(stream)

    # t = time.time()

    # if features.dtype == torch.int8 or features.dtype == torch.qint8:
    #     raise NotImplementedError("work in progress")
    # here filters is KRSC
    masks_ints = [m.item() for m in masks]
    out_channel = filters.shape[0]
    in_channel = filters.shape[-1]
    num_split = len(pair_mask_fwd_splits)
    filters = filters.reshape(out_channel, -1, filters.shape[-1])
    kv = filters.shape[1]
    mask_int_count = div_up(kv, 32)
    if is_int8:
        if is_subm:
            out_features = torch._empty_affine_quantized(size=(num_activate_out, out_channel), 
                scale=output_scale, zero_point=0, dtype=features.dtype, device=features.device)
            # out_features = torch.empty((num_activate_out, out_channel),
            #                         dtype=output_dtype,
            #                         device=features.device)
        else:
            out_features = torch._empty_affine_quantized(size=(num_activate_out, out_channel), 
                scale=output_scale, zero_point=0, dtype=features.dtype, device=features.device)
            ctx = tv.Context()
            ctx.set_cuda_stream(stream)
            torch_tensor_to_tv(out_features).zero_(ctx)
            # out_features = torch.zeros((num_activate_out, out_channel),
            #                         dtype=output_dtype,
            #                         device=features.device)
    else:
        if is_subm:
            out_features = torch.empty((num_activate_out, out_channel),
                                    dtype=output_dtype,
                                    device=features.device)
        else:
            out_features = torch.zeros((num_activate_out, out_channel),
                                    dtype=output_dtype,
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
            fp32_accum=fp32_accum,
            use_tf32=constants.SPCONV_ALLOW_TF32,
            bias=bias_tv, scale=scale_tv)

    mask_width = tune_res.algo_desp.tile_shape[0]
    if is_train:
        mask_output_fwd = torch.empty(
            [num_split,
             codeops.div_up(num_activate_out, mask_width), mask_int_count],
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
    # print(tune_res.algo_desp, "REF", features_tv.shape, filters.shape)
    # with tv.measure_and_print("f16 time"):
    bias_tv = tv.Tensor()
    if bias is not None:
        bias_tv = torch_tensor_to_tv(bias)
    alpha = 1.0
    if tune_res.algo_desp.is_int8_inference:
        alpha = output_scale
    with timer.record("implicit_gemm", stream):
        for j in range(num_split):
            beta = 0 if j == 0 else 1
            if bias is not None and not tune_res.algo_desp.is_int8_inference:
                beta = 1 
            if output_add is not None and tune_res.algo_desp.is_int8_inference:
                beta = output_add_scale / output_scale
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
                alpha=alpha,
                beta=beta,
                stream=stream,
                verbose=False,
                bias=bias_tv,
                act_type=act_type,
                act_alpha=act_alpha,
                act_beta=act_beta,
                scale=scale_tv,
                output_add=output_add_tv)
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
                           mask_output_fwd: Optional[torch.Tensor],
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
    if mask_output_fwd is None:
        raise ValueError("you must do bwd with net.train()")
    assert out_bp.is_contiguous()
    assert filters.is_contiguous()
    assert features.is_contiguous()
    stream = get_current_stream()

    if SPCONV_CPP_GEMM and CONV_CPP is not None:
        alloc = TorchAllocator(features.device)
        features_tv = torch_tensor_to_tv(features)
        pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
        pair_bwd_tv = torch_tensor_to_tv(pair_bwd)

        pair_mask_fwd_splits_tv = [
            torch_tensor_to_tv(t) for t in pair_mask_fwd_splits
        ]
        pair_mask_bwd_splits_tv = [
            torch_tensor_to_tv(t) for t in pair_mask_bwd_splits
        ]

        mask_argsort_fwd_splits_tv = [
            torch_tensor_to_tv(t) for t in mask_argsort_fwd_splits
        ]
        mask_argsort_bwd_splits_tv = [
            torch_tensor_to_tv(t) for t in mask_argsort_bwd_splits
        ]

        filters_tv = torch_tensor_to_tv(filters)
        out_bp_tv = torch_tensor_to_tv(out_bp)

        mask_output_fwd_tv = torch_tensor_to_tv(mask_output_fwd)

        mask = np.concatenate(masks)
        mask_tv = tv.from_numpy(mask).clone()

        timer_cpp = tv.CUDAKernelTimer(False)
        if timer._timer is not None:
            timer_cpp = timer._timer
        auto_fp32_accum = fp32_accum is None
        if fp32_accum is None:
            fp32_accum = False
        arch = get_arch()

        ConvGemmOps.implicit_gemm_backward(
            alloc, CONV_CPP, features_tv, filters_tv, out_bp_tv, pair_fwd_tv,
            pair_bwd_tv, pair_mask_fwd_splits_tv, pair_mask_bwd_splits_tv,
            mask_argsort_fwd_splits_tv, mask_argsort_bwd_splits_tv,
            mask_output_fwd_tv, mask_tv, arch, mask_width, is_subm, stream,
            timer_cpp, auto_fp32_accum, fp32_accum,
            use_tf32=constants.SPCONV_ALLOW_TF32)
        din = alloc.allocated[AllocKeys.DIn]
        dfilters = alloc.allocated[AllocKeys.DFilters]
        return din, dfilters

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
    need_dynamic_mask = kv > 32
    mask_int_count = div_up(kv, 32)

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
                                         in_channel, arch, need_dynamic_mask=need_dynamic_mask)
    wgrad_tune_res = CONV.get_tuned_algo(ConvOpType.kBackwardWeight,
                                         features_tv.dtype, dfilters_tv.dtype,
                                         dout_tv.dtype, out_channel,
                                         in_channel, arch, mask_width,
                                         need_dynamic_mask=need_dynamic_mask)

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
                                                fp32_accum=fp32_accum,
                                                use_tf32=constants.SPCONV_ALLOW_TF32)
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
            mask=mask_output_fwd_tv[0],
            mask_argsort=mask_argsort_fwd_split_tvs[0],
            indices=pair_fwd_tv,
            reverse_mask=False,
            mask_filter=masks[0].item(),
            mask_output=tv.Tensor(),
            mask_width=mask_width,
            stream=stream,
            use_tf32=constants.SPCONV_ALLOW_TF32)
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
            # for backward weight, beta = 0 because each split
            # handle different kernel locations.
            # TODO remove D iterator in backward weight kernel
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
                beta=0,
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
    if features.is_quantized:
        out_features = torch._empty_affine_quantized((num_activate_out, out_channel),
                                scale=features.q_scale(),
                                dtype=features.dtype,
                                device=features.device)
    else:
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


def indice_avgpool_implicit_gemm(features: torch.Tensor,
                                 indice_pairs: torch.Tensor, num_activate_out,
                                 calc_count: bool):
    # torch.cuda.synchronize()
    # t = time.time()
    stream = get_current_stream()
    # CONV.stream_synchronize(stream)
    # t = time.time()
    if not features.is_contiguous():
        features = features.contiguous()

    out_channel = features.shape[-1]
    if features.is_quantized:
        out_features = torch._empty_affine_quantized((num_activate_out, out_channel),
                                scale=features.q_scale(),
                                dtype=features.dtype,
                                device=features.device)
    else:
        out_features = torch.empty((num_activate_out, out_channel),
                                dtype=features.dtype,
                                device=features.device)

    assert features.is_cuda
    stream = get_current_stream()
    out_features_tv = torch_tensor_to_tv(out_features)
    features_tv = torch_tensor_to_tv(features)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    count_out = torch.Tensor()
    count_out_tv = tv.Tensor()
    if calc_count:
        count_out = torch.zeros((num_activate_out, ),
                                dtype=torch.int32,
                                device=features.device)
        count_out_tv = torch_tensor_to_tv(count_out)
    SpconvOps.avgpool_implicit_gemm_forward(out_features_tv, features_tv,
                                            indice_pairs_tv, count_out_tv,
                                            stream)

    # CONV.stream_synchronize(stream)
    # print("M", time.time() - t)

    return out_features, count_out


def indice_avgpool_implicit_gemm_backward(out_bp, indice_pairs, count_out):
    # torch.cuda.synchronize()
    # t = time.time()
    out_channel = out_bp.shape[-1]
    din = torch.zeros((indice_pairs.shape[1], out_bp.shape[1]),
                      dtype=out_bp.dtype,
                      device=out_bp.device)
    assert out_bp.is_cuda
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()

    stream = get_current_stream()
    count_out_tv = torch_tensor_to_tv(count_out)
    out_bp_tv = torch_tensor_to_tv(out_bp)
    din_tv = torch_tensor_to_tv(din)
    indice_pairs_tv = torch_tensor_to_tv(indice_pairs)
    SpconvOps.avgpool_implicit_gemm_backward(out_bp_tv, din_tv,
                                             indice_pairs_tv, count_out_tv,
                                             stream)
    return din


def maximum_value_int_(ten: torch.Tensor, value: int):
    stream = 0
    if not CPU_ONLY_BUILD:
        stream = get_current_stream()
    else:
        assert not ten.is_cuda
    SpconvOps.maximum_value_int(torch_tensor_to_tv(ten), value, stream)
