import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import sys
import time
from pathlib import Path
from cumm.gemm.algospec.core import GemmAlgo

import numpy as np
import pccm
import torch
import torch.nn.functional as F

from cumm import dtypes
from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT
from cumm.conv.bases import NCHW, NHWC, ConvIterAlgo, ConvOpType
from cumm.conv.main import ConvMainUnitTest, gen_gemm_kernels
from cumm.conv.params import ConvProblem
from cumm.gemm import kernel
import os
from spconv.core_cc.csrc.sparse.all import SpconvOps
from cumm.gemm.codeops import div_up
from spconv.constants import PACKAGE_ROOT
from spconv.core import ConvAlgo

from spconv.pytorch import ops
from spconv.algo import CONV, BestConvAlgoByProfile
from spconv.pytorch.cppcore import torch_tensor_to_tv


def reduce_mask_count(mask: np.ndarray, width: int):
    mask_length_32 = (div_up(mask.shape[0], width)) * width
    if mask.shape[0] < mask_length_32:
        mask_pad = np.zeros((mask_length_32, ), dtype=mask.dtype)
        mask_pad[:mask.shape[0]] = mask
        mask = mask_pad
    mask = mask.reshape(-1, width)
    maskr = np.bitwise_or.reduce(mask, axis=1)
    maskr_tv = tv.from_numpy(maskr)
    return SpconvOps.count_bits(maskr_tv).numpy().sum() * width


def reduce_mask_count_x(mask: np.ndarray, width: int):
    mask_length_32 = (div_up(mask.shape[0], width)) * width
    if mask.shape[0] < mask_length_32:
        mask_pad = np.zeros((mask_length_32, ), dtype=mask.dtype)
        mask_pad[:mask.shape[0]] = mask
        mask = mask_pad
    mask = mask.reshape(-1, width)
    maskr = np.bitwise_or.reduce(mask, axis=1)
    return maskr


def dev_subm_inds_v2(subm: bool = False, run_conv: bool = True):
    limit_input_n = 16384
    limit_input_n = None
    np.random.seed(484)

    with (PACKAGE_ROOT.parent / "test/data/test_spconv.pkl").open("rb") as f:
        voxels_np, indices_np, spatial_shape = pickle.load(f)
        from spconv.test_utils import generate_sparse_data
        voxels_np = voxels_np[:limit_input_n]
        indices_np = indices_np[:limit_input_n]

        spatial_shape = [19, 18, 17]
        sparse_dict = generate_sparse_data(spatial_shape, [1024], 128)

        voxels_np = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices_np = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)

        voxels = tv.from_numpy(voxels_np).cuda()
        indices = tv.from_numpy(indices_np).cuda()
        indices_th = torch.from_numpy(indices_np).cuda()
    print(spatial_shape, indices_np.shape)
    ndim = 3
    if subm:
        ksize = [3, 3, 3]
        kv = np.prod(ksize)
        padding = [1] * ndim
        stride = [1] * ndim
        dilation = [1] * ndim
        out_padding = [0] * ndim
    else:
        ksize = [2, 2, 2]
        kv = np.prod(ksize)
        padding = [0] * ndim
        stride = [1] * ndim
        dilation = [1] * ndim
        out_padding = [0] * ndim
    out_inds, pair_ref, indice_num_per_loc = ops.get_indice_pairs(
        indices_th, 1, spatial_shape, ConvAlgo.Native, ksize, stride, padding,
        dilation, out_padding, subm)
    indice_num_per_loc_np = indice_num_per_loc.cpu().numpy()
    indice_pairs_np = pair_ref.cpu().numpy()
    algo = ConvAlgo.MaskSplitImplicitGemm
    if algo == ConvAlgo.MaskImplicitGemm:
        num_split = 1
    else:
        num_split = 2
    for i in range(5):
        res = ops.get_indice_pairs_implicit_gemm(indices_th, 1, spatial_shape,
                                                 algo, ksize, stride, padding,
                                                 dilation, out_padding, subm)
    out_inds = res[0]
    num_inds_per_loc = res[1]
    pair_fwd = res[2]
    pair_fwd_x = pair_fwd.cpu().numpy().reshape(-1)
    pair_fwd_x[pair_fwd_x == -1] = 0
    loc_num_np = (pair_fwd_x > 0).reshape(kv, -1).sum(1)
    print(loc_num_np)
    print(indice_num_per_loc_np)

    pair_bwd = res[3]
    pair_mask_fwd_splits = res[4]
    pair_mask_bwd_splits = res[5]
    mask_argsort_fwd_splits = res[6]
    mask_argsort_bwd_splits = res[7]
    masks = res[8]
    pair_mask_fwd_splits_tv = [
        ops.torch_tensor_to_tv(t, dtype=tv.uint32)
        for t in pair_mask_fwd_splits
    ]
    valid_location_bitcount = [
        SpconvOps.count_bits(t) for t in pair_mask_fwd_splits_tv
    ]
    valid_location_count = sum(
        [t.cpu().numpy().sum() for t in valid_location_bitcount])
    reduce_length = 32
    split_mask_valid_count = sum([
        reduce_mask_count(t.cpu().numpy(), reduce_length)
        for t in pair_mask_fwd_splits_tv
    ])
    if subm:
        print("SUBM", valid_location_count, split_mask_valid_count,
              pair_fwd.numel())
    else:
        print("REGULAR", valid_location_count, split_mask_valid_count,
              pair_fwd.numel())
    # return

    if run_conv:
        C = 64
        K = 64
        desps = CONV.desps
        mask_output_fwd = torch.zeros([2, div_up(out_inds.shape[0], 32)],
                                      dtype=torch.int32,
                                      device=indices_th.device)
        mask_output_bwd = torch.zeros([2, div_up(indices.dim(0), 32)],
                                      dtype=torch.int32,
                                      device=indices_th.device)

        for desp in desps:
            if desp.algo != GemmAlgo.Simt.value:
                continue
            # if desp.op_type == ConvOpType.kBackwardWeight.value:
            #     continue
            # if desp.tile_shape !
            if desp.dtype_a == dtypes.int8.tv_dtype:
                inp = np.random.randint(-1, 1, size=[voxels_np.shape[0],
                                                     C]).astype(np.int8)
                weight = np.random.randint(-1, 1, size=[K, *ksize,
                                                        C]).astype(np.int8)
                output = np.random.randint(-1, 1, size=[
                    out_inds.shape[0], K
                ]).astype(dtypes.get_npdtype_from_tvdtype(desp.dtype_output))
            else:
                inp = np.random.uniform(-1, 1, size=[
                    voxels_np.shape[0], C
                ]).astype(dtypes.get_npdtype_from_tvdtype(desp.dtype_input))
                weight = np.random.uniform(-1, 1, size=[K, *ksize, C]).astype(
                    dtypes.get_npdtype_from_tvdtype(desp.dtype_weight))
                output = np.random.uniform(-1, 1, size=[
                    out_inds.shape[0], K
                ]).astype(dtypes.get_npdtype_from_tvdtype(desp.dtype_output))
            weight_ref = weight.transpose(1, 2, 3, 0, 4)
            weight_ref = np.ascontiguousarray(weight_ref).reshape(-1, K, C)
            if desp.op_type == ConvOpType.kBackwardInput.value:
                inp_tv = tv.zeros(inp.shape, desp.dtype_input, 0)
            else:
                inp_tv = tv.from_numpy(inp).cuda()
            if desp.op_type == ConvOpType.kBackwardWeight.value:
                weight_tv = tv.zeros(weight.shape, desp.dtype_weight, 0)
            else:
                weight_tv = tv.from_numpy(weight).cuda()
            # _ = tv.zeros([5000, 10], tv.float32, 0)
            if desp.op_type == ConvOpType.kForward.value:
                output_tv = tv.zeros(output.shape, desp.dtype_output, 0)
            else:
                output_tv = tv.from_numpy(output).cuda()
            torch.cuda.synchronize()
            t = time.time()
            spk = 1
            if desp.op_type == ConvOpType.kBackwardWeight.value:
                # TODO support splitk parallel
                spk = 32
            if subm:
                if desp.op_type == ConvOpType.kForward.value:
                    indice_pairs = pair_fwd
                elif desp.op_type == ConvOpType.kBackwardInput.value:
                    indice_pairs = pair_bwd
                else:
                    indice_pairs = pair_fwd
                mask_output = mask_output_fwd
                # print([bin(x.item()) for x in masks])
                for j in range(num_split):
                    beta = 1 if j == 1 else 0
                    mask_filter = 0xffffffff
                    mask_filter = masks[j].item()

                    reverse_mask = False
                    if desp.op_type == ConvOpType.kBackwardWeight.value:
                        mask_op = mask_output[j]
                    else:
                        mask_op = pair_mask_fwd_splits[j]
                    if desp.op_type == ConvOpType.kBackwardInput.value:
                        reverse_mask = True
                    CONV.run_with_tuned_result(
                        BestConvAlgoByProfile(desp, spk),
                        desp.op_type,
                        inp_tv,
                        weight_tv,
                        output_tv,
                        torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                        torch_tensor_to_tv(mask_argsort_fwd_splits[j]),
                        torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                        torch_tensor_to_tv(indice_pairs),
                        reverse_mask,
                        mask_filter=mask_filter,
                        mask_width=32,
                        beta=beta,
                        verbose=True,
                    )
            else:
                if desp.op_type == ConvOpType.kForward.value:
                    indice_pairs = pair_fwd  # inp -> out
                    mask_ops = pair_mask_fwd_splits
                    mask_argsorts = mask_argsort_fwd_splits
                    mask_output = mask_output_fwd
                elif desp.op_type == ConvOpType.kBackwardInput.value:
                    indice_pairs = pair_bwd  # out -> inp
                    mask_ops = pair_mask_bwd_splits
                    mask_argsorts = mask_argsort_bwd_splits
                    mask_output = mask_output_bwd

                    print([bin(x.item()) for x in masks])
                else:
                    indice_pairs = pair_fwd  # inp -> out
                    mask_ops = pair_mask_fwd_splits
                    mask_argsorts = mask_argsort_fwd_splits
                    mask_output = mask_output_fwd

                for j in range(2):
                    beta = 1 if j == 1 else 0
                    mask_filter = masks[j].item()
                    reverse_mask = False
                    if desp.op_type == ConvOpType.kBackwardWeight.value:
                        mask_op = mask_output[j]
                    else:
                        mask_op = mask_ops[j]

                    CONV.run_with_tuned_result(
                        BestConvAlgoByProfile(desp, spk),
                        desp.op_type,
                        inp_tv,
                        weight_tv,
                        output_tv,
                        torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                        torch_tensor_to_tv(mask_argsorts[j]),
                        torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                        torch_tensor_to_tv(indice_pairs),
                        reverse_mask,
                        mask_filter=mask_filter,
                        mask_width=32,
                        beta=beta,
                        verbose=True,
                    )

            torch.cuda.synchronize()
            duration = time.time() - t
            if desp.op_type == ConvOpType.kForward.value:
                output_ref = np.zeros_like(output, dtype=np.float32)
                # ref algorithm
                for filter_offset in range(kv):
                    if subm and filter_offset > kv // 2:
                        nhot = indice_num_per_loc_np[kv - 1 - filter_offset]
                    elif subm and filter_offset == kv // 2:
                        nhot = voxels.shape[0]
                    else:
                        nhot = indice_num_per_loc_np[filter_offset]
                    a_inds = indice_pairs_np[0][filter_offset][:nhot]
                    c_inds = indice_pairs_np[1][filter_offset][:nhot]
                    # print(a_inds_cpu[:10])
                    a = inp[a_inds]
                    cc = a.astype(
                        np.float32) @ weight_ref[filter_offset].T.astype(
                            np.float32)
                    output_ref[c_inds] += cc

                output_cpu = output_tv.cpu().numpy().astype(np.float32)
                duration = time.time() - t
                my = output_cpu.reshape(-1)
                print("ERROR", np.linalg.norm(output_ref.reshape(-1) - my))

            elif desp.op_type == ConvOpType.kBackwardInput.value:
                dinput_ref = np.zeros_like(inp, dtype=np.float32)
                # ref algorithm
                for filter_offset in range(kv):
                    if subm and filter_offset > kv // 2:
                        nhot = indice_num_per_loc_np[kv - 1 - filter_offset]
                    elif subm and filter_offset == kv // 2:
                        nhot = voxels.shape[0]
                    else:
                        nhot = indice_num_per_loc_np[filter_offset]
                    a_inds = indice_pairs_np[1][filter_offset][:nhot]
                    c_inds = indice_pairs_np[0][filter_offset][:nhot]

                    # print(a_inds_cpu[:10])
                    a = output[a_inds]
                    # NK @ KC
                    cc = a.astype(
                        np.float32) @ weight_ref[filter_offset].astype(
                            np.float32)
                    dinput_ref[c_inds] += cc
                din_cpu = inp_tv.cpu().numpy()
                print(
                    "ERROR",
                    np.linalg.norm(
                        din_cpu.reshape(-1) - dinput_ref.reshape(-1)))
            else:
                dw_ref = np.zeros_like(weight_ref,
                                       dtype=np.float32)  # KV, K, C
                for filter_offset in range(kv):
                    if subm and filter_offset > kv // 2:
                        nhot = indice_num_per_loc_np[kv - 1 - filter_offset]
                    elif subm and filter_offset == kv // 2:
                        nhot = voxels.shape[0]
                    else:
                        nhot = indice_num_per_loc_np[filter_offset]
                    o_inds = indice_pairs_np[1][filter_offset][:nhot]
                    i_inds = indice_pairs_np[0][filter_offset][:nhot]
                    # print(a_inds_cpu[:10])
                    out_gather = output[o_inds]  # [N, K]
                    inp_gather = inp[i_inds]  # [N, C]
                    # KN @ NC
                    dw_res = out_gather.astype(
                        np.float32).T @ inp_gather.astype(np.float32)
                    dw_ref[filter_offset] = dw_res
                # print(indice_pairs_np_test[0])
                dw_ref_kcrs = dw_ref.transpose(1, 0, 2)
                dw_cpu = weight_tv.cpu().numpy().reshape(K, np.prod(ksize), C)

                print(
                    "ERROR",
                    np.linalg.norm(
                        dw_cpu.reshape(-1) - dw_ref_kcrs.reshape(-1)))


if __name__ == "__main__":
    dev_subm_inds_v2()
