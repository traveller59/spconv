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

"""Test all gemm/conv kernels.
We can't test all kernels in network because auto-tuner will only find one best kernel.
"""


import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import sys
import time
from pathlib import Path
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType

import numpy as np
import pccm
import torch
import torch.nn.functional as F
from spconv.core_cc.csrc.sparse.convops import GemmTuneResult, ConvTuneResult
from spconv.pytorch.core import SparseConvTensor
from spconv.test_utils import TestCase
from cumm import tensorview as tv
from spconv.constants import SPCONV_ALLOW_TF32
from cumm.conv.bases import NCHW, NHWC, ConvIterAlgo, ConvOpType
import os
from cumm.gemm.codeops import div_up
from spconv.core import AlgoHint, ConvAlgo
from spconv.pytorch.conv import expand_nd
from spconv.pytorch import ops
from spconv.algo import GEMM, CONV, GEMM_CPP, CONV_CPP, BestAlgoByProfile, BestConvAlgoByProfile, GemmTunerSimple
from spconv.pytorch.cppcore import get_current_stream, torch_tensor_to_tv
from spconv.test_utils import generate_sparse_data, params_grid
import tqdm 
from spconv.constants import ALL_WEIGHT_IS_KRSC, SPCONV_CPP_GEMM
from spconv.core_cc.csrc.sparse.inference import InferenceOps
from spconv.pytorch import functional as Fsp
assert ALL_WEIGHT_IS_KRSC is True, "we only support KRSC in spconv >= 2.2"
from spconv.pytorch.hash import HashTable

# TODO remove or release this when tf32 op is ready
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

NUMPY_DTYPE_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,

}

class SparseConvTester:
    def __init__(self, algo: ConvAlgo, subm: bool, shape: List[int], bs: int, dtype: np.dtype, N: int, K: int, C: int, 
        ksize: int, stride: int, padding: int, dilation: int, check_bias: bool = False, check_act: bool = False) -> None:
        ndim = 3
        transpose = False
        self.shape = shape 
        self.bs = bs 
        self.dtype = dtype 
        self.dtype_th = NUMPY_DTYPE_TO_TORCH[dtype]
        self.K = K 
        self.C = C 
        self.ksize = expand_nd(ndim, ksize) 
        self.stride = expand_nd(ndim, stride) 
        self.padding = expand_nd(ndim, padding, ) 
        self.dilation = expand_nd(ndim, dilation) 
        self.N = N
        self.device = torch.device("cuda:0")
        op = expand_nd(ndim, 0)
        self.kv: int = np.prod(self.ksize)
        self.num_split = 1 if algo == ConvAlgo.MaskImplicitGemm else 2
        if not subm:
            if transpose:
                out_shape = ops.get_deconv_output_size(shape, self.ksize, self.stride,
                                                self.padding, self.dilation, op)
            else:
                out_shape = ops.get_conv_output_size(shape, self.ksize, self.stride,
                                                self.padding, self.dilation)
        else:
            out_shape = shape

        sparse_dict = generate_sparse_data(shape, [N] * bs, C)

        voxels_np = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices_np = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        indices_th = torch.from_numpy(indices_np).to(self.device)
        out_inds, pair_ref, indice_num_per_loc = ops.get_indice_pairs(
            indices_th, 1, shape, ConvAlgo.Native, self.ksize, self.stride, self.padding,
            self.dilation, op, subm)
        self.ref_out_inds = out_inds
        self.ref_out_inds_scalar = Fsp._indice_to_scalar(out_inds.long(), [bs, *out_shape])
        self.indice_num_per_loc_np = indice_num_per_loc.cpu().numpy()
        self.indice_pairs_np = pair_ref.cpu().numpy()
        self.pair_native = pair_ref
        self.indice_num_per_loc = indice_num_per_loc
        self.use_direct_table = True
        
        self.out_shape = out_shape
        if algo == ConvAlgo.Native:
            self.out_inds: torch.Tensor = out_inds
            self.num_inds_per_loc: torch.Tensor = indice_num_per_loc
            self.pair_fwd : torch.Tensor = torch.Tensor()
            self.pair_bwd: torch.Tensor = torch.Tensor()
            self.pair_mask_fwd_splits: List[torch.Tensor] = []
            self.pair_mask_bwd_splits: List[torch.Tensor] = []
            self.mask_argsort_fwd_splits: List[torch.Tensor] = []
            self.mask_argsort_bwd_splits: List[torch.Tensor] = []
            self.masks = np.array([])
        else:
            res = ops.get_indice_pairs_implicit_gemm(indices_th, bs, shape,
                                                    algo, self.ksize, self.stride, self.padding,
                                                    self.dilation, op, subm=subm, direct_table=self.use_direct_table)
            
            self.out_inds = res[0]
            self.num_inds_per_loc = res[1]
            self.pair_fwd = res[2]
            self.pair_bwd = res[3]
            self.pair_mask_fwd_splits = res[4]
            self.pair_mask_bwd_splits = res[5]
            self.mask_argsort_fwd_splits = res[6]
            self.mask_argsort_bwd_splits = res[7]
            self.masks = res[8]
        
        self.out_inds_scalar = Fsp._indice_to_scalar(self.out_inds.long(), [bs, *out_shape])

        table = HashTable(out_inds.device, torch.int64, torch.int32, self.out_inds.shape[0] * 2)
        # test coords -> test out indexes
        table.insert(self.out_inds_scalar, torch.arange(0, self.out_inds.shape[0], dtype=torch.int32, device=self.device))
        # out_order:  test_order_to_ref, test index for each ref coord
        out_order, is_empty = table.query(self.ref_out_inds_scalar)
        assert is_empty.int().sum().item() == 0, "shouldn't happen"
        self.out_order = out_order.cpu().numpy()

        # inp_table = HashTable(out_inds.device, torch.int64, torch.int32, self.ref_out_inds.shape[0] * 2)
        # inp_table.insert(self.ref_out_inds_scalar, torch.arange(0, self.ref_out_inds.shape[0], dtype=torch.int32, device=self.device))
        # # out_order:  ref index for each out coord
        # out_order, is_empty = inp_table.query(self.out_inds_scalar)


        self.voxels_np = voxels_np
        self.indices_np = indices_np
        self.check_bias = check_bias
        self.check_act = check_act

        self.subm = subm
        if dtype == np.int8:
            self.inp = np.random.randint(-2, 2, size=[voxels_np.shape[0],
                                                    C]).astype(np.int8)
            self.weight = np.random.randint(-2, 2, size=[K, *self.ksize,
                                                    C]).astype(np.int8)
            self.output = np.random.randint(-2, 2, size=[
                self.out_inds.shape[0], K
            ]).astype(dtype)
            self.bias = np.random.randint(-2, 2, size=[
                K
            ]).astype(dtype)

        else:
            self.inp = np.random.uniform(-1, 1, size=[
                voxels_np.shape[0], C
            ]).astype(dtype)
            self.weight = np.random.uniform(-1, 1, size=[K, *self.ksize, C]).astype(dtype)
            self.output = np.random.uniform(-1, 1, size=[
                self.out_inds.shape[0], K
            ]).astype(dtype)
            self.bias = np.random.uniform(-1, 1, size=[
                K
            ]).astype(dtype)

        self.weight_ref = self.weight.transpose(1, 2, 3, 0, 4)
        self.weight_ref = np.ascontiguousarray(self.weight_ref).reshape(-1, K, C)
        self.out_ref, self.din_ref, self.dw_ref = self._get_ref_output()
        if check_bias:
            self.out_ref += self.bias
            # relu
        if check_act:
            self.out_ref = np.maximum(self.out_ref, 0)
        self.dw_ref = np.ascontiguousarray(self.dw_ref.transpose(1, 0, 2).reshape(K, *self.ksize, C))
        self.arch = tv.get_compute_capability()

    def get_output_ref_spt(self):
        return SparseConvTensor(torch.from_numpy(self.out_ref).cuda(), self.ref_out_inds, self.out_shape, self.bs)


    def _get_ref_output(self):
        output_ref = np.zeros_like(self.output, dtype=np.float32)
        dinput_ref = np.zeros_like(self.inp, dtype=np.float32)
        dw_ref = np.zeros_like(self.weight_ref,
                                dtype=np.float32)  # KV, K, C

        for filter_offset in range(self.kv):
            if self.subm and filter_offset > self.kv // 2:
                nhot = self.indice_num_per_loc_np[self.kv - 1 - filter_offset]
            elif self.subm and filter_offset == self.kv // 2:
                nhot = self.voxels_np.shape[0]
            else:
                nhot = self.indice_num_per_loc_np[filter_offset]

            i_inds = self.indice_pairs_np[0][filter_offset][:nhot]
            o_inds = self.indice_pairs_np[1][filter_offset][:nhot]
            a = self.inp[i_inds]
            cc = a.astype(
                np.float32) @ self.weight_ref[filter_offset].T.astype(
                    np.float32)
            output_ref[o_inds] += cc
            # we use random output as dout here
            a = self.output[self.out_order][o_inds]
            # NK @ KC
            cc = a.astype(
                np.float32) @ self.weight_ref[filter_offset].astype(
                    np.float32)
            dinput_ref[i_inds] += cc
            # use random output and random inp as dout and inp
            out_gather = self.output[self.out_order][o_inds]  # [N, K]
            inp_gather = self.inp[i_inds]  # [N, C]
            # KN @ NC
            dw_res = out_gather.astype(
                np.float32).T @ inp_gather.astype(np.float32)
            dw_ref[filter_offset] = dw_res
        if self.dtype == np.int8:
            output_ref = np.clip(output_ref, -127, 127)
        return output_ref, dinput_ref, dw_ref

    def get_operands(self, op_type: ConvOpType):
        zeros_func = tv.zeros if not self.subm else tv.empty
        if op_type == ConvOpType.kBackwardInput:
            inp_tv = zeros_func(list(self.inp.shape), self.dtype, 0)
        else:
            inp_tv = tv.from_numpy(self.inp).cuda()
        if op_type == ConvOpType.kBackwardWeight:
            weight_tv = zeros_func(list(self.weight.shape), self.dtype, 0)
        else:
            weight_tv = tv.from_numpy(self.weight).cuda()
        if op_type == ConvOpType.kForward:
            output_tv = zeros_func(list(self.output.shape), self.dtype, 0)
        else:
            output_tv = tv.from_numpy(self.output).cuda()
        return inp_tv, weight_tv, output_tv

    def get_operands_torch(self, op_type: ConvOpType):
        zeros_func = torch.zeros if not self.subm else torch.empty
        if op_type == ConvOpType.kBackwardInput:
            inp_tv = zeros_func(list(self.inp.shape), dtype=self.dtype_th, device=self.device)
        else:
            inp_tv = torch.from_numpy(self.inp).cuda()
        if op_type == ConvOpType.kBackwardWeight:
            weight_tv = zeros_func(list(self.weight.shape), dtype=self.dtype_th, device=self.device)
        else:
            weight_tv = torch.from_numpy(self.weight).cuda()
        if op_type == ConvOpType.kForward:
            output_tv = zeros_func(list(self.output.shape), dtype=self.dtype_th, device=self.device)
        else:
            output_tv = torch.from_numpy(self.output).cuda()
        return inp_tv, weight_tv, output_tv

def _test_impgemm_conv_cuda(subm: bool):
    ndim = 3
    np.random.seed(50005)
    dtype_to_tol = {
        np.float32: (1e-2, 1e-2),
        np.float16: (1e-2, 1e-2),
        np.int8: (1e-4, 1e-4),
    }
    device = torch.device("cuda:0")
    shapes = [[19, 18, 17]]
    batchsizes = [1]
    dtypes = [np.float32, np.float16]
    # dtypes = [np.float16]

    # dtypes = [np.int8]
    test_case = TestCase()
    # in_channels = [32]
    # out_channels = [32, 48, 64]
    in_channels = [32, 47]
    out_channels = [32, 48, 62]
    # in_channels = [32]
    # out_channels = [32]

    multiple_base = 16
    if subm:
        ksizes = [3]
        strides = [1]
        paddings = [0]
        dilations = [1]
    else:
        ksizes = [2, 3]
        strides = [1, 2, 3]
        paddings = [0, 1]
        dilations = [1, 2]

    algos = [
        # ConvAlgo.MaskSplitImplicitGemm,
        ConvAlgo.MaskImplicitGemm,
    ]
    arch = torch.cuda.get_device_capability()
    force_nvrtc = False
    for shape, bs, C, K, k, s, p, d, algo, dtype in tqdm.tqdm(params_grid(
            shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, algos, dtypes)):
        shape_prod = np.prod(shape)
        num_batch = np.random.randint(int(0.2 * shape_prod), int(0.7 * shape_prod))
        # C = np.random.randint(int(0.3 * C), int(0.7 * C))
        # K = np.random.randint(int(0.3 * K), int(0.7 * K))
        multipler = max(C, K) / multiple_base
        multipler = max(multipler, 1.0)
        # print(num_batch)
        tester = SparseConvTester(algo, subm, shape, bs, dtype, num_batch, K, C, k, s, p, d, check_bias=True, check_act=True)
        bias = None
        act = tv.gemm.Activation.None_
        if tester.check_bias:
            bias = tv.from_numpy(tester.bias).cuda()
        atol, rtol = dtype_to_tol[dtype]
        mask_width_to_mask_out_fwd: Dict[int, torch.Tensor] = {}
        mask_width_to_mask_out_bwd: Dict[int, torch.Tensor] = {}
        op_types = [ConvOpType.kForward, ConvOpType.kBackwardInput]
        spk = 1
        for op_type in op_types:
            inp_tv, weight_tv, output_tv = tester.get_operands(op_type)
            if SPCONV_CPP_GEMM:
                avail_desps = CONV_CPP.get_all_available(inp_tv, weight_tv, output_tv, 
                    NHWC.layout_type.value, NHWC.layout_type.value, 
                    NHWC.layout_type.value, NHWC.interleave, NHWC.interleave, NHWC.interleave, arch, op_type.value, -1, True, False,
                        use_tf32=SPCONV_ALLOW_TF32)
            else:
                avail_desps = CONV.get_all_available(inp_tv, weight_tv, output_tv, NHWC, NHWC, NHWC, arch, op_type, -1,
                        use_tf32=SPCONV_ALLOW_TF32)
            if op_type == ConvOpType.kForward and tester.check_act:
                act = tv.gemm.Activation.ReLU
            else:
                act = tv.gemm.Activation.None_
            assert avail_desps
            for desp in avail_desps:
                if not subm:
                    if op_type == ConvOpType.kForward:
                        output_tv.zero_()
                    else:
                        inp_tv.zero_()
                # this algo must success
                mask_width = desp.tile_shape[0]
                # if mask_width != 32:
                #     continue
                if mask_width not in mask_width_to_mask_out_fwd:
                    mask_width_to_mask_out_fwd[mask_width] = torch.zeros([2, div_up(tester.out_inds.shape[0], mask_width)],
                                      dtype=torch.int32,
                                      device=tester.device)
                mask_output_fwd = mask_width_to_mask_out_fwd[mask_width]
                is_fwd = desp.op_type.value == ConvOpType.kForward.value
                bias_cur = bias 
                if op_type != ConvOpType.kForward:
                    bias_cur = None
                if subm:
                    if desp.op_type.value == ConvOpType.kForward.value:
                        indice_pairs = tester.pair_fwd
                    elif desp.op_type.value == ConvOpType.kBackwardInput.value:
                        indice_pairs = tester.pair_bwd
                    else:
                        indice_pairs = tester.pair_fwd
                    mask_output = mask_output_fwd
                    # print([bin(x.item()) for x in masks])
                    for j in range(tester.num_split):
                        beta = 1 if j > 0 else 0
                        if bias_cur is not None:
                            beta = 1
                        if j > 0:
                            bias_cur = None
                        mask_filter = tester.masks[j].item()
                        reverse_mask = False
                        if desp.op_type.value == ConvOpType.kBackwardWeight.value:
                            mask_op = mask_output[j]
                        else:
                            mask_op = tester.pair_mask_fwd_splits[j]
                        if desp.op_type.value == ConvOpType.kBackwardInput.value:
                            reverse_mask = True
                        mask_output_run = torch_tensor_to_tv(mask_output[j], dtype=tv.uint32)
                        if desp.op_type.value == ConvOpType.kBackwardWeight.value:
                            mask_output_run = tv.Tensor()
                        # force_nvrtc = desp.op_type.value == ConvOpType.kBackwardInput.value
                        # if force_nvrtc:
                        #     desp.is_nvrtc = True
                        # print(force_nvrtc, desp.op_type, op_type)
                        if SPCONV_CPP_GEMM:

                            CONV_CPP.run_with_tuned_result(
                                ConvTuneResult(desp, tester.arch, spk),
                                desp.op_type.value,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                                torch_tensor_to_tv(tester.mask_argsort_fwd_splits[j]),
                                mask_output_run,
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                                force_nvrtc=force_nvrtc,
                                bias=bias_cur if is_fwd and bias_cur is not None else tv.Tensor(),
                                act_type=act,
                            )
                        else:
                            CONV.run_with_tuned_result(
                                BestConvAlgoByProfile(desp, tester.arch, spk),
                                desp.op_type.value,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                                torch_tensor_to_tv(tester.mask_argsort_fwd_splits[j]),
                                mask_output_run,
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                                force_nvrtc=force_nvrtc,
                                bias=bias_cur if is_fwd else None,
                                act_type=act,
                            )

                else:
                    if mask_width not in mask_width_to_mask_out_bwd:
                        mask_width_to_mask_out_bwd[mask_width] = torch.zeros([2, div_up(tester.indices_np.shape[0], mask_width)],
                                        dtype=torch.int32,
                                        device=tester.device)
                    mask_output_bwd = mask_width_to_mask_out_bwd[mask_width]

                    if desp.op_type.value == ConvOpType.kForward.value:
                        indice_pairs = tester.pair_fwd  # inp -> out
                        mask_ops = tester.pair_mask_fwd_splits
                        mask_argsorts = tester.mask_argsort_fwd_splits
                        mask_output = mask_output_fwd
                    elif desp.op_type.value == ConvOpType.kBackwardInput.value:
                        indice_pairs = tester.pair_bwd  # out -> inp
                        mask_ops = tester.pair_mask_bwd_splits
                        mask_argsorts = tester.mask_argsort_bwd_splits
                        mask_output = mask_output_bwd
                    else:
                        indice_pairs = tester.pair_fwd  # inp -> out
                        mask_ops = tester.pair_mask_fwd_splits
                        mask_argsorts = tester.mask_argsort_fwd_splits
                        mask_output = mask_output_fwd

                    for j in range(tester.num_split):
                        # beta = 1 if j == 1 else 0
                        beta = 1 if j > 0 else 0
                        if bias_cur is not None:
                            beta = 1
                        if j > 0:
                            bias_cur = None
                        mask_filter = tester.masks[j].item()
                        reverse_mask = False
                        if desp.op_type.value == ConvOpType.kBackwardWeight.value:
                            mask_op = mask_output[j]
                        else:
                            mask_op = mask_ops[j]
                        if SPCONV_CPP_GEMM:

                            CONV_CPP.run_with_tuned_result(
                                ConvTuneResult(desp, tester.arch, spk),
                                desp.op_type.value,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                                torch_tensor_to_tv(mask_argsorts[j]),
                                torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                                force_nvrtc=force_nvrtc,
                                bias=bias if is_fwd and bias is not None else tv.Tensor(),
                                act_type=act,
                            )
                        else:
                            CONV.run_with_tuned_result(
                                BestConvAlgoByProfile(desp, tester.arch, spk),
                                desp.op_type.value,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                                torch_tensor_to_tv(mask_argsorts[j]),
                                torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                                force_nvrtc=force_nvrtc,
                                bias=bias if is_fwd else None,
                                act_type=act,
                            )

                out_ref = tester.out_ref
                din_ref = tester.din_ref
                dw_ref = tester.dw_ref
                if op_type == ConvOpType.kForward:
                    out_my = output_tv.cpu().numpy()
                    out_my = out_my[tester.out_order]
                    if dtype != np.float16:
                        test_case.assertAllClose(out_ref, out_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(out_ref.reshape(-1) - out_my.reshape(-1))
                        if (error_norm > 5):
                            print(f"{desp}, Error={error_norm}")
                        assert error_norm < 10 * multipler
                else:
                    din_my = inp_tv.cpu().numpy()
                    if dtype != np.float16:
                        test_case.assertAllClose(din_ref, din_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(din_ref.reshape(-1) - din_my.reshape(-1))
                        assert error_norm < 10 * multipler, f"{desp}, {error_norm}, {k}, {s}, {p}, {d}"
        inp_tv, weight_tv, output_tv = tester.get_operands(ConvOpType.kBackwardWeight)
        for spk in [1, 4, 16, 64]:
            for mask_width, mask_output in mask_width_to_mask_out_fwd.items():
                if SPCONV_CPP_GEMM:
                    avail_desps = CONV_CPP.get_all_available(inp_tv, weight_tv, output_tv, 
                        NHWC.layout_type.value, NHWC.layout_type.value, 
                        NHWC.layout_type.value, NHWC.interleave, NHWC.interleave, NHWC.interleave, arch, 
                        ConvOpType.kBackwardWeight.value, mask_width, True, False,
                        use_tf32=SPCONV_ALLOW_TF32)
                else:
                    avail_desps = CONV.get_all_available(inp_tv, weight_tv, output_tv, NHWC, NHWC, NHWC, arch, ConvOpType.kBackwardWeight, mask_width,
                        use_tf32=SPCONV_ALLOW_TF32)
                for desp in avail_desps:
                    weight_tv.zero_()
                    if subm:
                        indice_pairs = tester.pair_fwd
                        for j in range(tester.num_split):
                            beta = 0
                            mask_filter = tester.masks[j].item()
                            mask_op = mask_output[j]
                            mask_op_tv = torch_tensor_to_tv(mask_op, dtype=tv.uint32)
                            # mask_op_np = mask_op_tv.cpu().numpy()
                            # bit_ref = np.bitwise_or.reduce(mask_op_np, axis=0)
                            # bit_my = mask_filter
                            CONV.run_with_tuned_result(
                                BestConvAlgoByProfile(desp, tester.arch, spk),
                                desp.op_type.value,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                mask_op_tv,
                                torch_tensor_to_tv(tester.mask_argsort_fwd_splits[j]),
                                tv.Tensor(),
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask=False,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                            )
                    else:
                        indice_pairs = tester.pair_fwd  # inp -> out
                        mask_ops = tester.pair_mask_fwd_splits
                        mask_argsorts = tester.mask_argsort_fwd_splits
                        for j in range(tester.num_split):
                            # beta = 1 if j == 1 else 0
                            beta = 0
                            mask_filter = tester.masks[j].item()
                            reverse_mask = False
                            mask_op = mask_output[j]

                            CONV.run_with_tuned_result(
                                BestConvAlgoByProfile(desp, tester.arch, spk),
                                desp.op_type.value,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                                torch_tensor_to_tv(mask_argsorts[j]),
                                torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                            )
                    dw_ref = tester.dw_ref
                    dw_my = weight_tv.cpu().numpy()
                    if dtype != np.float16:
                        # print(desp, spk, K, C, mask_width, algo)
                        test_case.assertAllClose(dw_ref, dw_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(dw_ref.reshape(-1) - dw_my.reshape(-1))
                        # print(desp, error_norm)
                        if (error_norm > 5):
                            print(f"{desp}, Error={error_norm}, {spk}")
                        assert error_norm < 10 * multipler

def _test_native_conv_cuda(subm: bool):
    ndim = 3
    dtype_to_tol = {
        np.float32: (1e-4, 1e-4),
        np.float16: (1e-2, 1e-2),
        np.int8: (1e-4, 1e-4),
    }
    device = torch.device("cuda:0")
    shapes = [[19, 18, 17]]
    batchsizes = [1]
    dtypes = [np.float32, np.float16]
    test_case = TestCase()
    in_channels = [32, 47]
    out_channels = [32, 48, 62]
    if subm:
        ksizes = [3, 5]
        strides = [1]
        paddings = [0]
        dilations = [1]
    else:
        ksizes = [2, 3]
        strides = [1, 2, 3]
        paddings = [0, 1]
        dilations = [1, 2]
    multiple_base = 128

    arch = torch.cuda.get_device_capability()
    stream = get_current_stream()
    force_nvrtc = False
    for shape, bs, C, K, k, s, p, d, dtype in tqdm.tqdm(params_grid(
            shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, dtypes)):
        tester = SparseConvTester(ConvAlgo.Native, subm, shape, bs, dtype, 1500, K, C, k, s, p, d, check_bias=True, check_act=True)
        bias = None
        if tester.check_bias:
            bias = tv.from_numpy(tester.bias).cuda()
        atol, rtol = dtype_to_tol[dtype]
        multipler = max(C, K) / multiple_base
        multipler = max(multipler, 1.0)

        kv_center = tester.kv // 2
        kv = tester.kv
        pair_in = torch_tensor_to_tv(tester.pair_native)[0]
        pair_out = torch_tensor_to_tv(tester.pair_native)[1]

        op_types = [ConvOpType.kForward, ConvOpType.kBackwardInput, ConvOpType.kBackwardWeight]
        # op_types = [ConvOpType.kForward]

        indice_pair_num_cpu = tester.indice_num_per_loc_np
        spk = 1

        out_ref = tester.out_ref
        din_ref = tester.din_ref
        dw_ref = tester.dw_ref.reshape(K, -1, C)

        for op_type in op_types:
            inp_th, weight_th, output_th = tester.get_operands_torch(op_type)
            weight_th = weight_th.view(K, -1, C)
            inp_tv = torch_tensor_to_tv(inp_th)
            weight_tv = torch_tensor_to_tv(weight_th)
            output_tv = torch_tensor_to_tv(output_th)
            if op_type == ConvOpType.kForward:
                a = inp_tv
                c = output_tv
                b = weight_tv.select(1, tester.kv // 2)
                if SPCONV_CPP_GEMM:
                    avail_desps = GEMM_CPP.get_all_available(a, b, c, False, True, False, arch, ShuffleStrideType.ShuffleAC.value)
                else:
                    avail_desps = GEMM.get_all_available(a, b, c, False, True, False, arch, ShuffleStrideType.ShuffleAC)

                for desp in avail_desps:
                    if subm:
                        torch.mm(inp_th, weight_th[:, tester.kv // 2].T, out=output_th)
                            # output_th += bias_th
                    else:
                        output_tv.zero_()
                    inited = subm
                    # determine last valid subm indices, then apply 
                    for i, nhot in enumerate(indice_pair_num_cpu):
                        if subm and i == kv_center:
                            continue
                        if subm and i > kv_center:
                            nhot = indice_pair_num_cpu[kv - i - 1]
                        if nhot <= 0:
                            continue
                        inp_indices = pair_in[i].slice_first_axis(0, nhot)
                        out_indices = pair_out[i].slice_first_axis(0, nhot)
                        b = weight_tv.select(1, i)
                        # inp @ filter.T, NC @ KC
                        beta = 1.0 if inited else 0.0
                        if SPCONV_CPP_GEMM:
                            GEMM_CPP.run_with_tuned_result(
                                GemmTuneResult(desp, tester.arch, 1),
                                a,
                                b,
                                c,
                                False,
                                True,
                                False,
                                arch=arch,
                                stream_int=stream,
                                shuffle_type=ShuffleStrideType.ShuffleAC.value,
                                a_inds=inp_indices,
                                b_inds=tv.Tensor(),
                                c_inds=out_indices,
                                hint=AlgoHint.Fowrard.value,
                                alpha=1.0,
                                beta=beta,
                                force_nvrtc=force_nvrtc)
                        else:
                            GEMM.run_with_tuned_result(
                                BestAlgoByProfile(desp, tester.arch, 1),
                                a,
                                b,
                                c,
                                False,
                                True,
                                False,
                                arch=arch,
                                stream=stream,
                                shuffle_type=ShuffleStrideType.ShuffleAC,
                                a_inds=inp_indices,
                                c_inds=out_indices,
                                hint=AlgoHint.Fowrard.value,
                                alpha=1.0,
                                beta=beta,
                                force_nvrtc=force_nvrtc)
                        inited = True
                    if bias is not None and tester.check_act:
                        InferenceOps.bias_add_act_inplace(output_tv, bias, tv.gemm.Activation.ReLU, 0, 0)
                    else:
                        if bias is not None:
                            InferenceOps.bias_add_inplace(output_tv, bias, 0)
                        if tester.check_act:
                            InferenceOps.activation_inplace(output_tv, tv.gemm.Activation.ReLU, 0, 0)
                    out_my = output_tv.cpu().numpy()
                    if dtype != np.float16:
                        # error_norm = np.linalg.norm(out_ref.reshape(-1) - out_my.reshape(-1))
                        # assert error_norm < 1
                        # print(desp, K, C, k, error_norm)

                        test_case.assertAllClose(out_ref, out_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(out_ref.reshape(-1) - out_my.reshape(-1))
                        assert error_norm < 10 * multipler

            elif op_type == ConvOpType.kBackwardInput:
                a = output_tv
                b = weight_tv.select(1, tester.kv // 2)
                c = inp_tv
                if SPCONV_CPP_GEMM:
                    avail_desps = GEMM_CPP.get_all_available(a, b, c, False, False, False, arch, ShuffleStrideType.ShuffleAC.value,
                        use_tf32=SPCONV_ALLOW_TF32)
                else:
                    avail_desps = GEMM.get_all_available(a, b, c, False, False, False, arch, ShuffleStrideType.ShuffleAC,
                        use_tf32=SPCONV_ALLOW_TF32)

                for desp in avail_desps:
                    if subm:
                        torch.mm(output_th, weight_th[:, tester.kv // 2], out=inp_th)
                    else:
                        inp_tv.zero_()
                    inited = subm
                    for i, nhot in enumerate(indice_pair_num_cpu):
                        if subm and i == kv_center:
                            continue
                        if subm and i > kv_center:
                            nhot = indice_pair_num_cpu[kv - i - 1]
                        if nhot <= 0:
                            continue
                        inp_indices = pair_in[i].slice_first_axis(0, nhot)
                        out_indices = pair_out[i].slice_first_axis(0, nhot)
                        b = weight_tv.select(1, i)
                        # inp @ filter.T, NC @ KC
                        beta = 1.0 if inited else 0.0
                        if SPCONV_CPP_GEMM:
                            GEMM_CPP.run_with_tuned_result(
                                GemmTuneResult(desp, tester.arch, 1),
                                a,
                                b,
                                c,
                                False,
                                False,
                                False,
                                arch=arch,
                                stream_int=stream,
                                shuffle_type=ShuffleStrideType.ShuffleAC.value,
                                a_inds=out_indices,
                                b_inds=tv.Tensor(),
                                c_inds=inp_indices,
                                hint=AlgoHint.Fowrard.value,
                                alpha=1.0,
                                beta=beta,
                                force_nvrtc=force_nvrtc)
                        else:
                            GEMM.run_with_tuned_result(
                                BestAlgoByProfile(desp, tester.arch, 1),
                                a,
                                b,
                                c,
                                False,
                                False,
                                False,
                                arch=arch,
                                stream=stream,
                                shuffle_type=ShuffleStrideType.ShuffleAC,
                                a_inds=out_indices,
                                c_inds=inp_indices,
                                hint=AlgoHint.Fowrard.value,
                                alpha=1.0,
                                beta=beta,
                                force_nvrtc=force_nvrtc)

                        inited = True
                    din_my = inp_tv.cpu().numpy()
                    if dtype != np.float16:
                        # error_norm = np.linalg.norm(din_ref.reshape(-1) - din_my.reshape(-1))
                        # print(desp, K, C, k, error_norm)
                        test_case.assertAllClose(din_ref, din_my, atol=atol, rtol=rtol)
                        # assert error_norm < 1

                    else:
                        error_norm = np.linalg.norm(din_ref.reshape(-1) - din_my.reshape(-1))
                        assert error_norm < 10 * multipler
            else:
                a = output_tv
                b = inp_tv
                c = weight_tv.select(1, tester.kv // 2)
                if SPCONV_CPP_GEMM:
                    avail_desps = GEMM_CPP.get_all_available(a, b, c, True, False, False, arch, ShuffleStrideType.ShuffleAB.value,
                        use_tf32=SPCONV_ALLOW_TF32)
                else:
                    avail_desps = GEMM.get_all_available(a, b, c, True, False, False, arch, ShuffleStrideType.ShuffleAB,
                        use_tf32=SPCONV_ALLOW_TF32)

                for desp in avail_desps:
                    # print(desp, C, K, k, s, p, d)
                    # desp.is_nvrtc = True
                    inited = subm
                    weight_tv.zero_()
                    if subm:
                        torch.mm(output_th.T, inp_th, out=weight_th[:, kv_center])

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
                        a_inds = out_indices
                        b_inds = inp_indices
                        if SPCONV_CPP_GEMM:
                            GEMM_CPP.run_with_tuned_result(
                                GemmTuneResult(desp, tester.arch, 32),
                                a,
                                b,
                                weight_tv.select(1, i),
                                True,
                                False,
                                False,
                                arch=arch,
                                stream_int=stream,
                                shuffle_type=ShuffleStrideType.ShuffleAB.value,
                                a_inds=a_inds,
                                b_inds=b_inds,
                                c_inds=tv.Tensor(),
                                hint=AlgoHint.BackwardWeight.value,
                                alpha=1.0,
                                beta=beta,
                                force_nvrtc=force_nvrtc)

                        else:
                            GEMM.run_with_tuned_result(BestAlgoByProfile(desp, tester.arch, 32),
                                                    a,
                                                    b,
                                                    weight_tv.select(1, i),
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
                                                    force_nvrtc=force_nvrtc)

                    dw_my = weight_tv.cpu().numpy()
                    if dtype != np.float16:
                        error_norm = np.linalg.norm(dw_ref.reshape(-1) - dw_my.reshape(-1))
                        assert error_norm < 1 * multipler, f"{desp}, {error_norm}"
                    else:
                        error_norm = np.linalg.norm(dw_ref.reshape(-1) - dw_my.reshape(-1))
                        assert error_norm < 10 * multipler, f"{desp}, {error_norm}"


def test_all_algo_unit():
    # for i in range(5):
    _test_impgemm_conv_cuda(True)
    _test_impgemm_conv_cuda(False)
    _test_native_conv_cuda(True)
    _test_native_conv_cuda(False)


if __name__ == "__main__":
    test_all_algo_unit()