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

from typing import Dict, List, Optional, Set, Tuple, Union

from cumm.conv.bases import ConvLayout, ConvLayoutType, ConvOpType

from cumm.gemm.algospec.core import (GemmAlgo, ShuffleStrideType)
from cumm.tensorview.gemm import ConvAlgoDesp
from cumm.tensorview.gemm import ConvIterAlgo as ConvIterAlgoCpp
from cumm.tensorview.gemm import ConvOpType as ConvOpTypeCpp
from cumm.tensorview.gemm import ConvLayoutType as ConvLayoutTypeCpp
from cumm.tensorview.gemm import ShuffleStrideType as ShuffleStrideTypeCpp

from cumm.tensorview.gemm import ConvParams, GemmAlgoDesp, GemmParams
from cumm.gemm.main import GemmAlgoParams, gen_gemm_kernels
from cumm.conv.main import ConvAlgoParams, ConvIterAlgo, gen_gemm_kernels as gen_conv_kernels
from cumm import dtypes
from cumm.conv.bases import (NCHW, NHWC, ConvIterAlgo, ConvLayout,
                             ConvLayoutType, ConvMode, ConvOpType)
from cumm.gemm.core import MetaArray
from cumm.gemm.algospec import TensorOp


def _assign_gemm_desp_props(desp: Union[ConvAlgoDesp, GemmAlgoDesp],
                            p: Union[GemmAlgoParams, ConvAlgoParams]):
    desp.dtype_a = p.dtype_a.tv_dtype
    desp.dtype_b = p.dtype_b.tv_dtype
    desp.dtype_c = p.dtype_c.tv_dtype
    desp.dacc = p.dtype_acc.tv_dtype
    desp.dcomp = p.dtype_comp.tv_dtype
    desp.trans_a = p.trans_a
    desp.trans_b = p.trans_b
    desp.trans_c = p.trans_c

    desp.tile_shape = (p.ts[0], p.ts[1], p.ts[2])
    desp.warp_tile_shape = (p.wts[0], p.wts[1], p.wts[2])
    if p.tensorop is not None:
        desp.tensorop = (p.tensorop[0], p.tensorop[1], p.tensorop[2])
    desp.num_stage = p.num_stage
    desp.algo = p.algo.value
    desp.split_k_serial = p.splitk_serial
    desp.split_k_parallel = p.splitk_parallel
    desp.shuffle_type = ShuffleStrideTypeCpp(p.shuffle_stride.value)
    desp.access_per_vector = p.access_per_vector
    desp.is_nvrtc = p.is_nvrtc

def get_gemm_algo_desp_from_param(p: GemmAlgoParams):
    desp = GemmAlgoDesp()
    _assign_gemm_desp_props(desp, p)
    # here we must generate kernel for element-per-access data
    ker = gen_gemm_kernels(p)
    desp.element_per_access_a = ker.input_spec.input_iter_a.element_per_acc
    desp.element_per_access_b = ker.input_spec.input_iter_b.element_per_acc
    desp.element_per_access_c = ker.output_spec.out_iter.element_per_acc
    desp.min_arch = ker.min_arch()
    return desp


def get_conv_algo_desp_from_param(p: ConvAlgoParams):
    desp = ConvAlgoDesp(p.ndim, ConvOpTypeCpp(p.op_type.value))
    _assign_gemm_desp_props(desp, p)
    # conv attrs
    desp.ndim = p.ndim
    desp.op_type = ConvOpTypeCpp(p.op_type.value)
    desp.iter_algo = ConvIterAlgoCpp(p.iter_algo.value)
    desp.layout_i = ConvLayoutTypeCpp(p.layout_desp_input.layout_type.value)
    desp.layout_w = ConvLayoutTypeCpp(p.layout_desp_weight.layout_type.value)
    desp.layout_o = ConvLayoutTypeCpp(p.layout_desp_output.layout_type.value)
    desp.interleave_i = p.layout_desp_input.interleave
    desp.interleave_w = p.layout_desp_weight.interleave
    desp.interleave_o = p.layout_desp_output.interleave
    desp.mask_sparse = p.mask_sparse
    desp.increment_k_first = p.increment_k_first
    ker = gen_conv_kernels(p)
    desp.element_per_access_a = ker.input_spec.input_iter_a.element_per_acc
    desp.element_per_access_b = ker.input_spec.input_iter_b.element_per_acc
    desp.element_per_access_c = ker.output_spec.out_iter.element_per_acc
    desp.is_int8_inference = ker.int8_inference
    desp.dynamic_mask = ker.dynamic_mask

    desp.min_arch = ker.min_arch()
    return desp


def _assign_gemm_params(desp: Union[ConvAlgoDesp, GemmAlgoDesp],
                        p: Union[GemmAlgoParams, ConvAlgoParams]):
    p.dtype_a = dtypes.get_dtype_from_tvdtype(desp.dtype_a)
    p.dtype_b = dtypes.get_dtype_from_tvdtype(desp.dtype_b)
    p.dtype_c = dtypes.get_dtype_from_tvdtype(desp.dtype_c)
    p.dtype_acc = dtypes.get_dtype_from_tvdtype(desp.dacc)
    p.dtype_comp = dtypes.get_dtype_from_tvdtype(desp.dcomp)
    p.trans_a = desp.trans_a
    p.trans_b = desp.trans_b
    p.trans_c = desp.trans_c

    p.ts = MetaArray(*desp.tile_shape)
    p.wts = MetaArray(*desp.warp_tile_shape)
    if desp.tensorop[0] > 0:
        p.tensorop = TensorOp(
            (desp.tensorop[0], desp.tensorop[1], desp.tensorop[2]))
    p.num_stage = desp.num_stage
    p.algo = GemmAlgo(desp.algo)
    p.splitk_serial = desp.split_k_serial
    p.splitk_parallel = desp.split_k_parallel
    p.shuffle_stride = ShuffleStrideType(desp.shuffle_type.value)
    p.access_per_vector = desp.access_per_vector
    p.is_nvrtc = desp.is_nvrtc



def get_gemm_param_from_desp(desp: GemmAlgoDesp):
    p = GemmAlgoParams((0, 0, 0), (0, 0, 0), 0, "s8,s8,s8,s8,s8", False, False,
                          False, GemmAlgo.Simt)
    _assign_gemm_params(desp, p)
    return p


def get_conv_param_from_desp(desp: ConvAlgoDesp):
    p = ConvAlgoParams(desp.ndim, ConvOpType.kForward, ConvIterAlgo.Optimized,
                       (0, 0, 0), (0, 0, 0), 0, "s8,s8,s8,s8,s8", NHWC, NHWC, NHWC,
                       GemmAlgo.Simt)
    _assign_gemm_params(desp, p)
    # conv attrs
    p.ndim = desp.ndim
    p.op_type = ConvOpType(desp.op_type.value)
    p.iter_algo = ConvIterAlgo(desp.iter_algo.value)
    p.layout_desp_input = ConvLayout(ConvLayoutType(desp.layout_i.value),
                                     desp.interleave_i)
    p.layout_desp_weight = ConvLayout(ConvLayoutType(desp.layout_w.value),
                                      desp.interleave_w)
    p.layout_desp_output = ConvLayout(ConvLayoutType(desp.layout_o.value),
                                      desp.interleave_o)
    p.mask_sparse = desp.mask_sparse
    p.increment_k_first = desp.increment_k_first
    p.int8_inference = desp.is_int8_inference
    p.dynamic_mask = desp.dynamic_mask
    return p
