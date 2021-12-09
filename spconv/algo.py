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

from enum import Enum
from cumm import tensorview as tv
from typing import Dict, List, Set, Tuple, Union
from spconv.core_cc.cumm.gemm.main import GemmAlgoDesp, GemmMainUnitTest, GemmParams
from spconv.core_cc.cumm.conv.main import ConvAlgoDesp, ConvMainUnitTest, ConvParams
from cumm.conv.bases import ConvLayout, ConvLayoutType, ConvOpType
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType, get_min_arch_of_algo_str, get_available_algo_str_from_arch
from cumm.gemm.codeops import group_by, div_up
from spconv.constants import NDIM_DONT_CARE, SPCONV_BWD_SPLITK
from typing import Optional
import time
from threading import Lock
import contextlib
import numpy as np
from spconv.core import ConvAlgo, AlgoHint
from spconv.tools import CUDAKernelTimer

ALL_ALGO_DESPS = GemmMainUnitTest.get_all_algo_desp()
ALL_CONV_ALGO_DESPS = ConvMainUnitTest.get_all_conv_algo_desp()
_GEMM_STATIC_KEY = Tuple[bool, bool, bool, int, int, int, str, str]


class SimpleGemmAlgoMeta:
    def __init__(self, tile_ms: List[int], tile_ns: List[int],
                 tile_ks: List[int],
                 tile_shape_to_algos: Dict[int, List[int]]) -> None:
        self.tile_shape_to_algos = tile_shape_to_algos
        self.tile_ms = tile_ms
        self.tile_ns = tile_ns
        self.tile_ks = tile_ks


class BestAlgoByProfile:
    def __init__(self, algo_desp: GemmAlgoDesp, splitk: int = 1) -> None:
        self.algo_desp = algo_desp
        self.splitk = splitk


class BestConvAlgoByProfile:
    def __init__(self, algo_desp: ConvAlgoDesp, splitk: int = 1) -> None:
        self.algo_desp = algo_desp
        self.splitk = splitk


class SimpleGemm:
    def __init__(self, desps: List[GemmAlgoDesp]) -> None:
        self.desps = desps
        self.lock = Lock()
        self.static_key_to_desps = group_by(self.get_static_key, desps)
        self.static_key_to_meta: Dict[_GEMM_STATIC_KEY,
                                      SimpleGemmAlgoMeta] = {}
        for k, static_desps in self.static_key_to_desps.items():
            tile_shape_to_algos: Dict[int, List[int]] = {}
            tile_ms: Set[int] = set()
            tile_ns: Set[int] = set()
            tile_ks: Set[int] = set()
            for i, desp in enumerate(static_desps):
                ts = desp.tile_shape
                tile_ms.add(ts[0])
                tile_ns.add(ts[1])
                tile_ks.add(ts[2])
                tile_key = ts[0] | (ts[1] << 20) | (ts[2] << 40)
                if tile_key not in tile_shape_to_algos:
                    tile_shape_to_algos[tile_key] = []
                tile_shape_to_algos[tile_key].append(i)
            tile_ms_list = list(tile_ms)
            tile_ns_list = list(tile_ns)
            tile_ks_list = list(tile_ks)
            tile_ms_list.sort()
            tile_ns_list.sort()
            tile_ks_list.sort()
            self.static_key_to_meta[k] = SimpleGemmAlgoMeta(
                tile_ms_list, tile_ns_list, tile_ks_list, tile_shape_to_algos)

        self.nk_forward_cache: Dict[Tuple[int, int, int, int, int],
                                    BestAlgoByProfile] = {}  # for forward
        self.nk_dgrad_cache: Dict[Tuple[int, int,
                                        int, int, int], BestAlgoByProfile] = {
                                        }  # for backward weight

        self.mn_cache: Dict[Tuple[int, int, int, int, int],
                            BestAlgoByProfile] = {}  # for backward weight

    @staticmethod
    def get_static_key(d: GemmAlgoDesp) -> _GEMM_STATIC_KEY:
        return (d.trans_a, d.trans_b, d.trans_c, d.dtype_a, d.dtype_b,
                d.dtype_c, d.shuffle_type, d.algo)

    def device_synchronize(self):
        return GemmMainUnitTest.device_synchronize()

    def get_all_available(
            self,
            a: tv.Tensor,
            b: tv.Tensor,
            c: tv.Tensor,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            arch: Tuple[int, int],
            shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle):
        if trans_c:
            trans_a = not trans_a
            trans_b = not trans_b
            trans_a, trans_b = trans_b, trans_a
            a, b = b, a
            trans_c = False
        avail_algos = get_available_algo_str_from_arch(arch)
        finally_algos: List[GemmAlgoDesp] = []
        for algo in avail_algos:
            static_key = (trans_a, trans_b, trans_c, a.dtype, b.dtype, c.dtype,
                          shuffle_type.value, algo)
            desps = self.static_key_to_desps.get(static_key, None)
            if desps is None or len(desps) == 0:
                continue
            for desp in desps:
                # skip volta tensor op since it is very slow in architectures except volta.
                if arch >= (7, 5) and desp.algo == GemmAlgo.Volta.value:
                    continue
                lda = a.dim(1)
                ldb = b.dim(1)
                ldc = c.dim(1)
                if desp.supported_ldx(lda, ldb, ldc):
                    finally_algos.append(desp)
        return finally_algos

    def select(self,
               a: tv.Tensor,
               b: tv.Tensor,
               c: tv.Tensor,
               trans_a: bool,
               trans_b: bool,
               trans_c: bool,
               arch: Tuple[int, int],
               shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle,
               a_inds: tv.Tensor = tv.Tensor(),
               b_inds: tv.Tensor = tv.Tensor(),
               c_inds: tv.Tensor = tv.Tensor(),
               hint: int = AlgoHint.NoHint.value):
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape, trans_a,
                                               trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds.shape, b_inds.shape,
                                               c_inds.shape)
        if trans_c:
            trans_a = not trans_a
            trans_b = not trans_b
            trans_a, trans_b = trans_b, trans_a
            a, b = b, a
            trans_c = False
        avail_algos = get_available_algo_str_from_arch(arch)
        finally_algos: List[GemmAlgoDesp] = []
        for algo in avail_algos:
            static_key = (trans_a, trans_b, trans_c, a.dtype, b.dtype, c.dtype,
                          shuffle_type.value, algo)
            desps = self.static_key_to_desps.get(static_key, None)
            if desps is None or len(desps) == 0:
                continue
            meta = self.static_key_to_meta[static_key]
            # for shuffle stride algos, we need to make channel tile size as large as possible.
            # so if ShuffleAC, we need to make k largest.
            selected_algo_desps = GemmMainUnitTest.simple_select_tile_shape(
                m,
                n,
                k,
                meta.tile_ms,
                meta.tile_ns,
                meta.tile_ks,
                meta.tile_shape_to_algos,
                large_k_first=shuffle_type == shuffle_type.ShuffleAC)
            if not selected_algo_desps:
                candidate = desps
            else:
                candidate = [desps[i] for i in selected_algo_desps]
            # select by hint
            if hint == 0:
                return candidate[0]
            if hint & (AlgoHint.Fowrard.value | AlgoHint.BackwardInput.value):
                # m may be huge, n and k are small
                # don't need mixed precision
                # don't need splitk
                finally_algos = []
                if a.dtype == tv.float16:
                    dacc = tv.float16
                    dcomp = tv.float16
                    for can in candidate:
                        if can.dacc == dacc and can.dcomp == dcomp:
                            finally_algos.append(can)
                else:
                    finally_algos = candidate
            elif hint & AlgoHint.BackwardWeight.value:
                # k is huge
                # don't support i8
                # if f16, acc and comp must be f32
                finally_algos = []
                candidate_filtered: List[GemmAlgoDesp] = list(
                    filter(lambda x: x.split_k_serial, candidate))
                if not candidate_filtered:
                    candidate_filtered = candidate
                if a.dtype == tv.int8:
                    continue
                elif a.dtype == tv.float16:
                    dacc = tv.float32
                    dcomp = tv.float32
                    for can in candidate_filtered:
                        if can.dacc == dacc and can.dcomp == dcomp:
                            finally_algos.append(can)
                else:
                    finally_algos = candidate_filtered
            else:
                return candidate[0]
        # print(finally_algos)
        if finally_algos:
            return finally_algos[0]
        return None

    def get_tuned_algo(
            self,
            a_dtype: int,
            b_dtype: int,
            c_dtype: int,
            a_shape: List[int],
            b_shape: List[int],
            c_shape: List[int],
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            arch: Tuple[int, int],
            shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            a_inds_shape: Optional[List[int]] = None,
            b_inds_shape: Optional[List[int]] = None,
            c_inds_shape: Optional[List[int]] = None,
            hint: int = AlgoHint.NoHint.value):
        if a_inds_shape is None:
            a_inds_shape = []
        if b_inds_shape is None:
            b_inds_shape = []
        if c_inds_shape is None:
            c_inds_shape = []
        m, n, k = GemmMainUnitTest.extract_mnk(a_shape, b_shape, trans_a,
                                               trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds_shape, b_inds_shape,
                                               c_inds_shape)
        if hint & AlgoHint.BackwardWeight.value:
            key = (a_dtype, b_dtype, c_dtype, m, n)
            return self.mn_cache.get(key, None)
        elif hint & AlgoHint.BackwardInput.value:
            key = (a_dtype, b_dtype, c_dtype, n, k)
            return self.nk_dgrad_cache.get(key, None)
        elif hint & AlgoHint.Fowrard.value:
            key = (a_dtype, b_dtype, c_dtype, n, k)
            return self.nk_forward_cache.get(key, None)
        raise NotImplementedError

    def extract_mnk(
            self,
            a_shape: List[int],
            b_shape: List[int],
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            arch: Tuple[int, int],
            shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            a_inds_shape: Optional[List[int]] = None,
            b_inds_shape: Optional[List[int]] = None,
            c_inds_shape: Optional[List[int]] = None,
            hint: int = AlgoHint.NoHint.value):
        if a_inds_shape is None:
            a_inds_shape = []
        if b_inds_shape is None:
            b_inds_shape = []
        if c_inds_shape is None:
            c_inds_shape = []
        m, n, k = GemmMainUnitTest.extract_mnk(a_shape, b_shape, trans_a,
                                               trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds_shape, b_inds_shape,
                                               c_inds_shape)
        return m, n, k

    def tune_and_cache(
            self,
            a: tv.Tensor,
            b: tv.Tensor,
            c: tv.Tensor,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            arch: Tuple[int, int],
            shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            a_inds: tv.Tensor = tv.Tensor(),
            b_inds: tv.Tensor = tv.Tensor(),
            c_inds: tv.Tensor = tv.Tensor(),
            hint: int = AlgoHint.NoHint.value,
            alpha: float = 1.0,
            beta: float = 0.0,
            gather_data: tv.Tensor = tv.Tensor(),
            scatter_data: tv.Tensor = tv.Tensor(),
            # mm_func
            stream: int = 0):
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape, trans_a,
                                               trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds.shape, b_inds.shape,
                                               c_inds.shape)
        avail = self.get_all_available(a, b, c, trans_a, trans_b, trans_c,
                                       arch, shuffle_type)

        c_ = c.clone()
        times: List[float] = []
        best_gather_params = (-1, -1, -1, -1)
        best_scatter_params = (-1, -1, -1, -1)

        all_profile_res: List[BestAlgoByProfile] = []
        for desp in avail:
            c_.zero_()
            split_k_slices = 1
            # TODO better splitk selection
            if desp.split_k_serial and hint & AlgoHint.BackwardWeight.value:
                split_k_slices = max(min(32, k // 128), 1)
            params = GemmParams()
            params.a = a
            params.b = b
            params.c = c_
            params.a_inds = a_inds
            params.b_inds = b_inds
            params.c_inds = c_inds
            params.algo_desp = desp
            params.alpha = alpha
            params.beta = beta
            params.stream = stream
            if desp.split_k_serial and hint & AlgoHint.BackwardWeight.value:
                splitk_tests = SPCONV_BWD_SPLITK
            else:
                splitk_tests = [1]
            spk_speeds = []
            for spk in splitk_tests:
                this_times = []
                for j in range(3):
                    GemmMainUnitTest.stream_synchronize(stream)
                    t = time.time()
                    params.split_k_slices = spk
                    GemmMainUnitTest.matmul2(params)
                    GemmMainUnitTest.stream_synchronize(stream)
                    this_times.append(time.time() - t)
                times.append(np.mean(this_times[1:]))
                spk_speeds.append(times[-1])

                all_profile_res.append(BestAlgoByProfile(desp, splitk=spk))

        min_time = 1000
        min_idx = -1
        for i, t in enumerate(times):
            if t < min_time:
                min_time = t
                min_idx = i
        res = all_profile_res[min_idx]
        with self.lock:
            if hint & AlgoHint.BackwardWeight.value:
                key = (a.dtype, b.dtype, c.dtype, m, n)
                self.mn_cache[key] = res
            elif hint & AlgoHint.BackwardInput.value:
                key = (a.dtype, b.dtype, c.dtype, n, k)
                self.nk_dgrad_cache[key] = res
            elif hint & AlgoHint.Fowrard.value:
                key = (a.dtype, b.dtype, c.dtype, n, k)
                self.nk_forward_cache[key] = res
            else:
                raise NotImplementedError

        return res, min_time

    def run_with_tuned_result(
        self,
        profile_res: BestAlgoByProfile,
        a: tv.Tensor,
        b: tv.Tensor,
        c: tv.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        arch: Tuple[int, int],
        stream: int,
        shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle,
        a_inds: tv.Tensor = tv.Tensor(),
        b_inds: tv.Tensor = tv.Tensor(),
        c_inds: tv.Tensor = tv.Tensor(),
        hint: int = AlgoHint.NoHint.value,
        alpha: float = 1.0,
        beta: float = 0.0,
        gather_data: tv.Tensor = tv.Tensor(),
        workspace: tv.Tensor = tv.Tensor(),
        timer: CUDAKernelTimer = CUDAKernelTimer(False)):
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape, trans_a,
                                               trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds.shape, b_inds.shape,
                                               c_inds.shape)
        # GemmMainUnitTest.stream_synchronize(stream)
        algo_desp = profile_res.algo_desp
        assert algo_desp is not None
        split_k_slices = 1
        # TODO better splitk selection
        # if algo_desp.split_k_serial and hint & AlgoHint.BackwardWeight.value:
        #     split_k_slices = max(min(32, k // 128), 1)
        if profile_res.splitk > 1:
            split_k_slices = profile_res.splitk
        params = GemmParams()
        params.a = a
        params.b = b
        params.c = c
        params.a_inds = a_inds
        params.b_inds = b_inds
        params.c_inds = c_inds
        params.algo_desp = algo_desp
        params.split_k_slices = split_k_slices
        params.stream = stream
        params.alpha = alpha
        params.beta = beta
        params.workspace = workspace
        # gather = 0
        # if profile_res.external_gather and not gather_data.empty():
        #     GemmMainUnitTest.stream_synchronize(stream)
        #     tt = time.time()
        #     assert not gather_data.empty()
        #     params.a_inds = tv.Tensor()
        #     params.a = gather_data
        #     # print(profile_res.gather_params, gather_data.shape, a.shape, a_inds.shape)
        #     GATHER.gather(gather_data,
        #                    a,
        #                    a_inds,
        #                    *profile_res.gather_params,
        #                    stream=stream)
        #     GemmMainUnitTest.stream_synchronize(stream)
        #     gather = time.time() - tt
        if timer.enable:
            assert timer._timer is not None
            params.timer = timer._timer

        GemmMainUnitTest.matmul2(params)
        # GemmMainUnitTest.stream_synchronize(stream)
        return algo_desp


_CONV_STATIC_KEY = Tuple[int, int, int, int, int, int, int, int, int, str, int]


class SimpleConv:
    def __init__(self, desps: List[ConvAlgoDesp]) -> None:
        self.desps = desps
        self.lock = Lock()

        self.static_key_to_desps = group_by(self.get_static_key, desps)
        self.static_key_to_meta: Dict[_CONV_STATIC_KEY,
                                      SimpleGemmAlgoMeta] = {}
        for k, static_desps in self.static_key_to_desps.items():
            tile_shape_to_algos: Dict[int, List[int]] = {}
            tile_ms: Set[int] = set()
            tile_ns: Set[int] = set()
            tile_ks: Set[int] = set()
            for i, desp in enumerate(static_desps):
                ts = desp.tile_shape
                tile_ms.add(ts[0])
                tile_ns.add(ts[1])
                tile_ks.add(ts[2])
                tile_key = ts[0] | (ts[1] << 20) | (ts[2] << 40)
                if tile_key not in tile_shape_to_algos:
                    tile_shape_to_algos[tile_key] = []
                tile_shape_to_algos[tile_key].append(i)
            tile_ms_list = list(tile_ms)
            tile_ns_list = list(tile_ns)
            tile_ks_list = list(tile_ks)
            tile_ms_list.sort()
            tile_ns_list.sort()
            tile_ks_list.sort()
            self.static_key_to_meta[k] = SimpleGemmAlgoMeta(
                tile_ms_list, tile_ns_list, tile_ks_list, tile_shape_to_algos)

        self.kc_forward_cache: Dict[Tuple[int, int, int, int, int, int, int,
                                          int],
                                    BestConvAlgoByProfile] = {}  # for forward
        self.kc_dgrad_cache: Dict[Tuple[int, int, int, int, int, int, int,
                                        int], BestConvAlgoByProfile] = {
                                        }  # for backward weight
        self.kc_wgrad_cache: Dict[Tuple[int, int, int, int, int, int, int,
                                        int], BestConvAlgoByProfile] = {
                                        }  # for backward weight

    @staticmethod
    def get_static_key(d: ConvAlgoDesp) -> _CONV_STATIC_KEY:
        return (d.layout_i, d.layout_w, d.layout_o, d.interleave_i,
                d.interleave_w, d.interleave_o, d.dtype_input, d.dtype_weight,
                d.dtype_output, d.algo, d.op_type)

    def device_synchronize(self):
        return GemmMainUnitTest.device_synchronize()

    def get_all_available(self, inp: tv.Tensor, weight: tv.Tensor,
                          out: tv.Tensor, layout_i: ConvLayout,
                          layout_w: ConvLayout, layout_o: ConvLayout,
                          arch: Tuple[int, int], op_type: ConvOpType,
                          mask_width: int, fp32_accum: Optional[bool] = None):

        avail_algos = get_available_algo_str_from_arch(arch)
        finally_algos: List[ConvAlgoDesp] = []
        is_fp16 = inp.dtype == tv.float16 and weight.dtype == tv.float16 and out.dtype == tv.float16
        use_f32_as_accum = False
        kv = int(np.prod(weight.shape[1:-1]))
        # for 3d conv, if reduce axis is too large, may cause nan during 
        # forward.
        if is_fp16:
            if fp32_accum is None:
                if op_type == ConvOpType.kForward:
                    use_f32_as_accum = weight.dim(-1) * kv > 128 * 27
                elif op_type == ConvOpType.kBackwardInput:
                    use_f32_as_accum = weight.dim(0) * kv > 128 * 27
            else:
                use_f32_as_accum = fp32_accum
        for algo in avail_algos:
            static_key = (layout_i.layout_type.value,
                          layout_w.layout_type.value,
                          layout_o.layout_type.value, layout_i.interleave,
                          layout_w.interleave, layout_o.interleave, inp.dtype,
                          weight.dtype, out.dtype, algo, op_type.value)
            desps = self.static_key_to_desps.get(static_key, None)
            if desps is None or len(desps) == 0:
                continue
            for desp in desps:
                # skip volta tensor op since it is very slow in architectures except volta.
                if arch >= (7, 5) and desp.algo == GemmAlgo.Volta.value:
                    continue
                if arch >= (7, 0) and is_fp16:
                    # skip simt fp16 kernels if we have tensor core
                    if desp.algo == GemmAlgo.Simt:
                        continue
                    if use_f32_as_accum:
                        if desp.dacc == tv.float16:
                            continue
                        
                ldi = inp.dim(-1)
                ldw = weight.dim(-1)
                ldo = out.dim(-1)
                mask_width_valid = True
                if desp.op_type == ConvOpType.kBackwardWeight.value:
                    assert mask_width > 0
                    mask_width_valid = mask_width % desp.tile_shape[2] == 0
                if desp.supported_ldx_conv(ldi, ldw, ldo) and mask_width_valid:
                    finally_algos.append(desp)
        return finally_algos

    def get_tuned_algo(self,
                       op_type: ConvOpType,
                       i_dtype: int,
                       w_dtype: int,
                       o_dtype: int,
                       k: int,
                       c: int,
                       arch: Tuple[int, int],
                       mask_width: int = -1):
        if not op_type == ConvOpType.kBackwardWeight:
            # fwd and dgrad don't need
            mask_width = -1
        key = (i_dtype, w_dtype, o_dtype, k, c, arch[0], arch[1], mask_width)
        if op_type == ConvOpType.kForward:
            return self.kc_forward_cache.get(key, None)
        elif op_type == ConvOpType.kBackwardInput:
            return self.kc_dgrad_cache.get(key, None)
        elif op_type == ConvOpType.kBackwardWeight:
            return self.kc_wgrad_cache.get(key, None)
        raise NotImplementedError

    def query_workspace_size(self, desp: ConvAlgoDesp, splitk: int,
                             op_type: ConvOpType, N: int, C: int, K: int,
                             kv: int):
        mnk = ConvMainUnitTest.extract_mnk(op_type.value, N, C, K, kv, -1, -1,
                                           True)
        return desp.query_conv_workspace_size(mnk[0], mnk[1], mnk[2], splitk,
                                              kv)

    def tune_and_cache(self,
                       op_type: ConvOpType,
                       inp: tv.Tensor,
                       weight: tv.Tensor,
                       output: tv.Tensor,
                       layout_i: ConvLayout,
                       layout_w: ConvLayout,
                       layout_o: ConvLayout,
                       arch: Tuple[int, int],
                       mask: tv.Tensor,
                       mask_argsort: tv.Tensor,
                       indices: tv.Tensor,
                       reverse_mask: bool,
                       mask_filter: int = 0xffffffff,
                       mask_width: int = -1,
                       mask_output: tv.Tensor = tv.Tensor(),
                       alpha: float = 1.0,
                       beta: float = 0.0,
                       stream: int = 0,
                       fp32_accum: Optional[bool] = None):
        avail = self.get_all_available(inp, weight, output, layout_i, layout_w,
                                       layout_o, arch, op_type, mask_width, 
                                       fp32_accum)
        inp = inp.clone()
        weight = weight.clone()
        output = output.clone()

        channel_k = output.dim(1)
        channel_c = inp.dim(1)

        times: List[float] = []
        all_profile_res: List[BestConvAlgoByProfile] = []
        for desp in avail:
            # for sparse conv, ndim isn't used, so we just provide a constant value.
            params = ConvParams(NDIM_DONT_CARE, op_type.value)
            params.conv_algo_desp = desp
            params.input = inp
            params.weight = weight.view([channel_k, -1, channel_c])
            params.output = output

            params.mask_width = mask_width
            params.alpha = alpha
            params.beta = beta
            params.stream = stream
            params.mask_argsort = mask_argsort
            params.indices = indices
            params.mask = mask
            params.mask_output = mask_output
            if op_type == ConvOpType.kBackwardWeight:
                assert not mask_output.empty()
            if op_type == ConvOpType.kBackwardInput:
                params.reverse_mask = reverse_mask
            params.mask_filter = mask_filter
            if desp.split_k_serial and op_type == ConvOpType.kBackwardWeight:
                splitk_tests = SPCONV_BWD_SPLITK
                # splitk_tests = [1]
            else:
                splitk_tests = [1]
            spk_speeds = []
            for spk in splitk_tests:
                this_times = []
                for j in range(3):
                    GemmMainUnitTest.stream_synchronize(stream)
                    t = time.time()
                    params.split_k_slices = spk
                    ConvMainUnitTest.implicit_gemm2(params)
                    GemmMainUnitTest.stream_synchronize(stream)
                    this_times.append(time.time() - t)
                times.append(np.mean(this_times[1:]))
                spk_speeds.append(times[-1])

                all_profile_res.append(BestConvAlgoByProfile(desp, splitk=spk))
        if not all_profile_res:
            raise ValueError("can't find suitable algorithm for", op_type)
        min_time = 1000
        min_idx = -1
        for i, t in enumerate(times):
            if t < min_time:
                min_time = t
                min_idx = i
        res = all_profile_res[min_idx]
        if not op_type == ConvOpType.kBackwardWeight:
            # fwd and dgrad don't need
            mask_width = -1
        key = (inp.dtype, weight.dtype, output.dtype, channel_k, channel_c,
               arch[0], arch[1], mask_width)
        with self.lock:
            if op_type == ConvOpType.kForward:
                self.kc_forward_cache[key] = res
            elif op_type == ConvOpType.kBackwardInput:
                self.kc_dgrad_cache[key] = res
            elif op_type == ConvOpType.kBackwardWeight:
                self.kc_wgrad_cache[key] = res
            else:
                raise NotImplementedError
        return res, min_time

    def run_with_tuned_result(self,
                              profile_res: BestConvAlgoByProfile,
                              op_type: Union[ConvOpType, int],
                              inp: tv.Tensor,
                              weight: tv.Tensor,
                              output: tv.Tensor,
                              mask: tv.Tensor,
                              mask_argsort: tv.Tensor,
                              mask_output: tv.Tensor,
                              indices: tv.Tensor,
                              reverse_mask: bool,
                              mask_filter: int = 0xffffffff,
                              mask_width: int = -1,
                              alpha: float = 1.0,
                              beta: float = 0.0,
                              stream: int = 0,
                              workspace: tv.Tensor = tv.Tensor(),
                              verbose: bool = False,
                              timer: CUDAKernelTimer = CUDAKernelTimer(False)):
        channel_k = output.dim(1)
        channel_c = inp.dim(1)
        # GemmMainUnitTest.stream_synchronize(stream)
        algo_desp = profile_res.algo_desp
        assert algo_desp is not None
        split_k_slices = 1
        if profile_res.splitk > 1:
            split_k_slices = profile_res.splitk
        if isinstance(op_type, int):
            op_type_value = op_type
        else:
            op_type_value = op_type.value
        params = ConvParams(NDIM_DONT_CARE, op_type_value)
        params.conv_algo_desp = profile_res.algo_desp
        params.input = inp
        params.verbose = verbose
        params.weight = weight.view([channel_k, -1, channel_c])
        params.output = output
        params.split_k_slices = split_k_slices
        params.alpha = alpha
        params.beta = beta
        params.stream = stream
        params.mask_argsort = mask_argsort
        params.indices = indices
        params.mask = mask
        params.mask_filter = mask_filter
        params.mask_width = mask_width
        params.mask_filter = mask_filter
        params.mask_output = mask_output
        params.reverse_mask = reverse_mask
        if timer.enable:
            assert timer._timer is not None
            params.timer = timer._timer
        # torch.cuda.synchronize()
        # t = time.time()
        params.workspace = workspace
        ConvMainUnitTest.implicit_gemm2(params)
        # torch.cuda.synchronize()
        # dura = time.time() - t
        # print("F", algo_desp, dura)

        # GemmMainUnitTest.stream_synchronize(stream)
        return algo_desp

    def stream_synchronize(self, stream: int):
        return GemmMainUnitTest.stream_synchronize(stream)


GEMM = SimpleGemm(ALL_ALGO_DESPS)
CONV = SimpleConv(ALL_CONV_ALGO_DESPS)

if __name__ == "__main__":
    print(len(ALL_CONV_ALGO_DESPS))
    print(ALL_CONV_ALGO_DESPS[0])
