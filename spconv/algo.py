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
from typing import Dict, List, Set, Tuple
from spconv.core_cc.cumm.gemm.main import GemmAlgoDesp, GemmMainUnitTest, GemmParams
# from spconv.core_cc.cumm.gemm.gather import GatherAll, ScatterAll
from cumm.gemm.algospec.core import ShuffleStrideType, get_min_arch_of_algo_str, get_available_algo_str_from_arch
from cumm.gemm.codeops import group_by, div_up
from typing import Optional
import time

import numpy as np


class ConvAlgo(Enum):
    Native = "Native"
    MaskImplicitGemm = "MaskImplicitGemm"
    MaskSplitImplicitGemm = "MaskSplitImplicitGemm"


class AlgoHint(Enum):
    NoHint = 0b000
    Fowrard = 0b001
    BackwardInput = 0b010
    BackwardWeight = 0b100


ALL_ALGO_DESPS = GemmMainUnitTest.get_all_algo_desp()

_GEMM_STATIC_KEY = Tuple[bool, bool, bool, int, int, int, str, str]

# GATHER = GatherAll()
# SCATTER = ScatterAll()


class SimpleGemmAlgoMeta:
    def __init__(self, tile_ms: List[int], tile_ns: List[int],
                 tile_ks: List[int],
                 tile_shape_to_algos: Dict[int, List[int]]) -> None:
        self.tile_shape_to_algos = tile_shape_to_algos
        self.tile_ms = tile_ms
        self.tile_ns = tile_ns
        self.tile_ks = tile_ks


class BestAlgoByProfile:
    def __init__(self,
                 algo_desp: GemmAlgoDesp,
                 external_gather: bool,
                 external_scatter: bool,
                 gather_params: Optional[Tuple[int, int, int, int]] = None,
                 scatter_params: Optional[Tuple[int, int, int, int]] = None,
                 splitk: int = 1) -> None:
        self.algo_desp = algo_desp
        self.external_gather = external_gather
        self.external_scatter = external_scatter
        self.gather_params = gather_params
        self.scatter_params = scatter_params
        self.splitk = splitk


class SimpleGemm:
    def __init__(self, desps: List[GemmAlgoDesp]) -> None:
        self.desps = desps

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

        self.nk_forward_cache: Dict[Tuple[int, int],
                            BestAlgoByProfile] = {}  # for forward
        self.nk_dgrad_cache: Dict[Tuple[int, int],
                            BestAlgoByProfile] = {}  # for backward weight

        self.mn_cache: Dict[Tuple[int, int],
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
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape,
                                               trans_a, trans_b, trans_c,
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

    def get_profiled_algo(
            self,
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
        m, n, k = GemmMainUnitTest.extract_mnk(a_shape, b_shape,
                                               trans_a, trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds_shape, b_inds_shape,
                                               c_inds_shape)
        if hint & AlgoHint.BackwardWeight.value:
            key = (m, n)
            return self.mn_cache.get(key, None)
        elif hint & AlgoHint.BackwardInput.value:
            key = (n, k)
            return self.nk_dgrad_cache.get(key, None)
        elif hint & AlgoHint.Fowrard.value:
            key = (n, k)
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
        m, n, k = GemmMainUnitTest.extract_mnk(a_shape, b_shape,
                                               trans_a, trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds_shape, b_inds_shape,
                                               c_inds_shape)
        return m, n, k

    def profile_and_cache(
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
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape,
                                               trans_a, trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds.shape, b_inds.shape,
                                               c_inds.shape)
        if hint & AlgoHint.BackwardWeight.value:
            key = (m, n)
        else:
            key = (n, k)

        avail = self.get_all_available(a, b, c, trans_a, trans_b, trans_c,
                                       arch, shuffle_type)
        c_ = c.clone()
        times: List[float] = []
        # gather_algos: List[GemmAlgoDesp] = []
        # find fastest gather algo for this input
        best_gather_params = (-1, -1, -1, -1)
        best_scatter_params = (-1, -1, -1, -1)
        # gather_data_ = tv.Tensor()
        # if not gather_data.empty(
        # ) and not hint & AlgoHint.BackwardWeight.value:
        #     # run gather here
        #     all_gather_params = GATHER.get_all_gather_params()
        #     gather_data_ = gather_data.clone()
        #     gather_times: List[float] = []

        #     for gather_params in all_gather_params:
        #         if GATHER.supported(gather_params[2], a.dim(1), a.dtype):
        #             this_times = []
        #             for j in range(10):
        #                 GemmMainUnitTest.stream_synchronize(stream)
        #                 t = time.time()
        #                 GATHER.gather(gather_data_, a, a_inds, *gather_params)
        #                 GemmMainUnitTest.stream_synchronize(stream)
        #                 this_times.append(time.time() - t)
        #             gather_times.append(np.mean(this_times[5:]))

        #     min_time = 1000
        #     min_idx = -1
        #     for i, t in enumerate(gather_times):
        #         if t < min_time:
        #             min_time = t
        #             min_idx = i
        #     best_gather_params = all_gather_params[min_idx]

        # if not scatter_data.empty(
        # ) and not hint & AlgoHint.BackwardWeight.value:
        #     # run gather here
        #     all_scatter_params = SCATTER.get_all_scatter_params()
        #     scatter_data_ = scatter_data.clone()
        #     scatter_times: List[float] = []

        #     for params in all_scatter_params:
        #         if SCATTER.supported_scatter(*params, a.dim(1), a.dtype):
        #             this_times = []
        #             for j in range(10):
        #                 GemmMainUnitTest.stream_synchronize(stream)
        #                 t = time.time()
        #                 SCATTER.scatter(c_, scatter_data_, c_inds, *params)
        #                 GemmMainUnitTest.stream_synchronize(stream)
        #                 this_times.append(time.time() - t)
        #             scatter_times.append(np.mean(this_times[5:]))

        #     min_time = 1000
        #     min_idx = -1
        #     for i, t in enumerate(scatter_times):
        #         if t < min_time:
        #             min_time = t
        #             min_idx = i
        #     best_scatter_params = all_scatter_params[min_idx]


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
                splitk_tests = [1, 2, 4, 8, 16, 32, 64]
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

                all_profile_res.append(
                    BestAlgoByProfile(desp, False, False, best_gather_params, best_scatter_params, splitk=spk))
            # if desp.split_k_serial:
            #     print(a.shape, b.shape, spk_speeds)
            # if not gather_data.empty(
            # ) and not hint & AlgoHint.BackwardWeight.value:
            #     # run gather here
            #     for spk in splitk_tests:
            #         this_times = []
            #         for j in range(3):

            #             GemmMainUnitTest.stream_synchronize(stream)
            #             t = time.time()
            #             params.a_inds = tv.Tensor()
            #             params.a = gather_data_
            #             params.split_k_slices = spk
            #             GATHER.gather(gather_data_,
            #                         a,
            #                         a_inds,
            #                         *best_gather_params,
            #                         stream=stream)
            #             GemmMainUnitTest.matmul2(params)
            #             GemmMainUnitTest.stream_synchronize(stream)
            #             this_times.append(time.time() - t)

            #         times.append(np.mean(this_times[1:]))
            #         # print("G", times[-1], times[-2])
            #         all_profile_res.append(
            #             BestAlgoByProfile(desp,
            #                             True,
            #                             False,
            #                             best_gather_params, best_scatter_params,
            #                             splitk=spk))

        min_time = 1000
        min_idx = -1
        for i, t in enumerate(times):
            if t < min_time:
                min_time = t
                min_idx = i
        res = all_profile_res[min_idx]
        if hint & AlgoHint.BackwardWeight.value:
            key = (m, n)
            self.mn_cache[key] = res
        elif hint & AlgoHint.BackwardInput.value:
            key = (n, k)
            self.nk_dgrad_cache[key] = res
        elif hint & AlgoHint.Fowrard.value:
            key = (n, k)
            self.nk_forward_cache[key] = res
        else:
            raise NotImplementedError

        return res, min_time

    def run_profile(
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
        workspace: tv.Tensor = tv.Tensor()):
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape,
                                               trans_a, trans_b, trans_c,
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

        GemmMainUnitTest.matmul2(params)
        # GemmMainUnitTest.stream_synchronize(stream)
        return algo_desp


GEMM = SimpleGemm(ALL_ALGO_DESPS)

if __name__ == "__main__":
    print(len(ALL_ALGO_DESPS))
    print(ALL_ALGO_DESPS[0])

    a = tv.zeros([64000, 32], dtype=tv.float16)
    b = tv.zeros([32, 64], dtype=tv.float16)
    c = tv.zeros([64000, 64], dtype=tv.float16)
    a_inds = tv.zeros([64000], dtype=tv.int32)
    c_inds = tv.zeros([64000], dtype=tv.int32)
    t = time.time()
    for i in range(100):
        algo = GEMM.select(a,
                           c,
                           b,
                           True,
                           False,
                           False, (7, 5),
                           ShuffleStrideType.ShuffleAB,
                           a_inds=a_inds,
                           b_inds=c_inds)
    print((time.time() - t) / 100)
    print(algo)
