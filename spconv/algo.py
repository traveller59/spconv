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

import contextlib
import time
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple, Union
from spconv.core_cc.cumm.common import CompileInfo
import numpy as np
from cumm import tensorview as tv
from cumm.conv.bases import ConvLayout, ConvLayoutType, ConvOpType
from cumm.conv.kernel import ConvKernel
from cumm.gemm.kernel import GemmKernel

from cumm.gemm.algospec.core import (GemmAlgo, ShuffleStrideType,
                                     get_available_algo_str_from_arch,
                                     get_min_arch_of_algo_str)
from cumm.gemm.codeops import div_up, group_by
from cumm.nvrtc import CummNVRTCModule, get_cudadevrt_path
from cumm.tensorview.gemm import ConvAlgoDesp
from cumm.tensorview.gemm import ConvOpType as ConvOpTypeCpp

from cumm.tensorview.gemm import ConvParams, GemmAlgoDesp, GemmParams
from cumm import dtypes

from spconv.constants import (NDIM_DONT_CARE, SPCONV_BWD_SPLITK,
                              SPCONV_NVRTC_MODE, SPCONV_DEBUG_NVRTC_KERNELS)
from spconv.core import ALL_IMPGEMM_PARAMS, AlgoHint, ConvAlgo, ALL_NATIVE_PARAMS
from spconv.core_cc.cumm.conv.main import ConvMainUnitTest
from spconv.core_cc.cumm.gemm.main import GemmMainUnitTest
from spconv.cppconstants import COMPILED_CUDA_GEMM_ARCHS
from cumm.tensorview.gemm import NVRTCParams
from spconv.tools import CUDAKernelTimer
from cumm.gemm.constants import NVRTCConstants, NVRTCMode

from spconv import algocore

from cumm.conv.main import gen_gemm_kernels as gen_conv_kernels
from cumm.gemm.main import gen_gemm_kernels
from spconv.core_cc.csrc.sparse.convops import GemmTuneResult, ConvTuneResult
from spconv.core_cc.csrc.sparse.convops.gemmops import GemmTunerSimple as GemmTunerSimpleBase
from spconv.core_cc.csrc.sparse.convops.convops import ConvTunerSimple as ConvTunerSimpleBase

ALL_ALGO_DESPS = GemmMainUnitTest.get_all_algo_desp()
ALL_CONV_ALGO_DESPS = ConvMainUnitTest.get_all_conv_algo_desp()


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
                 arch: Tuple[int, int],
                 splitk: int = 1) -> None:
        self.algo_desp = algo_desp
        self.splitk = splitk
        self.arch = arch


class BestConvAlgoByProfile:

    def __init__(self,
                 algo_desp: ConvAlgoDesp,
                 arch: Tuple[int, int],
                 splitk: int = 1) -> None:
        self.algo_desp = algo_desp
        self.splitk = splitk
        self.arch = arch


def _get_nvrtc_params(mod: CummNVRTCModule, ker: Union[GemmKernel, ConvKernel],
                      kernel_name: str):
    nvrtc_mode = SPCONV_NVRTC_MODE
    nvrtc_params = tv.gemm.NVRTCParams()
    nvrtc_params.cumodule = mod.get_cpp_object()
    nvrtc_params.mode = nvrtc_mode.value
    nvrtc_params.num_threads = ker.num_threads
    nvrtc_params.smem_size = ker.smem_size
    ns = ker.namespace

    if nvrtc_mode == NVRTCMode.DynamicParallism:
        nvrtc_params.kernel_name = mod.get_lowered_name(f"{ns}::nvrtc_kernel")

    elif nvrtc_mode == NVRTCMode.KernelAndCPU:
        nvrtc_params.kernel_name = mod.get_lowered_name(f"{ns}::{kernel_name}")
        nvrtc_params.init_kernel_name = mod.get_lowered_name(
            f"{ns}::nvrtc_kernel_cpu_out")
        nvrtc_params.param_size = mod.const_values[
            f"{ns}::{NVRTCConstants.SIZEOF_KEY}"]

        nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                              tv.uint8, 0)
        nvrtc_params.param_storage_cpu = tv.empty([nvrtc_params.param_size],
                                                  tv.uint8,
                                                  -1,
                                                  pinned=True)

    elif nvrtc_mode == NVRTCMode.Direct:
        nvrtc_params.kernel_name = mod.get_lowered_name(f"{ns}::{kernel_name}")
    elif nvrtc_mode == NVRTCMode.ConstantMemory:
        nvrtc_params.kernel_name = mod.get_lowered_name(f"{ns}::{kernel_name}")
        nvrtc_params.init_kernel_name = mod.get_lowered_name(
            f"{ns}::nvrtc_kernel_cpu_out")
        nvrtc_params.param_size = mod.const_values[
            f"{ns}::{NVRTCConstants.SIZEOF_KEY}"]
        nvrtc_params.constant_name = mod.get_lowered_name(
            f"&{ns}::{NVRTCConstants.CONSTANT_PARAM_KEY}")
        nvrtc_params.param_storage = tv.empty([nvrtc_params.param_size],
                                              tv.uint8, 0)
    else:
        raise NotImplementedError
    return nvrtc_params

class GemmTunerSimple(GemmTunerSimpleBase):
    def __init__(self, desps: List[GemmAlgoDesp]) -> None:
        super().__init__(desps)
        self._nvrtc_caches: Dict[Tuple[str, Tuple[int, int], int], NVRTCParams] = {}
    
    def _compile_nvrtc_module(self, desp: GemmAlgoDesp):
        params = algocore.get_gemm_param_from_desp(desp)
        kernel = gen_gemm_kernels(params, SPCONV_NVRTC_MODE)
        kernel.namespace = "spconv"
        custom_names = []
        if SPCONV_NVRTC_MODE == NVRTCMode.ConstantMemory:
            custom_names = [
                f"&{kernel.namespace}::{NVRTCConstants.CONSTANT_PARAM_KEY}"
            ]
        cudadevrt = ""
        if SPCONV_NVRTC_MODE == NVRTCMode.DynamicParallism:
            cudadevrt_p = get_cudadevrt_path()
            assert cudadevrt_p is not None, "DynamicParallism must have cudadevrt"
            cudadevrt = str(cudadevrt_p)
        mod = CummNVRTCModule([kernel],
                              cudadevrt_path=cudadevrt,
                              custom_names=custom_names)
        mod.load()
        return mod, kernel

    def cached_get_nvrtc_params(self, desp: GemmAlgoDesp, arch: Tuple[int, int], stream_int: int) -> NVRTCParams:
        
        key = (str(desp), arch, stream_int)
        if key in self._nvrtc_caches:
            return self._nvrtc_caches[key]
        mod, ker = self._compile_nvrtc_module(desp)
        print(f"Can't find algo {desp} in prebuilt. compile with nvrtc...")
        nvrtc_params = _get_nvrtc_params(mod, ker, "gemm_kernel")
        self._nvrtc_caches[key] = nvrtc_params
        return nvrtc_params

class ConvTunerSimple(ConvTunerSimpleBase):
    def __init__(self, desps: List[ConvAlgoDesp]) -> None:
        super().__init__(desps)
        self._nvrtc_caches: Dict[Tuple[str, Tuple[int, int], int], NVRTCParams] = {}
    
    def _compile_nvrtc_module(self, desp: ConvAlgoDesp):
        params = algocore.get_conv_param_from_desp(desp)
        kernel = gen_conv_kernels(params, SPCONV_NVRTC_MODE)
        kernel.namespace = "spconv"
        custom_names = []
        if SPCONV_NVRTC_MODE == NVRTCMode.ConstantMemory:
            custom_names = [
                f"&{kernel.namespace}::{NVRTCConstants.CONSTANT_PARAM_KEY}"
            ]
        cudadevrt = ""
        if SPCONV_NVRTC_MODE == NVRTCMode.DynamicParallism:
            cudadevrt_p = get_cudadevrt_path()
            assert cudadevrt_p is not None, "DynamicParallism must have cudadevrt"
            cudadevrt = str(cudadevrt_p)
        # mod = CummNVRTCModule([kernel],
        #                       cudadevrt_path=cudadevrt,
        #                       verbose=True,
        #                       custom_names=custom_names,
        #                       verbose_path="/home/yy/Projects/spconv-release/spconv/build/dev_nvrtc_int8")
        mod = CummNVRTCModule([kernel],
                              cudadevrt_path=cudadevrt,
                              verbose=False,
                              custom_names=custom_names)

        mod.load()
        return mod, kernel

    def cached_get_nvrtc_params(self, desp: ConvAlgoDesp, arch: Tuple[int, int], stream_int: int) -> NVRTCParams:
        key = (str(desp), arch, stream_int)
        if key in self._nvrtc_caches:
            return self._nvrtc_caches[key]
        mod, ker = self._compile_nvrtc_module(desp)
        print(f"Can't find algo {desp} in prebuilt. compile with nvrtc...")
        nvrtc_params = _get_nvrtc_params(mod, ker, "conv_kernel")
        self._nvrtc_caches[key] = nvrtc_params
        return nvrtc_params

_GEMM_STATIC_KEY = Tuple[bool, bool, bool, int, int, int, int]

class SimpleGemm:

    def __init__(self, prebuilt_desps: List[GemmAlgoDesp]) -> None:
        all_desps = [
            algocore.get_gemm_algo_desp_from_param(p)
            for p in ALL_NATIVE_PARAMS
        ]
        self.prebuilt_desps = prebuilt_desps
        self.prebuilt_desp_names = {str(d) for d in prebuilt_desps}
        if SPCONV_DEBUG_NVRTC_KERNELS:
            self.prebuilt_desp_names.clear()
        self.lock = Lock()
        self.static_key_to_desps = group_by(self.get_static_key, all_desps)
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
        self._nvrtc_caches: Dict[Tuple[str, Tuple[int, int]], NVRTCParams] = {}

    @staticmethod
    def get_static_key(d: GemmAlgoDesp) -> _GEMM_STATIC_KEY:
        return (d.trans_a, d.trans_b, d.trans_c, d.dtype_a, d.dtype_b,
                d.dtype_c, d.shuffle_type.value)

    def device_synchronize(self):
        return GemmMainUnitTest.device_synchronize()

    @staticmethod
    def _compile_nvrtc_module(desp: GemmAlgoDesp):
        params = algocore.get_gemm_param_from_desp(desp)
        kernel = gen_gemm_kernels(params, SPCONV_NVRTC_MODE)
        kernel.namespace = "spconv"
        custom_names = []
        if SPCONV_NVRTC_MODE == NVRTCMode.ConstantMemory:
            custom_names = [
                f"&{kernel.namespace}::{NVRTCConstants.CONSTANT_PARAM_KEY}"
            ]
        cudadevrt = ""
        if SPCONV_NVRTC_MODE == NVRTCMode.DynamicParallism:
            cudadevrt_p = get_cudadevrt_path()
            assert cudadevrt_p is not None, "DynamicParallism must have cudadevrt"
            cudadevrt = str(cudadevrt_p)
        mod = CummNVRTCModule([kernel],
                              cudadevrt_path=cudadevrt,
                              custom_names=custom_names)
        mod.load()
        return mod, kernel

    def _cached_get_nvrtc_params(self, desp: GemmAlgoDesp, arch: Tuple[int,
                                                                       int]):
        key = (str(desp), arch)
        if key in self._nvrtc_caches:
            return self._nvrtc_caches[key]
        mod, ker = self._compile_nvrtc_module(desp)
        print(f"Can't find algo {desp} in prebuilt. compile with nvrtc...")
        nvrtc_params = _get_nvrtc_params(mod, ker, "gemm_kernel")
        self._nvrtc_caches[key] = nvrtc_params
        return nvrtc_params

    def get_all_available(
            self,
            a: tv.Tensor,
            b: tv.Tensor,
            c: tv.Tensor,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            arch: Tuple[int, int],
            shuffle_type: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            use_tf32: bool = True):
        if trans_c:
            trans_a = not trans_a
            trans_b = not trans_b
            trans_a, trans_b = trans_b, trans_a
            a, b = b, a
            trans_c = False
        avail_algos = get_available_algo_str_from_arch(arch)
        finally_algos: List[GemmAlgoDesp] = []
        # print(self.static_key_to_desps)
        static_key = (trans_a, trans_b, trans_c, a.dtype, b.dtype, c.dtype,
                        shuffle_type.value)
        # for algo in avail_algos:
        #     static_key = (trans_a, trans_b, trans_c, a.dtype, b.dtype, c.dtype,
        #                   shuffle_type.value)
            # print(static_key)
        desps = self.static_key_to_desps.get(static_key, None)
        if desps is None or len(desps) == 0:
            return finally_algos
        # print(desps)
        for desp in desps:
            if arch < desp.min_arch:
                continue
            # skip volta tensor op since it is very slow in architectures except volta.
            if arch >= (7, 5) and desp.algo == GemmAlgo.Volta.value:
                continue
            if not use_tf32:
                if desp.tensorop[0] > 0 and a.dtype == tv.float32 and b.dtype == tv.float32:
                    continue
            lda = a.stride[0]
            ldb = b.stride[0]
            ldc = c.stride[0]
            if desp.supported_ldx(lda, ldb, ldc):
                if desp.is_nvrtc:
                    if not CompileInfo.algo_can_be_nvrtc_compiled(desp.min_arch):
                        continue
                if not CompileInfo.arch_is_compiled_gemm(arch):
                    # use PTX of possible
                    if not CompileInfo.gemm_algo_can_use_ptx(desp.min_arch, arch):
                        if CompileInfo.algo_can_be_nvrtc_compiled(desp.min_arch):
                            # compiled kernel can't use PTX, for example, desp need at least sm_80 and only sm_75+PTX is compiled
                            # all sm_80 code of this desp is invalid, we must use nvrtc.
                            # only desp <= sm_75 can use virtual PTX code to generate sm_80 code.
                            desp = desp.copy()
                            desp.is_nvrtc = True
                        else:
                            continue
                if SPCONV_DEBUG_NVRTC_KERNELS:
                    desp.is_nvrtc = True
                finally_algos.append(desp)
        return finally_algos


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
            stream: int = 0,
            use_tf32: bool = True):
        m, n, k = GemmMainUnitTest.extract_mnk(a.shape, b.shape, trans_a,
                                               trans_b, trans_c,
                                               shuffle_type.value,
                                               a_inds.shape, b_inds.shape,
                                               c_inds.shape)
        avail = self.get_all_available(a, b, c, trans_a, trans_b, trans_c,
                                       arch, shuffle_type, use_tf32)
        # c may be weight, may non-contiguous.
        # cumm.tensorview.Tensor don't support non-contiguous clone
        c_ = c.clone_whole_storage()
        times: List[float] = []
        best_gather_params = (-1, -1, -1, -1)
        best_scatter_params = (-1, -1, -1, -1)

        all_profile_res: List[BestAlgoByProfile] = []
        # print(avail)
        for desp in avail:
            c_.zero_whole_storage_()
            split_k_slices = 1
            # TODO better splitk selection
            if desp.split_k_serial and hint & AlgoHint.BackwardWeight.value:
                split_k_slices = max(min(32, k // 128), 1)
            params = GemmParams()
            if desp.is_nvrtc or str(desp) not in self.prebuilt_desp_names:
                params.nvrtc_params = self._cached_get_nvrtc_params(desp, arch)
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

                all_profile_res.append(
                    BestAlgoByProfile(desp, arch, splitk=spk))

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

    def run_with_tuned_result(self,
                              profile_res: BestAlgoByProfile,
                              a: tv.Tensor,
                              b: tv.Tensor,
                              c: tv.Tensor,
                              trans_a: bool,
                              trans_b: bool,
                              trans_c: bool,
                              arch: Tuple[int, int],
                              stream: int,
                              shuffle_type: ShuffleStrideType,
                              a_inds: tv.Tensor = tv.Tensor(),
                              b_inds: tv.Tensor = tv.Tensor(),
                              c_inds: tv.Tensor = tv.Tensor(),
                              hint: int = AlgoHint.NoHint.value,
                              alpha: float = 1.0,
                              beta: float = 0.0,
                              gather_data: tv.Tensor = tv.Tensor(),
                              workspace: tv.Tensor = tv.Tensor(),
                              timer: CUDAKernelTimer = CUDAKernelTimer(False),
                              force_nvrtc: bool = False,
                              bias: Optional[tv.Tensor] = None,
                              act_alpha: float = 0.0,
                              act_beta: float = 0.0,
                              act_type: tv.gemm.Activation = tv.gemm.Activation.None_):
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
        is_not_static = str(algo_desp) not in self.prebuilt_desp_names
        if algo_desp.is_nvrtc or is_not_static or force_nvrtc:
            params.nvrtc_params = self._cached_get_nvrtc_params(
                algo_desp, profile_res.arch)

        params.a = a
        params.b = b
        params.c = c
        if bias is not None:
            params.d = bias
        params.a_inds = a_inds
        params.b_inds = b_inds
        params.c_inds = c_inds
        params.algo_desp = algo_desp
        params.split_k_slices = split_k_slices
        params.stream = stream
        params.alpha = alpha
        params.beta = beta
        params.act_alpha = act_alpha
        params.act_beta = act_beta
        params.act_type = act_type
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


_CONV_STATIC_KEY = Tuple[int, int, int, int, int, int, int, int, int, int]


class SimpleConv:

    def __init__(self, prebuilt_desps: List[ConvAlgoDesp]) -> None:
        all_desps = [
            algocore.get_conv_algo_desp_from_param(p)
            for p in ALL_IMPGEMM_PARAMS
        ]
        self.all_desps = all_desps
        self.prebuilt_desps = prebuilt_desps
        self.prebuilt_desp_names = {str(d) for d in prebuilt_desps}
        
        self.lock = Lock()

        self.static_key_to_desps = group_by(self.get_static_key, all_desps)
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
                                          int, bool],
                                    BestConvAlgoByProfile] = {}  # for forward
        self.kc_dgrad_cache: Dict[Tuple[int, int, int, int, int, int, int,
                                        int, bool], BestConvAlgoByProfile] = {
                                        }  # for backward weight
        self.kc_wgrad_cache: Dict[Tuple[int, int, int, int, int, int, int,
                                        int, bool], BestConvAlgoByProfile] = {
                                        }  # for backward weight

        self._nvrtc_caches: Dict[Tuple[str, Tuple[int, int]], NVRTCParams] = {}

    @staticmethod
    def get_static_key(d: ConvAlgoDesp) -> _CONV_STATIC_KEY:
        return (d.layout_i.value, d.layout_w.value, d.layout_o.value,
                d.interleave_i, d.interleave_w, d.interleave_o, d.dtype_input,
                d.dtype_weight, d.dtype_output, d.op_type.value)

    def device_synchronize(self):
        return GemmMainUnitTest.device_synchronize()

    def get_all_available(self,
                          inp: tv.Tensor,
                          weight: tv.Tensor,
                          out: tv.Tensor,
                          layout_i: ConvLayout,
                          layout_w: ConvLayout,
                          layout_o: ConvLayout,
                          arch: Tuple[int, int],
                          op_type: ConvOpType,
                          mask_width: int,
                          fp32_accum: Optional[bool] = None,
                          use_tf32: bool = True,
                          bias: tv.Tensor = tv.Tensor(),
                          scale: tv.Tensor = tv.Tensor()):
        avail_algos = get_available_algo_str_from_arch(arch)
        finally_algos: List[ConvAlgoDesp] = []
        is_fp16 = inp.dtype == tv.float16 and weight.dtype == tv.float16 # and out.dtype == tv.float16
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
        # use_f32_as_accum = False
        static_key = (layout_i.layout_type.value,
                        layout_w.layout_type.value,
                        layout_o.layout_type.value, layout_i.interleave,
                        layout_w.interleave, layout_o.interleave, inp.dtype,
                        weight.dtype, out.dtype, op_type.value)
        desps = self.static_key_to_desps.get(static_key, None)
        # for d in self.all_desps:
        #     print(d)
        # print(len(desps))
        # breakpoint()
        if desps is None or len(desps) == 0:
            return finally_algos
        for desp in desps:
            if arch < desp.min_arch:
                continue
            # skip volta tensor op since it is very slow in architectures except volta.
            if arch >= (7, 5) and desp.algo == GemmAlgo.Volta.value:
                continue
            if not use_tf32:
                if (desp.tensorop[0] > 0 and inp.dtype == tv.float32 
                        and weight.dtype == tv.float32 and out.dtype == tv.float32):
                    continue
            if arch >= (7, 0) and is_fp16:
                if desp.algo == GemmAlgo.Simt:
                    continue
                if use_f32_as_accum:
                    if desp.dacc == tv.float16:
                        continue

            ldi = inp.dim(-1)
            ldw = weight.dim(-1)
            ldo = out.dim(-1)
            mask_width_valid = True
            if desp.op_type.value == ConvOpType.kBackwardWeight.value:
                assert mask_width > 0
                mask_width_valid = mask_width % desp.tile_shape[2] == 0
            require_dynamic_mask = kv > 32
            if desp.supported_ldx_conv(ldi, ldw, ldo) and mask_width_valid:
                if not bias.empty() and not scale.empty():
                    # int8 inference, bias/scale dtype must equal to compute dtype in gemm
                    assert bias.dtype == scale.dtype
                    if desp.dcomp != bias.dtype:
                        continue
                    if not desp.is_int8_inference:
                        continue 
                else:
                    if desp.is_int8_inference:
                        continue
                if desp.is_nvrtc:
                    if not CompileInfo.algo_can_be_nvrtc_compiled(desp.min_arch):
                        continue
                if not CompileInfo.arch_is_compiled_gemm(arch):
                    # use PTX of possible
                    if not CompileInfo.gemm_algo_can_use_ptx(desp.min_arch, arch):
                        if CompileInfo.algo_can_be_nvrtc_compiled(desp.min_arch):
                            # compiled kernel can't use PTX, for example, desp need at least sm_80 and only sm_75+PTX is compiled
                            # all sm_80 code of this desp is invalid, we must use nvrtc.
                            # only desp <= sm_75 can use virtual PTX code to generate sm_80 code.
                            desp = desp.copy()
                            desp.is_nvrtc = True
                        else:
                            continue
                if SPCONV_DEBUG_NVRTC_KERNELS:
                    desp.is_nvrtc = True
                if require_dynamic_mask:
                    if not desp.dynamic_mask:
                        continue 
                else:
                    if desp.dynamic_mask:
                        continue 
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
                       mask_width: int = -1,
                       need_dynamic_mask: bool = False):
        if not op_type == ConvOpType.kBackwardWeight:
            # fwd and dgrad don't need
            mask_width = -1
        key = (i_dtype, w_dtype, o_dtype, k, c, arch[0], arch[1], mask_width, need_dynamic_mask)
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

    @staticmethod
    def _compile_nvrtc_module(desp: ConvAlgoDesp):
        params = algocore.get_conv_param_from_desp(desp)
        kernel = gen_conv_kernels(params, SPCONV_NVRTC_MODE)
        kernel.namespace = "spconv"
        custom_names = []
        if SPCONV_NVRTC_MODE == NVRTCMode.ConstantMemory:
            custom_names = [
                f"&{kernel.namespace}::{NVRTCConstants.CONSTANT_PARAM_KEY}"
            ]
        cudadevrt = ""
        if SPCONV_NVRTC_MODE == NVRTCMode.DynamicParallism:
            cudadevrt_p = get_cudadevrt_path()
            assert cudadevrt_p is not None, "DynamicParallism must have cudadevrt"
            cudadevrt = str(cudadevrt_p)
        mod = CummNVRTCModule([kernel],
                              cudadevrt_path=cudadevrt,
                              verbose=False,
                              custom_names=custom_names)
        mod.load()
        return mod, kernel

    def _cached_get_nvrtc_params(self, desp: ConvAlgoDesp, arch: Tuple[int,
                                                                       int]):
        key = (str(desp), arch)
        if key in self._nvrtc_caches:
            return self._nvrtc_caches[key]
        print(f"Can't find algo {desp} in prebuilt. compile with nvrtc...")
        mod, ker = self._compile_nvrtc_module(desp)
        nvrtc_params = _get_nvrtc_params(mod, ker, "conv_kernel")
        self._nvrtc_caches[key] = nvrtc_params
        return nvrtc_params

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
                       fp32_accum: Optional[bool] = None,
                       use_tf32: bool = True,
                       bias: tv.Tensor = tv.Tensor(),
                       scale: tv.Tensor = tv.Tensor()):
        avail = self.get_all_available(inp, weight, output, layout_i, layout_w,
                                       layout_o, arch, op_type, mask_width,
                                       fp32_accum, use_tf32, bias, scale)
        inp = inp.clone()
        weight = weight.clone()
        output = output.clone()
        # print(len(avail), inp.dtype, weight.dtype, output.dtype, bias.dtype, scale.dtype, bias.empty(), scale.empty())
        channel_k = output.dim(1)
        channel_c = inp.dim(1)
        weight = weight.view([channel_k, -1, channel_c])
        need_dynamic_mask = weight.dim(1) > 32
        times: List[float] = []
        all_profile_res: List[BestConvAlgoByProfile] = []
        group_by_algo = {}
        for desp in avail:
            # for sparse conv, ndim isn't used, so we just provide a constant value.
            params = ConvParams(NDIM_DONT_CARE, ConvOpTypeCpp(op_type.value))
            if desp.is_nvrtc or str(desp) not in self.prebuilt_desp_names:
                params.nvrtc_params = self._cached_get_nvrtc_params(desp, arch)

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
            if desp.is_int8_inference:
                params.bias = bias 
                params.scale = scale
            # if op_type == ConvOpType.kBackwardWeight:
            #     assert not mask_output.empty()
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
                for j in range(4):
                    params.split_k_slices = spk
                    with tv.measure_duration(stream=stream) as measure:
                        if desp.is_nvrtc and str(
                                desp) not in self.prebuilt_desp_names:
                            tv.gemm.run_nvrtc_conv_kernel(params)
                        else:
                            ConvMainUnitTest.implicit_gemm2(params)
                    this_times.append(measure.duration)
                times.append(np.mean(this_times[1:]))
                spk_speeds.append(times[-1])
                if desp.algo not in group_by_algo:
                    group_by_algo[desp.algo] = 10000.0
                group_by_algo[desp.algo] = min(times[-1], group_by_algo[desp.algo])
                all_profile_res.append(
                    BestConvAlgoByProfile(desp, arch, splitk=spk))
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
               arch[0], arch[1], mask_width, need_dynamic_mask)
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
                              timer: CUDAKernelTimer = CUDAKernelTimer(False),
                              force_nvrtc: bool = False,
                              bias: Optional[tv.Tensor] = None,
                              act_alpha: float = 0.0,
                              act_beta: float = 0.0,
                              act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                              scale: Optional[tv.Tensor] = None,
                              output_add: Optional[tv.Tensor] = None):
        
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
        params = ConvParams(NDIM_DONT_CARE, ConvOpTypeCpp(op_type_value))
        is_not_static = str(
                algo_desp) not in self.prebuilt_desp_names
        if force_nvrtc or algo_desp.is_nvrtc or is_not_static:
            params.nvrtc_params = self._cached_get_nvrtc_params(
                algo_desp, profile_res.arch)
        params.conv_algo_desp = profile_res.algo_desp
        params.input = inp
        params.verbose = verbose
        params.weight = weight.view([channel_k, -1, channel_c])
        params.output = output

        params.split_k_slices = split_k_slices
        params.alpha = alpha
        params.beta = beta
        params.act_alpha = act_alpha
        params.act_beta = act_beta
        params.act_type = act_type
        params.stream = stream
        params.mask_argsort = mask_argsort
        params.indices = indices
        params.mask = mask

        params.mask_filter = mask_filter
        params.mask_width = mask_width
        params.mask_filter = mask_filter
        params.mask_output = mask_output
        params.reverse_mask = reverse_mask
        if bias is not None:
            params.bias = bias
        if output_add is not None and algo_desp.is_int8_inference:
            params.output_add = output_add
        if scale is not None and algo_desp.is_int8_inference:
            params.scale = scale
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

GEMM_CPP = GemmTunerSimple([
            algocore.get_gemm_algo_desp_from_param(p)
            for p in ALL_NATIVE_PARAMS])
CONV_CPP = ConvTunerSimple([
            algocore.get_conv_algo_desp_from_param(p)
            for p in ALL_IMPGEMM_PARAMS])

if __name__ == "__main__":
    for desp in ALL_CONV_ALGO_DESPS:
        print(desp, desp.min_arch)
