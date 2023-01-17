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

from cumm import tensorview as tv
import torch
from typing import Dict, Optional, List, Union
from spconv.constants import AllocKeys
from spconv.cppconstants import COMPILED_CUDA_ARCHS
import sys
from spconv.core_cc.csrc.sparse.alloc import ExternalAllocator
from spconv.core_cc.csrc.sparse.convops import ExternalSpconvMatmul
from spconv.core_cc.cumm.common import CompileInfo
import warnings

import numpy as np

_TORCH_DTYPE_TO_TV = {
    torch.float32: tv.float32,
    torch.float64: tv.float64,
    torch.float16: tv.float16,
    torch.int32: tv.int32,
    torch.int64: tv.int64,
    torch.int8: tv.int8,
    torch.int16: tv.int16,
    torch.uint8: tv.uint8,
    torch.qint8: tv.int8,
}

_TORCH_UINT_WORKAROUNDS = {
    tv.uint32: tv.int32,
    tv.uint16: tv.int16,
    tv.uint64: tv.int64
}

_TH_QTYPES = {torch.qint8}

_TV_DTYPE_TO_TORCH = {v: k for k, v in _TORCH_DTYPE_TO_TV.items()}
_TV_DTYPE_TO_TORCH.update({
    tv.uint32: torch.int32,
    tv.uint16: torch.int16,
    tv.uint64: torch.int64

})

_TV_DTYPE_TO_TORCHQ = _TV_DTYPE_TO_TORCH.copy()
_TV_DTYPE_TO_TORCHQ[tv.int8] = torch.qint8

_ALL_INTS = {
    tv.int32, tv.int16, tv.int8, tv.int64, tv.uint64, tv.uint8, tv.uint32,
    tv.uint16
}


def torch_tensor_to_tv(ten: torch.Tensor,
                       dtype: Optional[int] = None,
                       shape: Optional[List[int]] = None,
                       stride: Optional[List[int]] = None):
    # assert ten.is_contiguous(), "must be contiguous tensor"
    ptr = ten.data_ptr()
    device = ten.device
    if device.type == "cpu":
        tv_device = -1
    elif device.type == "cuda":
        tv_device = 0
    else:
        raise NotImplementedError
    if dtype is None:
        dtype = _TORCH_DTYPE_TO_TV[ten.dtype]
    if stride is None:
        stride = list(ten.stride())
    if shape is None:
        shape = list(ten.shape)
    else:
        if not ten.is_contiguous():
            msg = "if you provide custom shape for non-contig tensor, stride must not None"
            assert stride is not None, msg
        else:
            # custom shape, if tensor is contiguous, we use from_blob and calc strides
            return tv.from_blob(ptr, shape, dtype, tv_device)
    return tv.from_blob_strided(ptr, shape, stride, dtype, tv_device)


def torch_tensors_to_tv(*tens: torch.Tensor):
    return (torch_tensor_to_tv(t) for t in tens)


def get_current_stream():
    return torch.cuda.current_stream().cuda_stream


def get_arch():
    arch = torch.cuda.get_device_capability()
    if not CompileInfo.arch_is_compatible(arch) and not CompileInfo.algo_can_use_ptx((0, 0), arch):
        warnings.warn(
            f"[WARNING]your gpu arch {arch} isn't compiled in prebuilt, "
            f"may cause invalid device function error. "
            f"available: {COMPILED_CUDA_ARCHS}")
    return arch


class TorchAllocator(ExternalAllocator):

    def __init__(self, gpudevice: torch.device, is_quantized: bool = False) -> None:
        super().__init__()
        self.gpudevice = gpudevice
        self.cpudevice = torch.device("cpu")
        self.allocated: Dict[Union[str, int], torch.Tensor] = {}
        self.is_quantized = is_quantized
        self._tv_dtype_to_torch = _TV_DTYPE_TO_TORCH
        if is_quantized:
            self._tv_dtype_to_torch = _TV_DTYPE_TO_TORCHQ


    def zeros(self, name: str, shape: List[int], dtype: int,
              device: int, stream: int = 0, is_temp_memory: bool = False, scale: float = 1.0) -> tv.Tensor:
        # TODO free memory by name if its already free by pointer.
        # provide a name if you want to access it after c++ function exit.
        dtype_bkp = dtype
        th_dtype = self._tv_dtype_to_torch[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        if self.is_quantized:
            ten = torch._empty_affine_quantized(shape, scale=scale, zero_point=0, dtype=th_dtype, device=dev)
        else:
            ten = torch.empty(shape, dtype=th_dtype, device=dev).zero_()
        ten_tv = torch_tensor_to_tv(ten, dtype_bkp)
        if self.is_quantized:
            # no _zeros_affine_quantized available, so we need to zero_ here.
            ctx = tv.Context()
            ctx.set_cuda_stream(stream)
            ten_tv.zero_(ctx)
        self.allocated[ten_tv.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten_tv

    def empty(self, name: str, shape: List[int], dtype: int,
              device: int, stream: int = 0, is_temp_memory: bool = False, scale: float = 1.0) -> tv.Tensor:
        dtype_bkp = dtype
        th_dtype = self._tv_dtype_to_torch[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        if self.is_quantized:
            ten = torch._empty_affine_quantized(shape, scale=scale, zero_point=0, dtype=th_dtype, device=dev)
        else:
            ten = torch.empty(shape, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten, dtype_bkp)
        self.allocated[ten_tv.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten_tv

    def full_int(self, name: str, shape: List[int], value: int, dtype: int,
                 device: int, stream: int = 0, is_temp_memory: bool = False) -> tv.Tensor:
        if dtype in _TORCH_UINT_WORKAROUNDS and value < 0:
            raise NotImplementedError("you can't use full for unsigned dtypes")
        dtype_bkp = dtype
        th_dtype = self._tv_dtype_to_torch[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        if self.is_quantized:
            assert th_dtype not in _TH_QTYPES
        ten = torch.full(shape, value, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten, dtype_bkp)
        self.allocated[ten_tv.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten_tv

    def full_float(self, name: str, shape: List[int], value: float, dtype: int,
                   device: int, stream: int = 0, is_temp_memory: bool = False) -> tv.Tensor:
        if dtype in _TORCH_UINT_WORKAROUNDS and value < 0:
            raise NotImplementedError("you can't use full for unsigned dtypes")
        dtype_bkp = dtype
        th_dtype = self._tv_dtype_to_torch[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        if self.is_quantized:
            assert th_dtype not in _TH_QTYPES
        ten = torch.full(shape, value, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten, dtype_bkp)
        self.allocated[ten_tv.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten_tv

    def get_tensor_by_name(self, name: str):
        return torch_tensor_to_tv(self.allocated[name])

    def free(self, ten: tv.Tensor):
        if ten.storage_bytesize() != ten.bytesize():
            raise ValueError("you can't free a sliced tensor.")
        if ten.byte_pointer() in self.allocated:
            self.allocated.pop(ten.byte_pointer())
            return
        raise ValueError("can't find your tensor in cache.")

    def free_noexcept(self, ten: tv.Tensor):
        # for c++ scope guard, free will be called in c++ destructor
        if ten.storage_bytesize() != ten.bytesize():
            return
        if ten.byte_pointer() in self.allocated:
            self.allocated.pop(ten.byte_pointer())
            return


class TorchSpconvMatmul(ExternalSpconvMatmul):

    def __init__(self, alloc: TorchAllocator) -> None:
        super().__init__()
        self.alloc = alloc

    def indice_conv_init_gemm(self, features_n: str, filters_n: str,
                              all_weight_is_krsc: bool, is_kc_not_ck: bool,
                              kv_center: int, out_channel: int, stream_int: int = 0):
        features = self.alloc.allocated[features_n]
        filters = self.alloc.allocated[filters_n]
        if not all_weight_is_krsc:
            filters = filters.reshape(-1, *filters.shape[-2:])
            if not is_kc_not_ck:
                out_features = torch.mm(features, filters[kv_center])
            else:
                out_features = torch.mm(features, filters[kv_center].T)
        else:
            filters = filters.reshape(out_channel, -1, filters.shape[-1])
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
        self.alloc.allocated[AllocKeys.OutFeatures] = out_features
        # print(filters.shape, features.shape, all_weight_is_krsc, out_features.shape, out_features.is_contiguous())

        return torch_tensor_to_tv(out_features)

    def indice_conv_cpu_gemm(self, inp_buffer_n: str, out_buffer_n: str, filters_n: str,
                             all_weight_is_krsc: bool,
                             is_kc_not_ck: bool, nhot: int, index: int):
        kv_dim = 1 if all_weight_is_krsc else 0
        inp_buffer = self.alloc.allocated[inp_buffer_n]
        filters = self.alloc.allocated[filters_n]
        if not all_weight_is_krsc:
            filters = filters.reshape(-1, *filters.shape[-2:])
        else:
            filters = filters.reshape(filters.shape[0], -1, filters.shape[-1])
        out_buffer = self.alloc.allocated[out_buffer_n]
        filters_i = filters.select(kv_dim, index)
        filters_cur = filters_i if not is_kc_not_ck else filters_i.T
        if inp_buffer.dtype == torch.float16:
            inp_buffer_np = torch_tensor_to_tv(inp_buffer).numpy_view()
            filters_np = torch_tensor_to_tv(filters).numpy_view()
            filters_i_np = filters_np[
                index] if not all_weight_is_krsc else filters_np[:, index]
            filters_cur_np = filters_i_np if not is_kc_not_ck else filters_i_np.T
            out_buffer_np = torch_tensor_to_tv(out_buffer).numpy_view()
            np.matmul(inp_buffer_np[:nhot],
                      filters_cur_np,
                      out=out_buffer_np[:nhot])
        else:
            torch.mm(inp_buffer[:nhot], filters_cur, out=out_buffer[:nhot])

    def indice_conv_bwd_init_gemm(self, features_n: str, filters_n: str,
                                  out_bp_n: str, dfilters_n: str,
                                  all_weight_is_krsc: bool, is_kc_not_ck: bool,
                                  kv_center: int, stream_int: int = 0):
        features = self.alloc.allocated[features_n]
        filters = self.alloc.allocated[filters_n]
        out_bp = self.alloc.allocated[out_bp_n]
        dfilters = self.alloc.allocated[dfilters_n]
        if not all_weight_is_krsc:
            filters = filters.reshape(-1, *filters.shape[-2:])
            dfilters = dfilters.reshape(-1, *filters.shape[-2:])

        else:
            filters = filters.reshape(filters.shape[0], -1, filters.shape[-1])
            dfilters = dfilters.reshape(filters.shape[0], -1, filters.shape[-1])

        if not all_weight_is_krsc:
            if not is_kc_not_ck:
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
        self.alloc.allocated[AllocKeys.DIn] = din
        return torch_tensor_to_tv(din)

    def indice_conv_bwd_cpu_gemm(self, inp_buffer_n: str, 
                             out_buffer_n: str, filters_n: str, dfilters_n: str,all_weight_is_krsc: bool,
                             is_kc_not_ck: bool, nhot: int, index: int):
        kv_dim = 1 if all_weight_is_krsc else 0
        inp_buffer = self.alloc.allocated[inp_buffer_n]
        out_buffer = self.alloc.allocated[out_buffer_n]
        filters = self.alloc.allocated[filters_n]
        dfilters = self.alloc.allocated[dfilters_n]
        if not all_weight_is_krsc:
            filters = filters.reshape(-1, *filters.shape[-2:])
            dfilters = dfilters.reshape(-1, *filters.shape[-2:])

        else:
            filters = filters.reshape(filters.shape[0], -1, filters.shape[-1])
            dfilters = dfilters.reshape(filters.shape[0], -1, filters.shape[-1])

        filters_i = filters.select(kv_dim, index)
        dfilters_i = dfilters.select(kv_dim, index)

        filters_KC = filters_i if is_kc_not_ck else filters_i.T
        if is_kc_not_ck:
            # KN @ NC
            torch.mm(out_buffer[:nhot].T, inp_buffer[:nhot], out=dfilters_i)
        else:
            # CN @ NK
            torch.mm(inp_buffer[:nhot].T, out_buffer[:nhot], out=dfilters_i)
        # NK @ KC
        torch.mm(out_buffer[:nhot], filters_KC, out=inp_buffer[:nhot])

if __name__ == "__main__":
    a = torch.rand(2, 2)
    atv = torch_tensor_to_tv(a)
    print(atv.numpy_view())
