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
from spconv.cppconstants import COMPILED_CUDA_ARCHS
import sys 
from spconv.core_cc.csrc.sparse.alloc import ExternalAllocator

_TORCH_DTYPE_TO_TV = {
    torch.float32: tv.float32,
    torch.float64: tv.float64,
    torch.float16: tv.float16,
    torch.int32: tv.int32,
    torch.int64: tv.int64,
    torch.int8: tv.int8,
    torch.int16: tv.int16,
    torch.uint8: tv.uint8,
}
_TV_DTYPE_TO_TORCH = {v: k for k, v in _TORCH_DTYPE_TO_TV.items()}

_TORCH_UINT_WORKAROUNDS = {tv.uint32: tv.int32, tv.uint16: tv.int16, tv.uint64: tv.int64}
_ALL_INTS = {tv.int32, tv.int16, tv.int8, tv.int64, tv.uint64, tv.uint8, tv.uint32, tv.uint16}

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
    if arch not in COMPILED_CUDA_ARCHS:
        print(f"[WARNING]your gpu arch {arch} isn't compiled in prebuilt, "
                f"may cause invalid device function. "
                f"available: {COMPILED_CUDA_ARCHS}", file=sys.stderr)
    return arch

class TorchAllocator(ExternalAllocator):
    def __init__(self, gpudevice: torch.device) -> None:
        super().__init__()
        self.gpudevice = gpudevice
        self.cpudevice = torch.device("cpu:0")
        self.allocated: Dict[Union[str, int], torch.Tensor] = {}

    def zeros(self, name: str, shape: List[int], dtype: int, device: int) -> tv.Tensor:
        # provide a name if you want to access it after c++ function exit.
        torch_uint_workaround = dtype in _TORCH_UINT_WORKAROUNDS
        dtype_bkp = dtype
        if dtype in _TORCH_UINT_WORKAROUNDS:
            assert name == "", "must be temp memory for uint dtypes"
            dtype = _TORCH_UINT_WORKAROUNDS[dtype]        
        th_dtype = _TV_DTYPE_TO_TORCH[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        ten = torch.zeros(shape, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten)
        self.allocated[ten.data_ptr()] = ten
        if name:
            self.allocated[name] = ten
        if torch_uint_workaround:
            return ten_tv.type_view(dtype_bkp)
        return ten_tv

    def empty(self, name: str, shape: List[int], dtype: int, device: int) -> tv.Tensor:
        torch_uint_workaround = dtype in _TORCH_UINT_WORKAROUNDS
        dtype_bkp = dtype
        if dtype in _TORCH_UINT_WORKAROUNDS:
            assert name == "", "must be temp memory for uint dtypes"
            dtype = _TORCH_UINT_WORKAROUNDS[dtype]        
        th_dtype = _TV_DTYPE_TO_TORCH[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        ten = torch.empty(shape, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten)
        self.allocated[ten.data_ptr()] = ten
        if name:
            self.allocated[name] = ten
        if torch_uint_workaround:
            return ten_tv.type_view(dtype_bkp)
        return ten_tv

    def full_int(self, name: str, shape: List[int], value: int, dtype: int, device: int) -> tv.Tensor:
        if dtype in _TORCH_UINT_WORKAROUNDS and value < 0:
            raise NotImplementedError("you can't use full for unsigned dtypes")
        torch_uint_workaround = dtype in _TORCH_UINT_WORKAROUNDS
        dtype_bkp = dtype
        if dtype in _TORCH_UINT_WORKAROUNDS:
            assert name == "", "must be temp memory for uint dtypes"
            dtype = _TORCH_UINT_WORKAROUNDS[dtype]        

        th_dtype = _TV_DTYPE_TO_TORCH[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        ten = torch.full(shape, value, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten)
        self.allocated[ten.data_ptr()] = ten
        if name:
            self.allocated[name] = ten
        if name:
            self.allocated[name] = ten
        if torch_uint_workaround:
            return ten_tv.type_view(dtype_bkp)
        return ten_tv

    def full_float(self, name: str, shape: List[int], value: float, dtype: int, device: int) -> tv.Tensor:
        if dtype in _TORCH_UINT_WORKAROUNDS and value < 0:
            raise NotImplementedError("you can't use full for unsigned dtypes")
        torch_uint_workaround = dtype in _TORCH_UINT_WORKAROUNDS
        dtype_bkp = dtype
        if dtype in _TORCH_UINT_WORKAROUNDS:
            assert name == "", "must be temp memory for uint dtypes"
            dtype = _TORCH_UINT_WORKAROUNDS[dtype]        
        th_dtype = _TV_DTYPE_TO_TORCH[dtype]
        if device == -1:
            dev = self.cpudevice
        else:
            dev = self.gpudevice
        ten = torch.full(shape, value, dtype=th_dtype, device=dev)
        ten_tv = torch_tensor_to_tv(ten)
        self.allocated[ten.data_ptr()] = ten
        if name:
            self.allocated[name] = ten
        if torch_uint_workaround:
            return ten_tv.type_view(dtype_bkp)
        return ten_tv

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


if __name__ == "__main__":
    a = torch.rand(2, 2)
    atv = torch_tensor_to_tv(a)
    print(atv.numpy_view())
