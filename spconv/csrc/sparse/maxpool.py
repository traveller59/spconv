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
from cumm.conv.bases import ConvEnum
from cumm.gemm.core.metaarray import MetaArray, seq
from cumm import dtypes
import pccm 
from cumm.gemm.layout import TensorGeneric, to_stride
from cumm.common import TensorView, TensorViewHashKernel, TensorViewKernel, ThrustLib, GemmBasic
from cumm.gemm import codeops
from typing import List 
from cumm.conv.params import ConvProblem
from cumm.gemm.mask_iters import MaskTileIterator, MaskTileIteratorParams
import numpy as np 
from cumm.gemm import (thread_map)

class IndiceMaxPool(pccm.Class):
    # TODO optimize this function
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewKernel, TensorView, GemmBasic)
    
    @pccm.cuda.cuda_global_function
    def forward_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")

        code.arg("out_features", f"T*") 
        code.arg("in_features", f"const T*")
        code.arg("out_indices", "const int*")
        code.arg("in_indices", "const int*")
        code.arg("size", "int")
        code.arg("num_features", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopY<int>(size)) {{
            int in_idx = in_indices[i];
            int out_idx = out_indices[i];
            auto in_ptr = in_features + in_idx * num_features;
            auto out_ptr = out_features + out_idx * num_features;
            for (int j : tv::KernelLoopX<int>(num_features)) {{
                auto in = in_ptr[j];
                auto out = out_ptr[j];
                if (in > out){{
                    out_ptr[j] = in;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def backward_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.arg("out_features", f"const T*") 
        code.arg("in_features", f"const T*")
        code.arg("dout_features", f"const T*") 
        code.arg("din_features", f"T*")
        code.arg("out_indices", "const int*")
        code.arg("in_indices", "const int*")
        code.arg("size", "int")
        code.arg("num_features", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopY<int>(size)) {{
            int in_idx_offset = in_indices[i] * num_features;
            int out_idx_offset = out_indices[i] * num_features;
            auto in_ptr = in_features + in_idx_offset;
            auto out_ptr = out_features + out_idx_offset;
            auto din_ptr = din_features + in_idx_offset;
            auto dout_ptr = dout_features + out_idx_offset;
            for (int j : tv::KernelLoopX<int>(num_features)) {{
                auto in = in_ptr[j];
                auto out = out_ptr[j];
                if (in == out){{
                    din_ptr[j] = din_ptr[j] + dout_ptr[j];
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.static_function
    def forward(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        auto nhot = out_inds.dim(0);
        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            constexpr int MaxThreads = 512;
            tv::cuda::Launch launcher(1);
            bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(out.dim(1), [](int my, int expect){{return my >= expect;}}, [&](auto V){{
                // if out.dim(1) > value in list above, run this function.
                // if a value is found, other value won't be executed.
                constexpr int NumFeatures = TV_DECLTYPE(V)::value;
                constexpr int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                constexpr int NumFeatures = 16;
                constexpr int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }}
            launcher(forward_kernel<T>, out.data_ptr<T>(), in.data_ptr<const T>(),
                out_inds.data_ptr<const int>(), in_inds.data_ptr<const int>(), nhot, out.dim(1));

        }});
        """)
        return code

    @pccm.cuda.static_function
    def backward(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("dout", "tv::Tensor")
        code.arg("din", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        auto nhot = out_inds.dim(0);

        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            constexpr int MaxThreads = 512;
            tv::cuda::Launch launcher(1);
            bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(out.dim(1), [](int my, int expect){{return my >= expect;}}, [&](auto V){{
                // if out.dim(1) > value in list above, run this function.
                // if a value is found, other value won't be executed.
                constexpr int NumFeatures = TV_DECLTYPE(V)::value;
                constexpr int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                constexpr int NumFeatures = 16;
                constexpr int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }}
            launcher(backward_kernel<T>, out.data_ptr<const T>(), in.data_ptr<const T>(),
                dout.data_ptr<const T>(), din.data_ptr<T>(),
                out_inds.data_ptr<const int>(), in_inds.data_ptr<const int>(), nhot, out.dim(1));
        }});
        """)
        return code
