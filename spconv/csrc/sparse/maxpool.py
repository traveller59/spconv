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
from cumm.gemm.core.metaarray import MetaArray, seq
from cumm import dtypes
import pccm
from cumm.gemm.layout import TensorGeneric, to_stride
from cumm.common import TensorView, GemmDTypes, TensorViewKernel, ThrustLib, GemmBasic
from cumm.gemm import codeops
from typing import List
from cumm.conv.params import ConvProblem
from cumm.gemm.mask_iters import MaskTileIterator, MaskTileIteratorParams
import numpy as np
from cumm.gemm import (thread_map)
from spconv.csrc.sparse.cpu_core import OMPLib
from cumm.constants import CUMM_CPU_ONLY_BUILD
from ..utils.launch import LaunchUtils

class IndiceMaxPool(pccm.Class):
    # TODO optimize this function
    def __init__(self):
        super().__init__()
        self.add_include("limits")
        self.add_dependency(TensorViewKernel, TensorView, GemmBasic, LaunchUtils)
        self.add_static_const("kMaxGridYZDim", "int", "65535")


    @pccm.cuda.cuda_global_function
    def forward_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"T*")
        code.arg("in_features", f"const T*")
        code.arg("out_indices", "const int*")
        code.arg("in_indices", "const int*")
        code.arg("size", "int")
        code.arg("num_features", "int")
        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

        for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            int in_idx = in_indices[i];
            int out_idx = out_indices[i];
            auto in_ptr = in_features + in_idx * num_features;
            auto out_ptr = out_features + out_idx * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
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
    def forward_implicit_gemm_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"T*")
        code.arg("in_features", f"const T*")
        code.arg("indices", "const int*")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_indices", "int")
        code.arg("lowest", "T")
        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

        for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            auto out_ptr = out_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                auto indices_ptr = indices + i;
                int in_idx = indices_ptr[0];
                T in, in_temp;
                in = lowest;
                bool valid = in_idx != -1;
                in_temp = valid ? in_features[in_idx * num_features + j] : lowest;
                in = (in < in_temp && valid) ? in_temp: in;
                indices_ptr += num_indices;
                for (int k = 1; k < RS; ++k){{
                    in_idx = indices_ptr[0];
                    valid = in_idx != -1;
                    in_temp = valid ? in_features[in_idx * num_features + j] : lowest;
                    in = (in < in_temp && valid) ? in_temp: in;
                    indices_ptr += num_indices;
                }}
                out_ptr[j] = in;
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def backward_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"const T*")
        code.arg("in_features", f"const T*")
        code.arg("dout_features", f"const T*")
        code.arg("din_features", f"T*")
        code.arg("out_indices", "const int*")
        code.arg("in_indices", "const int*")
        code.arg("size", "int")
        code.arg("num_features", "int")

        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
        for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            int in_idx_offset = in_indices[i] * num_features;
            int out_idx_offset = out_indices[i] * num_features;
            auto in_ptr = in_features + in_idx_offset;
            auto out_ptr = out_features + out_idx_offset;
            auto din_ptr = din_features + in_idx_offset;
            auto dout_ptr = dout_features + out_idx_offset;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                auto in = in_ptr[j];
                auto out = out_ptr[j];
                if (in == out){{
                    din_ptr[j] = din_ptr[j] + dout_ptr[j];
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def backward_implicit_gemm_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"const T*")
        code.arg("in_features", f"const T*")
        code.arg("dout_features", f"const T*")
        code.arg("din_features", f"T*")
        code.arg("indices_bwd", "const int*")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_indices", "int")

        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

        for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            auto in_ptr = in_features + i * num_features;
            auto din_ptr = din_features + i * num_features;
            
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                auto indices_ptr = indices_bwd + i;
                int out_idx = indices_ptr[0];
                T in = in_ptr[j];
                T sum_val = T(0);
                // if idx invalid, we only need to ensure in not equal to out.
                T out = out_idx != -1 ? out_features[out_idx * num_features + j] : T(0);
                T dout = out_idx != -1 ? dout_features[out_idx * num_features + j] : T(0);
                bool valid = in == out && out_idx != -1;
                sum_val = valid ? sum_val + dout : sum_val;

                indices_ptr += num_indices;
                for (int k = 1; k < RS; ++k){{
                    out_idx = indices_ptr[0];
                    out = out_idx != -1 ? out_features[out_idx * num_features + j] : T(0);
                    dout = out_idx != -1 ? dout_features[out_idx * num_features + j] : T(0);
                    valid = in == out && out_idx != -1;
                    sum_val = valid ? sum_val + dout : sum_val;
                    indices_ptr += num_indices;
                }}
                din_ptr[j] = sum_val;
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def forward_avgpool_implicit_gemm_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"T*")
        code.arg("in_features", f"const T*")
        code.arg("indices", "const int*")
        code.arg("count_out", "int*")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_indices", "int")

        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
        for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            auto out_ptr = out_features + i * num_features;
            auto indices_ptr = indices + i;
            int in_idx = 0;
            int count = 0;
            for (int k = 0; k < RS; ++k){{
                in_idx = indices_ptr[0];
                count += int(in_idx != -1);
                indices_ptr += num_indices;
            }}
            if (count_out != nullptr){{
                count_out[i] = count;
            }}
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                indices_ptr = indices + i;
                int in_idx;
                T in, in_temp;
                in = T(0);
                for (int k = 0; k < RS; ++k){{
                    in_idx = indices_ptr[0];
                    bool valid = in_idx != -1;
                    in_temp = valid ? in_features[in_idx * num_features + j] : T(0);
                    in += in_temp;
                    indices_ptr += num_indices;
                }}
                out_ptr[j] = count > 0 ? in / T(count) : T(0);
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def backward_avgpool_implicit_gemm_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("dout_features", f"const T*")
        code.arg("din_features", f"T*")
        code.arg("indices_bwd", "const int*")
        code.arg("count_out", "const int*")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_indices", "int")

        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

        for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            auto din_ptr = din_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                auto indices_ptr = indices_bwd + i;
                int out_idx = 0;
                T sum_val = T(0);
                for (int k = 0; k < RS; ++k){{
                    out_idx = indices_ptr[0];
                    bool valid = out_idx != -1;
                    T dout = valid ? dout_features[out_idx * num_features + j] : T(0);
                    int count = valid ? count_out[out_idx] : T(0);
                    sum_val += dout * T(count);
                    indices_ptr += num_indices;
                }}
                din_ptr[j] = sum_val;
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
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t, int8_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, out.dim(1));
            int num_blocks_X = std::get<0>(launchdims);
            int num_blocks_Y = std::get<1>(launchdims);
            dim3 blocks;
            dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
            if (num_blocks_Y > kMaxGridYZDim){{
                blocks = dim3(num_blocks_X * num_blocks_Y);
            }}else{{
                blocks = dim3(num_blocks_X, num_blocks_Y);
            }}
            tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
            tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){{
                constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;
                launcher(forward_kernel<T, OneDim>, out.data_ptr<T>(), in.data_ptr<const T>(),
                    out_inds.data_ptr<const int>(), in_inds.data_ptr<const int>(), nhot, out.dim(1),
                    num_blocks_X, num_blocks_Y);
            }});
            TV_CHECK_CUDA_ERR_V2("max pool fwd failed!!!");
        }});
        """)
        return code

    @pccm.cuda.static_function
    def forward_implicit_gemm(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        auto nhot = out.dim(0);

        tv::check_shape(inds, {{-1, nhot}});
        tv::check_shape(in, {{-1, out.dim(1)}});

        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t, int8_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, out.dim(1));
            int num_blocks_X = std::get<0>(launchdims);
            int num_blocks_Y = std::get<1>(launchdims);
            dim3 blocks;
            dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
            if (num_blocks_Y > kMaxGridYZDim){{
                blocks = dim3(num_blocks_X * num_blocks_Y);
            }}else{{
                blocks = dim3(num_blocks_X, num_blocks_Y);
            }}
            tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
            T lowest = std::numeric_limits<T>::lowest();
            lowest = T(0);
            tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){{
                constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;
                launcher(forward_implicit_gemm_kernel<T, OneDim>, out.data_ptr<T>(), in.data_ptr<const T>(),
                    inds.data_ptr<const int>(), out.dim(1), inds.dim(0), inds.dim(1), lowest, 
                    num_blocks_X, num_blocks_Y);
            }});

            TV_CHECK_CUDA_ERR_V2("max pool fwd failed!!!");

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
            auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, out.dim(1));
            int num_blocks_X = std::get<0>(launchdims);
            int num_blocks_Y = std::get<1>(launchdims);
            dim3 blocks;
            dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
            if (num_blocks_Y > kMaxGridYZDim){{
                blocks = dim3(num_blocks_X * num_blocks_Y);
            }}else{{
                blocks = dim3(num_blocks_X, num_blocks_Y);
            }}
            tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
            tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){{
                constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;
                launcher(backward_kernel<T, OneDim>, out.data_ptr<const T>(), in.data_ptr<const T>(),
                    dout.data_ptr<const T>(), din.data_ptr<T>(),
                    out_inds.data_ptr<const int>(), in_inds.data_ptr<const int>(), nhot, out.dim(1),
                    num_blocks_X, num_blocks_Y);

            }});
            TV_CHECK_CUDA_ERR_V2("max pool backward failed!!!");
        }});
        """)
        return code

    @pccm.cuda.static_function
    def backward_implicit_gemm(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("dout", "tv::Tensor")
        code.arg("din", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        auto nhot = in.dim(0);

        tv::check_shape(inds, {{-1, nhot}});
        tv::check_shape(in, {{-1, out.dim(1)}});

        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, out.dim(1));
            int num_blocks_X = std::get<0>(launchdims);
            int num_blocks_Y = std::get<1>(launchdims);
            dim3 blocks;
            dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
            if (num_blocks_Y > kMaxGridYZDim){{
                blocks = dim3(num_blocks_X * num_blocks_Y);
            }}else{{
                blocks = dim3(num_blocks_X, num_blocks_Y);
            }}
            tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
            tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){{
                constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;
                launcher(backward_implicit_gemm_kernel<T, OneDim>, out.data_ptr<const T>(), in.data_ptr<const T>(),
                    dout.data_ptr<const T>(), din.data_ptr<T>(),
                    inds.data_ptr<const int>(), out.dim(1), inds.dim(0), inds.dim(1),
                    num_blocks_X, num_blocks_Y);
            }});
            TV_CHECK_CUDA_ERR_V2("max pool fwd failed!!!");
        }});
        """)
        return code

    @pccm.cuda.static_function
    def forward_avgpool_implicit_gemm(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("count_out", "tv::Tensor")

        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        auto nhot = out.dim(0);

        tv::check_shape(inds, {{-1, nhot}});
        tv::check_shape(in, {{-1, out.dim(1)}});

        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t, int8_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, out.dim(1));
            int num_blocks_X = std::get<0>(launchdims);
            int num_blocks_Y = std::get<1>(launchdims);
            dim3 blocks;
            dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
            if (num_blocks_Y > kMaxGridYZDim){{
                blocks = dim3(num_blocks_X * num_blocks_Y);
            }}else{{
                blocks = dim3(num_blocks_X, num_blocks_Y);
            }}
            tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
            tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){{
                constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;
                launcher(forward_avgpool_implicit_gemm_kernel<T, OneDim>, out.data_ptr<T>(), in.data_ptr<const T>(),
                    inds.data_ptr<const int>(), count_out.data_ptr<int>(), out.dim(1), inds.dim(0), inds.dim(1),
                    num_blocks_X, num_blocks_Y);

            }});
            TV_CHECK_CUDA_ERR_V2("avg pool fwd failed!!!");

        }});
        """)
        return code

    @pccm.cuda.static_function
    def backward_avgpool_implicit_gemm(self):
        code = pccm.FunctionCode()
        code.arg("dout", "tv::Tensor")
        code.arg("din", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("count_out", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        auto nhot = din.dim(0);
        TV_ASSERT_RT_ERR(!count_out.empty(), "count out must not empty")
        tv::check_shape(inds, {{-1, nhot}});
        tv::check_shape(din, {{-1, dout.dim(1)}});
        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(dout.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, dout.dim(1));
            int num_blocks_X = std::get<0>(launchdims);
            int num_blocks_Y = std::get<1>(launchdims);
            dim3 blocks;
            dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
            if (num_blocks_Y > kMaxGridYZDim){{
                blocks = dim3(num_blocks_X * num_blocks_Y);
            }}else{{
                blocks = dim3(num_blocks_X, num_blocks_Y);
            }}
            tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
            tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){{
                constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;

                launcher(backward_avgpool_implicit_gemm_kernel<T, OneDim>, 
                    dout.data_ptr<const T>(), din.data_ptr<T>(),
                    inds.data_ptr<const int>(), count_out.data_ptr<const int>(),
                    dout.dim(1), inds.dim(0), inds.dim(1),
                    num_blocks_X, num_blocks_Y);
            }});
            TV_CHECK_CUDA_ERR_V2("avg pool bwd failed!!!");

        }});
        """)
        return code

class IndiceMaxPoolCPU(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmDTypes)
        if CUMM_CPU_ONLY_BUILD:
            self.add_dependency(OMPLib)
        self.add_include("tensorview/parallel/all.h")

    @pccm.static_function
    def forward(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        int nhot = out_inds.dim(0);
        int num_features = in.dim(1);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto out_features = out.data_ptr<T>();
            auto in_features = in.data_ptr<const T>();

            auto in_indices = in_inds.data_ptr<const int>();
            auto out_indices = out_inds.data_ptr<const int>();
            tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){{
                for (int i = begin; i < end; i += step) {{
                    int in_idx = in_indices[i];
                    int out_idx = out_indices[i];
                    auto in_ptr = in_features + in_idx * num_features;
                    auto out_ptr = out_features + out_idx * num_features;
                    for (int j = 0; j < num_features; ++j) {{
                        auto in = in_ptr[j];
                        auto out = out_ptr[j];
                        if (in > out){{
                            out_ptr[j] = in;
                        }}
                    }}
                }}
            }});
        }});
        """)
        return code

    @pccm.static_function
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
        int nhot = out_inds.dim(0);
        int num_features = in.dim(1);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto out_features = out.data_ptr<const T>();
            auto in_features = in.data_ptr<const T>();
            auto dout_features = dout.data_ptr<const T>();
            auto din_features = din.data_ptr<T>();

            auto in_indices = in_inds.data_ptr<const int>();
            auto out_indices = out_inds.data_ptr<const int>();
            tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){{
                for (int i = begin; i < end; i += step) {{
                    int in_idx_offset = in_indices[i] * num_features;
                    int out_idx_offset = out_indices[i] * num_features;
                    auto in_ptr = in_features + in_idx_offset;
                    auto out_ptr = out_features + out_idx_offset;
                    auto din_ptr = din_features + in_idx_offset;
                    auto dout_ptr = dout_features + out_idx_offset;
                    for (int j = 0; j < num_features; ++j) {{
                        auto in = in_ptr[j];
                        auto out = out_ptr[j];
                        if (in == out){{
                            din_ptr[j] = din_ptr[j] + dout_ptr[j];
                        }}
                    }}
                }}
            }});

        }});
        """)
        return code
