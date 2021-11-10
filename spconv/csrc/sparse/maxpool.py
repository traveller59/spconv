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
from spconv.csrc.sparse.cpu_core import OMPLib
from cumm.constants import CUMM_CPU_ONLY_BUILD


class IndiceMaxPool(pccm.Class):
    # TODO optimize this function
    def __init__(self):
        super().__init__()
        self.add_include("limits")
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
    def forward_implicit_gemm_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")

        code.arg("out_features", f"T*")
        code.arg("in_features", f"const T*")
        code.arg("indices", "const int*")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_indices", "int")
        code.arg("lowest", "T")

        code.raw(f"""
        for (int i : tv::KernelLoopY<int>(num_indices)) {{
            auto out_ptr = out_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features)) {{
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

    @pccm.cuda.cuda_global_function
    def backward_implicit_gemm_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")

        code.arg("out_features", f"const T*")
        code.arg("in_features", f"const T*")
        code.arg("dout_features", f"const T*")
        code.arg("din_features", f"T*")
        code.arg("indices_bwd", "const int*")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_indices", "int")

        code.raw(f"""

        for (int i : tv::KernelLoopY<int>(num_indices)) {{
            auto in_ptr = in_features + i * num_features;
            auto din_ptr = din_features + i * num_features;
            
            for (int j : tv::KernelLoopX<int>(num_features)) {{
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
                int NumFeatures = TV_DECLTYPE(V)::value;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                int NumFeatures = 16;
                int Num0 = MaxThreads / NumFeatures;
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
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            constexpr int MaxThreads = 512;
            tv::cuda::Launch launcher(1);
            bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(out.dim(1), [](int my, int expect){{return my >= expect;}}, [&](auto V){{
                // if out.dim(1) > value in list above, run this function.
                // if a value is found, other value won't be executed.
                int NumFeatures = TV_DECLTYPE(V)::value;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                int NumFeatures = 16;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }}
            T lowest = std::numeric_limits<T>::lowest();
            lowest = T(0);
            launcher(forward_implicit_gemm_kernel<T>, out.data_ptr<T>(), in.data_ptr<const T>(),
                inds.data_ptr<const int>(), out.dim(1), inds.dim(0), inds.dim(1), lowest);
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
                int NumFeatures = TV_DECLTYPE(V)::value;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                int NumFeatures = 16;
                int Num0 = MaxThreads / NumFeatures;
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
            constexpr int MaxThreads = 512;
            tv::cuda::Launch launcher(1);
            bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(out.dim(1), [](int my, int expect){{return my >= expect;}}, [&](auto V){{
                // if out.dim(1) > value in list above, run this function.
                // if a value is found, other value won't be executed.
                int NumFeatures = TV_DECLTYPE(V)::value;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                int NumFeatures = 16;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), NumFeatures), tv::div_up(nhot, Num0));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }}
            launcher(backward_implicit_gemm_kernel<T>, out.data_ptr<const T>(), in.data_ptr<const T>(),
                dout.data_ptr<const T>(), din.data_ptr<T>(),
                inds.data_ptr<const int>(), out.dim(1), inds.dim(0), inds.dim(1));
        }});
        """)
        return code


class IndiceMaxPoolCPU(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
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
        tv::dispatch<float, double>(out.dtype(), [&](auto I){{
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
        tv::dispatch<float, double>(out.dtype(), [&](auto I){{
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
