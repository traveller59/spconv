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

import pccm
from cumm.common import TensorView, GemmDTypes, TensorViewKernel, ThrustLib, GemmBasic
from spconv.csrc.sparse.cpu_core import OMPLib
from cumm.constants import CUMM_CPU_ONLY_BUILD

class InferenceOpsKernel(pccm.ParameterizedClass):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewKernel, GemmBasic)

    @pccm.cuda.cuda_global_function
    def bias_add_inplace_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")

        code.arg("out_features", f"T*")
        code.arg("bias", f"const T*")
        code.arg("size", "int")
        code.arg("num_features", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopY<int>(size)) {{
            auto out_ptr = out_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features)) {{
                out_ptr[j] = bias[j] + out_ptr[j];
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def bias_add_act_inplace_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")

        code.arg("out_features", f"T*")
        code.arg("bias", f"const T*")
        code.arg("act_type", f"tv::gemm::Activation")
        code.arg("alpha", f"T")
        code.arg("beta", f"T")
        code.arg("size", "int")
        code.arg("num_features", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopY<int>(size)) {{
            auto out_ptr = out_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features)) {{
                T o = out_ptr[j] + bias[j];
                switch (act_type){{
                    case tv::gemm::Activation::kNone:
                        break;
                    case tv::gemm::Activation::kReLU:{{
                        o = o >= T(0) ? o : T(0);
                    }}
                    case tv::gemm::Activation::kLeakyReLU:{{
                        o = o >= T(0) ? o : o * alpha;
                    }}
                    default: ;
                }}
                out_ptr[j] = o;
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def activation_inplace_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")

        code.arg("out_features", f"T*")
        code.arg("act_type", f"tv::gemm::Activation")
        code.arg("alpha", f"T")
        code.arg("beta", f"T")
        code.arg("size", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(size)) {{
            T o = out_features[i];
            switch (act_type){{
                case tv::gemm::Activation::kNone:
                    break;
                case tv::gemm::Activation::kReLU:{{
                    out_features[i] = o >= T(0) ? o : T(0);
                }}
                case tv::gemm::Activation::kLeakyReLU:{{
                    out_features[i] = o >= T(0) ? o : o * alpha;
                }}
                default: ;
            }}
        }}
        """)
        return code


class InferenceOps(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.kernel = InferenceOpsKernel()
        self.add_include("tensorview/gemm/core/constants.h")

    if CUMM_CPU_ONLY_BUILD:
        _DECORATOR = pccm.static_function
    else:
        _DECORATOR = pccm.cuda.static_function

    @pccm.pybind.mark
    @_DECORATOR
    def bias_add_act_inplace(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("bias", "tv::Tensor")
        code.arg("act_type", f"tv::gemm::Activation", "tv::gemm::Activation::kNone", "cumm.tensorview.gemm.Activation = Activation.None_")
        code.arg("alpha", f"float", "0.0")
        code.arg("beta", f"float", "0.0")
        code.arg("stream", "std::uintptr_t", "0")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"""
            TV_THROW_RT_ERR("this function don't support cpu only build.")
            """)
            return code
        code.add_param_class("ker", self.kernel)
        code.raw(f"""
        auto nhot = out.dim(0);
        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        TV_ASSERT_RT_ERR(bias.dim(0) == out.dim(1), "error");
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            constexpr int MaxThreads = 512;
            tv::cuda::Launch launcher(1);
            bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(out.dim(1), [](int my, int expect){{return my >= expect;}}, [&](auto V){{
                // if out.dim(1) > value in list above, run this function.
                // if a value is found, other value won't be executed.
                int NumFeatures = TV_DECLTYPE(V)::value;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), int64_t(NumFeatures)), tv::div_up(nhot, int64_t(Num0)));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }});
            if (!found){{
                int NumFeatures = 16;
                int Num0 = MaxThreads / NumFeatures;
                dim3 blocks(tv::div_up(out.dim(1), int64_t(NumFeatures)), tv::div_up(nhot, int64_t(Num0)));
                dim3 threads(NumFeatures, Num0);
                launcher = tv::cuda::Launch(blocks, threads, cudastream);
            }}
            if (act_type == tv::gemm::Activation::kNone){{
                launcher(ker::bias_add_inplace_kernel<T>, out.data_ptr<T>(), bias.data_ptr<const T>(),
                    nhot, out.dim(1));
            }}else{{
                launcher(ker::bias_add_act_inplace_kernel<T>, out.data_ptr<T>(), bias.data_ptr<const T>(),
                    act_type, T(alpha), T(beta), nhot, out.dim(1));
            }}

        }});
        """)
        return code

    @pccm.pybind.mark
    @_DECORATOR
    def bias_add_inplace(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("bias", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        return bias_add_act_inplace(out, bias, tv::gemm::Activation::kNone, 0, 0, stream);
        """)
        return code


    @pccm.pybind.mark
    @_DECORATOR
    def activation_inplace(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("act_type", f"tv::gemm::Activation")
        code.arg("alpha", f"float")
        code.arg("beta", f"float")
        code.arg("stream", "std::uintptr_t", "0")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"""
            TV_THROW_RT_ERR("this function don't support cpu only build.")
            """)
            return code
        code.add_param_class("ker", self.kernel)

        code.raw(f"""
        auto nhot = out.size();
        auto cudastream = reinterpret_cast<cudaStream_t>(stream);
        tv::cuda::Launch launcher = tv::cuda::Launch(nhot, cudastream);
        tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            launcher(ker::activation_inplace_kernel<T>, out.data_ptr<T>(), act_type, T(alpha), T(beta),
                nhot);
        }});
        """)
        return code
