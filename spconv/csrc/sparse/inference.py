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
from cumm.common import TensorView, GemmDTypes, TensorViewKernel, TensorViewNVRTC, ThrustLib, GemmBasic
from spconv.csrc.sparse.cpu_core import OMPLib
from ..utils.launch import LaunchUtils
from cumm.constants import CUMM_CPU_ONLY_BUILD

class InferenceOpsKernel(pccm.ParameterizedClass):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewKernel, TensorViewNVRTC, GemmBasic, LaunchUtils)

    @pccm.cuda.cuda_global_function
    def bias_add_inplace_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"T*")
        code.arg("bias", f"const T*")
        code.arg("size", "int")
        code.arg("num_features", "int")
        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;


        for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            auto out_ptr = out_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                out_ptr[j] = bias[j] + out_ptr[j];
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def bias_add_act_inplace_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("OneDim", "bool", "false")

        code.arg("out_features", f"T*")
        code.arg("bias", f"const T*")
        code.arg("act_type", f"tv::gemm::Activation")
        code.arg("alpha", f"T")
        code.arg("beta", f"T")
        code.arg("size", "int")
        code.arg("num_features", "int")
        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")

        code.raw(f"""
        int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

        namespace op = tv::arrayops;
        using nv_scalar_t = tv::equivalent_data_type_t<T>;
        using MathOp = op::MathScalarOp<nv_scalar_t>;
        for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {{
            auto out_ptr = out_features + i * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {{
                T o = out_ptr[j] + bias[j];
                auto* o_nv = reinterpret_cast<nv_scalar_t*>(&o);
                switch (act_type){{
                    case tv::gemm::Activation::kNone:
                        break;
                    case tv::gemm::Activation::kReLU:{{
                        o = o >= T(0) ? o : T(0);
                        break;
                    }}
                    case tv::gemm::Activation::kLeakyReLU:{{
                        o = o >= T(0) ? o : o * alpha;
                        break;
                    }}
                    case tv::gemm::Activation::kSigmoid:{{
                        auto e = MathOp::exp(MathOp::neg(*o_nv));
                        o = T(1) / (T(1) + *reinterpret_cast<T*>( &e ));
                        break;
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
        namespace op = tv::arrayops;
        using nv_scalar_t = tv::equivalent_data_type_t<T>;
        using MathOp = op::MathScalarOp<nv_scalar_t>;

        for (int i : tv::KernelLoopX<int>(size)) {{
            T o = out_features[i];
            auto* o_nv = reinterpret_cast<nv_scalar_t*>(&o);

            switch (act_type){{
                case tv::gemm::Activation::kNone:
                    break;
                case tv::gemm::Activation::kReLU:{{
                    out_features[i] = o >= T(0) ? o : T(0);
                    break;
                }}
                case tv::gemm::Activation::kLeakyReLU:{{
                    out_features[i] = o >= T(0) ? o : o * alpha;
                    break;
                }}
                case tv::gemm::Activation::kSigmoid:{{
                    auto e = MathOp::exp(MathOp::neg(*o_nv));
                    out_features[i] = T(1) / (T(1) + *reinterpret_cast<T*>( &e ));
                    break;
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
        if not CUMM_CPU_ONLY_BUILD:
            self.add_dependency(LaunchUtils)

        self.kernel = InferenceOpsKernel()
        self.add_include("tensorview/gemm/core/constants.h")
        self.add_static_const("kMaxGridYZDim", "int", "65535")

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
        tv::dispatch<float, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
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
                if (act_type == tv::gemm::Activation::kNone){{
                    launcher(ker::bias_add_inplace_kernel<T, OneDim>, out.data_ptr<T>(), bias.data_ptr<const T>(),
                        nhot, out.dim(1), num_blocks_X, num_blocks_Y);
                }}else{{
                    launcher(ker::bias_add_act_inplace_kernel<T, OneDim>, out.data_ptr<T>(), bias.data_ptr<const T>(),
                        act_type, T(alpha), T(beta), nhot, out.dim(1), num_blocks_X, num_blocks_Y);
                }}
            }});
            TV_CHECK_CUDA_ERR_V2("bias add act failed!!!");
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
        tv::dispatch<float, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            launcher(ker::activation_inplace_kernel<T>, out.data_ptr<T>(), act_type, T(alpha), T(beta),
                nhot);
            TV_CHECK_CUDA_ERR_V2("bias add act failed!!!");
        }});
        """)
        return code
