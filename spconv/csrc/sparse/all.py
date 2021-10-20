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

from cumm.common import TensorViewKernel, ThrustLib
from cumm.conv.bases import ConvOpType, NHWC
from cumm.conv.params import ConvProblem
from cumm import dtypes
import pccm 
from ccimport import compat
from .pointops import Point2Voxel, Point2VoxelCPU
from .indices import SparseConvIndicesKernel, CudaCommonKernel
from .maxpool import IndiceMaxPool

class SpconvOps(pccm.Class):
    def __init__(self):
        super().__init__()
        self.ndims = [1, 2, 3, 4]
        for ndim in self.ndims:
            p2v = Point2Voxel(dtypes.float32,  ndim)
            p2v_cpu = Point2VoxelCPU(dtypes.float32, ndim)
            self.add_param_class(f"ops{ndim}d", p2v, f"Point2Voxel{ndim}D")
            self.add_param_class(f"ops_cpu{ndim}d", p2v_cpu, f"Point2Voxel{ndim}DCPU")

            problem = ConvProblem(ndim, ConvOpType.kForward, NHWC, NHWC, NHWC)
            indices = SparseConvIndicesKernel(problem, dtypes.int32)
            # self.add_param_class("ops", indices, "SpconvIndices")
            cuda_funcs = [self.generate_subm_conv_inds, 
                self.generate_conv_inds_stage1, self.generate_conv_inds_stage1_5, self.generate_conv_inds_stage2, self.sort_1d_by_key]
            self.add_impl_only_param_class(cuda_funcs, f"ops{ndim}d", indices, f"SpconvIndices{ndim}D")


    @pccm.pybind.mark
    @pccm.cuda.static_function
    def generate_conv_inds_stage1(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, indice_pairs_uniq, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")
        code.arg("transposed", f"bool", "false")

        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.raw(f"""
        int ndim = indices.dim(1) - 1;
        TV_ASSERT_RT_ERR(output_dims.size() == ndim && input_dims.size() == ndim &&
            ksize.size() == ndim && stride.size() == ndim && dilation.size() == ndim &&
            padding.size() == ndim, "your params size not equal to ndim", ndim);
        """)

        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                tv::array<int, {ndim}> output_dims_, input_dims_;
                tv::array<int, {ndim}> ksize_, stride_, padding_, dilation_;
                for (int i = 0; i < {ndim}; ++i){{
                    output_dims_[i] = output_dims[i];
                    input_dims_[i] = input_dims[i];
                    ksize_[i] = ksize[i];
                    stride_[i] = stride[i];
                    padding_[i] = padding[i];
                    dilation_[i] = dilation[i];
                }}
                return SpconvIndices{ndim}D::generate_conv_inds_stage1(indices,
                    indice_pairs, indice_pairs_uniq, indice_num_per_loc,
                    batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code# .ret("int")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def generate_conv_inds_stage1_5(self):
        code = pccm.FunctionCode()
        code.arg("indice_pairs_uniq", "tv::Tensor")
        code.arg("ndim", "int")
        code.arg("uniq_size", "int64_t")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                return SpconvIndices{ndim}D::generate_conv_inds_stage1_5(indice_pairs_uniq, uniq_size, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def generate_conv_inds_stage2(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata", "tv::Tensor")
        code.arg("indice_pairs, indice_pairs_uniq, out_inds", "tv::Tensor")
        code.arg("num_out_act", "int")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")
        code.arg("transposed", f"bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.raw(f"""
        int ndim = indices.dim(1) - 1;
        TV_ASSERT_RT_ERR(output_dims.size() == ndim && input_dims.size() == ndim &&
            ksize.size() == ndim && stride.size() == ndim && dilation.size() == ndim &&
            padding.size() == ndim, "your params size not equal to ndim", ndim);
        """)

        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                tv::array<int, {ndim}> output_dims_, input_dims_;
                tv::array<int, {ndim}> ksize_, stride_, padding_, dilation_;
                for (int i = 0; i < {ndim}; ++i){{
                    output_dims_[i] = output_dims[i];
                    input_dims_[i] = input_dims[i];
                    ksize_[i] = ksize[i];
                    stride_[i] = stride[i];
                    padding_[i] = padding[i];
                    dilation_[i] = dilation[i];
                }}
                return SpconvIndices{ndim}D::generate_conv_inds_stage2(indices, hashdata,
                    indice_pairs, indice_pairs_uniq, out_inds, num_out_act,
                    batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code.ret("int")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def generate_subm_conv_inds(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"std::vector<int>")
        code.arg("ksize, dilation", f"std::vector<int>")
        code.arg("indice_pair_mask", "tv::Tensor", "tv::Tensor()", "cumm.tensorview.Tensor = Tensor()")
        code.arg("backward", "bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int = 0")
        code.raw(f"""
        int ndim = indices.dim(1) - 1;
        TV_ASSERT_RT_ERR(input_dims.size() == ndim &&
            ksize.size() == ndim && dilation.size() == ndim, "your params size not equal to ndim", ndim);
        """)
        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                tv::array<int, {ndim}> input_dims_;
                tv::array<int, {ndim}> ksize_, dilation_;
                for (int i = 0; i < {ndim}; ++i){{
                    input_dims_[i] = input_dims[i];
                    ksize_[i] = ksize[i];
                    dilation_[i] = dilation[i];
                }}
                return SpconvIndices{ndim}D::generate_subm_conv_inds(indices, hashdata,
                    indice_pairs, out_inds, indice_num_per_loc,
                    batch_size, input_dims_, 
                    ksize_, dilation_, indice_pair_mask, backward,
                    stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def maxpool_forward(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        return IndiceMaxPool::forward(out, inp, out_inds, in_inds, stream);
        """)
        return code

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def maxpool_backward(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("dout", "tv::Tensor")
        code.arg("dinp", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        return IndiceMaxPool::backward(out, inp, dout, dinp, out_inds, in_inds, stream);
        """)
        return code

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def sort_1d_by_key(self):
        code = pccm.FunctionCode()
        code.add_dependency(ThrustLib, TensorViewKernel)
        code.add_param_class("cudakers", CudaCommonKernel())
        code.arg("data", "tv::Tensor")
        code.raw(f"""
        tv::Tensor indices({{data.dim(0)}}, tv::int32, 0);
        tv::cuda::Launch launcher(data.dim(0));
        launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
        tv::dispatch<int32_t, uint32_t, int64_t, uint64_t>(data.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
            thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
            auto thrust_ctx = thrust::cuda::par.on(0);
            thrust::sort_by_key(thrust_ctx, ptr_tr, ptr_tr + data.dim(0), ptr_k);
        }});
        return indices;
        """)
        return code.ret("tv::Tensor")
