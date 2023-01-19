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

from typing import List
from cumm.common import TensorView, TensorViewCPU, TensorViewKernel, ThrustLib, GemmBasicHost, CppTimer
import cumm
from cumm.conv.bases import ConvOpType, NHWC
from cumm.conv.params import ConvProblem
from cumm import dtypes
from cumm.constants import CUMM_CPU_ONLY_BUILD
import pccm
from pccm.__version__ import __version__ as pccm_version
from ccimport import compat
from .pointops import Point2Voxel, Point2VoxelCPU
from .indices import SparseConvIndicesKernel, CudaCommonKernel, SparseConvIndicesCPU
from .maxpool import IndiceMaxPool, IndiceMaxPoolCPU
from .gather import GatherCPU
from .alloc import ExternalAllocator, ThrustAllocator
from spconv.constants import SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE, AllocKeys
import re
import os 
from cumm.gemm.codeops import dispatch
class CustomThrustLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(ThrustLib)
        # https://github.com/NVIDIA/thrust/issues/1401#issuecomment-806403746
        if compat.InLinux:
            self.build_meta.add_public_cflags("nvcc", "-Xcompiler -fno-gnu-unique", "-Xcompiler -fvisibility=hidden")


class ThrustCustomAllocatorV2(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("functional", "memory")
        self.add_pybind_member("alloc_func",
                               "std::function<std::uintptr_t(std::size_t)>",
                               pyanno="Callable[[int], int]")
        self.add_typedef("value_type", "char")

    @pccm.member_function
    def allocate(self):
        code = pccm.FunctionCode()
        code.arg("num_bytes", "std::ptrdiff_t")
        code.ret("char*")
        code.raw(f"""
        if (alloc_func){{
            char* result = reinterpret_cast<char*>(alloc_func(num_bytes));
            return result;
        }}
        else{{
            TV_THROW_RT_ERR("set alloc function first.");
        }}
        """)
        return code

    @pccm.member_function
    def deallocate(self):
        code = pccm.FunctionCode()
        code.arg("ptr", "char *")
        code.arg("num_bytes", "size_t")
        return code        

def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

class HashCoreHost(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/hash/hash_core.h")

class SpconvOps(pccm.Class):
    if CUMM_CPU_ONLY_BUILD:
        _STATIC_FUNCTION = pccm.static_function
    else:
        _STATIC_FUNCTION = pccm.cuda.static_function
    def __init__(self):
        super().__init__()
        self.add_dependency(ThrustCustomAllocatorV2, ExternalAllocator, GemmBasicHost, ThrustAllocator)
        self.ndims = [1, 2, 3, 4]
        self.cuda_common_kernel = CudaCommonKernel()
        for ndim in self.ndims:
            p2v = Point2Voxel(dtypes.float32, ndim)
            p2v_cpu = Point2VoxelCPU(dtypes.float32, ndim)
            self.add_param_class(f"ops_cpu{ndim}d", p2v_cpu,
                                 f"Point2Voxel{ndim}DCPU")

            problem = ConvProblem(ndim, ConvOpType.kForward, NHWC, NHWC, NHWC)
            indices = SparseConvIndicesKernel(problem, dtypes.int32)
            indices_cpu = SparseConvIndicesCPU(problem, dtypes.int32)
            self.add_param_class(f"ops_cpu{ndim}d", indices_cpu,
                                 f"SpconvIndicesCPU{ndim}D")
            # self.add_param_class("ops", indices, "SpconvIndices")
            if not CUMM_CPU_ONLY_BUILD:
                self.add_param_class(f"ops{ndim}d", p2v, f"Point2Voxel{ndim}D")
                cuda_funcs = [
                    self.generate_subm_conv_inds,
                    self.generate_conv_inds_stage1,
                    self.generate_conv_inds_stage1_5,
                    self.generate_conv_inds_stage2, self.sort_1d_by_key,
                    self.generate_conv_inds_mask_stage1,
                    self.generate_conv_inds_mask_stage2,
                    self.unique_hash, self.assign_output_direct_hash, 
                    self.generate_conv_inds_mask_stage1_direct_table,
                    self.generate_conv_inds_stage2_mask_direct_table
                ]
                self.add_impl_only_param_class(cuda_funcs, f"ops{ndim}d",
                                               indices,
                                               f"SpconvIndices{ndim}D")
        defines: List[str] = []
        # static constexpr in c++ < 17 may cause 
        # undefined symbol. use macro instead.
        for name in dir(AllocKeys):
            if not name.startswith("__"):
                v = getattr(AllocKeys, name)
                defines.append(f"#define SPCONV_ALLOC_{to_snake_case(name).upper()} {pccm.literal(v)}")
        define_str = "\n".join(defines)
        self.add_global_code(define_str)
        self.build_meta.add_global_cflags("cl", "/DNOMINMAX")
        cuda_ver = os.environ.get("CUMM_CUDA_VERSION", "")
        if cuda_ver:
            cuda_ver_items = cuda_ver.split(".")
            if len(cuda_ver_items) == 1:
                cuda_ver_num = int(cuda_ver)
                cuda_ver_tuple = (cuda_ver_num // 10, cuda_ver_num % 10)
            else:
                cuda_ver_vec = list(map(int, cuda_ver.split(".")))
                cuda_ver_tuple = (cuda_ver_vec[0], cuda_ver_vec[1])
            if cuda_ver_tuple[0] < 11:
                self.build_meta.add_global_cflags("nvcc", "-w")

        # for name in dir(AllocKeys):
        #     if not name.startswith("__"):
        #         v = getattr(AllocKeys, name)
        #         self.add_static_const("k" + name, "auto", f"tv::make_const_string({pccm.literal(v)})")

    @pccm.pybind.mark
    @pccm.static_function
    def cumm_version(self):
        """get cumm version when build spconv.
        """
        code = pccm.FunctionCode()
        code.raw(f"""
        return \"{cumm.__version__}\";
        """)
        return code.ret("std::string")

    @pccm.pybind.mark
    @pccm.static_function
    def is_cpu_only_build(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return {pccm.literal(CUMM_CPU_ONLY_BUILD)};
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def pccm_version(self):
        """get pccm version when build spconv.
        """
        code = pccm.FunctionCode()
        code.raw(f"""
        return \"{pccm_version}\";
        """)
        return code.ret("std::string")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_stage1(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, indice_pairs_uniq, indice_num_per_loc",
                 "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")
        code.arg("transposed", f"bool", "false")

        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
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

        return code  # .ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_stage1_5(self):
        code = pccm.FunctionCode()
        code.arg("indice_pairs_uniq", "tv::Tensor")
        code.arg("ndim", "int")
        code.arg("uniq_size", "int64_t")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                return SpconvIndices{ndim}D::generate_conv_inds_stage1_5(indice_pairs_uniq, uniq_size, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_stage2(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg("indice_pairs, indice_pairs_uniq, indice_pairs_uniq_before_sort, out_inds", "tv::Tensor")
        code.arg("indice_num_per_loc", "tv::Tensor")
        code.arg("num_out_act", "int")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")
        code.arg("transposed", f"bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("use_bound_algo", "bool", "false")

        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

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
                return SpconvIndices{ndim}D::generate_conv_inds_stage2(indices, 
                    hashdata_k, hashdata_v, indice_pairs,
                    indice_pairs_uniq, indice_pairs_uniq_before_sort, out_inds, 
                    indice_num_per_loc, num_out_act,
                    batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int,
                    use_bound_algo);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code.ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_mask_stage1(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs_bwd, indice_pairs_uniq, indice_num_per_loc",
                 "tv::Tensor")
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
                return SpconvIndices{ndim}D::generate_conv_inds_mask_stage1(indices,
                    indice_pairs_bwd, indice_pairs_uniq, indice_num_per_loc,
                    batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code  # .ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_mask_stage1_direct_table(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg("indice_pairs_bwd, indice_pairs_uniq, indice_num_per_loc",
                 "tv::Tensor")
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
                return SpconvIndices{ndim}D::generate_conv_inds_mask_stage1_direct_table(indices,
                    hashdata_k, hashdata_v, indice_pairs_bwd, indice_pairs_uniq, 
                    indice_num_per_loc, batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code  # .ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def unique_hash(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.arg("hashdata_k, hashdata_v, uniq_cnt, out_indices_offset", "tv::Tensor")
        code.arg("num_out_bound", "int")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.raw(f"""
        return SpconvIndices3D::unique_hash(hashdata_k, hashdata_v, 
            uniq_cnt, out_indices_offset, num_out_bound, stream_int);
        """)
        return code.ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def assign_output_direct_hash(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.arg("out_indices_offset, out_indices", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")

        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.raw(f"""
        int ndim = out_indices.dim(1) - 1;
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
                return SpconvIndices{ndim}D::assign_output_direct_hash(
                    out_indices_offset, out_indices, batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_mask_stage2(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg(
            "indice_pairs_fwd, indice_pairs_bwd, indice_pairs_uniq, indice_pairs_uniq_before_sort, out_inds",
            "tv::Tensor")
        code.arg("mask_fwd, mask_bwd", "tv::Tensor")
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
                return SpconvIndices{ndim}D::generate_conv_inds_stage2_mask(
                    indices, hashdata_k, hashdata_v,
                    indice_pairs_fwd, indice_pairs_bwd, 
                    indice_pairs_uniq, indice_pairs_uniq_before_sort,
                    out_inds, mask_fwd, mask_bwd,
                    num_out_act, batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code.ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_conv_inds_stage2_mask_direct_table(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg(
            "indice_pairs_fwd, indice_pairs_bwd, indice_pairs_uniq, indice_pairs_uniq_before_sort, out_inds",
            "tv::Tensor")
        code.arg("mask_fwd, mask_bwd", "tv::Tensor")
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
                return SpconvIndices{ndim}D::generate_conv_inds_stage2_mask_direct_table(
                    indices, hashdata_k, hashdata_v,
                    indice_pairs_fwd, indice_pairs_bwd, 
                    indice_pairs_uniq, indice_pairs_uniq_before_sort,
                    out_inds, mask_fwd, mask_bwd,
                    num_out_act, batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def generate_subm_conv_inds(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"std::vector<int>")
        code.arg("ksize, dilation", f"std::vector<int>")
        code.arg("indice_pair_mask", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
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
                return SpconvIndices{ndim}D::generate_subm_conv_inds(indices, 
                    hashdata_k, hashdata_v,
                    indice_pairs, out_inds, indice_num_per_loc,
                    batch_size, input_dims_, 
                    ksize_, dilation_, indice_pair_mask, backward,
                    stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.static_function
    def generate_conv_inds_cpu(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")
        code.arg("transposed", f"bool", "false")
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
                return SpconvIndicesCPU{ndim}D::generate_conv_inds(indices,
                    indice_pairs, out_inds, indice_num_per_loc,
                    batch_size, output_dims_, input_dims_, 
                    ksize_, stride_, padding_, dilation_, transposed);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")

        return code.ret("int")

    @pccm.pybind.mark
    @pccm.static_function
    def generate_subm_conv_inds_cpu(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"std::vector<int>")
        code.arg("ksize, dilation", f"std::vector<int>")

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
                return SpconvIndicesCPU{ndim}D::generate_subm_conv_inds(indices,
                    indice_pairs, out_inds, indice_num_per_loc,
                    batch_size, input_dims_, 
                    ksize_, dilation_);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("int")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def maxpool_forward(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
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
    @_STATIC_FUNCTION
    def maxpool_backward(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
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
    @_STATIC_FUNCTION
    def indice_maxpool(self):
        code = pccm.FunctionCode()
        code.arg("out_features, features", "tv::Tensor")
        code.arg("indice_pairs", "tv::Tensor")
        code.arg("indice_pair_num", "tv::Tensor")
        code.arg("num_activate_out", "int")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPoolCPU)
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        tv::check_shape(out_features, {{-1, features.dim(1)}});

        auto indice_pair_num_cpu = indice_pair_num.cpu();
        auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();

        """)
        with code.for_("int i = 0; i < indice_pair_num.dim(0); ++i"):
            code.raw(f"""
            int nhot = indice_pair_num_cpu_ptr[i];
            nhot = std::min(nhot, int(indice_pairs.dim(2)));
            if (nhot <= 0){{
                continue;
            }}
            auto inp_indices = indice_pairs[0][i].slice_first_axis(0, nhot);
            auto out_indices = indice_pairs[1][i].slice_first_axis(0, nhot);
            if (features.is_cpu()){{
                IndiceMaxPoolCPU::forward(out_features, features, out_indices, inp_indices);
            }}
            """)
            if not CUMM_CPU_ONLY_BUILD:
                with code.else_():
                    code.raw(f"""
                    IndiceMaxPool::forward(out_features, features, out_indices, inp_indices, stream);
                    """)
            else:
                code.raw(f"""
                TV_THROW_RT_ERR("not implemented in cpu-only spconv!!! ")
                """)
        return code


    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def indice_maxpool_backward(self):
        code = pccm.FunctionCode()
        code.arg("din, features, out_features, out_bp", "tv::Tensor")
        code.arg("indice_pairs", "tv::Tensor")
        code.arg("indice_pair_num", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPoolCPU)
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        tv::check_shape(din, features.shape());
        auto indice_pair_num_cpu = indice_pair_num.cpu();
        auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();

        """)
        with code.for_("int i = 0; i < indice_pair_num.dim(0); ++i"):
            code.raw(f"""
            int nhot = indice_pair_num_cpu_ptr[i];
            nhot = std::min(nhot, int(indice_pairs.dim(2)));

            if (nhot <= 0){{
                continue;
            }}
            auto inp_indices = indice_pairs[0][i].slice_first_axis(0, nhot);
            auto out_indices = indice_pairs[1][i].slice_first_axis(0, nhot);
            if (features.is_cpu()){{
                IndiceMaxPoolCPU::backward(out_features, features, out_bp, din, out_indices, inp_indices);
            }}
            """)
            if not CUMM_CPU_ONLY_BUILD:
                with code.else_():
                    code.raw(f"""
                    IndiceMaxPool::backward(out_features, features, out_bp, din, out_indices, inp_indices, stream);
                    """)
            else:
                code.raw(f"""
                TV_THROW_RT_ERR("not implemented in cpu-only spconv!!! ")
                """)
        return code

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def maxpool_implicit_gemm_forward(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        return IndiceMaxPool::forward_implicit_gemm(out, inp, inds, stream);
        """)
        return code

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def maxpool_implicit_gemm_backward(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("dout", "tv::Tensor")
        code.arg("dinp", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        return IndiceMaxPool::backward_implicit_gemm(out, inp, dout, dinp, inds, stream);
        """)
        return code

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def avgpool_implicit_gemm_forward(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("count_out", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        return IndiceMaxPool::forward_avgpool_implicit_gemm(out, inp, inds, count_out, stream);
        """)
        return code

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def avgpool_implicit_gemm_backward(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("dout", "tv::Tensor")
        code.arg("dinp", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.arg("count_out", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.add_dependency(IndiceMaxPool)
        code.raw(f"""
        return IndiceMaxPool::backward_avgpool_implicit_gemm(dout, dinp, inds, count_out, stream);
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def maxpool_forward_cpu(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.add_dependency(IndiceMaxPoolCPU)
        code.raw(f"""
        return IndiceMaxPoolCPU::forward(out, inp, out_inds, in_inds);
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def maxpool_backward_cpu(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("dout", "tv::Tensor")
        code.arg("dinp", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("in_inds", "tv::Tensor")
        code.add_dependency(IndiceMaxPoolCPU)
        code.raw(f"""
        return IndiceMaxPoolCPU::backward(out, inp, dout, dinp, out_inds, in_inds);
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def gather_cpu(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.add_dependency(GatherCPU)
        code.raw(f"""
        return GatherCPU::gather(out, inp, inds);
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def scatter_add_cpu(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("inp", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.add_dependency(GatherCPU)
        code.raw(f"""
        return GatherCPU::scatter_add(out, inp, inds);
        """)
        return code

    def sort_1d_by_key_allocator_template(self, use_allocator: bool):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("data", "tv::Tensor")
        if not use_allocator:
            code.arg("alloc_func", "std::function<std::uintptr_t(std::size_t)>")
        else:
            code.arg("allocator", "ThrustAllocator&")

        code.arg("indices",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.arg("mask_count", "int", "1", pyanno="int")
        code.arg("do_sort", "bool", "true")

        code.add_dependency(CustomThrustLib, TensorViewKernel)
        code.add_param_class("cudakers", self.cuda_common_kernel)
        if not use_allocator:
            code.raw(f"""
            ThrustCustomAllocatorV2 allocator{{alloc_func}};
            """)
        code.raw(f"""
        cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
        if (indices.empty()){{
            indices = tv::empty({{data.dim(0)}}, tv::int32, 0);
        }}
        tv::cuda::Launch launcher(data.dim(0), stream_cu);
        launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
        if (!do_sort){{
            return indices;
        }}
        // auto timer = tv::CUDATimer();
        """)
        # nested tv::dispatch may cause compiler bug in msvc.
        for dtype in dispatch(code, [dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64], "data.dtype()"):
            code.raw(f"""
            using T_ = {dtype};
            tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){{
                constexpr int I = TV_DECLTYPE(IV)::value;
                // we can't use thrust::tuple in mp_repeat_c directly because
                // thrust tuple actually has fixed size template arguments.
                using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
                thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
                thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
                auto thrust_ctx = thrust::cuda::par.on(stream_cu);
                auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
                thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
            }});
            """)
        code.raw(f"""
        // tv::ssprint("SORT BY KEY TIME", data.dim(0), timer.report() / 1000.0);
        return indices;
        """)
        return code.ret("tv::Tensor")


    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def sort_1d_by_key_allocator(self):
        return self.sort_1d_by_key_allocator_template(False)

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def sort_1d_by_key_allocator_v2(self):
        return self.sort_1d_by_key_allocator_template(True)

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def sort_1d_by_key_split(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("data", "tv::Tensor")
        code.arg("mask", "tv::Tensor")

        code.arg("indices",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.arg("mask_output", "bool", "false")

        code.code_after_include = f"""
        template <typename T> struct MaskedElementComp {{
            T mask_;
            TV_HOST_DEVICE_INLINE T operator()(const T &x, const T &y) const {{
                return (x & mask_) < (y & mask_);
            }}
        }};
        template <typename T> __global__ void mask_input(T* inp, T mask, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                inp[i] &= mask;
            }}
        }}
        """
        code.add_dependency(CustomThrustLib, TensorViewKernel)
        code.add_param_class("cudakers", self.cuda_common_kernel)
        code.raw(f"""
        cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
        // auto timer = tv::CudaContextTimer<>();
        if (indices.empty()){{
            indices = tv::empty({{data.dim(0)}}, tv::int32, 0);
        }}
        tv::cuda::Launch launcher(data.dim(0), stream_cu);
        launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
        tv::dispatch<int32_t, uint32_t, int64_t, uint64_t>(data.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto masks_ptr = mask.data_ptr<T>();
            MaskedElementComp<T> op_comp{{masks_ptr[0]}};
            thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
            thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
            auto thrust_ctx = thrust::cuda::par.on(stream_cu);
            thrust::sort_by_key(thrust_ctx, ptr_tr, ptr_tr + data.dim(0), ptr_k, op_comp);
            if (mask_output){{
                launcher(mask_input<T>, data.data_ptr<T>(), masks_ptr[0], data.dim(0));
            }}
        }});
        // tv::ssprint("SORT BY KEY MASKED TIME", timer.report() / 1000.0);
        return indices;
        """)
        return code.ret("tv::Tensor")

    def sort_1d_by_key_split_allocator_template(self, use_allocator: bool):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("data", "tv::Tensor")
        if not use_allocator:
            code.arg("alloc_func", "std::function<std::uintptr_t(std::size_t)>")
        else:
            code.arg("allocator", "ThrustAllocator&")

        code.arg("mask", "tv::Tensor")

        code.arg("indices",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.arg("mask_output", "bool", "false")

        code.code_after_include = f"""
        template <typename T> struct MaskedElementComp {{
            T mask_;
            TV_HOST_DEVICE_INLINE T operator()(const T &x, const T &y) const {{
                return (x & mask_) < (y & mask_);
            }}
        }};
        template <typename T> __global__ void mask_input(T* inp, T mask, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                inp[i] &= mask;
            }}
        }}
        """
        code.add_dependency(CustomThrustLib, TensorViewKernel)
        code.add_param_class("cudakers", self.cuda_common_kernel)
        if not use_allocator:
            code.raw(f"""
            ThrustCustomAllocatorV2 allocator{{alloc_func}};
            """)
        code.raw(f"""
        cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
        // auto timer = tv::CudaContextTimer<>();
        if (indices.empty()){{
            indices = tv::empty({{data.dim(0)}}, tv::int32, 0);
        }}
        tv::cuda::Launch launcher(data.dim(0), stream_cu);
        launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
        tv::dispatch<int32_t, uint32_t, int64_t, uint64_t>(data.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto masks_ptr = mask.data_ptr<T>();
            MaskedElementComp<T> op_comp{{masks_ptr[0]}};
            thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
            thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
            // auto thrust_ctx = thrust::cuda::par.on(stream_cu);
            auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
            thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k, op_comp);
            if (mask_output){{
                launcher(mask_input<T>, data.data_ptr<T>(), masks_ptr[0], data.dim(0));
            }}
        }});
        // tv::ssprint("SORT_BY_KEY_MASKED", timer.report() / 1000.0);
        return indices;
        """)
        return code.ret("tv::Tensor")



    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def sort_1d_by_key_split_allocator(self):
        return self.sort_1d_by_key_split_allocator_template(False)

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def sort_1d_by_key_split_allocator_v2(self):
        return self.sort_1d_by_key_split_allocator_template(True)

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def count_bits(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.add_dependency(TensorViewKernel)
        code.arg("a", "tv::Tensor")
        code.code_after_include = f"""
        __global__ void count_bits_kernel_64(const uint64_t* data, int32_t* out, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                out[i] = __popcll(reinterpret_cast<const unsigned long long*>(data)[i]);
            }}
        }}
        __global__ void count_bits_kernel(const uint32_t* data, int32_t* out, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                out[i] = __popc(data[i]);
            }}
        }}

        int numberOfSetBits(uint32_t i)
        {{
            // https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
            // Java: use int, and use >>> instead of >>. Or use Integer.bitCount()
            // C or C++: use uint32_t
            i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
            i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  // quads
            i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
            return (i * 0x01010101) >> 24;          // horizontal sum of bytes
        }}

        int numberOfSetBits(uint64_t i)
        {{
            return numberOfSetBits(uint32_t(i)) + numberOfSetBits(uint32_t(i >> 32));
        }}
        """
        code.raw(f"""
        tv::Tensor res(a.shape(), tv::int32, a.device());
        tv::dispatch<uint32_t, uint64_t>(a.dtype(), [&](auto I){{
            auto res_ptr = res.data_ptr<int>();
            using T = TV_DECLTYPE(I);
            auto a_ptr = a.data_ptr<const T>();
            if (a.device() == -1){{
                for (int i = 0; i < a.size(); ++i){{
                    res_ptr[i] = numberOfSetBits(a_ptr[i]);
                }}
            }}else{{
                tv::cuda::Launch launcher(a.size());
                tv::if_constexpr<std::is_same<T, uint64_t>::value>([=](auto _)mutable{{
                    launcher(_(count_bits_kernel_64), a_ptr, res_ptr, int(a.size()));
                }}, [=](auto _)mutable{{
                    launcher(_(count_bits_kernel), a_ptr, res_ptr, int(a.size()));
                }});
            }}
        }});
        return res;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def reverse_bits(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.add_dependency(TensorViewKernel)
        code.arg("a", "tv::Tensor")
        code.code_after_include = f"""
        __global__ void reverse_bits_kernel_64(const uint64_t* data, uint64_t* out, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                out[i] = __brevll(reinterpret_cast<const unsigned long long*>(data)[i]);
            }}
        }}

        __global__ void reverse_bits_kernel(const uint32_t* data, uint32_t* out, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                out[i] = __brev(data[i]);
            }}
        }}

        uint32_t reverse(uint32_t x)
        {{
            x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
            x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
            x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
            x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
            x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
            return x;
        }}

        int reverse(uint64_t i)
        {{
            return (reverse(uint32_t(i)) << 32) | reverse(uint32_t(i >> 32));
        }}
        """
        code.raw(f"""
        tv::Tensor res(a.shape(), a.dtype(), a.device());
        tv::dispatch<uint32_t, uint64_t>(a.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto res_ptr = res.data_ptr<T>();
            auto a_ptr = a.data_ptr<const T>();
            if (a.device() == -1){{
                for (int i = 0; i < a.size(); ++i){{
                    res_ptr[i] = reverse(a_ptr[i]);
                }}
            }}else{{
                tv::cuda::Launch launcher(a.size());
                tv::if_constexpr<std::is_same<T, uint64_t>::value>([=](auto _)mutable{{
                    launcher(_(reverse_bits_kernel_64), a_ptr, res_ptr, int(a.size()));
                }}, [=](auto _)mutable{{
                    launcher(_(reverse_bits_kernel), a_ptr, res_ptr, int(a.size()));
                }});
            }}
        }});
        return res;
        """)
        return code.ret("tv::Tensor")

    # cpu only build can't use pccm.cuda
    __CUDA_DECORATOR = pccm.static_function
    if not CUMM_CPU_ONLY_BUILD:
        __CUDA_DECORATOR = _STATIC_FUNCTION

    @pccm.pybind.mark 
    @__CUDA_DECORATOR
    def maximum_value_int(self):
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_param_class("cudakers", self.cuda_common_kernel)
        code.arg("data", "tv::Tensor")
        code.arg("value", "int")
        code.arg("stream_int", "std::uintptr_t")

        code.raw(f"""
        auto size = data.size();
        using ints_t = std::tuple<int32_t, int16_t, int8_t, int64_t, uint32_t, uint64_t, uint16_t, uint8_t>;
        """)    
        with code.block("", start="tv::Dispatch<ints_t>()(data.dtype(), [&](auto I){", end="});"):
            code.raw(f"""
            using T = TV_DECLTYPE(I);

            auto ptr = data.data_ptr<T>();
            """)
            with code.if_("data.is_cpu()"):
                code.raw(f"""
                for (int i = 0; i < size; ++i){{
                    ptr[i] = std::max(ptr[i], T(value));
                }}
                """)
            with code.else_():
                if not CUMM_CPU_ONLY_BUILD:
                    code.raw(f"""
                    tv::cuda::Launch lanucher(size, reinterpret_cast<cudaStream_t>(stream_int));
                    lanucher(cudakers::maximum_value_kernel<T>, ptr, value, size);
                    """)
                else:
                    code.raw(f"""
                    TV_THROW_RT_ERR("only support cpu.");
                    """)
        return code 
        
    @pccm.pybind.mark
    @_STATIC_FUNCTION
    def sort_1d_by_key(self):
        code = pccm.FunctionCode()
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()
        code.arg("data", "tv::Tensor")
        code.arg("indices",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("stream", "std::uintptr_t", "0", pyanno="int")
        code.code_after_include = f"""
        template <typename T> struct SmallOrEqualTo {{
            TV_HOST_DEVICE_INLINE T operator()(const T &x, const T &y) const {{
                return x < y;
            }}
        }};
        template <typename T> __global__ void mask_input(T* inp, T mask, int size){{
            for (int i : tv::KernelLoopX<int>(size)){{
                inp[i] &= mask;
            }}
        }}

        """
        code.add_dependency(CustomThrustLib, TensorViewKernel)
        code.add_param_class("cudakers", self.cuda_common_kernel)
        code.raw(f"""
        cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
        if (indices.empty()){{
            indices = tv::empty({{data.dim(0)}}, tv::int32, 0);
        }}
        tv::cuda::Launch launcher(data.dim(0), stream_cu);
        launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
        auto timer = tv::CUDATimer();
        tv::dispatch<int32_t, uint32_t, int64_t, uint64_t>(data.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
            thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
            auto thrust_ctx = thrust::cuda::par.on(stream_cu);
            thrust::stable_sort_by_key(thrust_ctx, ptr_tr, ptr_tr + data.dim(0), ptr_k, SmallOrEqualTo<uint32_t>());
        }});
        // tv::ssprint("SORT BY KEY TIME", data.dim(0), timer.report() / 1000.0);
        return indices;
        """)
        return code.ret("tv::Tensor")


    @pccm.pybind.mark
    @pccm.static_function
    def calc_point2voxel_meta_data(self):
        code = pccm.FunctionCode()
        code.arg("vsize_xyz", f"std::vector<float>")
        code.arg("coors_range_xyz", f"std::vector<float>")
        code.raw(f"""
        int ndim = vsize_xyz.size();
        TV_ASSERT_RT_ERR(vsize_xyz.size() == ndim &&
            coors_range_xyz.size() == ndim * 2, "your params size not equal to ndim", ndim);
        """)
        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                std::array<float, {ndim}> vsize_xyz_;
                std::array<float, {ndim * 2}> coors_range_xyz_;
                for (int i = 0; i < {ndim}; ++i){{
                    vsize_xyz_[i] = vsize_xyz[i];
                    coors_range_xyz_[i] = coors_range_xyz[i];
                    coors_range_xyz_[i + {ndim}] = coors_range_xyz[i + {ndim}];
                }}
                auto res = Point2Voxel{ndim}DCPU::calc_meta_data(vsize_xyz_, coors_range_xyz_);
                std::vector<float> vsize({ndim}), coors_range({ndim * 2});
                std::vector<int> grid_size({ndim}), grid_stride({ndim});

                for (int i = 0; i < {ndim}; ++i){{
                    vsize[i] = std::get<0>(res)[i];
                    grid_size[i] = std::get<1>(res)[i];
                    grid_stride[i] = std::get<2>(res)[i];
                    coors_range[i] = std::get<3>(res)[i];
                    coors_range[i + {ndim}] = std::get<3>(res)[i + {ndim}];
                }}
                return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret(
            "std::tuple<std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>>"
        )

    @pccm.pybind.mark
    @pccm.static_function
    def point2voxel_cpu(self):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("voxels, indices, num_per_voxel, densehashdata, pc_voxel_id", "tv::Tensor")
        code.arg("vsize", f"std::vector<float>")
        code.arg("grid_size, grid_stride", f"std::vector<int>")
        code.arg("coors_range", f"std::vector<float>")

        code.arg("empty_mean", "bool", "false")
        code.arg("clear_voxels", "bool", "true")

        code.raw(f"""
        int ndim = vsize.size();
        TV_ASSERT_RT_ERR(vsize.size() == ndim && grid_stride.size() == ndim && 
            coors_range.size() == ndim * 2 && grid_size.size() == ndim, 
            "your params size not equal to ndim", ndim);
        // voxels: []
        """)
        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                std::array<float, {ndim}> vsize_;
                std::array<int, {ndim}> grid_size_, grid_stride_;
                std::array<float, {ndim * 2}> coors_range_;
                for (int i = 0; i < {ndim}; ++i){{
                    vsize_[i] = vsize[i];
                    grid_size_[i] = grid_size[i];
                    grid_stride_[i] = grid_stride[i];
                    coors_range_[i] = coors_range[i];
                    coors_range_[i + {ndim}] = coors_range[i + {ndim}];
                }}
                if (empty_mean){{
                    return Point2Voxel{ndim}DCPU::point_to_voxel_empty_mean_static(points, voxels, indices, 
                        num_per_voxel, densehashdata, pc_voxel_id,
                        vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels);
                }} else{{
                    return Point2Voxel{ndim}DCPU::point_to_voxel_static(points, voxels, indices, 
                        num_per_voxel, densehashdata, pc_voxel_id,
                        vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels);
                }}
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")

    @pccm.pybind.mark
    @pccm.static_function
    def point2voxel_cuda(self):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("voxels, indices, num_per_voxel, hashdata, point_indice_data, pc_voxel_id",
                 "tv::Tensor")
        code.arg("vsize", f"std::vector<float>")
        code.arg("grid_size", f"std::vector<int>")
        code.arg("grid_stride", f"std::vector<int64_t>")
        code.arg("coors_range", f"std::vector<float>")

        code.arg("empty_mean", "bool", "false")
        code.arg("clear_voxels", "bool", "true")
        code.arg("stream_int", f"std::uintptr_t", "0")
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.raw(f"""
        int ndim = vsize.size();
        TV_ASSERT_RT_ERR(vsize.size() == ndim && grid_stride.size() == ndim && 
            coors_range.size() == ndim * 2 && grid_size.size() == ndim, 
            "your params size not equal to ndim", ndim);
        // voxels: []
        """)
        for ndim in self.ndims:
            code.raw(f"""
            if (ndim == {ndim}){{
                std::array<float, {ndim}> vsize_;
                std::array<int, {ndim}> grid_size_;
                std::array<int64_t, {ndim}> grid_stride_;

                std::array<float, {ndim * 2}> coors_range_;
                for (int i = 0; i < {ndim}; ++i){{
                    vsize_[i] = vsize[i];
                    grid_size_[i] = grid_size[i];
                    grid_stride_[i] = grid_stride[i];
                    coors_range_[i] = coors_range[i];
                    coors_range_[i + {ndim}] = coors_range[i + {ndim}];
                }}
                return Point2Voxel{ndim}D::point_to_voxel_hash_static(points, voxels, indices, 
                    num_per_voxel, hashdata, point_indice_data, pc_voxel_id,
                    vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels, 
                    empty_mean, stream_int);
            }}
            """)
        code.raw(f"""TV_THROW_RT_ERR("unknown ndim", ndim);""")
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")

    @pccm.pybind.mark
    @pccm.static_function
    def get_int32_max(self):
        code = pccm.FunctionCode()
        code.raw(f"return std::numeric_limits<int>::max();")
        return code.ret("int")

    @pccm.static_function
    def get_conv_output_size(self):
        code = pccm.FunctionCode()
        code.arg("input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation", f"std::vector<int>")
        code.raw(f"""
        int ndim = input_dims.size();
        std::vector<int> out_dims;
        for (int i = 0; i < ndim; ++i){{
            if (ksize[i] == -1){{
                out_dims.push_back(1);
            }}else{{
                auto size = (input_dims[i] + 2 * padding[i] - dilation[i] *
                    (ksize[i] - 1) - 1) / stride[i] + 1;
                out_dims.push_back(size);
            }}
        }}
        return out_dims;
        """)
        return code.ret("std::vector<int>")

    @pccm.static_function
    def get_deconv_output_size(self):
        code = pccm.FunctionCode()
        code.arg("input_dims", f"std::vector<int>")
        code.arg("ksize, stride, padding, dilation, output_padding", f"std::vector<int>")
        code.raw(f"""
        int ndim = input_dims.size();
        std::vector<int> out_dims;
        for (int i = 0; i < ndim; ++i){{
            if (ksize[i] == -1){{
                TV_THROW_INVALID_ARG("kernel size can't be -1");
            }}else{{
                auto size = (input_dims[i] - 1) * stride[i] - 2 * padding[i] + ksize[
                    i] + output_padding[i];
                out_dims.push_back(size);
            }}
        }}
        return out_dims;
        """)
        return code.ret("std::vector<int>")

    @_STATIC_FUNCTION
    def apply_thrust_unique_to_indice_pairs_uniq(self):
        code = pccm.code()
        code.arg("data", "tv::Tensor")
        code.arg("allocator", "ThrustAllocator&")
        code.arg("stream_int", f"std::uintptr_t", "0")
        if CUMM_CPU_ONLY_BUILD:
            return code.make_invalid()

        code.add_dependency(CustomThrustLib)
        code.raw(f"""
        int num_out_act = 0;
        int uniq_size = data.dim(0);
        tv::dispatch<int32_t, int64_t>(data.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
            auto thrust_ctx = thrust::cuda::par(allocator).on(reinterpret_cast<cudaStream_t>(stream_int));
            thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
            auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
            num_out_act = new_end - ptr_tr - 1;
        }});
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.pybind.mark 
    @pccm.static_function
    def get_handcrafted_max_act_out(self):
        code = pccm.code()
        code.arg("num_act_in", "size_t")
        code.arg("ksize, stride, padding, dilation", "std::vector<int>")
        code.raw(f"""
        int res = num_act_in;
        for (int i = 0; i < ksize.size(); ++i){{
            if (ksize[i] <= stride[i]){{
                res *= 1;
            }}
            else if (ksize[i] > stride[i]){{
                res *= tv::div_up(ksize[i], stride[i]);
            }}
            else{{
                res *= ksize[i];
            }}
        }}
        return res;
        """)
        return code.ret("int")

    @pccm.pybind.mark 
    @pccm.static_function
    def get_indice_gen_workspace_size(self):
        code = pccm.code()
        code.arg("kv", "size_t")
        code.arg("num_act_in", "size_t")
        code.arg("num_act_out_bound", "size_t")
        code.arg("max_act_out_in_theory", "size_t")
        code.arg("subm, use_int64_hash_k, direct_table", "bool")
        code.raw(f"""
        int hash_size = 2 * num_act_out_bound;
        if (direct_table){{
            hash_size = tv::align_up(int({SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE} * max_act_out_in_theory), 2);
        }}
        size_t res = 0;
        if (subm){{
            res = hash_size * (use_int64_hash_k ? 3 : 2) * sizeof(int) + 1 * sizeof(int);
        }}else{{
            size_t pair_single_size = kv * num_act_in; // 40000
            size_t ind_uniq_and_bkp_size = (pair_single_size + 1) * 2 * (use_int64_hash_k ? sizeof(int64_t) : sizeof(int32_t));
            hash_size = hash_size * (use_int64_hash_k ? 3 : 2) * sizeof(int);
            res = ind_uniq_and_bkp_size + hash_size + 1 * sizeof(int);
        }}
        return res;
        """)
        return code.ret("std::size_t")


    @pccm.pybind.mark 
    @pccm.static_function
    def get_indice_gen_tensors_from_workspace(self):
        code = pccm.code()
        code.arg("workspace", "uint8_t*")
        code.arg("kv", "size_t")
        code.arg("num_act_in", "size_t")
        code.arg("num_act_out_bound", "size_t")
        code.arg("max_act_out_in_theory", "size_t")
        code.arg("subm, use_int64_hash_k, direct_table", "bool")
        code.raw(f"""
        std::unordered_map<std::string, tv::Tensor> res;
        auto ws_prev = workspace;
        auto expected_size = get_indice_gen_workspace_size(kv, num_act_in, num_act_out_bound, 
            max_act_out_in_theory, subm, use_int64_hash_k, direct_table);
        int hash_size = 2 * num_act_out_bound;
        if (direct_table){{
            hash_size = tv::align_up(int({SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE} * max_act_out_in_theory), 2);
        }}
        if (use_int64_hash_k){{
            auto ten = tv::from_blob(workspace, {{int64_t(hash_size)}}, tv::int64, 0);
            res.insert({{{pccm.literal(AllocKeys.HashKOrKV)}, ten}});
            workspace += ten.nbytes();
            auto ten2 = tv::from_blob(workspace, {{int64_t(hash_size)}}, tv::int32, 0);
            res.insert({{{pccm.literal(AllocKeys.HashV)}, ten2}});
            workspace += ten2.nbytes();
        }}else{{
            auto ten = tv::from_blob(workspace, {{2, int64_t(hash_size)}}, tv::int32, 0);
            res.insert({{{pccm.literal(AllocKeys.HashKOrKV)}, ten}});
            workspace += ten.nbytes();
        }}
        if (!subm){{
            size_t pair_single_size = kv * int64_t(num_act_in);
            auto ten = tv::from_blob(workspace, {{int64_t(pair_single_size + 1)}}, use_int64_hash_k ? tv::int64 : tv::int32, 0);
            res.insert({{{pccm.literal(AllocKeys.IndicePairsUniq)}, ten}});
            workspace += ten.nbytes();
            auto ten2 = tv::from_blob(workspace, {{int64_t(pair_single_size + 1)}}, use_int64_hash_k ? tv::int64 : tv::int32, 0);
            res.insert({{{pccm.literal(AllocKeys.IndicePairsUniqBackup)}, ten2}});
            workspace += ten2.nbytes();
        }}
        auto uniq_cnt = tv::from_blob(workspace, {{1}}, tv::int32, 0);
        res.insert({{{pccm.literal(AllocKeys.TightUniqueCount)}, uniq_cnt}});
        workspace += uniq_cnt.nbytes();

        TV_ASSERT_RT_ERR(workspace - ws_prev == expected_size, "this shouldn't happen", kv, num_act_in,num_act_out_bound,  max_act_out_in_theory,
            subm, use_int64_hash_k, direct_table, "expected", expected_size, workspace - ws_prev);
        return res;
        """)
        return code.ret("std::unordered_map<std::string, tv::Tensor>")

    @pccm.pybind.mark 
    @pccm.static_function
    def get_indice_pairs_implicit_gemm(self):
        code = pccm.code()
        code.add_dependency(HashCoreHost)
        code.arg("allocator", "ExternalAllocator&")
        code.arg("indices", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"std::vector<int>")
        code.arg("algo", "int")
        code.arg("ksize, stride, padding, dilation, out_padding", f"std::vector<int>")
        code.arg("subm, transposed, is_train", f"bool")

        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("num_out_act_bound", f"int", "-1")
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)",
                 "cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")
        code.arg("direct_table", f"bool", "false")
        code.arg("do_sort", f"bool", "true")
        code.arg("preallocated", f"std::unordered_map<std::string, tv::Tensor>", 
            "std::unordered_map<std::string, tv::Tensor>{}", 
            "Dict[str, cumm.tensorview.Tensor] = {}")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"""
            TV_THROW_RT_ERR("this function can only be used with CUDA.");
            """)
            return code.ret("std::tuple<tv::Tensor, int>")
        code.raw(f"""
        auto tvctx = tv::Context();
        tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
        auto conv_algo = static_cast<tv::gemm::SparseConvAlgo>(algo);
        int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
        int mask_int_count = tv::div_up(kv, 32);
        // if (mask_int_count > 1 && mask_int_count < 4)
        //     mask_int_count = 4;
        // TV_ASSERT_RT_ERR(mask_int_count == 1 || mask_int_count == 4, "Not Implement too large kernel");
        // TV_ASSERT_RT_ERR(kv <= 32, "currently only support ksize < 32");
        std::vector<int> out_shape;
        if (!subm){{
            if (transposed){{
                out_shape = get_deconv_output_size(input_dims, ksize, stride, padding, dilation, out_padding);
            }}else{{
                out_shape = get_conv_output_size(input_dims, ksize, stride, padding, dilation);
            }}
        }}else{{
            out_shape = input_dims;
        }}
        for (auto& v : out_shape){{
            if (v <= 0){{
                TV_THROW_RT_ERR("your out spatial shape", out_shape, "ratch zero!, input shape:", input_dims);
            }}
        }}
        std::vector<int64_t> output_dims_i64(out_shape.begin(), out_shape.end());
        int64_t out_spatial_volume = std::accumulate(output_dims_i64.begin(),
          output_dims_i64.end(), int64_t(1), std::multiplies<int64_t>()) * batch_size;
        bool use_int64_hash_k = out_spatial_volume >= int64_t(std::numeric_limits<int>::max());
        tv::DType indice_uniq_dtype = use_int64_hash_k ? tv::int64 : tv::int32;
        TV_ASSERT_RT_ERR(conv_algo == tv::gemm::SparseConvAlgo::kMaskImplicitGemm || 
            conv_algo == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm, "only support implicit gemm");
        bool is_mask_split = conv_algo == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm;
        int mask_split_count = is_mask_split ? 2 : 1;
        int64_t num_act_in = indices.dim(0);
        """)
        code.raw(f"""
        tv::Tensor pair;
        if (subm){{
            if (preallocated.find({pccm.literal(AllocKeys.PairFwd)}) != preallocated.end()){{
                pair = preallocated.at({pccm.literal(AllocKeys.PairFwd)});
            }}
            else{{
                if (is_train){{
                    // query pair for fwd and bwd
                    pair = allocator.full_int({pccm.literal(AllocKeys.PairFwd)}, 
                        {{2, kv, num_act_in}}, -1, indices.dtype(), indices.device(), stream_int);
                }}else{{
                    // query pair fwd only
                    pair = allocator.full_int({pccm.literal(AllocKeys.PairFwd)}, 
                        {{1, kv, num_act_in}}, -1, indices.dtype(), indices.device(), stream_int);
                }}
            }}
        }}else{{
            if (is_train){{
                // query pair bwd
                pair = allocator.full_int({pccm.literal(AllocKeys.PairBwd)}, 
                    {{kv, num_act_in}}, -1, indices.dtype(), indices.device(), stream_int);
            }}else{{
                // don't need pair bwd, empty
                pair = tv::Tensor();
            }}
        }}
        """)

        code.raw(f"""
        tv::Tensor indice_num_per_loc;
        if (preallocated.find({pccm.literal(AllocKeys.IndiceNumPerLoc)}) != preallocated.end()){{
            indice_num_per_loc = preallocated.at({pccm.literal(AllocKeys.IndiceNumPerLoc)});
        }}
        else{{
            indice_num_per_loc = allocator.zeros({pccm.literal(AllocKeys.IndiceNumPerLoc)}, 
             {{kv}}, indices.dtype(), indices.device(), stream_int);
        }}
        tv::Tensor mask_tensor = tv::zeros({{mask_split_count}}, tv::uint32, -1);
        auto mask_tensor_ptr = mask_tensor.data_ptr<uint32_t>();

        if (is_mask_split){{
            TV_ASSERT_RT_ERR(mask_int_count == 1, "not support for kv > 32");
            auto kv_div_2 = kv / 2;
            auto remain = kv - kv_div_2;
            uint64_t mask_np_1 = 1;
            uint64_t first = ((mask_np_1 << remain) - 1);
            uint64_t second = ((mask_np_1 << kv_div_2) - 1) << remain;
            mask_tensor_ptr[0] = uint32_t(first);
            mask_tensor_ptr[1] = uint32_t(second);
        }}
        else{{
            mask_tensor_ptr[0] = 0xffffffff;
        }}
        tv::Tensor out_inds;
        ThrustAllocator thrustalloc(allocator);
        int num_act_out = 0;
        """)
        
        with code.if_("subm"):
            code.raw(f"""
            ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
            out_inds = indices;
            num_act_out = indices.dim(0);
            int hash_size = out_inds.dim(0) * 2;
            """)
            code.raw(f"""
            tv::Tensor hash_k, hash_v;
            if (use_int64_hash_k){{
                hash_k_guard = allocator.empty_guard({{hash_size}}, 
                    tv::int64, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                hash_v_gurad = allocator.empty_guard({{hash_size}}, 
                    tv::int32, 0, {pccm.literal(AllocKeys.HashV)});
                hash_k = hash_k_guard->tensor;
                hash_v = hash_v_gurad->tensor;
            }}else{{
                if (preallocated.find({pccm.literal(AllocKeys.HashKOrKV)}) != preallocated.end()){{
                    auto hash_kv = preallocated.at({pccm.literal(AllocKeys.HashKOrKV)});
                    hash_k = hash_kv[0];
                    hash_v = hash_kv[1];
                }}else{{
                    hash_kv_gurad = allocator.empty_guard({{2, hash_size}}, 
                        tv::int32, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                    hash_k = hash_kv_gurad->tensor[0];
                    hash_v = hash_kv_gurad->tensor[1];
                }}
            }}
            """)
            code.raw(f"""
            tv::Tensor pair_mask;
            if (preallocated.find({pccm.literal(AllocKeys.PairMask)}) != preallocated.end()){{
                pair_mask = preallocated.at({pccm.literal(AllocKeys.PairMask)});
            }}else{{
                pair_mask = allocator.empty({pccm.literal(AllocKeys.PairMask)}, 
                    {{mask_split_count, num_act_in, mask_int_count}}, tv::uint32, 0, stream_int);
            }}
            generate_subm_conv_inds(indices, hash_k, hash_v, pair, out_inds, indice_num_per_loc,
                batch_size, input_dims, ksize, dilation, pair_mask, is_train, stream_int);
            auto mask_argsort = allocator.empty({pccm.literal(AllocKeys.MaskArgSort)}, 
                {{mask_split_count, num_act_in}}, tv::int32, 0, stream_int);
            for (int j = 0; j < mask_split_count; ++j){{
                sort_1d_by_key_allocator_v2(pair_mask[j], thrustalloc, mask_argsort[j], stream_int, mask_int_count, do_sort);
            }}
        """)
        with code.else_():
            code.raw(f"""
            // auto start = tv::CPUEvent().record(stream_int);
            auto pair_bwd = pair;
            auto pair_size = kv * num_act_in;
            ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
            ExternalAllocator::guard_t indice_pairs_uniq_guard, indice_pairs_uniq_bkp_guard;
            tv::Tensor hash_k, hash_v, indice_pairs_uniq;
            int max_num_act = get_handcrafted_max_act_out(num_act_in, ksize, stride, padding, dilation);
            if (transposed){{
                max_num_act = pair_size;
            }}
            int hash_size = int(max_num_act * {SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE});
            if (direct_table){{
                if (use_int64_hash_k){{
                    // temp memory don't need to be fixed, static alloc will check
                    // that tensor is large enough.
                    hash_k_guard = allocator.empty_guard({{hash_size}}, 
                        tv::int64, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                    hash_v_gurad = allocator.empty_guard({{hash_size}}, 
                        tv::int32, 0, {pccm.literal(AllocKeys.HashV)});
                    hash_k = hash_k_guard->tensor;
                    hash_v = hash_v_gurad->tensor;
                }}else{{
                    hash_kv_gurad = allocator.empty_guard({{2, hash_size}}, 
                        tv::int32, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                    hash_k = hash_kv_gurad->tensor[0];
                    hash_v = hash_kv_gurad->tensor[1];
                }}
            }}
            indice_pairs_uniq_guard = allocator.empty_guard({{int64_t(pair_size + 1)}}, 
                indice_uniq_dtype, 0, {pccm.literal(AllocKeys.IndicePairsUniq)});
            
            indice_pairs_uniq = indice_pairs_uniq_guard->tensor;
            // auto indice_pairs_uniq_bkp = indice_pairs_uniq_guard->tensor[1];
            indice_pairs_uniq_bkp_guard = allocator.empty_guard({{int64_t(pair_size + 1)}}, 
                indice_uniq_dtype, 0, {pccm.literal(AllocKeys.IndicePairsUniqBackup)});
            auto indice_pairs_uniq_bkp = indice_pairs_uniq_bkp_guard->tensor;
            {{
                tv::CUDAKernelTimerGuard timer_guard("gen_conv_inds_stage1", 
                    timer, reinterpret_cast<cudaStream_t>(stream_int));
                if (direct_table){{
                    generate_conv_inds_mask_stage1_direct_table(indices, 
                    hash_k, hash_v, pair_bwd, indice_pairs_uniq_bkp,
                    indice_num_per_loc, batch_size, out_shape, input_dims, ksize,
                    stride, padding, dilation, transposed, stream_int);
                }}else{{
                    generate_conv_inds_mask_stage1(indices, pair_bwd, indice_pairs_uniq,
                        indice_num_per_loc, batch_size, out_shape, input_dims, ksize,
                        stride, padding, dilation, transposed, stream_int);
                    indice_pairs_uniq_bkp.copy_(indice_pairs_uniq, tvctx);
                }}
            }}
            // TODO pytorch unique run faster.
            {{
                tv::CUDAKernelTimerGuard timer_guard(std::string("unique_") + std::to_string(indice_pairs_uniq.dim(0)), 
                    timer, reinterpret_cast<cudaStream_t>(stream_int));
                if (direct_table){{
                    auto uniqcnt = allocator.zeros_guard({{1}}, tv::int32, 0, 
                        {pccm.literal(AllocKeys.TightUniqueCount)}, stream_int);
                    num_act_out = unique_hash(hash_k, hash_v, uniqcnt->tensor, 
                        indice_pairs_uniq, num_out_act_bound, stream_int);
                }}else{{
                    num_act_out = apply_thrust_unique_to_indice_pairs_uniq(indice_pairs_uniq, thrustalloc, stream_int);
                }}
            }}
            // tv::ssprint("HASH SIZE", hash_size, num_act_out);
            if (num_act_out == 0){{
                std::stringstream ss;
                ss << R"(Your points vanished here, this usually because you provide 
conv params that may ignore some input points. Example: 
    spatial_shape=[8, 200, 200]
    ksize=3
    stride=2
    padding=[0, 1, 1]
    dilation=1
    Coordinates=[[0, 7, 153, 142]]
these params will cause ALL points in z == 7 dropped because of padding_z=0.
enlarge your spatial shape or change your conv param to make sure 
every input point has a corresponding output point.
Your Conv Params: )" << "\\n";
                tv::sstream_print<'\\0'>(ss, "    spatial_shape=", input_dims, "\\n");
                tv::sstream_print<'\\0'>(ss, "    ksize=", ksize, "\\n");
                tv::sstream_print<'\\0'>(ss, "    stride=", stride, "\\n");
                tv::sstream_print<'\\0'>(ss, "    padding=", padding, "\\n");
                tv::sstream_print<'\\0'>(ss, "    dilation=", dilation, "\\n");
                tv::ssprint(ss.str());
                throw std::runtime_error(ss.str());
            }}
            if (num_out_act_bound > 0 && num_act_out > num_out_act_bound){{
                num_act_out = num_out_act_bound;
            }}
            indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_act_out);
            // for fixed size allocator, all memory alloc size must be fixed.
            tv::Tensor pair_fwd, pair_mask_fwd, pair_mask_bwd;
            {{
                tv::CUDAKernelTimerGuard timer_guard("alloc_stage2", 
                    timer, reinterpret_cast<cudaStream_t>(stream_int));
                out_inds = allocator.empty({pccm.literal(AllocKeys.OutIndices)}, 
                    {{num_act_out, indices.dim(1)}}, indices.dtype(), 0, stream_int);
                pair_fwd = allocator.full_int({pccm.literal(AllocKeys.PairFwd)}, 
                    {{kv, num_act_out}}, -1, indices.dtype(), indices.device(), stream_int);
                pair_mask_fwd = allocator.zeros({pccm.literal(AllocKeys.PairMask)}, 
                    {{mask_split_count, num_act_out, mask_int_count}}, tv::uint32, 0, stream_int);
                pair_mask_bwd = tv::Tensor();
                if (is_train){{
                    pair_mask_bwd = allocator.zeros({pccm.literal(AllocKeys.PairMaskBwd)}, 
                        {{mask_split_count, indices.dim(0), mask_int_count}}, tv::uint32, 0, stream_int);
                }}
            }}
            if (!direct_table){{
                int hash_size = int(num_act_out * 2);
                if (use_int64_hash_k){{
                    // temp memory don't need to be fixed, static alloc will check
                    // that tensor is large enough.
                    hash_k_guard = allocator.empty_guard({{hash_size}}, 
                        tv::int64, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                    hash_v_gurad = allocator.empty_guard({{hash_size}}, 
                        tv::int32, 0, {pccm.literal(AllocKeys.HashV)});
                    hash_k = hash_k_guard->tensor;
                    hash_v = hash_v_gurad->tensor;
                }}else{{
                    hash_kv_gurad = allocator.empty_guard({{2, hash_size}}, 
                        tv::int32, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                    hash_k = hash_kv_gurad->tensor[0];
                    hash_v = hash_kv_gurad->tensor[1];
                }}
            }}
            {{
                tv::CUDAKernelTimerGuard timer_guard(std::string("gen_conv_inds_stage2_") + std::to_string(num_act_out), 
                    timer, reinterpret_cast<cudaStream_t>(stream_int));
                if (direct_table){{
                    assign_output_direct_hash(indice_pairs_uniq, out_inds, 
                        batch_size, out_shape, 
                        input_dims, ksize, stride, padding, dilation, stream_int);
                    generate_conv_inds_stage2_mask_direct_table(indices, hash_k, hash_v, pair_fwd, pair_bwd,
                        indice_pairs_uniq, indice_pairs_uniq_bkp, 
                        out_inds, pair_mask_fwd, pair_mask_bwd, num_act_out,
                        batch_size, out_shape, input_dims, ksize, stride, padding, dilation,
                        transposed, stream_int);
                }}else{{
                    generate_conv_inds_mask_stage2(indices, hash_k, hash_v, pair_fwd, pair_bwd,
                        indice_pairs_uniq, indice_pairs_uniq_bkp, 
                        out_inds, pair_mask_fwd, pair_mask_bwd, num_act_out,
                        batch_size, out_shape, input_dims, ksize, stride, padding, dilation,
                        transposed, stream_int);
                }}
            }}
            """)
            code.raw(f"""
            auto mask_argsort_fwd = allocator.empty({pccm.literal(AllocKeys.MaskArgSort)}, 
                {{mask_split_count, num_act_out}}, tv::int32, 0, stream_int);
            tv::Tensor mask_argsort_bwd = tv::Tensor();
            if (is_train){{
                mask_argsort_bwd = allocator.zeros({pccm.literal(AllocKeys.MaskArgSortBwd)}, 
                    {{mask_split_count, num_act_in}}, tv::int32, 0, stream_int);
            }}
            {{
                tv::CUDAKernelTimerGuard timer_guard("gen_conv_inds_sort", 
                    timer, reinterpret_cast<cudaStream_t>(stream_int));
                if (is_mask_split){{
                    TV_ASSERT_RT_ERR(do_sort, "not implemented for now");
                    for (int j = 0; j < mask_split_count; ++j){{
                        auto mask_tensor_sub = mask_tensor.slice_first_axis(j, j + 1);
                        if (!is_train){{
                            sort_1d_by_key_split_allocator_v2(pair_mask_fwd[j], thrustalloc, 
                                mask_tensor_sub, mask_argsort_fwd[j], stream_int);
                        }}else{{
                            sort_1d_by_key_split_allocator_v2(pair_mask_fwd[j], thrustalloc, 
                                mask_tensor_sub, mask_argsort_fwd[j], stream_int);
                            sort_1d_by_key_split_allocator_v2(pair_mask_bwd[j], thrustalloc, 
                                mask_tensor_sub, mask_argsort_bwd[j], stream_int);
                        }}
                    }}
                }}else{{
                    if (!is_train){{
                        sort_1d_by_key_allocator_v2(pair_mask_fwd[0], thrustalloc, 
                            mask_argsort_fwd[0], stream_int, mask_int_count, do_sort);
                    }}else{{
                        sort_1d_by_key_allocator_v2(pair_mask_fwd[0], thrustalloc, 
                            mask_argsort_fwd[0], stream_int, mask_int_count, do_sort);
                        sort_1d_by_key_allocator_v2(pair_mask_bwd[0], thrustalloc, 
                            mask_argsort_bwd[0], stream_int, mask_int_count, do_sort);
                    }}
                }}
            }}
            """)
        code.raw(f"""
        return std::make_tuple(mask_tensor, num_act_out);
        """)
        return code.ret("std::tuple<tv::Tensor, int>")

    @pccm.pybind.mark 
    @pccm.static_function
    def get_indice_pairs(self):
        code = pccm.code()
        code.arg("allocator", "ExternalAllocator&")
        code.arg("indices", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"std::vector<int>")
        code.arg("algo", "int")
        code.arg("ksize, stride, padding, dilation, out_padding", f"std::vector<int>")
        code.arg("subm, transposed", f"bool")
        code.arg("stream_int", f"std::uintptr_t", "0")
        code.arg("num_out_act_bound", f"int", "-1")
        code.arg("num_input_act_bound", f"int", "-1")

        
        code.raw(f"""
        int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
        auto conv_algo = static_cast<tv::gemm::SparseConvAlgo>(algo);
        TV_ASSERT_RT_ERR(conv_algo == tv::gemm::SparseConvAlgo::kNative, "only support kNative");
        if (num_out_act_bound > 0){{
            TV_ASSERT_RT_ERR(num_input_act_bound > 0 && indices.dim(0) <= num_input_act_bound, 
                "out bound and input bound must both larger than zero");
        }}

        std::vector<int> out_shape;
        if (!subm){{
            if (transposed){{
                out_shape = get_deconv_output_size(input_dims, ksize, stride, padding, dilation, out_padding);
            }}else{{
                out_shape = get_conv_output_size(input_dims, ksize, stride, padding, dilation);
            }}
        }}else{{
            out_shape = input_dims;
        }}
        for (auto& v : out_shape){{
            if (v <= 0){{
                TV_THROW_RT_ERR("your out spatial shape", out_shape, "ratch zero!, input shape:", input_dims);
            }}
        }}
        std::vector<int64_t> output_dims_i64(out_shape.begin(), out_shape.end());
        int64_t out_spatial_volume = std::accumulate(output_dims_i64.begin(),
          output_dims_i64.end(), int64_t(1), std::multiplies<int64_t>()) * batch_size;
        bool use_int64_hash_k = out_spatial_volume >= int64_t(std::numeric_limits<int>::max());
        tv::DType indice_uniq_dtype = use_int64_hash_k ? tv::int64 : tv::int32;

        tv::Tensor pair;
        int64_t num_act_in_bounded = indices.dim(0);

        if (num_out_act_bound > 0){{
            // we need stable pair stride for bounded output
            num_act_in_bounded = num_input_act_bound;
        }}
        pair = allocator.full_int({pccm.literal(AllocKeys.PairFwd)}, 
            {{2, kv, num_act_in_bounded}}, -1, indices.dtype(), indices.device(), stream_int);
        
        auto indice_num_per_loc = allocator.zeros({pccm.literal(AllocKeys.IndiceNumPerLoc)}, 
            {{kv}}, indices.dtype(), indices.device(), stream_int);
        tv::Tensor out_inds;
        int num_act_out = -1;
        """)
        with code.if_("subm"):
            code.raw(f"""
            num_act_out = indices.dim(0);
            if (indices.is_cpu()){{
                generate_subm_conv_inds_cpu(indices, pair, out_inds, indice_num_per_loc,
                    batch_size, input_dims, ksize, dilation);
            }}
            """)
            if not CUMM_CPU_ONLY_BUILD:
                code.raw(f"""
                else {{
                    ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
                    out_inds = indices;
                    int num_points = out_inds.dim(0);
                    tv::Tensor hash_k, hash_v;
                    if (use_int64_hash_k){{
                        hash_k_guard = allocator.empty_guard({{num_points * 2}}, 
                            tv::int64, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                        hash_v_gurad = allocator.empty_guard({{num_points * 2}}, 
                            tv::int32, 0, {pccm.literal(AllocKeys.HashV)});
                        hash_k = hash_k_guard->tensor;
                        hash_v = hash_v_gurad->tensor;
                    }}else{{
                        hash_kv_gurad = allocator.empty_guard({{2, num_points * 2}}, 
                            tv::int32, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                        hash_k = hash_kv_gurad->tensor[0];
                        hash_v = hash_kv_gurad->tensor[1];
                    }}
                    generate_subm_conv_inds(indices, hash_k, hash_v, pair, out_inds, indice_num_per_loc,
                        batch_size, input_dims, ksize, dilation, tv::Tensor(), false, stream_int);
                }}
                """)
            else:
                code.raw(f"""
                else {{
                    TV_THROW_RT_ERR("not implemented for CPU ONLY build.")
                }}
                """)
        with code.else_():
            code.raw(f"""
            if (indices.is_cpu()){{
                TV_ASSERT_RT_ERR(num_out_act_bound <= 0, "cpu algo don't support out bound")
                out_inds = allocator.empty({pccm.literal(AllocKeys.OutIndices)}, 
                    {{kv * indices.dim(0), indices.dim(1)}}, indices.dtype(), -1);
                num_act_out = generate_conv_inds_cpu(indices, pair, out_inds, indice_num_per_loc,
                    batch_size, out_shape, input_dims, ksize, 
                    stride, padding, dilation, transposed);
            }}
            """)
            if not CUMM_CPU_ONLY_BUILD:
                code.raw(f"""
                else {{
                    ThrustAllocator thrustalloc(allocator);

                    auto tvctx = tv::Context();
                    tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));

                    auto indice_pairs_uniq_guard = allocator.empty_guard(
                        {{int64_t(pair.numel() / 2 + 1)}}, indice_uniq_dtype, 0, 
                        {pccm.literal(AllocKeys.IndicePairsUniq)});
                    auto indice_pairs_uniq = indice_pairs_uniq_guard->tensor;
                    auto indice_pairs_uniq_bkp_guard = allocator.empty_guard(
                        {{int64_t(pair.numel() / 2 + 1)}}, indice_uniq_dtype, 0,
                        {pccm.literal(AllocKeys.IndicePairsUniqBackup)});

                    generate_conv_inds_stage1(indices, pair, indice_pairs_uniq,
                        indice_num_per_loc, batch_size, out_shape, input_dims, ksize,
                        stride, padding, dilation, transposed, stream_int);
                    indice_pairs_uniq_bkp_guard->tensor.copy_(indice_pairs_uniq, tvctx);

                    // TODO pytorch unique may be faster?
                    num_act_out = apply_thrust_unique_to_indice_pairs_uniq(indice_pairs_uniq, thrustalloc, stream_int);
                    if (num_act_out == 0){{
                        std::stringstream ss;
                        ss << R"(Your points vanished here, this usually because you provide 
conv params that may ignore some input points. Example: 
    spatial_shape=[8, 200, 200]
    ksize=3
    stride=2
    padding=[0, 1, 1]
    dilation=1
    Coordinates=[[0, 7, 153, 142]]
these params will cause ALL points in z == 7 dropped because of padding_z=0.
enlarge your spatial shape or change your conv param to make sure 
every input point has a corresponding output point.
Your Conv Params: )" << "\\n";
                        tv::sstream_print<'\\0'>(ss, "    spatial_shape=", input_dims, "\\n");
                        tv::sstream_print<'\\0'>(ss, "    ksize=", ksize, "\\n");
                        tv::sstream_print<'\\0'>(ss, "    stride=", stride, "\\n");
                        tv::sstream_print<'\\0'>(ss, "    padding=", padding, "\\n");
                        tv::sstream_print<'\\0'>(ss, "    dilation=", dilation, "\\n");
                        tv::ssprint(ss.str());
                        throw std::runtime_error(ss.str());
                    }}

                    bool use_bound_algo = false;
                    int64_t num_out_bounded = num_act_out;
                    if (num_out_act_bound > 0 && num_act_out > num_out_act_bound){{
                        num_act_out = num_out_act_bound;
                        use_bound_algo = true;
                    }}
                    if (num_out_act_bound > 0 ){{
                        num_out_bounded = num_out_act_bound;
                    }}

                    indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_act_out);
                    out_inds = allocator.empty({pccm.literal(AllocKeys.OutIndices)}, 
                        {{num_out_bounded, indices.dim(1)}}, indices.dtype(), 0, stream_int);
                    ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
                    tv::Tensor hash_k, hash_v;
                    if (use_int64_hash_k){{
                        hash_k_guard = allocator.empty_guard({{num_act_out * 2}}, 
                            tv::int64, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                        hash_v_gurad = allocator.empty_guard({{num_act_out * 2}}, 
                            tv::int32, 0, {pccm.literal(AllocKeys.HashV)});
                        hash_k = hash_k_guard->tensor;
                        hash_v = hash_v_gurad->tensor;
                    }}else{{
                        hash_kv_gurad = allocator.empty_guard({{2, num_act_out * 2}}, 
                            tv::int32, 0, {pccm.literal(AllocKeys.HashKOrKV)});
                        hash_k = hash_kv_gurad->tensor[0];
                        hash_v = hash_kv_gurad->tensor[1];
                    }}
                    num_act_out = generate_conv_inds_stage2(indices, hash_k, hash_v, pair,
                        indice_pairs_uniq, indice_pairs_uniq_bkp_guard->tensor, 
                        out_inds, indice_num_per_loc, num_act_out,
                        batch_size, out_shape, input_dims, ksize, stride, padding, dilation,
                        transposed, stream_int, use_bound_algo);
                }}
                """)
            else:
                code.raw(f"""
                else {{
                    TV_THROW_RT_ERR("not implemented for CPU ONLY build.")
                }}
                """)
        code.raw(f"""
        return num_act_out;
        """)
        return code.ret("int")
