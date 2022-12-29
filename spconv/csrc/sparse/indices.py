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
from cumm.common import TensorView, TensorViewHashKernel, TensorViewKernel, ThrustLib
from cumm.gemm import codeops
from typing import List
from cumm.conv.params import ConvProblem
import numpy as np


class CudaCommonKernel(pccm.ParameterizedClass):
    # we need to use PClass instead of Class
    # because cuda global function can't be put in class body.
    def __init__(self) -> None:
        super().__init__()
        self.add_include("tensorview/cuda/launch.h")
        self.add_include("tensorview/cuda/kernel_utils.h")

    @pccm.cuda.cuda_global_function
    def arange_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.arg("data", f"T*")
        code.arg("size", f"int")
        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(size)) {{
            data[i] = T(i);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def fill_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.arg("data", f"T*")
        code.arg("val", f"T")
        code.arg("size", f"int")
        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(size)) {{
            data[i] = T(val);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def maximum_value_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.arg("data", f"T*")
        code.arg("val", f"T")
        code.arg("size", f"int")
        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(size)) {{
            data[i] = max(data[i], val);
        }}
        """)
        return code


class ConvOutLocIter(pccm.ParameterizedClass):

    def __init__(self, problem: ConvProblem, use_i64: bool = False):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_param_class("lociter", problem, "ConvProblem")
        if use_i64:
            layout_npq = TensorGeneric(problem.ndim + 1, False, dtypes.int64)
        else:
            layout_npq = TensorGeneric(problem.ndim + 1, False)
        layout_rs = TensorGeneric(problem.ndim, False)

        self.add_param_class("lociter", layout_npq, "LayoutNPQ")
        self.add_param_class("lociter_rs", layout_rs, "LayoutRS")

        self.ndim = problem.ndim
        self.add_member("problem_", f"ConvProblem")
        self.add_member("count_", f"tv::array<int, {self.ndim}>")
        self.add_member("layout_npq", f"LayoutNPQ")
        self.add_member("layout_rs", f"LayoutRS")

    @pccm.constructor(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def ctor(self):
        code = pccm.FunctionCode()
        code.arg("problem", f"ConvProblem const&")
        code.ctor_init("problem_", f"problem")
        zeros = ", ".join(["0"] * self.ndim)
        code.ctor_init("count_", f"{{{zeros}}}")
        pqs = codeops.unpack("problem.output_dims", range(self.ndim))
        rss = codeops.unpack("problem.ksize", range(self.ndim))

        code.ctor_init("layout_npq",
                       f"LayoutNPQ::from_shape({{problem.N, {pqs}}})")
        code.ctor_init("layout_rs", f"LayoutRS::from_shape({{{rss}}})")

        return code

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          name="operator++")
    def increment(self):
        code = pccm.FunctionCode()
        for i in range(self.ndim - 1, -1, -1):
            code.raw(f"""
            if (++count_[{i}] < problem_.ksize[{i}]){{
                return *this;
            }}
            count_[{i}] = 0;
            """)
        code.raw("return *this;")
        return code.ret(f"{self.class_name}&")

    @pccm.member_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def set_filter_offset(self):
        code = pccm.FunctionCode()
        code.arg("filter_offset", "int")
        code.raw(f"""
        layout_rs.inverse(filter_offset, count_);
        """)
        return code

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def nhw_to_npq(self):
        code = pccm.FunctionCode()
        code.arg("nhw_offset", "const int*")
        code.nontype_targ("NoStride", "bool")
        for i in range(self.ndim):
            code.raw(f"""
            int r_{i} = count_[{i}];
            int h_{i} = (nhw_offset[{i + 1}] + problem_.padding[{i}] - 
                r_{i} * problem_.dilation[{i}]) / (NoStride ? 1 : problem_.stride[{i}]);
            """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{nhw_offset[0], {h0h1h2}}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 1}>")

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def npq_to_nhw(self):
        code = pccm.FunctionCode()
        code.arg("npq_offset", "const int*")
        for i in range(self.ndim):
            code.raw(f"""
            int r_{i} = count_[{i}];
            int h_{i} = npq_offset[{i + 1}] * problem_.stride[{i}] - problem_.padding[{i}] + r_{i} * problem_.dilation[{i}];
            """)
        h0h1h2 = codeops.unpack_str("h", range(self.ndim))
        code.raw(f"""
        return {{npq_offset[0], {h0h1h2}}};
        """)
        return code.ret(f"tv::array<int, {self.ndim + 1}>")

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def query_npq(self):
        code = pccm.FunctionCode()
        code.arg("nhw_offset", "const int*")
        code.arg("npq_offset", f"tv::array<int, {self.ndim + 1}>&")
        code.ret("bool")
        code.raw(f"""
        auto npq_no_stride = nhw_to_npq<true>(nhw_offset);
        npq_offset[0] = npq_no_stride[0];
        """)
        hw_valid = []  # type: List[str]
        stride_valid = []  # type: List[str]
        for i in range(self.ndim):
            code.raw(
                f"npq_offset[{i + 1}] = npq_no_stride[{i + 1}] / problem_.stride[{i}];"
            )
            hw_valid.append(
                (f"npq_offset[{i + 1}] >= 0 && "
                 f"npq_offset[{i + 1}] < problem_.output_dims[{i}]"))
            stride_valid.append(
                f"!(npq_no_stride[{i + 1}] % problem_.stride[{i}])")
        code.raw(f"""
        return npq_no_stride[0] < problem_.N && 
            {' && '.join(hw_valid)} &&
            {' && '.join(stride_valid)};
        """)
        return code

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def query_npq_no_stride(self):
        code = pccm.FunctionCode()
        code.arg("nhw_offset", "const int*")
        code.arg("npq_offset", f"tv::array<int, {self.ndim + 1}>&")
        code.ret("bool")
        code.raw(f"""
        npq_offset = nhw_to_npq<true>(nhw_offset);
        """)
        hw_valid = []  # type: List[str]
        for i in range(self.ndim):
            hw_valid.append(
                (f"npq_offset[{i + 1}] >= 0 && "
                 f"npq_offset[{i + 1}] < problem_.output_dims[{i}]"))
        code.raw(f"""
        return npq_offset[0] < problem_.N && 
            {' && '.join(hw_valid)};
        """)
        return code

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def query_nhw(self):
        code = pccm.FunctionCode()
        code.arg("npq_offset", "const int*")
        code.arg("nhw_offset", f"tv::array<int, {self.ndim + 1}>&")
        code.ret("bool")
        code.raw(f"""
        nhw_offset = npq_to_nhw(npq_offset);
        """)
        hw_valid = []  # type: List[str]
        for i in range(self.ndim):
            hw_valid.append(
                (f"nhw_offset[{i + 1}] >= 0 && "
                 f"nhw_offset[{i + 1}] < problem_.input_dims[{i}]"))
        code.raw(f"""
        return nhw_offset[0] < problem_.N && 
            {' && '.join(hw_valid)};
        """)
        return code

    @pccm.member_function(header_only=True,
                          attrs=["TV_HOST_DEVICE_INLINE"],
                          const=True)
    def query_nhw_out(self):
        code = pccm.FunctionCode()
        code.arg("npq_offset", "const int*")
        code.arg("nhw_offset", f"tv::array<int, {self.ndim + 1}>&")
        code.ret("bool")
        code.raw(f"""
        nhw_offset = npq_to_nhw(npq_offset);
        """)
        hw_valid = []  # type: List[str]
        for i in range(self.ndim):
            hw_valid.append(
                (f"nhw_offset[{i + 1}] >= 0 && "
                 f"nhw_offset[{i + 1}] < problem_.output_dims[{i}]"))
        code.raw(f"""
        return nhw_offset[0] < problem_.N && 
            {' && '.join(hw_valid)};
        """)
        return code


class SparseConvIndicesKernel(pccm.ParameterizedClass):

    def __init__(self, problem: ConvProblem, dtype_indices: dtypes.DType):
        super().__init__()
        self.add_dependency(TensorView, TensorViewKernel, TensorViewHashKernel)
        self.loc_iter = ConvOutLocIter(problem)
        self.loc_iter_64 = ConvOutLocIter(problem, True)

        self.add_param_class("spinds", self.loc_iter, "ConvLocIter")
        self.add_param_class("spinds64", self.loc_iter_64, "ConvLocIter64")
        self.add_param_class("spinds", problem, "ConvProblem")
        self.add_param_class("cudakers", CudaCommonKernel())
        self.add_include("tensorview/hash/ops.h")
        self.ndim = problem.ndim
        self.dtype_indices = dtype_indices
        self.dtype_indices_uniq = dtype_indices

        assert dtype_indices == dtypes.int32 or dtype_indices == dtypes.int64

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage1(self):
        code = pccm.FunctionCode()
        code.targ("TIndiceUniq")
        code.targ("TConvLocIter")
        code.arg("loc_iter", f"TConvLocIter")  # [N, ndim + 1]
        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("indice_pairs_for_uniq",
                 f"TIndiceUniq*")  # [kernelProd * MaxSize + 1]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]

        code.arg("num_indices_in", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
        code.arg("transposed", "bool")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        loc_iter.set_filter_offset(filter_offset);
        // int indices_pair_size_mul_RS = indices_pair_size * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
        for (int i : tv::KernelLoopX<int>(num_indices_in)) {{
            tv::array<int, {self.ndim + 1}> npq_offset;
            bool valid;
            if (transposed){{
                valid = loc_iter.query_nhw_out(indices_in + i * {self.ndim + 1}, npq_offset);
            }}else{{
                valid = loc_iter.query_npq(indices_in + i * {self.ndim + 1}, npq_offset);
            }}
            if (valid){{
                int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                int64_t offset = loc_iter.layout_npq(npq_offset);
                if (old_num < indices_pair_size){{
                    indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                    // indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = offset;
                    indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = offset;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def build_conv_hash_table(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.targ("TLayoutNPQ")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indices_out", f"int*")  # [N, ndim + 1]
        code.arg(
            "indice_pairs_for_uniq",
            f"const typename TTable::key_type*")  # [2, kernelProd, MaxSize]
        code.arg("layout_npq", f"TLayoutNPQ")  # [N, ndim + 1]

        code.arg("num_indices", "int")

        code.raw(f"""
        for (int output_index : tv::KernelLoopX<int>(num_indices)) {{
            auto output_coord_offset = indice_pairs_for_uniq[output_index];
            layout_npq.inverse(output_coord_offset, indices_out + {self.ndim + 1} * output_index);
            table.insert(output_coord_offset, output_index);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def arange_hash_table_and_assign_out(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.targ("TLayoutNPQ")
        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indices_out", f"int*")  # [N, ndim + 1]
        code.arg("count", f"int*")  # [N, ndim + 1]
        code.arg("limit", f"int")  # [N, ndim + 1]
        code.arg("layout_npq", f"TLayoutNPQ")  # [N, ndim + 1]

        code.raw(f"""
        
        auto key_ptr = table.key_ptr();
        auto value_ptr = table.value_ptr();

        for (auto i : tv::KernelLoopX<int>(table.size())) {{
            auto output_coord_offset = key_ptr[i];
            if (output_coord_offset != TTable::empty_key) {{
                auto output_index = tv::cuda::atomicAggInc(count);
                if (output_index < limit){{
                    value_ptr[i] = output_index;
                    layout_npq.inverse(output_coord_offset, indices_out + {self.ndim + 1} * output_index);
                }}else{{
                    value_ptr[i] = -1;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def arange_hash_table(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("out_indices_offset",
                 f"typename TTable::key_type *")  # [N, ndim + 1]

        code.arg("count", f"int*")  # [N, ndim + 1]
        code.arg("limit", f"int")  # [N, ndim + 1]
        code.raw(f"""
        
        auto key_ptr = table.key_ptr();
        auto value_ptr = table.value_ptr();

        for (auto i : tv::KernelLoopX<int>(table.size())) {{
            auto output_coord_offset = key_ptr[i];
            if (output_coord_offset != TTable::empty_key) {{
                auto output_index = tv::cuda::atomicAggInc(count);
                value_ptr[i] = output_index < limit ? output_index : -1;
                out_indices_offset[output_index] = output_coord_offset;
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def assign_out_indices(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.targ("TLayoutNPQ")
        code.arg("indices_out", f"int*")  # [N, ndim + 1]
        code.arg("out_indices_offset", f"const T*")  # [N, ndim + 1]
        code.arg("layout_npq", f"TLayoutNPQ")  # [N, ndim + 1]
        code.arg("size", f"int")  # [N, ndim + 1]
        code.raw(f"""
        for (auto i : tv::KernelLoopX<int>(size)) {{
            layout_npq.inverse(out_indices_offset[i], indices_out + {self.ndim + 1} * i);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_uniq_before_sort",
                 f"const typename TTable::key_type*")  # [kernelProd, MaxSize]
        code.arg("indice_pairs_out_part", f"int*")  # [kernelProd, MaxSize]
        code.arg("num_indices_in", "int")
        code.arg("indices_pair_size", "int")
        code.raw(f"""
        int filter_offset = blockIdx.y;
        auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
        auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * indices_pair_size;
        for (int i : tv::KernelLoopX<int>(num_indices_in)) {{
            {self.dtype_indices} output_coord_offset = indice_pairs_uniq_before_sort_filter[i];
            if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){{
                auto table_offset = table.lookup_offset(output_coord_offset);
                if (table_offset != -1){{
                    indice_pairs_out_part_filter[i] = table.value_ptr()[table_offset];
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2_bounded(self):
        """if we bound output indices, some pair may be invalid,
        so we need to atomicAdd and assign again.
        here we will use indice_pairs_uniq as temp memory of 
        indice_pairs_in_part.
        """
        code = pccm.FunctionCode()
        code.targ("TTable")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_uniq_before_sort",
                 f"const typename TTable::key_type*")  # [kernelProd, MaxSize]
        code.arg("indice_pairs_in_part_temp",
                 f"const int*")  # [kernelProd, MaxSize]
        code.arg("indice_pairs_in_part", f"int*")  # [kernelProd, MaxSize]
        code.arg("indice_pairs_out_part", f"int*")  # [kernelProd, MaxSize]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]
        code.arg("num_indices_in", "int")
        code.arg("indices_pair_size", "int")
        code.raw(f"""
        int filter_offset = blockIdx.y;
        auto indice_pairs_in_part_filter = indice_pairs_in_part + filter_offset * indices_pair_size;
        auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;

        auto indice_pairs_in_part_temp_filter = indice_pairs_in_part_temp + filter_offset * indices_pair_size;
        auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * indices_pair_size;

        for (int i : tv::KernelLoopX<int>(num_indices_in)) {{
            {self.dtype_indices} output_coord_offset = indice_pairs_uniq_before_sort_filter[i];
            if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){{
                auto table_offset = table.lookup_offset(output_coord_offset);
                if (table_offset != -1){{
                    int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                    indice_pairs_in_part_filter[old_num] = indice_pairs_in_part_temp_filter[i];
                    indice_pairs_out_part_filter[old_num] = table.value_ptr()[table_offset];
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage1_mask(self):
        code = pccm.FunctionCode()
        code.targ("TIndiceUniq")
        code.targ("TConvLocIter")

        code.arg("loc_iter", f"TConvLocIter")  # [N, ndim + 1]
        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs_bwd",
                 f"{self.dtype_indices}*")  # [kernelProd, MaxSize]
        code.arg("indice_pairs_for_uniq",
                 f"TIndiceUniq*")  # [kernelProd * MaxSize + 1]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]

        code.arg("num_indices_in", "int")

        code.arg("RS", "int")
        code.arg("transposed", "bool")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        loc_iter.set_filter_offset(filter_offset);
        // int indices_pair_size_mul_RS = num_indices_in * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            tv::array<int, {self.ndim + 1}> npq_offset;
            bool valid;
            if (transposed){{
                valid = loc_iter.query_nhw_out(indices_in + input_index * {self.ndim + 1}, npq_offset);
            }}else{{
                valid = loc_iter.query_npq(indices_in + input_index * {self.ndim + 1}, npq_offset);
            }}
            if (valid){{
                // int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                TIndiceUniq output_coord_offset = loc_iter.layout_npq(npq_offset);
                // if (old_num < indices_pair_size){{
                // indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                // indice_pairs_bwd[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
                // indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = output_coord_offset;
                
                indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
                // }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage1_mask_direct_table(self):
        code = pccm.FunctionCode()
        code.targ("TIndiceUniq")
        code.targ("TTable")
        code.targ("TConvLocIter")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("loc_iter", f"TConvLocIter")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs_bwd",
                 f"{self.dtype_indices}*")  # [kernelProd, MaxSize]
        code.arg("indice_pairs_for_uniq",
                 f"TIndiceUniq*")  # [kernelProd * MaxSize + 1]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]

        code.arg("num_indices_in", "int")

        code.arg("RS", "int")
        code.arg("transposed", "bool")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        loc_iter.set_filter_offset(filter_offset);
        // int indices_pair_size_mul_RS = num_indices_in * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            tv::array<int, {self.ndim + 1}> npq_offset;
            bool valid;
            if (transposed){{
                valid = loc_iter.query_nhw_out(indices_in + input_index * {self.ndim + 1}, npq_offset);
            }}else{{
                valid = loc_iter.query_npq(indices_in + input_index * {self.ndim + 1}, npq_offset);
            }}
            if (valid){{
                // int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                TIndiceUniq output_coord_offset = loc_iter.layout_npq(npq_offset);
                // if (old_num < indices_pair_size){{
                // indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                // indice_pairs_bwd[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
                // indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = output_coord_offset;
                
                table.insert_key_only(output_coord_offset);
                indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
                // }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2_mask(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.nontype_targ("CheckValueValid", "bool")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_fwd",
                 f"int*")  # [kernelProd, MaxSize], inp -> out
        code.arg("indice_pairs_bwd",
                 f"int*")  # [kernelProd, MaxSize], out -> inp
        code.arg("indice_pairs_uniq_before_sort",
                 f"const typename TTable::key_type*")  # [kernelProd, MaxSize]
        code.arg("mask_fwd", f"uint32_t*")  # [kernelProd]
        code.arg("mask_bwd", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices_in", "int")
        code.arg("num_indices_out", "int")
        code.arg("mask_int_count", "int")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        int filter_pointer_offset = filter_offset / 32;
        uint32_t filter_mask_fwd = (1u << (filter_offset % 32));
        // TODO following rule for even kernel size is wrong. 
        // uint32_t filter_mask_bwd = (1u << (gridDim.y - 1 - filter_offset));

        auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
        auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
        auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;

        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
           auto output_coord_offset = indice_pairs_uniq_before_sort_filter[input_index];
            if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){{
                
                auto table_offset = table.lookup_offset(output_coord_offset);
                if (table_offset != -1){{
                    auto output_index = table.value_ptr()[table_offset];
                    bool valid = CheckValueValid ? output_index >= 0 : true;
                    if (valid){{
                        atomicOr(mask_fwd + output_index * mask_int_count + filter_pointer_offset, filter_mask_fwd);
                        // atomicOr(mask_bwd + input_index, filter_mask_bwd);
                        indice_pairs_fwd_filter[output_index] = input_index;
                        if (indice_pairs_bwd != nullptr){{
                            indice_pairs_bwd_filter[input_index] = output_index;
                        }}
                    }}
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2_mask_output(self):
        code = pccm.FunctionCode()
        code.arg("indice_pairs_bwd",
                 f"int*")  # [kernelProd, MaxSize], out -> inp
        code.arg("mask_bwd", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices_in", "int")
        code.arg("kv", "int")
        code.arg("mask_int_count", "int")

        code.raw(f"""
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            for (int mask_offset = 0; mask_offset < mask_int_count; ++mask_offset){{
                uint32_t mask = 0;
                for (int filter_offset = mask_offset * 32; filter_offset < mask_offset * 32 +  32 && filter_offset < kv; ++filter_offset){{
                    auto val = indice_pairs_bwd[filter_offset * num_indices_in + input_index];
                    mask |= (val != -1) << (filter_offset % 32);
                }}
                mask_bwd[input_index * mask_int_count + mask_offset] = mask;
            }}

        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2_inference_mask(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.nontype_targ("CheckValueValid", "bool")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_fwd",
                 f"int*")  # [kernelProd, MaxSize], inp -> out
        code.arg("indice_pairs_bwd",
                 f"int*")  # [kernelProd, MaxSize], out -> inp
        code.arg("indice_pairs_uniq_before_sort",
                 f"const typename TTable::key_type*")  # [kernelProd, MaxSize]

        code.arg("mask_fwd", f"uint32_t*")  # [kernelProd]
        code.arg("num_indices_in", "int")
        code.arg("num_indices_out", "int")
        code.arg("mask_int_count", "int")

        # TODO use block instead of filter_offset?
        code.raw(f"""
        int filter_offset = blockIdx.y;
        int filter_pointer_offset = filter_offset / 32;
        uint32_t filter_mask_fwd = (1u << (filter_offset % 32));

        auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
        // auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
        auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            auto output_coord_offset = indice_pairs_uniq_before_sort_filter[input_index];
            if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){{
                auto table_offset = table.lookup_offset(output_coord_offset);
                if (table_offset != -1){{
                    auto output_index = table.value_ptr()[table_offset];
                    bool valid = CheckValueValid ? output_index >= 0 : true;
                    if (valid){{
                        atomicOr(mask_fwd + output_index * mask_int_count + filter_pointer_offset, filter_mask_fwd);
                        indice_pairs_fwd_filter[output_index] = input_index;
                    }}
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def build_subm_conv_hash_table(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.targ("TLayoutNPQ")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indices_in", f"const int*")  # [N, ndim + 1]

        code.arg("layout_npq", f"TLayoutNPQ")

        code.arg("num_indices", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(num_indices)) {{
            table.insert(layout_npq(indices_in + i * {self.ndim + 1}), i);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def clean_indices_uniq(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.arg("indice_pairs_for_uniq", f"T*")
        code.arg("size", f"size_t")
        code.raw(f"""
        for (size_t i : tv::KernelLoopX<size_t>(size)) {{
            indice_pairs_for_uniq[i] = std::numeric_limits<T>::max();
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_subm_conv_indices(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.targ("TConvLocIter")
        code.arg("loc_iter", f"TConvLocIter")  # [N, ndim + 1]
        code.arg("table", f"TTable")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]

        code.arg("num_indices_in", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
        code.raw(f"""
        int filter_offset = blockIdx.y;
        loc_iter.set_filter_offset(filter_offset);
        int indices_pair_size_mul_RS = indices_pair_size * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;

        int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
        if (filter_offset == (RS / 2)){{
            for (int i : tv::KernelLoopX<int>(num_indices_in)) {{
                indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
                indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
            }}
        }} else {{
            for (int i : tv::KernelLoopX<int>(num_indices_in)) {{
                tv::array<int, {self.ndim + 1}> npq_offset;
                if (loc_iter.query_npq_no_stride(indices_in + i * {self.ndim + 1}, npq_offset)){{
                    auto offset = loc_iter.layout_npq(npq_offset);
                    // auto item = table.lookup(offset); // performance bound
                    auto table_offset = table.lookup_offset(offset); // performance bound
                    if (table_offset != -1){{
                        auto v = table.value_ptr()[table_offset];
                        int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                        indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                        indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = v;
                        indice_pairs[filter_offset_mul_indices_pair_size_1 + old_num] = v;
                        indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
                    }}
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_subm_conv_indices_mask(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.targ("TConvLocIter")
        code.arg("loc_iter", f"TConvLocIter")  # [N, ndim + 1]
        code.arg("table", f"TTable")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("mask", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
        code.arg("is_train", "bool")
        code.arg("mask_int_count", "int", "1")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        uint32_t filter_mask_out = (1u << (filter_offset % 32));
        uint32_t filter_mask_out_offset = filter_offset / 32;
        uint32_t filter_mask_in = (1u << ((RS - 1 - filter_offset) % 32));
        uint32_t filter_mask_in_offset = (RS - 1 - filter_offset) / 32;
        // uint32_t filter_mask_center = (1u << (RS / 2));
        loc_iter.set_filter_offset(filter_offset);
        int indices_pair_size_mul_RS = indices_pair_size * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;

        int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
        if (filter_offset == (RS / 2)){{
            for (int i : tv::KernelLoopX<int>(num_indices)) {{
                // atomicOr(mask + i, filter_mask_center);
                indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
                if (is_train){{
                    indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
                }}
            }}
        }} else {{
            for (int output_index : tv::KernelLoopX<int>(num_indices)) {{
                // find input offset from output offset
                tv::array<int, {self.ndim + 1}> nhw_offset;
                // table: input indice coord to output index (or output indice coord to input index)
                if (loc_iter.query_nhw(indices_in + output_index * {self.ndim + 1}, nhw_offset)){{
                    auto offset = loc_iter.layout_npq(nhw_offset);
                    // auto item = table.lookup(offset);
                    auto table_offset = table.lookup_offset(offset); // performance bound
                    if (table_offset != -1){{
                        auto input_index = table.value_ptr()[table_offset]; // we find a input indice idx.
                        atomicOr(mask + output_index * mask_int_count + filter_mask_out_offset, filter_mask_out);
                        atomicOr(mask + input_index * mask_int_count + filter_mask_in_offset, filter_mask_in);
                        // for this output, we set correct input idx.
                        indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                        if (is_train){{
                            indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + input_index] = output_index;
                        }}
                        // the output in "input location" connect this output idx in another location.
                        indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                        if (is_train){{
                            indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
                        }}
                    }}
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_subm_conv_indices_split_mask(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.targ("TConvLocIter")
        code.arg("loc_iter", f"TConvLocIter")  # [N, ndim + 1]

        code.arg("table", f"TTable")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("mask1", f"uint32_t*")  # [kernelProd]
        code.arg("mask2", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
        code.arg("is_train", "bool")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        uint32_t filter_mask_out = (1u << (filter_offset));
        uint32_t filter_mask_in = (1u << (RS - 1 - filter_offset));
        // uint32_t filter_mask_center = (1u << (RS / 2));

        loc_iter.set_filter_offset(filter_offset);
        auto indice_ptr_inv = indice_pairs + indices_pair_size * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
        int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
        if (filter_offset == (RS / 2)){{
            for (int i : tv::KernelLoopX<int>(num_indices)) {{
                indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
                if (is_train){{
                    indice_ptr_inv[filter_offset_mul_indices_pair_size + i] = i;
                }}
            }}
        }} else {{
            for (int output_index : tv::KernelLoopX<int>(num_indices)) {{
                // find input offset from output offset
                tv::array<int, {self.ndim + 1}> nhw_offset;
                // table: input indice coord to output index (or output indice coord to input index)
                if (loc_iter.query_nhw(indices_in + output_index * {self.ndim + 1}, nhw_offset)){{
                    auto offset = loc_iter.layout_npq(nhw_offset);
                    auto table_offset = table.lookup_offset(offset); // performance bound
                    if (table_offset != -1){{
                        auto input_index = table.value_ptr()[table_offset]; // we find a input indice idx.
                        atomicOr(mask1 + output_index, filter_mask_out);
                        atomicOr(mask2 + input_index, filter_mask_in);
                        // for this output, we set correct input idx.
                        indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                        // the output in "input location" connect this output idx in another location.
                        indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                        if (is_train){{
                            indice_ptr_inv[filter_offset_mul_indices_pair_size + input_index] = output_index;
                            indice_ptr_inv[filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
                        }}
                    }}
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.static_function
    def generate_conv_inds_stage1(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, indice_pairs_uniq, indice_num_per_loc",
                 "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.arg("transposed", f"bool", "false")

        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        // TODO stream
        // TODO handle num input == 0
        int kv = ksize.op<tv::arrayops::prod>();
        
        TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");

        // indice_pairs: [2, kv, num_act_in]
        // indice_pairs_uniq: [num_act_in * kv + 1]
        tv::check_shape(indice_pairs, {{2, kv, -1}});

        // TV_ASSERT_RT_ERR(indice_pairs.dim(-1) == indices.dim(0), "error");

        tv::check_shape(indice_num_per_loc, {{kv}});
        int64_t uniq_size = indice_pairs.size() / 2 + 1;
        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= uniq_size, "error");
        TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");

        tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
        // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();
        tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
        """)
        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);
            tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){{
                using T = TV_DECLTYPE(I);
                TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<T>::max(), 
                    "kernel volume must smaller than max value of T");
                launcher_clean_uniq(clean_indices_uniq<T>, indice_pairs_uniq.data_ptr<T>(), uniq_size);
                launcher_num_act_in(calc_conv_indices_stage1<T, {loc_type}>, loc_iter, indices.data_ptr<const int>(), 
                    indice_pairs.data_ptr<{self.dtype_indices}>(), 
                    indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
                    indice_pairs.dim(2), kv, transposed);
            }});
            """)
        return code  # .ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_stage1_5(self):
        code = pccm.FunctionCode()
        code.add_dependency(ThrustLib)
        code.arg("indice_pairs_uniq", "tv::Tensor")
        code.arg("uniq_size", "int64_t")
        code.arg("stream_int", f"std::uintptr_t", "0")
        code.raw(f"""
        int num_out_act = 0;
        tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            thrust::device_ptr<T> ptr_tr(indice_pairs_uniq.data_ptr<T>());
            auto thrust_ctx = thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(stream_int));
            thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
            auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
            num_out_act = new_end - ptr_tr - 1;
        }});
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_stage2(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg(
            "indice_pairs, indice_pairs_uniq, indice_pairs_uniq_before_sort, out_inds",
            "tv::Tensor")
        code.arg("indice_num_per_loc", "tv::Tensor")

        code.arg("num_out_act", "int")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.arg("transposed", f"bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0")
        code.arg("use_bound_algo", "bool", "false")

        code.raw(f"""
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        // use_bound_algo = true;
        // TODO stream
        // TODO handle num input == 0
        int kv = ksize.op<tv::arrayops::prod>();
        TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
        TV_ASSERT_RT_ERR(hashdata_k.dtype() == indice_pairs_uniq.dtype(), "error");
        TV_ASSERT_RT_ERR(hashdata_v.dtype() == tv::int32, "error");
        auto ctx = tv::Context();
        ctx.set_cuda_stream(custream);

        // indice_pairs: [2, kv, num_act_in_bounded]
        // indice_pairs_uniq: [indice_pairs.size() / 2 + 1]
        // out_inds: [MaxSize, {self.ndim + 1}]
        // auto timer = tv::CudaContextTimer<>();
        int64_t uniq_size = indice_pairs.size() / 2 + 1;
        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= num_out_act, "error");
        TV_ASSERT_RT_ERR(out_inds.dim(0) >= num_out_act && out_inds.dim(1) == {self.ndim + 1}, "error");
        tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();

        // TODO handle invalid num_out_act
        indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
        tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
        """)
        with code.block(
                "",
                "tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){",
                "});"):
            code.raw(f"""
            using V = {self.dtype_indices};
            using K = TV_DECLTYPE(I);
            using table_t =
                tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                            tv::hash::default_empty_key_v<K>, false>;
            TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_out_act, "hash size not enough");
            table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
            tv::hash::clear_map_split(hash, custream);
            // hash.clear(custream);
            """)
            for x in codeops.dispatch_ints(code, [0, 1],
                                           "int(use_int32)"):
                loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
                code.raw(f"""
                {loc_type} loc_iter(problem);
                lanucher_build_hash(build_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, 
                    out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const K>(), 
                    loc_iter.layout_npq, num_out_act);
                """)
            code.raw(f"""
            if (!use_bound_algo){{
                launcher_num_act_in(calc_conv_indices_stage2<table_t>, hash, 
                    indice_pairs_uniq_before_sort.data_ptr<const K>(),
                    indice_pairs[1].data_ptr<int>(), 
                    indices.dim(0), 
                    indice_pairs.dim(2));
            }}else{{
                indice_num_per_loc.zero_(ctx);
                // copy previous pair in to indice_pairs_uniq
                // we need to ensure size of indice_pairs_uniq larger than pair in
                TV_ASSERT_RT_ERR({pccm.literal(self.dtype_indices == dtypes.int32)}, "error");
                tv::Tensor indice_pairs_in_temp = tv::from_blob(indice_pairs_uniq.raw_data(), {{indice_pairs.dim(1), indice_pairs.dim(2)}}, 
                    indice_pairs.dtype(), indice_pairs.device());
                indice_pairs_in_temp.copy_(indice_pairs[0].view(-1), ctx);
                launcher_num_act_in(calc_conv_indices_stage2_bounded<table_t>, hash, 
                    indice_pairs_uniq_before_sort.data_ptr<const K>(),
                    indice_pairs_in_temp.data_ptr<const int>(),
                    indice_pairs[0].data_ptr<int>(), 
                    indice_pairs[1].data_ptr<int>(), 
                    indice_num_per_loc.data_ptr<int>(),
                    indices.dim(0), 
                    indice_pairs.dim(2));
            }}
            """)
        code.raw(f"""
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_mask_stage1(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs_bwd, indice_pairs_uniq", "tv::Tensor")
        code.arg("indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.arg("transposed", f"bool", "false")

        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        // TODO stream
        // TODO handle num input == 0
        int kv = ksize.op<tv::arrayops::prod>();
        int num_act_in = indices.dim(0);
        // indice_pairs_bwd: [kv, num_act_in] or empty
        // indice_pairs_uniq: [kv * num_act_in + 1]
        if (!indice_pairs_bwd.empty()){{
            tv::check_shape(indice_pairs_bwd, {{kv, num_act_in}});
        }}
        tv::check_shape(indice_num_per_loc, {{kv}});
        int64_t uniq_size = kv * num_act_in + 1;

        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) == uniq_size, "error");

        tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
        // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();
        tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
        """)

        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);
            tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){{
                using T = TV_DECLTYPE(I);
                TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<T>::max(), 
                    "kernel volume must smaller than max value of T");
                launcher_clean_uniq(clean_indices_uniq<T>, indice_pairs_uniq.data_ptr<T>(), uniq_size);
                launcher_num_act_in(calc_conv_indices_stage1_mask<T, {loc_type}>, loc_iter, indices.data_ptr<const int>(), 
                    indice_pairs_bwd.data_ptr<{self.dtype_indices}>(), 
                    indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
                    kv, transposed);
            }});
            """)
        return code  # .ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_mask_stage1_direct_table(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg("indice_pairs_bwd, indice_pairs_uniq", "tv::Tensor")
        code.arg("indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.arg("transposed", f"bool", "false")

        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        // TODO stream
        // TODO handle num input == 0
        int kv = ksize.op<tv::arrayops::prod>();
        int num_act_in = indices.dim(0);
        // indice_pairs_bwd: [kv, num_act_in] or empty
        // indice_pairs_uniq: [kv * num_act_in + 1]
        if (!indice_pairs_bwd.empty()){{
            tv::check_shape(indice_pairs_bwd, {{kv, num_act_in}});
        }}
        tv::check_shape(indice_num_per_loc, {{kv}});
        int64_t uniq_size = kv * num_act_in + 1;

        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) == uniq_size, "error");

        tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
        // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
        bool use_int32 = problem.check_npq_not_overflow();

        """)
        with code.block(
                "",
                "tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){",
                "});"):
            code.raw(f"""
            using V = {self.dtype_indices};
            using K = TV_DECLTYPE(I);
            using table_t =
                tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                            tv::hash::default_empty_key_v<K>, false>;
            table_t table = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
            tv::hash::clear_map_split(table, reinterpret_cast<cudaStream_t>(stream_int));
            using T = TV_DECLTYPE(I);
            TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<T>::max(), 
                "kernel volume must smaller than max value of T");
            launcher_clean_uniq(clean_indices_uniq<T>, indice_pairs_uniq.data_ptr<T>(), uniq_size);
            """)
            for x in codeops.dispatch_ints(code, [0, 1],
                                           "int(use_int32)"):
                loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
                code.raw(f"""
                {loc_type} loc_iter(problem);
                launcher_num_act_in(calc_conv_indices_stage1_mask_direct_table<T, table_t, {loc_type}>, table, 
                    loc_iter, indices.data_ptr<const int>(), 
                    indice_pairs_bwd.data_ptr<{self.dtype_indices}>(), 
                    indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), 
                    indices.dim(0),
                    kv, transposed);
                """)
        return code

    def generate_conv_inds_stage2_mask_template(self, is_direct_table: bool):
        """here indice_pairs_uniq may be bounded, some
        points may be dropped.
        """
        code = pccm.FunctionCode()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg(
            "indice_pairs_fwd, indice_pairs_bwd, indice_pairs_uniq, indice_pairs_uniq_before_sort, out_inds",
            "tv::Tensor")
        code.arg("mask_fwd, mask_bwd", "tv::Tensor")

        code.arg("num_out_act", "int")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.arg("transposed", f"bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        // TODO stream
        // TODO handle num input == 0
        int kv = ksize.op<tv::arrayops::prod>();
        int mask_int_count = tv::div_up(kv, 32);
        // indice_pairs_bwd: [kv, num_act_in]  or empty
        // indice_pairs_fwd: [kv, num_act_out]
        auto ctx = tv::Context();
        ctx.set_cuda_stream(custream);
        int num_act_in = indices.dim(0);
        int num_act_out = num_out_act;
        """)
        if not is_direct_table:
            code.raw(f"""
            TV_ASSERT_RT_ERR(hashdata_k.dtype() == indice_pairs_uniq.dtype(), "error");
            """)
        code.raw(f"""
        TV_ASSERT_RT_ERR(hashdata_v.dtype() == tv::int32, "error");
        // out_inds: [num_out_act, {self.ndim + 1}]
        // auto timer = tv::CudaContextTimer<>();
        if (!indice_pairs_bwd.empty()){{
            tv::check_shape(indice_pairs_bwd, {{kv, num_act_in}});
        }}
        tv::check_shape(indice_pairs_fwd, {{kv, num_act_out}});
        tv::check_shape(out_inds, {{num_out_act, {self.ndim + 1}}});

        tv::cuda::Launch launcher_num_act_in(num_act_in, custream);
        launcher_num_act_in.blocks.y = kv;
        tv::cuda::Launch launcher_num_act_in_no_y(num_act_in, custream);

        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);

        tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
        bool use_int32 = problem.check_npq_not_overflow();

        // TODO handle invalid num_out_act
        """)
        if not is_direct_table:
            code.raw(f"""
            indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
            """)
        with code.block(
                "",
                start=
                "tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){",
                end="});"):
            code.raw(f"""
            using V = {self.dtype_indices};
            using K = TV_DECLTYPE(I);
            using table_t =
                tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                            tv::hash::default_empty_key_v<K>, false>;
            TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_out_act, "hash size not enough");
            table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
            """)
            if not is_direct_table:
                code.raw(f"""
                tv::hash::clear_map_split(hash, custream);
                """)
                # direct table built in stage 1.
                for x in codeops.dispatch_ints(code, [0, 1],
                                               "int(use_int32)"):
                    loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
                    code.raw(f"""
                    {loc_type} loc_iter(problem);
                    lanucher_build_hash(build_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, 
                        out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const K>(), 
                        loc_iter.layout_npq, num_out_act);
                    """)
            code.raw(f"""
            if (!mask_bwd.empty()){{
                launcher_num_act_in(calc_conv_indices_stage2_mask<table_t, {pccm.literal(is_direct_table)}>, hash, 
                    indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
                    indice_pairs_uniq_before_sort.data_ptr<K>(),
                    mask_fwd.data_ptr<uint32_t>(), mask_bwd.data_ptr<uint32_t>(),
                    num_act_in, indice_pairs_fwd.dim(1),
                    mask_int_count);
                launcher_num_act_in_no_y(calc_conv_indices_stage2_mask_output, 
                    indice_pairs_bwd.data_ptr<int>(), 
                    mask_bwd.data_ptr<uint32_t>(),
                    num_act_in, kv,
                    mask_int_count);
                if (mask_fwd.dim(0) == 2){{
                    mask_fwd[1].copy_(mask_fwd[0], ctx);
                }}
                if (mask_bwd.dim(0) == 2){{
                    mask_bwd[1].copy_(mask_bwd[0], ctx);
                }}
            }}else{{
                launcher_num_act_in(calc_conv_indices_stage2_inference_mask<table_t, {pccm.literal(is_direct_table)}>, hash, 
                    indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
                    indice_pairs_uniq_before_sort.data_ptr<K>(),
                    mask_fwd.data_ptr<uint32_t>(),
                    num_act_in, indice_pairs_fwd.dim(1),
                    mask_int_count);
                if (mask_fwd.dim(0) == 2){{
                    mask_fwd[1].copy_(mask_fwd[0], ctx);
                }}
            }}
            """)
        code.raw(f"""
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_stage2_mask(self):
        """here indice_pairs_uniq may be bounded, some
        points may be dropped.
        """
        return self.generate_conv_inds_stage2_mask_template(False)

    @pccm.cuda.static_function
    def generate_conv_inds_stage2_mask_direct_table(self):
        """here indice_pairs_uniq may be bounded, some
        points may be dropped.
        """
        return self.generate_conv_inds_stage2_mask_template(True)

    @pccm.cuda.static_function
    def unique_and_assign_output_direct_hash(self):
        """unique by hash
        """
        code = pccm.FunctionCode()
        code.arg("hashdata_k, hashdata_v, uniq_cnt", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("num_out_bound", "int")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")

        code.arg("stream_int", f"std::uintptr_t", "0")
        code.raw(f"""
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        tv::cuda::Launch lanucher_build_hash(hashdata_k.size(), custream);
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();
        auto tvctx = tv::Context();
        tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
        if (num_out_bound <= 0){{
            num_out_bound = hashdata_k.size();
        }}
        """)
        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);
            tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){{
                using V = {self.dtype_indices};
                using K = TV_DECLTYPE(I);
                using table_t =
                    tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                tv::hash::default_empty_key_v<K>, false>;
                table_t table = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
                lanucher_build_hash(arange_hash_table_and_assign_out<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, table, 
                    out_inds.data_ptr<int>(), uniq_cnt.data_ptr<int>(), num_out_bound,
                    loc_iter.layout_npq);
            }});
            """)
        code.raw(f"""
        auto uniq_cnt_cpu = uniq_cnt.cpu(tvctx);
        return std::min(uniq_cnt_cpu.data_ptr<int>()[0], num_out_bound);
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def unique_hash(self):
        """unique by hash
        """
        code = pccm.FunctionCode()
        code.arg("hashdata_k, hashdata_v, uniq_cnt, out_indices_offset",
                 "tv::Tensor")
        code.arg("num_out_bound", "int")
        code.arg("stream_int", f"std::uintptr_t", "0")
        code.raw(f"""
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        tv::cuda::Launch lanucher_build_hash(hashdata_k.size(), custream);
        auto tvctx = tv::Context();
        tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
        if (num_out_bound <= 0){{
            num_out_bound = out_indices_offset.dim(0);
        }}
        tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){{
            using V = {self.dtype_indices};
            using K = TV_DECLTYPE(I);
            using table_t =
                tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                            tv::hash::default_empty_key_v<K>, false>;
            table_t table = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
            lanucher_build_hash(arange_hash_table<table_t>, table, 
                out_indices_offset.data_ptr<K>(),
                uniq_cnt.data_ptr<int>(), num_out_bound);
        }});
        auto uniq_cnt_cpu = uniq_cnt.cpu(tvctx);
        return std::min(uniq_cnt_cpu.data_ptr<int>()[0], num_out_bound);
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def assign_output_direct_hash(self):
        """unique by hash
        """
        code = pccm.FunctionCode()
        code.arg("out_indices_offset", "tv::Tensor")
        code.arg("out_inds", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")

        code.arg("stream_int", f"std::uintptr_t", "0")
        code.raw(f"""
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        tv::cuda::Launch lanucher_build_hash(out_inds.dim(0), custream);
        TV_ASSERT_RT_ERR(out_indices_offset.dim(0) >= out_inds.dim(0), "error");
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();

        auto tvctx = tv::Context();
        tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
        """)
        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);
            tv::dispatch<int32_t, int64_t>(out_indices_offset.dtype(), [&](auto I){{
                using K = TV_DECLTYPE(I);
                lanucher_build_hash(assign_out_indices<K, std::decay_t<decltype(loc_iter.layout_npq)>>, out_inds.data_ptr<int>(),
                    out_indices_offset.data_ptr<const K>(),
                    loc_iter.layout_npq, out_inds.dim(0));
            }});
            """)
        return code

    @pccm.cuda.static_function
    def generate_subm_conv_inds(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata_k, hashdata_v", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, dilation", f"tv::array<int, {self.ndim}>")
        code.arg("indice_pair_mask", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("is_train", "bool", "true")
        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        int num_act_in_real = indices.dim(0);
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        auto ctx = tv::Context();
        ctx.set_cuda_stream(custream);
        if (!indice_pair_mask.empty()){{
            // TV_ASSERT_INVALID_ARG(ksize.op<tv::arrayops::prod>() <= 32, "for now only support 32bit mask");
        }}
        // TODO stream
        // TODO handle num input == 0
        tv::array<int, {self.ndim}> stride, padding;
        for (int i = 0; i < {self.ndim}; ++i){{
            TV_ASSERT_RT_ERR(ksize[i] % 2 == 1, "subm only support odd ksize");
            stride[i] = 1;
            padding[i] = (ksize[i] / 2) * dilation[i];
        }}
        int kv = ksize.op<tv::arrayops::prod>();
        int mask_int_count = tv::div_up(kv, 32);
        TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
        // indice_pairs: [1 or 2, kv, num_act_in] if mask else [2, kv, num_act_in]
        // out_inds: [MaxSize, {self.ndim + 1}]

        TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
        tv::cuda::Launch launcher_num_act_in(num_act_in_real, custream);
        launcher_num_act_in.blocks.y = (kv / 2) + 1;
        // launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, input_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();
        tv::cuda::Launch lanucher_build_hash(num_act_in_real, custream);
        """)
        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);
            tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){{
                using V = {self.dtype_indices};
                using K = TV_DECLTYPE(I);
                TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<K>::max(), 
                    "kernel volume must smaller than max value of K");

                using table_t =
                    tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                tv::hash::default_empty_key_v<K>, false>;
                TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_act_in_real, "hash size not enough");
                table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
                tv::hash::clear_map_split(hash, custream);
                lanucher_build_hash(build_subm_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, indices.data_ptr<const int>(),
                    loc_iter.layout_npq, num_act_in_real);
                if (!indice_pair_mask.empty()){{
                    TV_ASSERT_RT_ERR(indice_pairs.ndim() == 3, "error");
                    TV_ASSERT_RT_ERR(indice_pairs.dim(0) == (is_train ? 2 : 1), "error");
                    TV_ASSERT_INVALID_ARG(indice_pair_mask.ndim() == 3, "error");
                    // indice_pair_mask: [mask_split_count, num_act_in, num_mask_per_point]
                    if (indice_pair_mask.dim(0) == 2){{
                        auto mask_0 = indice_pair_mask[0].slice_first_axis(0, num_act_in_real);
                        auto mask_1 = indice_pair_mask[1].slice_first_axis(0, num_act_in_real);
                        tv::cuda::Launch lanucher_fill(num_act_in_real, custream);
                        lanucher_fill(cudakers::fill_kernel<uint32_t>, mask_0.data_ptr<uint32_t>(), (1 << (kv / 2)), indices.dim(0));
                        mask_1.zero_(ctx);
                        auto kernel = &calc_subm_conv_indices_split_mask<table_t, {loc_type}>;
                        launcher_num_act_in(kernel, loc_iter, hash,  
                            indices.data_ptr<const int>(), indice_pairs.data_ptr<int>(), 
                            mask_0.data_ptr<uint32_t>(), mask_1.data_ptr<uint32_t>(), 
                            indices.dim(0), indice_pairs.dim(2), kv, is_train);

                    }}else{{
                        // indice_pair_mask: [1, num_act_in, num_mask_per_point]
                        tv::cuda::Launch lanucher_fill(num_act_in_real, custream);
                        if (mask_int_count == 1){{
                            lanucher_fill(cudakers::fill_kernel<uint32_t>, indice_pair_mask.data_ptr<uint32_t>(), (1 << (kv / 2)), indices.dim(0));
                        }}
                        else{{
                            lanucher_fill(init_subm_multiple_mask_int_kernel<uint32_t>, 
                                    indice_pair_mask.data_ptr<uint32_t>(), kv / 2, indices.dim(0), mask_int_count);
                        }}
                        TV_ASSERT_RT_ERR(indice_pair_mask.dim(0) == 1, "error");
                        launcher_num_act_in(calc_subm_conv_indices_mask<table_t, {loc_type}>, loc_iter, hash, 
                            indices.data_ptr<const int>(), indice_pairs.data_ptr<int>(), 
                            indice_pair_mask.data_ptr<uint32_t>(), indices.dim(0), indice_pairs.dim(2), kv, is_train, mask_int_count);
                    }}
                }}else{{
                    TV_ASSERT_RT_ERR(indice_pairs.ndim() == 3, "error");
                    TV_ASSERT_RT_ERR(indice_pairs.dim(0) == 2, "error");
                    launcher_num_act_in(calc_subm_conv_indices<table_t, {loc_type}>, loc_iter, hash, indices.data_ptr<const int>(), 
                        indice_pairs.data_ptr<int>(), 
                        indice_num_per_loc.data_ptr<int>(), indices.dim(0), indice_pairs.dim(2), kv);
                }}
            }});
        return indices.dim(0);
        """)

        return code.ret("int")

    @pccm.cuda.cuda_global_function
    def init_subm_multiple_mask_int_kernel(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.arg("ptr", "T*")
        code.arg("set_bit", "int")
        code.arg("length", "int")
        code.arg("mask_int_count", "int")
        code.raw(f"""
            int initial_offset = blockIdx.x * blockDim.x + threadIdx.x;
            int bit_offset = set_bit / 32;
            int bit_residue = set_bit % 32;
            for(int offset : tv::KernelLoopX<int>(length)){{
                for (int i=0; i < mask_int_count; ++i)
                    ptr[offset * mask_int_count + i] = (i == bit_offset) * (1 << bit_residue);
            }}
        """)
        return code


class SparseConvIndicesCPU(pccm.ParameterizedClass):

    def __init__(self, problem: ConvProblem, dtype_indices: dtypes.DType):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("unordered_map")
        self.loc_iter = ConvOutLocIter(problem)
        self.loc_iter_64 = ConvOutLocIter(problem, True)
        self.add_param_class("spinds", self.loc_iter, "ConvLocIter")
        self.add_param_class("spinds64", self.loc_iter_64, "ConvLocIter64")
        self.add_param_class("spinds", problem, "ConvProblem")

        self.ndim = problem.ndim
        self.dtype_indices = dtype_indices
        self.dtype_indices_uniq = dtype_indices

        assert dtype_indices == dtypes.int32 or dtype_indices == dtypes.int64

    @pccm.static_function
    def generate_subm_conv_inds(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, dilation", f"tv::array<int, {self.ndim}>")
        code.raw(f"""
        tv::array<int, {self.ndim}> stride, padding;
        for (int i = 0; i < {self.ndim}; ++i){{
            TV_ASSERT_RT_ERR(ksize[i] % 2 == 1, "subm only support odd ksize");
            stride[i] = 1;
            padding[i] = (ksize[i] / 2) * dilation[i];
        }}
        int kv = ksize.op<tv::arrayops::prod>();
        TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<{self.dtype_indices}>::max(), 
            "kernel volume must smaller than max value of {self.dtype_indices}");
        ConvProblem problem(batch_size, 1, 1, input_dims, input_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();
        """)
        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);
            int indices_pair_size = indice_pairs.dim(2);
            int indices_pair_size_mul_RS = indices_pair_size * kv;
            auto indice_pairs_ptr = indice_pairs.data_ptr<{self.dtype_indices}>();
            std::unordered_map<{self.dtype_indices}, {self.dtype_indices}> hash;
            auto indices_ptr = indices.data_ptr<const {self.dtype_indices}>();
            int indice_in_num = indices.dim(0);
            for (int i = 0; i < indice_in_num; ++i){{
                {self.dtype_indices} index = loc_iter.layout_npq(indices_ptr);
                hash.insert({{index, i}});
                indices_ptr += {self.ndim + 1};
            }}
            for (int filter_offset = 0; filter_offset < (kv / 2 + 1); ++filter_offset){{
                int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
                int filter_offset_mul_indices_pair_size_1 = (kv - 1 - filter_offset) * indices_pair_size;
                if (filter_offset == kv / 2){{
                    for (int i = 0; i < indice_in_num; ++i){{
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size + i] = i;
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
                    }}
                }}else{{
                    indices_ptr = indices.data_ptr<const {self.dtype_indices}>();
                    auto indice_num_per_loc_ptr = indice_num_per_loc.data_ptr<{self.dtype_indices}>() + filter_offset;
                    for (int i = 0; i < indice_in_num; ++i){{
                        tv::array<int, {self.ndim + 1}> npq_offset;
                        if (loc_iter.query_npq_no_stride(indices_ptr, npq_offset)){{
                            auto index = loc_iter.layout_npq(npq_offset);
                            auto iter = hash.find(index);
                            if (iter != hash.end()){{
                                auto old_num = indice_num_per_loc_ptr[0]++;
                                indice_pairs_ptr[filter_offset_mul_indices_pair_size + old_num] = i;
                                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = iter->second;
                                indice_pairs_ptr[filter_offset_mul_indices_pair_size_1 + old_num] = iter->second;
                                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
                            }}
                        }}
                        indices_ptr += {self.ndim + 1};
                    }}
                }}
                ++loc_iter;
            }}
            """)
        code.raw(f"""
        return indices.dim(0);
        """)
        return code.ret("int")

    @pccm.static_function
    def generate_conv_inds(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("output_dims, input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, stride, padding, dilation",
                 f"tv::array<int, {self.ndim}>")
        code.arg("transposed", f"bool", "false")
        code.raw(f"""
        int kv = ksize.op<tv::arrayops::prod>();
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        bool use_int32 = problem.check_npq_not_overflow();
        int num_act = 0;

        """)
        for x in codeops.dispatch_ints(code, [0, 1], "int(use_int32)"):
            loc_type = "ConvLocIter" if x == 1 else "ConvLocIter64"
            code.raw(f"""
            {loc_type} loc_iter(problem);

            int indices_pair_size = indice_pairs.dim(2);
            int indices_pair_size_mul_RS = indices_pair_size * kv;
            auto indice_pairs_ptr = indice_pairs.data_ptr<{self.dtype_indices}>();
            std::unordered_map<{self.dtype_indices}, {self.dtype_indices}> hash;
            auto indices_ptr = indices.data_ptr<const {self.dtype_indices}>();
            auto out_inds_ptr = out_inds.data_ptr<{self.dtype_indices}>();
            TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<{self.dtype_indices}>::max(), 
                "kernel volume must smaller than max value of {self.dtype_indices}");
            int indice_in_num = indices.dim(0);
            {self.dtype_indices} hashval;
            for (int filter_offset = 0; filter_offset < kv; ++filter_offset){{
                int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
                indices_ptr = indices.data_ptr<const {self.dtype_indices}>();
                auto indice_num_per_loc_ptr = indice_num_per_loc.data_ptr<{self.dtype_indices}>() + filter_offset;
                for (int i = 0; i < indice_in_num; ++i){{
                    tv::array<int, {self.ndim + 1}> npq_offset;
                    bool valid;
                    if (transposed){{
                        valid = loc_iter.query_nhw_out(indices_ptr, npq_offset);
                    }}else{{
                        valid = loc_iter.query_npq(indices_ptr, npq_offset);
                    }}
                    if (valid){{
                        auto index = loc_iter.layout_npq(npq_offset);
                        auto iter = hash.find(index);
                        if (iter == hash.end()){{
                            hashval = num_act++;
                            hash.insert({{index, hashval}});
                            for (int k = 0; k < {self.ndim + 1}; ++k){{
                                out_inds_ptr[k] = npq_offset[k];
                            }}
                            out_inds_ptr += {self.ndim + 1};
                        }}else{{
                            hashval = iter->second;
                        }}
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]] = i;
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]++] = hashval;
                    }}
                    indices_ptr += {self.ndim + 1};
                }}
                ++loc_iter;
            }}
            """)
        code.raw(f"""
        return num_act;
        """)
        return code.ret("int")
