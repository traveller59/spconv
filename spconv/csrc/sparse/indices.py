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
from cumm.common import TensorView, TensorViewHashKernel, TensorViewKernel, ThrustLib
from cumm.gemm import codeops
from typing import List
from cumm.conv.params import ConvProblem
import numpy as np


class CudaCommonKernel(pccm.ParameterizedClass):
    # we need to use PClass instead of Class
    # because cuda global function can't be put in class body.
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


class ConvOutLocIter(pccm.ParameterizedClass):
    def __init__(self, problem: ConvProblem):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_param_class("lociter", problem, "ConvProblem")
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
        self.add_dependency(TensorView, TensorViewKernel, TensorViewHashKernel,
                            ThrustLib)
        self.loc_iter = ConvOutLocIter(problem)
        self.add_param_class("spinds", self.loc_iter, "ConvLocIter")
        self.add_param_class("spinds", problem, "ConvProblem")
        self.add_param_class("cudakers", CudaCommonKernel())

        self.ndim = problem.ndim
        self.dtype_indices = dtype_indices
        self.dtype_indices_uniq = dtype_indices

        assert dtype_indices == dtypes.int32 or dtype_indices == dtypes.int64

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage1(self):
        code = pccm.FunctionCode()
        code.arg("loc_iter", f"ConvLocIter")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("indice_pairs_for_uniq",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]

        code.arg("num_indices_in", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
        code.arg("transposed", "bool")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        loc_iter.set_filter_offset(filter_offset);
        int indices_pair_size_mul_RS = indices_pair_size * RS;
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
                {self.dtype_indices} offset = loc_iter.layout_npq(npq_offset);
                if (old_num < indices_pair_size){{
                    indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                    indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = offset;
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

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indices_out", f"int*")  # [N, ndim + 1]
        code.arg("indice_pairs_for_uniq",
                 f"const {self.dtype_indices}*")  # [2, kernelProd, MaxSize]

        code.arg("layout_npq",
                 f"spinds::LayoutNPQ")  # [2, kernelProd, MaxSize]

        code.arg("num_indices", "int")

        code.raw(f"""
        for (int output_index : tv::KernelLoopX<int>(num_indices)) {{
            {self.dtype_indices} output_coord_offset = indice_pairs_for_uniq[output_index];
            layout_npq.inverse(output_coord_offset, indices_out + {self.ndim + 1} * output_index);
            table.insert(output_coord_offset, output_index);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_out_part", f"int*")  # [2, kernelProd, MaxSize]
        code.arg("num_indices_in", "int")
        code.arg("indices_pair_size", "int")
        # TODO use block instead of filter_offset?
        code.raw(f"""
        int filter_offset = blockIdx.y;
        auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
        for (int i : tv::KernelLoopX<int>(num_indices_in)) {{
            {self.dtype_indices} output_coord_offset = indice_pairs_out_part_filter[i];
            if (output_coord_offset > -1){{
                auto ptr = table.lookup_ptr(output_coord_offset);
                if (ptr){{
                    indice_pairs_out_part_filter[i] = ptr->second;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage1_mask(self):
        code = pccm.FunctionCode()
        code.arg("loc_iter", f"ConvLocIter")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs_bwd",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("indice_pairs_for_uniq",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("indice_num_per_loc", f"int*")  # [kernelProd]

        code.arg("num_indices_in", "int")

        code.arg("RS", "int")
        code.arg("transposed", "bool")

        code.raw(f"""
        int filter_offset = blockIdx.y;
        loc_iter.set_filter_offset(filter_offset);
        int indices_pair_size_mul_RS = num_indices_in * RS;
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
                int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                {self.dtype_indices} output_coord_offset = loc_iter.layout_npq(npq_offset);
                // if (old_num < indices_pair_size){{
                // indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;

                indice_pairs_bwd[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
                indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = output_coord_offset;
                // }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2_mask(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_fwd",
                 f"int*")  # [kernelProd, MaxSize], inp -> out
        code.arg("indice_pairs_bwd",
                 f"int*")  # [kernelProd, MaxSize], out -> inp
        code.arg("mask_fwd", f"uint32_t*")  # [kernelProd]
        code.arg("mask_bwd", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices_in", "int")
        code.arg("num_indices_out", "int")

        # TODO use block instead of filter_offset?
        code.raw(f"""
        int filter_offset = blockIdx.y;
        uint32_t filter_mask_fwd = (1u << (filter_offset));
        // TODO following rule for even kernel size is wrong. 
        // uint32_t filter_mask_bwd = (1u << (gridDim.y - 1 - filter_offset));

        auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
        auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            {self.dtype_indices} output_coord_offset = indice_pairs_bwd_filter[input_index];
            if (output_coord_offset > -1){{
                auto ptr = table.lookup_ptr(output_coord_offset);
                if (ptr){{
                    auto output_index = ptr->second;
                    atomicOr(mask_fwd + output_index, filter_mask_fwd);
                    // atomicOr(mask_bwd + input_index, filter_mask_bwd);
                    indice_pairs_fwd_filter[output_index] = input_index;
                    indice_pairs_bwd_filter[input_index] = output_index;
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

        # TODO use block instead of filter_offset?
        code.raw(f"""
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            uint32_t mask = 0;
            for (int filter_offset = 0; filter_offset < kv; ++filter_offset){{
                auto val = indice_pairs_bwd[filter_offset * num_indices_in + input_index];
                mask |= (val != -1) << filter_offset;
            }}
            mask_bwd[input_index] = mask;
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_conv_indices_stage2_inference_mask(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indice_pairs_fwd",
                 f"int*")  # [kernelProd, MaxSize], inp -> out
        code.arg("indice_pairs_bwd",
                 f"int*")  # [kernelProd, MaxSize], out -> inp
        code.arg("mask_fwd", f"uint32_t*")  # [kernelProd]
        code.arg("num_indices_in", "int")
        code.arg("num_indices_out", "int")

        # TODO use block instead of filter_offset?
        code.raw(f"""
        int filter_offset = blockIdx.y;
        uint32_t filter_mask_fwd = (1u << (filter_offset));

        auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
        auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
        for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {{
            {self.dtype_indices} output_coord_offset = indice_pairs_bwd_filter[input_index];
            if (output_coord_offset > -1){{
                auto ptr = table.lookup_ptr(output_coord_offset);
                if (ptr){{
                    auto output_index = ptr->second;
                    atomicOr(mask_fwd + output_index, filter_mask_fwd);
                    indice_pairs_fwd_filter[output_index] = input_index;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def build_subm_conv_hash_table(self):
        code = pccm.FunctionCode()
        code.targ("TTable")

        code.arg("table", f"TTable")  # [N, ndim + 1]
        code.arg("indices_in", f"const int*")  # [N, ndim + 1]

        code.arg("layout_npq", f"spinds::LayoutNPQ")

        code.arg("num_indices", "int")

        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(num_indices)) {{
            {self.dtype_indices} index = layout_npq(indices_in + i * {self.ndim + 1});
            table.insert(index, i);
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def clean_indices_uniq(self):
        code = pccm.FunctionCode()
        code.arg("indice_pairs_for_uniq", f"{self.dtype_indices}*")
        code.arg("size", f"{self.dtype_indices}")
        code.raw(f"""
        for ({self.dtype_indices} i : tv::KernelLoopX<{self.dtype_indices}>(size)) {{
            indice_pairs_for_uniq[i] = std::numeric_limits<{self.dtype_indices}>::max();
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def calc_subm_conv_indices(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("loc_iter", f"ConvLocIter")  # [N, ndim + 1]
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
                    {self.dtype_indices} offset = loc_iter.layout_npq(npq_offset);
                    auto item = table.lookup(offset); // performance bound
                    if (!item.empty()){{
                        int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                        indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                        indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = item.second;
                        indice_pairs[filter_offset_mul_indices_pair_size_1 + old_num] = item.second;
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
        code.arg("loc_iter", f"ConvLocIter")  # [N, ndim + 1]
        code.arg("table", f"TTable")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("mask", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
        code.raw(f"""
        int filter_offset = blockIdx.y;
        uint32_t filter_mask_out = (1u << (filter_offset));
        uint32_t filter_mask_in = (1u << (RS - 1 - filter_offset));
        // uint32_t filter_mask_center = (1u << (RS / 2));

        loc_iter.set_filter_offset(filter_offset);
        int indices_pair_size_mul_RS = indices_pair_size * RS;
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;

        int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
        if (filter_offset == (RS / 2)){{
            for (int i : tv::KernelLoopX<int>(num_indices)) {{
                // atomicOr(mask + i, filter_mask_center);
                indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
                indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
            }}
        }} else {{
            for (int output_index : tv::KernelLoopX<int>(num_indices)) {{
                // find input offset from output offset
                tv::array<int, {self.ndim + 1}> nhw_offset;
                // table: input indice coord to output index (or output indice coord to input index)
                if (loc_iter.query_nhw(indices_in + output_index * {self.ndim + 1}, nhw_offset)){{
                    {self.dtype_indices} offset = loc_iter.layout_npq(nhw_offset);
                    auto item = table.lookup(offset);
                    if (!item.empty()) {{
                        auto input_index = item.second; // we find a input indice idx.
                        atomicOr(mask + output_index, filter_mask_out);
                        atomicOr(mask + input_index, filter_mask_in);
                        // for this output, we set correct input idx.
                        indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                        indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + input_index] = output_index;
                        // the output in "input location" connect this output idx in another location.
                        indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                        indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
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
        code.arg("loc_iter", f"ConvLocIter")  # [N, ndim + 1]
        code.arg("table", f"TTable")  # [N, ndim + 1]

        code.arg("indices_in", f"const int*")  # [N, ndim + 1]
        code.arg("indice_pairs",
                 f"{self.dtype_indices}*")  # [2, kernelProd, MaxSize]
        code.arg("mask1", f"uint32_t*")  # [kernelProd]
        code.arg("mask2", f"uint32_t*")  # [kernelProd]

        code.arg("num_indices", "int")
        code.arg("indices_pair_size", "int")

        code.arg("RS", "int")
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
                indice_ptr_inv[filter_offset_mul_indices_pair_size + i] = i;
            }}
        }} else {{
            for (int output_index : tv::KernelLoopX<int>(num_indices)) {{
                // find input offset from output offset
                tv::array<int, {self.ndim + 1}> nhw_offset;
                // table: input indice coord to output index (or output indice coord to input index)
                if (loc_iter.query_nhw(indices_in + output_index * {self.ndim + 1}, nhw_offset)){{
                    {self.dtype_indices} offset = loc_iter.layout_npq(nhw_offset);
                    auto item = table.lookup(offset);
                    if (!item.empty()) {{
                        auto input_index = item.second; // we find a input indice idx.
                        atomicOr(mask1 + output_index, filter_mask_out);
                        atomicOr(mask2 + input_index, filter_mask_in);
                        // for this output, we set correct input idx.
                        indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                        // the output in "input location" connect this output idx in another location.
                        indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                        indice_ptr_inv[filter_offset_mul_indices_pair_size + input_index] = output_index;
                        indice_ptr_inv[filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
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
        int kv = tv::arrayops::prod(ksize);
        TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
        TV_ASSERT_RT_ERR(tv::arrayops::prod(input_dims) <= std::numeric_limits<{self.dtype_indices}>::max(), 
            "kernel volume must smaller than max value of {self.dtype_indices}");
        // indice_pairs: [2, kv, indices.dim(0)]
        // indice_pairs_uniq: [indice_pairs.size() / 2 + 1]
        tv::check_shape(indice_pairs, {{2, kv, indices.dim(0)}});
        tv::check_shape(indice_num_per_loc, {{kv}});
        int64_t uniq_size = indice_pairs.size() / 2 + 1;
        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= uniq_size, "error");
        TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
        int64_t expected_out_size = indices.dim(0) * kv;
        tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
        // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);
        tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
        launcher_clean_uniq(clean_indices_uniq, indice_pairs_uniq.data_ptr<{self.dtype_indices}>(), uniq_size);
        launcher_num_act_in(calc_conv_indices_stage1, loc_iter, indices.data_ptr<const int>(), 
            indice_pairs.data_ptr<{self.dtype_indices}>(), 
            indice_pairs_uniq.data_ptr<{self.dtype_indices}>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
            indice_pairs.dim(2), kv, transposed);
        // thrust::device_ptr<{self.dtype_indices}> ptr_tr(indice_pairs_uniq.data_ptr<{self.dtype_indices}>());
        // auto thrust_ctx = thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(stream_int));
        // thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
        // auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
        // auto num_out_act = new_end - ptr_tr - 1;
        // return num_out_act;
        """)
        return code  # .ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_stage1_5(self):
        code = pccm.FunctionCode()
        code.arg("indice_pairs_uniq", "tv::Tensor")
        code.arg("uniq_size", "int64_t")
        code.arg("stream_int", f"std::uintptr_t", "0")
        code.raw(f"""
        thrust::device_ptr<{self.dtype_indices}> ptr_tr(indice_pairs_uniq.data_ptr<{self.dtype_indices}>());
        auto thrust_ctx = thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(stream_int));
        thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
        auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
        auto num_out_act = new_end - ptr_tr - 1;
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_stage2(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata", "tv::Tensor")
        code.arg("indice_pairs, indice_pairs_uniq, out_inds", "tv::Tensor")
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
        int kv = tv::arrayops::prod(ksize);
        TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
        // indice_pairs: [2, kv, indices.dim(0)]
        // indice_pairs_uniq: [indice_pairs.size() / 2 + 1]
        // out_inds: [MaxSize, {self.ndim + 1}]
        // auto timer = tv::CudaContextTimer<>();
        int64_t uniq_size = indice_pairs.size() / 2 + 1;
        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= num_out_act, "error");
        TV_ASSERT_RT_ERR(out_inds.dim(0) >= num_out_act && out_inds.dim(1) == {self.ndim + 1}, "error");
        tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);
        
        // TODO handle invalid num_out_act
        indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
        tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
        using V = {self.dtype_indices};
        using KeyType = {self.dtype_indices};
        constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();
        using table_t =
            tv::hash::LinearHashTable<KeyType, V, tv::hash::Murmur3Hash<KeyType>,
                                        kEmptyKey, false>;
        using pair_t = typename table_t::value_type;
        TV_ASSERT_RT_ERR(hashdata.dim(0) >= num_out_act, "hash size not enough");
        table_t hash = table_t(hashdata.data_ptr<pair_t>(), hashdata.dim(0));
        hash.clear(custream);
        lanucher_build_hash(build_conv_hash_table<table_t>, hash, 
            out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const {self.dtype_indices}>(), 
            loc_iter.layout_npq, num_out_act);
        launcher_num_act_in(calc_conv_indices_stage2<table_t>, hash, 
            indice_pairs[1].data_ptr<int>(), indices.dim(0), 
            indice_pairs.dim(2));
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_mask_stage1(self):
        code = pccm.FunctionCode()
        code.arg("indices", "tv::Tensor")
        code.arg("indice_pairs_bwd, indice_pairs_uniq, indice_num_per_loc",
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
        int kv = tv::arrayops::prod(ksize);
        TV_ASSERT_RT_ERR(tv::arrayops::prod(input_dims) <= std::numeric_limits<{self.dtype_indices}>::max(), 
            "kernel volume must smaller than max value of {self.dtype_indices}");
        // indice_pairs_bwd: [kv, indices.dim(0)]
        // indice_pairs_uniq: [indice_pairs_bwd.size() + 1]
        tv::check_shape(indice_pairs_bwd, {{kv, indices.dim(0)}});
        tv::check_shape(indice_num_per_loc, {{kv}});
        int64_t uniq_size = indice_pairs_bwd.size() + 1;

        TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= uniq_size, "error");

        int64_t expected_out_size = indices.dim(0) * kv;
        tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
        // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
        launcher_num_act_in.blocks.y = kv;
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);
        tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
        launcher_clean_uniq(clean_indices_uniq, indice_pairs_uniq.data_ptr<{self.dtype_indices}>(), uniq_size);
        launcher_num_act_in(calc_conv_indices_stage1_mask, loc_iter, indices.data_ptr<const int>(), 
            indice_pairs_bwd.data_ptr<{self.dtype_indices}>(), 
            indice_pairs_uniq.data_ptr<{self.dtype_indices}>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
            kv, transposed);
        """)
        return code  # .ret("int")

    @pccm.cuda.static_function
    def generate_conv_inds_stage2_mask(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata", "tv::Tensor")
        code.arg(
            "indice_pairs_fwd, indice_pairs_bwd, indice_pairs_uniq, out_inds",
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
        int kv = tv::arrayops::prod(ksize);
        // indice_pairs_bwd: [kv, indices.dim(0)]
        // indice_pairs_fwd: [kv, out_inds.dim(0)]
        auto ctx = tv::Context();
        ctx.set_cuda_stream(custream);

        // out_inds: [MaxSize, {self.ndim + 1}]
        // auto timer = tv::CudaContextTimer<>();
        tv::check_shape(indice_pairs_bwd, {{kv, indices.dim(0)}});
        tv::check_shape(indice_pairs_fwd, {{kv, num_out_act}});
        tv::check_shape(out_inds, {{num_out_act, {self.ndim + 1}}});

        tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
        launcher_num_act_in.blocks.y = kv;
        tv::cuda::Launch launcher_num_act_in_no_y(indices.dim(0), custream);

        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);
        
        // TODO handle invalid num_out_act
        indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
        tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
        using V = {self.dtype_indices};
        using KeyType = {self.dtype_indices};
        constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();
        using table_t =
            tv::hash::LinearHashTable<KeyType, V, tv::hash::Murmur3Hash<KeyType>,
                                        kEmptyKey, false>;
        using pair_t = typename table_t::value_type;
        TV_ASSERT_RT_ERR(hashdata.dim(0) >= num_out_act, "hash size not enough");
        table_t hash = table_t(hashdata.data_ptr<pair_t>(), hashdata.dim(0));
        hash.clear(custream);
        lanucher_build_hash(build_conv_hash_table<table_t>, hash, 
            out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const {self.dtype_indices}>(), 
            loc_iter.layout_npq, num_out_act);
        if (!mask_bwd.empty()){{
            // auto timer = tv::CudaContextTimer<>();
            launcher_num_act_in(calc_conv_indices_stage2_mask<table_t>, hash, 
                indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
                mask_fwd.data_ptr<uint32_t>(), mask_bwd.data_ptr<uint32_t>(),
                indice_pairs_bwd.dim(1), indice_pairs_fwd.dim(1));
            // tv::ssprint("calc_conv_indices_stage2_mask", timer.report() / 1000.0);
            launcher_num_act_in_no_y(calc_conv_indices_stage2_mask_output, indice_pairs_bwd.data_ptr<int>(), 
                mask_bwd.data_ptr<uint32_t>(),
                indice_pairs_bwd.dim(1), kv);
            // tv::ssprint("calc_conv_indices_stage2_mask_output", timer.report() / 1000.0);
            if (mask_fwd.dim(0) == 2){{
                mask_fwd[1].copy_(mask_fwd[0], ctx);
            }}
            if (mask_bwd.dim(0) == 2){{
                mask_bwd[1].copy_(mask_bwd[0], ctx);
            }}
        }}else{{
            launcher_num_act_in(calc_conv_indices_stage2_inference_mask<table_t>, hash, 
                indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
                mask_fwd.data_ptr<uint32_t>(),
                indice_pairs_bwd.dim(1), indice_pairs_fwd.dim(1));
            if (mask_fwd.dim(0) == 2){{
                mask_fwd[1].copy_(mask_fwd[0], ctx);
            }}
        }}
        return num_out_act;
        """)
        return code.ret("int")

    @pccm.cuda.static_function
    def generate_subm_conv_inds(self):
        code = pccm.FunctionCode()
        code.arg("indices, hashdata", "tv::Tensor")
        code.arg("indice_pairs, out_inds, indice_num_per_loc", "tv::Tensor")
        code.arg("batch_size", "int")
        code.arg("input_dims", f"tv::array<int, {self.ndim}>")
        code.arg("ksize, dilation", f"tv::array<int, {self.ndim}>")
        code.arg("indice_pair_mask", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("backward", "bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);
        auto ctx = tv::Context();
        ctx.set_cuda_stream(custream);
        if (!indice_pair_mask.empty()){{
            TV_ASSERT_INVALID_ARG(tv::arrayops::prod(ksize) < 32, "for now only support 32bit mask");
        }}
        // TODO stream
        // TODO handle num input == 0
        tv::array<int, {self.ndim}> stride, padding;
        for (int i = 0; i < {self.ndim}; ++i){{
            TV_ASSERT_RT_ERR(ksize[i] % 2 == 1, "subm only support odd ksize");
            stride[i] = 1;
            padding[i] = (ksize[i] / 2) * dilation[i];
        }}
        int kv = tv::arrayops::prod(ksize);
        TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
        // indice_pairs: [2, kv, indices.dim(0)]
        // out_inds: [MaxSize, {self.ndim + 1}]
        // auto timer = tv::CudaContextTimer<>();
        TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
        tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
        launcher_num_act_in.blocks.y = (kv / 2) + 1;
        // launcher_num_act_in.blocks.y = kv;
        TV_ASSERT_RT_ERR(tv::arrayops::prod(input_dims) <= std::numeric_limits<{self.dtype_indices}>::max(), 
            "kernel volume must smaller than max value of {self.dtype_indices}");
        ConvProblem problem(batch_size, 1, 1, input_dims, input_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);

        tv::cuda::Launch lanucher_build_hash(indices.dim(0), custream);
        using V = {self.dtype_indices};
        using KeyType = {self.dtype_indices};
        constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();

        using table_t =
            tv::hash::LinearHashTable<KeyType, V, tv::hash::Murmur3Hash<KeyType>,
                                        kEmptyKey, false>;
        using pair_t = typename table_t::value_type;
        TV_ASSERT_RT_ERR(hashdata.dim(0) >= indices.dim(0), "hash size not enough");
        table_t hash = table_t(hashdata.data_ptr<pair_t>(), hashdata.dim(0));
        hash.clear(custream);
        // tv::ssprint("clear hash time", hashdata.dim(0), timer.report() / 1000.0);

        lanucher_build_hash(build_subm_conv_hash_table<table_t>, hash, indices.data_ptr<const int>(),
            loc_iter.layout_npq, indices.dim(0));
        // tv::ssprint("build_hash time", timer.report() / 1000.0);
        if (!indice_pair_mask.empty()){{
            TV_ASSERT_INVALID_ARG(indice_pair_mask.ndim() == 2, "error");
            if (indice_pair_mask.dim(0) == 2){{
                auto mask_0 = indice_pair_mask[0];
                tv::cuda::Launch lanucher_fill(mask_0.size(), custream);
                lanucher_fill(cudakers::fill_kernel<uint32_t>, mask_0.data_ptr<uint32_t>(), (1 << (kv / 2)), mask_0.size());
                indice_pair_mask[1].zero_(ctx);
                auto kernel = &calc_subm_conv_indices_split_mask<table_t>;
                launcher_num_act_in(kernel, loc_iter, hash,  
                    indices.data_ptr<int>(), indice_pairs.data_ptr<int>(), 
                    indice_pair_mask[0].data_ptr<uint32_t>(), indice_pair_mask[1].data_ptr<uint32_t>(), 
                    indices.dim(0), indice_pairs.dim(2), kv);
            }}else{{
                tv::cuda::Launch lanucher_fill(indice_pair_mask.size(), custream);
                lanucher_fill(cudakers::fill_kernel<uint32_t>, indice_pair_mask.data_ptr<uint32_t>(), (1 << (kv / 2)), indice_pair_mask.size());
                TV_ASSERT_RT_ERR(indice_pair_mask.dim(0) == 1, "error");
                launcher_num_act_in(calc_subm_conv_indices_mask<table_t>, loc_iter, hash, 
                    indices.data_ptr<int>(), indice_pairs.data_ptr<int>(), 
                    indice_pair_mask.data_ptr<uint32_t>(), indices.dim(0), indice_pairs.dim(2), kv);
            }}
        }}else{{
            launcher_num_act_in(calc_subm_conv_indices<table_t>, loc_iter, hash, indices.data_ptr<int>(), 
                indice_pairs.data_ptr<int>(), 
                indice_num_per_loc.data_ptr<int>(), indices.dim(0), indice_pairs.dim(2), kv);
        }}
        // tv::ssprint("gem subm conv inds time", timer.report() / 1000.0);
        return indices.dim(0);
        """)

        return code.ret("int")


class SparseConvIndicesCPU(pccm.ParameterizedClass):
    def __init__(self, problem: ConvProblem, dtype_indices: dtypes.DType):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("unordered_map")
        self.loc_iter = ConvOutLocIter(problem)
        self.add_param_class("spinds", self.loc_iter, "ConvLocIter")
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
        int kv = tv::arrayops::prod(ksize);

        ConvProblem problem(batch_size, 1, 1, input_dims, input_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);
        int indices_pair_size = indice_pairs.dim(2);
        int indices_pair_size_mul_RS = indices_pair_size * kv;
        auto indice_pairs_ptr = indice_pairs.data_ptr<{self.dtype_indices}>();
        std::unordered_map<{self.dtype_indices}, {self.dtype_indices}> hash;
        auto indices_ptr = indices.data_ptr<{self.dtype_indices}>();
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
                indices_ptr = indices.data_ptr<{self.dtype_indices}>();
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
        int kv = tv::arrayops::prod(ksize);
        ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
        ConvLocIter loc_iter(problem);
        int indices_pair_size = indice_pairs.dim(2);
        int indices_pair_size_mul_RS = indices_pair_size * kv;
        auto indice_pairs_ptr = indice_pairs.data_ptr<{self.dtype_indices}>();
        std::unordered_map<{self.dtype_indices}, {self.dtype_indices}> hash;
        auto indices_ptr = indices.data_ptr<{self.dtype_indices}>();
        auto out_inds_ptr = out_inds.data_ptr<{self.dtype_indices}>();

        int indice_in_num = indices.dim(0);
        int num_act = 0;
        {self.dtype_indices} hashval;
        for (int filter_offset = 0; filter_offset < kv; ++filter_offset){{
            int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
            indices_ptr = indices.data_ptr<{self.dtype_indices}>();
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
        return num_act;
        """)
        return code.ret("int")
