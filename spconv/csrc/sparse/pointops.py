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
from cumm.common import TensorView, TensorViewHashKernel
from cumm.gemm import codeops
from typing import List
from cumm.conv.params import ConvProblem
import numpy as np


class Point2VoxelCommon(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, ndim: int, zyx: bool = True):
        super().__init__()
        self.add_dependency(TensorView)
        self.dtype = dtype
        self.ndim = ndim
        self.zyx = zyx
        ret_str = f"std::array<int, {self.ndim}>"
        retf_str = f"std::array<float, {self.ndim}>"
        retf2_str = f"std::array<float, {self.ndim * 2}>"
        self.calc_meta_ret = f"std::tuple<{retf_str}, {ret_str}, {ret_str}, {retf2_str}>"

    @pccm.static_function
    def calc_meta_data(self):
        code = pccm.FunctionCode()
        code.arg("vsize_xyz", f"std::array<float, {self.ndim}>")
        code.arg("coors_range_xyz", f"std::array<float, {self.ndim * 2}>")
        code.raw(f"""
        std::array<float, {self.ndim}> vsize;
        std::array<int, {self.ndim}> grid_size, grid_stride;

        std::array<float, {self.ndim * 2}> coors_range;

        """)
        if self.zyx:
            code.raw(f"""
            for (int i = 0; i < {self.ndim}; ++i){{
                vsize[{self.ndim - 1} - i] = vsize_xyz[i];
                coors_range[{self.ndim - 1} - i] = coors_range_xyz[i];
                coors_range[{2 * self.ndim - 1} - i] = coors_range_xyz[i + {self.ndim}];
            }}
            """)
        else:
            code.raw(f"""
            for (int i = 0; i < {self.ndim}; ++i){{
                vsize[i] = vsize_xyz[i];
                coors_range[i] = coors_range_xyz[i];
                coors_range[i + {self.ndim}] = coors_range_xyz[i + {self.ndim}];
            }}
            """)
        code.raw(f"""
        int64_t prod = 1;
        for (size_t i = 0; i < {self.ndim}; ++i) {{
            grid_size[i] =
                std::round((coors_range[{self.ndim} + i] - coors_range[i]) / vsize[i]);
        }}
        for (int i = {self.ndim} - 1; i >= 0; --i) {{
            grid_stride[i] = prod;
            prod *= grid_size[i];
        }}
        return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
        """)
        ret_str = f"std::array<int, {self.ndim}>"
        retf_str = f"std::array<float, {self.ndim}>"
        retf2_str = f"std::array<float, {self.ndim * 2}>"

        return code.ret(
            f"std::tuple<{retf_str}, {ret_str}, {ret_str}, {retf2_str}>")

    @pccm.static_function
    def array2tvarray(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("N", "size_t")
        code.arg("arr", "std::array<T, N>")
        code.raw(f"""
        tv::array<T, N> tarr;
        for (int i = 0; i < N; ++i){{
            tarr[i] = arr[i];
        }}
        return tarr;
        """)
        return code.ret("tv::array<T, N>")

    @pccm.static_function
    def tvarray2array(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("N", "size_t")
        code.arg("arr", "tv::array<T, N>")
        code.raw(f"""
        std::array<T, N> tarr;
        for (int i = 0; i < N; ++i){{
            tarr[i] = arr[i];
        }}
        return tarr;
        """)
        return code.ret("std::array<T, N>")


class Point2VoxelKernel(pccm.ParameterizedClass, pccm.pybind.PybindClassMixin):
    """this class don't support multi-thread. 
    create p2v for every thread.
    """
    def __init__(self,
                 dtype: dtypes.DType,
                 ndim: int,
                 layout: TensorGeneric,
                 zyx: bool = True):
        super().__init__()
        self.add_dependency(TensorView, TensorViewHashKernel)
        self.add_param_class("layout_ns", layout, "Layout")
        self.dtype = dtype
        self.ndim = ndim
        self.zyx = zyx

    @pccm.cuda.cuda_global_function
    def build_hash_table(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", "TTable")
        code.arg("points", f"{self.dtype} const*")
        code.arg("points_indice_data", f"int64_t *")

        code.arg("point_stride", f"int")
        code.arg("vsize", f"tv::array<float, {self.ndim}>")
        code.arg("coors_range", f"tv::array<float, {self.ndim * 2}>")
        code.arg("grid_bound", f"tv::array<int, {self.ndim}>")
        code.arg("grid_stride", f"tv::array<int, {self.ndim}>")

        code.arg("num_points", f"int")
        point_xyz = f"{self.ndim - 1} - j"
        if not self.zyx:
            point_xyz = f"j"
        # if zyx, the coors_range and grid_bound is zyx too,
        # generated indices is zyx.
        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(num_points)){{
            bool failed = false;
            int c;
            int64_t prod = 0;
        #pragma unroll
            for (int j = 0; j < {self.ndim}; ++j) {{
                c = floor((points[i * point_stride + {point_xyz}] - coors_range[j]) /
                            vsize[j]);
                if ((c < 0 || c >= grid_bound[j])) {{
                    failed = true;
                }}
                prod += grid_stride[j] * c;
            }}
            if (!failed){{
                points_indice_data[i] = prod;
                table.insert(prod, i);
            }}else{{
                points_indice_data[i] = -1;
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def assign_table(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", "TTable")
        code.arg("indices", f"int*")
        code.arg("count", f"int*")
        code.arg("layout", f"Layout")
        code.arg("max_voxels", f"int")

        code.raw(f"""
        auto data = table.data();
        for (int i : tv::KernelLoopX<int>(table.size())){{
            auto &item = data[i];
            if (!item.empty()) {{
                item.second = tv::cuda::atomicAggInc(count);
                if (item.second < max_voxels){{
                    layout.inverse(item.first, indices + item.second * {self.ndim});
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def generate_voxel(self):
        code = pccm.FunctionCode()
        code.targ("TTable")
        code.arg("table", "TTable")
        code.arg("points", f"{self.dtype} const*")

        code.arg("points_indice_data", f"const int64_t*")
        code.arg("voxels", f"{self.dtype} *")
        code.arg("num_per_voxel", f"int *")
        code.arg("points_voxel_id", f"int64_t*")

        code.arg("point_stride", f"int")
        code.arg("max_points_per_voxel", f"int")
        code.arg("max_voxels", f"int")

        code.arg("vsize", f"tv::array<float, {self.ndim}>")
        code.arg("coors_range", f"tv::array<float, {self.ndim * 2}>")
        code.arg("grid_bound", f"tv::array<int, {self.ndim}>")
        code.arg("grid_stride", f"tv::array<int, {self.ndim}>")

        code.arg("num_points", f"int")
        # TODO add backward?
        code.raw(f"""
        int voxel_stride0 = point_stride * max_points_per_voxel;
        for (int i : tv::KernelLoopX<int>(num_points)){{
            int64_t prod = points_indice_data[i];
            int voxel_id = -1;
            if (prod != -1){{
                auto voxel_index_pair = table.lookup(prod);
                if (!voxel_index_pair.empty() &&
                    voxel_index_pair.second < max_voxels) {{
                    voxel_id = voxel_index_pair.second;
                    int old = atomicAdd(num_per_voxel + voxel_index_pair.second, 1);
                    if (old < max_points_per_voxel) {{
                        for (int j = 0; j < point_stride; ++j) {{
                            voxels[voxel_index_pair.second * voxel_stride0 + old * point_stride + j] = points[i * point_stride + j];
                        }}
                    }}
                }}
            }}
            points_voxel_id[i] = voxel_id;
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def voxel_empty_fill_mean(self):
        code = pccm.FunctionCode()
        code.arg("voxels", f"{self.dtype} *")

        code.arg("num_per_voxel", f"int *")
        code.arg("num_voxels", f"int")
        code.arg("num_points_per_voxel", f"int")
        code.arg("num_voxel_features", f"int")

        code.raw(f"""
        int voxel_stride = num_points_per_voxel * num_voxel_features;
        for (int i : tv::KernelLoopX<int>(num_voxels)){{
            int count = min(num_points_per_voxel, num_per_voxel[i]);
            num_per_voxel[i] = count;
            for (int j = 0; j < num_voxel_features; ++j){{
                auto voxel_ptr = voxels + i * voxel_stride + j;
                {self.dtype} sum_val = 0;
                for (int k = 0; k < count; ++k){{
                    sum_val += voxel_ptr[0];
                    voxel_ptr += num_voxel_features;
                }}
                sum_val = count == 0 ? 0 : sum_val / count;
                for (int k = count; k < num_points_per_voxel; ++k){{
                    voxel_ptr[0] = sum_val;
                    voxel_ptr += num_voxel_features;
                }}
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def limit_num_per_voxel_value(self):
        code = pccm.FunctionCode()
        code.arg("num_per_voxel", f"int *")
        code.arg("num_voxels, num_points_per_voxel", f"int")
        code.raw(f"""
        for (int i : tv::KernelLoopX<int>(num_voxels)){{
            int count = min(num_points_per_voxel, num_per_voxel[i]);
            num_per_voxel[i] = count;
        }}
        """)
        return code


class Point2Voxel(pccm.ParameterizedClass, pccm.pybind.PybindClassMixin):
    def __init__(self, dtype: dtypes.DType, ndim: int, zyx: bool = True):
        super().__init__()
        self.add_dependency(TensorView)
        self.p2v_c = Point2VoxelCommon(dtype, ndim, zyx)
        self.add_param_class("p2v_c", self.p2v_c, "Point2VoxelCommon")
        layout = TensorGeneric(ndim, True)
        self.add_param_class("layout_ns", layout, "Layout")
        self.dtype = dtype
        self.ndim = ndim
        self.zyx = zyx
        cuda_funcs = [
            self.point_to_voxel_hash, self.point_to_voxel_hash_static
        ]
        self.add_impl_only_param_class(
            cuda_funcs, "kernel", Point2VoxelKernel(dtype, ndim, layout, zyx))

        self.add_pybind_member("hashdata",
                               "tv::Tensor",
                               readwrite=False,
                               pyanno="cumm.tensorview.Tensor")
        self.add_pybind_member("point_indice_data",
                               "tv::Tensor",
                               readwrite=False,
                               pyanno="cumm.tensorview.Tensor")

        self.add_pybind_member("voxels", "tv::Tensor", readwrite=False)
        self.add_pybind_member("indices", "tv::Tensor", readwrite=False)
        self.add_pybind_member("num_per_voxel", "tv::Tensor", readwrite=False)
        self.add_member("vsize", f"tv::array<float, {self.ndim}>")
        self.add_member("coors_range", f"tv::array<float, {self.ndim * 2}>")
        self.add_member("grid_size", f"tv::array<int, {self.ndim}>")
        self.add_member("grid_stride", f"tv::array<int, {self.ndim}>")

    @pccm.pybind.mark_prop_getter(prop_name="grid_size")
    @pccm.member_function
    def get_grid_size(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::array<int, {self.ndim}> res;
        for (int i = 0; i < {self.ndim}; ++i){{
            res[i] = grid_size[i];
        }}
        return res;
        """)
        return code.ret(f"std::array<int, {self.ndim}>")

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.FunctionCode()
        code.arg("vsize_xyz", f"std::array<float, {self.ndim}>")
        code.arg("coors_range_xyz", f"std::array<float, {self.ndim * 2}>")
        code.arg("num_point_features", f"int")
        code.arg("max_num_voxels, max_num_points_per_voxel", f"int")
        if self.zyx:
            code.raw(f"""
            for (int i = 0; i < {self.ndim}; ++i){{
                vsize[{self.ndim - 1} - i] = vsize_xyz[i];
                coors_range[{self.ndim - 1} - i] = coors_range_xyz[i];
                coors_range[{2 * self.ndim - 1} - i] = coors_range_xyz[i + {self.ndim}];
            }}
            """)
        else:
            code.raw(f"""
            for (int i = 0; i < {self.ndim}; ++i){{
                vsize[i] = vsize_xyz[i];
                coors_range[i] = coors_range_xyz[i];
                coors_range[i + {self.ndim}] = coors_range_xyz[i + {self.ndim}];
            }}
            """)
        # if zyx, grid_size is zyx.
        code.raw(f"""
        int64_t prod = 1;
        for (size_t i = 0; i < {self.ndim}; ++i) {{
            grid_size[i] =
                std::round((coors_range[{self.ndim} + i] - coors_range[i]) / vsize[i]);
        }}
        for (int i = {self.ndim} - 1; i >= 0; --i) {{
            grid_stride[i] = prod;
            prod *= grid_size[i];
        }}
        voxels = tv::zeros({{max_num_voxels, max_num_points_per_voxel, num_point_features}}, tv::type_v<{self.dtype}>, 0);
        indices = tv::zeros({{max_num_voxels, {self.ndim}}}, tv::int32, 0);
        num_per_voxel = tv::zeros({{max_num_voxels}}, tv::int32, 0);
        hashdata = tv::zeros({{1}}, tv::custom128, 0);
        point_indice_data = tv::zeros({{1}}, tv::int64, 0);
        """)
        return code

    @pccm.pybind.mark
    @pccm.cuda.member_function
    def point_to_voxel_hash(self):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("clear_voxels", "bool", "true")
        code.arg("empty_mean", "bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        tv::Tensor points_voxel_id = tv::empty({{points.dim(0)}}, tv::int64, 0);
        int64_t expected_hash_data_num = points.dim(0) * 2;
        if (hashdata.dim(0) < expected_hash_data_num){{
            hashdata = tv::zeros({{expected_hash_data_num}}, tv::custom128, 0);
        }}
        if (point_indice_data.dim(0) < points.dim(0)){{
            point_indice_data = tv::zeros({{points.dim(0)}}, tv::int64, 0);
        }}
        return point_to_voxel_hash_static(points, voxels, indices, num_per_voxel, 
            hashdata, point_indice_data, points_voxel_id, Point2VoxelCommon::tvarray2array(vsize), 
            Point2VoxelCommon::tvarray2array(grid_size), Point2VoxelCommon::tvarray2array(grid_stride), 
            Point2VoxelCommon::tvarray2array(coors_range), clear_voxels, empty_mean, stream_int);
        """)
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def point_to_voxel_hash_static(self):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("voxels, indices, num_per_voxel, hashdata, point_indice_data, points_voxel_id",
                 "tv::Tensor")
        code.arg("vsize", f"std::array<float, {self.ndim}>")
        code.arg("grid_size, grid_stride", f"std::array<int, {self.ndim}>")
        code.arg("coors_range", f"std::array<float, {self.ndim * 2}>")
        code.arg("clear_voxels", "bool", "true")
        code.arg("empty_mean", "bool", "false")

        code.arg("stream_int", f"std::uintptr_t", "0")

        code.raw(f"""
        auto vsize_tv = Point2VoxelCommon::array2tvarray(vsize);
        auto grid_size_tv = Point2VoxelCommon::array2tvarray(grid_size);
        auto grid_stride_tv = Point2VoxelCommon::array2tvarray(grid_stride);
        auto coors_range_tv = Point2VoxelCommon::array2tvarray(coors_range);
        auto custream = reinterpret_cast<cudaStream_t>(stream_int);

        auto ctx = tv::Context();
        ctx.set_cuda_stream(custream);

        TV_ASSERT_INVALID_ARG(points.ndim() == 2 && points.dim(1) >= {self.ndim}, "error");
        using V = int64_t;
        using KeyType = int64_t;
        constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();
        if (clear_voxels){{
            voxels.zero_(ctx);
        }}
        using table_t =
            tv::hash::LinearHashTable<KeyType, V, tv::hash::Murmur3Hash<KeyType>,
                                        kEmptyKey, false>;
        using pair_t = typename table_t::value_type;
        // int64_t expected_hash_data_num = int64_t(tv::hash::align_to_power2(points.dim(0) * 2));
        int64_t expected_hash_data_num = points.dim(0) * 2;
        TV_ASSERT_RT_ERR(hashdata.dim(0) >= expected_hash_data_num, "hash table too small")
        TV_ASSERT_RT_ERR(point_indice_data.dim(0) >= points.dim(0), "point_indice_data too small")
        num_per_voxel.zero_(ctx);
        table_t hash = table_t(hashdata.data_ptr<pair_t>(), expected_hash_data_num);
        hash.clear(custream);
        auto launcher = tv::cuda::Launch(points.dim(0), custream);
        launcher(kernel::build_hash_table<table_t>, hash, points.data_ptr<const {self.dtype}>(),
                point_indice_data.data_ptr<int64_t>(),
                points.dim(1), vsize_tv, coors_range_tv, grid_size_tv, grid_stride_tv, points.dim(0));

        auto table_launcher = tv::cuda::Launch(hash.size(), custream);
        tv::Tensor count = tv::zeros({{1}}, tv::int32, 0);
        Layout layout = Layout::from_shape(grid_size_tv);
        table_launcher(kernel::assign_table<table_t>, hash, indices.data_ptr<int>(),
                        count.data_ptr<int>(),
                        layout, voxels.dim(0));
        auto count_cpu = count.cpu();
        int count_val = count_cpu.item<int32_t>();
        count_val = count_val > voxels.dim(0) ? voxels.dim(0) : count_val;
        launcher(kernel::generate_voxel<table_t>, hash, points.data_ptr<const {self.dtype}>(),
                point_indice_data.data_ptr<const int64_t>(), voxels.data_ptr<{self.dtype}>(),
                num_per_voxel.data_ptr<int>(), points_voxel_id.data_ptr<int64_t>(), points.dim(1), voxels.dim(1), 
                voxels.dim(0), vsize_tv, coors_range_tv,
                grid_size_tv, grid_stride_tv, points.dim(0));
        auto voxel_launcher = tv::cuda::Launch(count_val, custream);
        if (empty_mean){{
            launcher(kernel::voxel_empty_fill_mean, voxels.data_ptr<{self.dtype}>(),
                    num_per_voxel.data_ptr<int>(), count_val, 
                    voxels.dim(1), voxels.dim(2));
        }}else{{
            launcher(kernel::limit_num_per_voxel_value, num_per_voxel.data_ptr<int>(), count_val, 
                    voxels.dim(1));
        }}
        return std::make_tuple(voxels.slice_first_axis(0, count_val), 
            indices.slice_first_axis(0, count_val), 
            num_per_voxel.slice_first_axis(0, count_val));

        """)
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")


class Point2VoxelCPU(pccm.ParameterizedClass, pccm.pybind.PybindClassMixin):
    def __init__(self, dtype: dtypes.DType, ndim: int, zyx: bool = True):
        super().__init__()
        self.add_dependency(TensorView)
        layout = TensorGeneric(ndim, True)
        self.add_param_class("layout_ns", layout, "Layout")
        self.dtype = dtype
        self.ndim = ndim
        self.zyx = zyx
        self.p2v_c = Point2VoxelCommon(dtype, ndim, zyx)
        self.add_param_class("p2v_c", self.p2v_c, "Point2VoxelCommon")

        self.add_pybind_member("densehashdata",
                               "tv::Tensor",
                               readwrite=False,
                               pyanno="cumm.tensorview.Tensor")

        self.add_pybind_member("voxels", "tv::Tensor", readwrite=False)
        self.add_pybind_member("indices", "tv::Tensor", readwrite=False)
        self.add_pybind_member("num_per_voxel", "tv::Tensor", readwrite=False)

        self.add_member("vsize", f"tv::array<float, {self.ndim}>")
        self.add_member("coors_range", f"tv::array<float, {self.ndim * 2}>")
        self.add_member("grid_size", f"tv::array<int, {self.ndim}>")
        self.add_member("grid_stride", f"tv::array<int, {self.ndim}>")

    @pccm.pybind.mark_prop_getter(prop_name="grid_size")
    @pccm.member_function
    def get_grid_size(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::array<int, {self.ndim}> res;
        for (int i = 0; i < {self.ndim}; ++i){{
            res[i] = grid_size[i];
        }}
        return res;
        """)
        return code.ret(f"std::array<int, {self.ndim}>")

    @pccm.pybind.mark
    @pccm.static_function
    def calc_meta_data(self):
        code = pccm.FunctionCode()
        code.arg("vsize_xyz", f"std::array<float, {self.ndim}>")
        code.arg("coors_range_xyz", f"std::array<float, {self.ndim * 2}>")
        code.raw(f"""
        return Point2VoxelCommon::calc_meta_data(vsize_xyz, coors_range_xyz);
        """)
        return code.ret(self.p2v_c.calc_meta_ret)

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.FunctionCode()
        code.arg("vsize_xyz", f"std::array<float, {self.ndim}>")
        code.arg("coors_range_xyz", f"std::array<float, {self.ndim * 2}>")
        code.arg("num_point_features", f"int")
        code.arg("max_num_voxels, max_num_points_per_voxel", f"int")
        if self.zyx:
            code.raw(f"""
            for (int i = 0; i < {self.ndim}; ++i){{
                vsize[{self.ndim - 1} - i] = vsize_xyz[i];
                coors_range[{self.ndim - 1} - i] = coors_range_xyz[i];
                coors_range[{2 * self.ndim - 1} - i] = coors_range_xyz[i + {self.ndim}];
            }}
            """)
        else:
            code.raw(f"""
            for (int i = 0; i < {self.ndim}; ++i){{
                vsize[i] = vsize_xyz[i];
                coors_range[i] = coors_range_xyz[i];
                coors_range[i + {self.ndim}] = coors_range_xyz[i + {self.ndim}];
            }}
            """)
        code.raw(f"""
        int64_t prod = 1;
        for (size_t i = 0; i < {self.ndim}; ++i) {{
            grid_size[i] =
                std::round((coors_range[{self.ndim} + i] - coors_range[i]) / vsize[i]);
        }}
        for (int i = {self.ndim} - 1; i >= 0; --i) {{
            grid_stride[i] = prod;
            prod *= grid_size[i];
        }}
        voxels = tv::zeros({{max_num_voxels, max_num_points_per_voxel, num_point_features}}, tv::type_v<{self.dtype}>, -1);
        indices = tv::zeros({{max_num_voxels, {self.ndim}}}, tv::int32, -1);
        num_per_voxel = tv::zeros({{max_num_voxels}}, tv::int32, -1);
        tv::TensorShape grid_shape(grid_size.data(), grid_size.data() + {self.ndim});
        densehashdata = tv::zeros(grid_shape, tv::int32, -1);
        auto densehashdata_ptr = densehashdata.data_ptr<int>();
        for (int i= 0; i < densehashdata.size(); ++i){{
            densehashdata_ptr[i] = -1;
        }}
        """)
        return code

    def point_to_voxel_static_template(self, mean: bool = False):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("voxels, indices, num_per_voxel, densehashdata, points_voxel_id", "tv::Tensor")
        code.arg("vsize", f"std::array<float, {self.ndim}>")
        code.arg("grid_size, grid_stride", f"std::array<int, {self.ndim}>")
        code.arg("coors_range", f"std::array<float, {self.ndim * 2}>")

        code.arg("clear_voxels", "bool", "true")

        point_xyz = f"{self.ndim - 1} - j"
        if not self.zyx:
            point_xyz = f"j"
        code.raw(f"""
        auto max_num_voxels = voxels.dim(0);
        auto max_num_points_per_voxel = voxels.dim(1);
        num_per_voxel.zero_();
        if (clear_voxels){{
            voxels.zero_();
        }}
        auto points_voxel_id_ptr = points_voxel_id.data_ptr<int64_t>();
        int res_voxel_num = 0;
        int num_features = points.dim(1);
        auto N = points.dim(0);
        int c;
        TV_ASSERT_RT_ERR(num_features == voxels.dim(2), "your points num features doesn't equal to voxel.");
        tv::dispatch<float, double>(points.dtype(), [&](auto I){{
            using T = decltype(I);
            auto points_rw = points.tview<T, 2>();
            auto coors_rw = indices.tview<int, 2>();
            auto voxels_rw = voxels.tview<{self.dtype}, 3>();
            auto num_points_per_voxel_rw = num_per_voxel.tview<int, 1>();
            
            int coor[{self.ndim}];
            auto coor_to_voxelidx_rw = densehashdata.tview<int, {self.ndim}>();
            int voxelidx, num;
            bool failed;
            int voxel_num = 0;
            for (int i = 0; i < N; ++i) {{
                failed = false;
                for (int j = 0; j < {self.ndim}; ++j) {{
                    c = floor((points_rw(i, {point_xyz}) - coors_range[j]) / vsize[j]);
                    if ((c < 0 || c >= grid_size[j])) {{
                        failed = true;
                        break;
                    }}
                    coor[j] = c;
                }}
                if (failed){{
                    points_voxel_id_ptr[i] = -1;
                    continue;
                }}
                voxelidx = coor_to_voxelidx_rw({codeops.unpack("coor", range(self.ndim))});
                
                if (voxelidx == -1) {{
                    voxelidx = voxel_num;
                    if (voxel_num >= max_num_voxels){{
                        points_voxel_id_ptr[i] = -1;
                        continue;
                    }}
                    voxel_num += 1;
                    coor_to_voxelidx_rw({codeops.unpack("coor", range(self.ndim))}) = voxelidx;
                    for (int k = 0; k < {self.ndim}; ++k) {{
                        coors_rw(voxelidx, k) = coor[k];
                    }}
                }}
                points_voxel_id_ptr[i] = voxelidx;
                num = num_points_per_voxel_rw(voxelidx);
                if (num < max_num_points_per_voxel) {{
                    // voxel_point_mask_rw(voxelidx, num) = {self.dtype}(1);
                    for (int k = 0; k < num_features; ++k) {{
                        voxels_rw(voxelidx, num, k) = points_rw(i, k);
                    }}
                    num_points_per_voxel_rw(voxelidx) += 1;
                }}
            }}
            std::vector<{self.dtype}> mean_value(num_features);
            for (int i = 0; i < voxel_num; ++i) {{
                coor_to_voxelidx_rw({codeops.unpack("coors_rw", range(self.ndim), left="(i, ", right=")")}) = -1;
                if TV_IF_CONSTEXPR ({pccm.boolean(mean)}){{
                    num = num_points_per_voxel_rw(i);
                    if (num > 0){{
                        mean_value.clear();
                        for (int j = 0; j < num; ++j) {{
                            for (int k = 0; k < num_features; ++k) {{
                                mean_value[k] += voxels_rw(i, j, k);
                            }}
                        }}
                        for (int k = 0; k < num_features; ++k){{
                            mean_value[k] /= num;
                        }}
                        for (int j = num; j < max_num_points_per_voxel; ++j) {{
                            for (int k = 0; k < num_features; ++k) {{
                                voxels_rw(i, j, k) = mean_value[k];
                            }}
                        }}
                    }}
                }}
            }}
            res_voxel_num = voxel_num;
        }});
        return std::make_tuple(voxels.slice_first_axis(0, res_voxel_num), 
            indices.slice_first_axis(0, res_voxel_num), 
            num_per_voxel.slice_first_axis(0, res_voxel_num));
        """)
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")

    @pccm.static_function
    def array2tvarray(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("N", "size_t")
        code.arg("arr", "std::array<T, N>")
        code.raw(f"""
        tv::array<T, N> tarr;
        for (int i = 0; i < N; ++i){{
            tarr[i] = arr[i];
        }}
        return tarr;
        """)
        return code.ret("tv::array<T, N>")

    @pccm.static_function
    def tvarray2array(self):
        code = pccm.FunctionCode()
        code.targ("T")
        code.nontype_targ("N", "size_t")
        code.arg("arr", "tv::array<T, N>")
        code.raw(f"""
        std::array<T, N> tarr;
        for (int i = 0; i < N; ++i){{
            tarr[i] = arr[i];
        }}
        return tarr;
        """)
        return code.ret("std::array<T, N>")

    @pccm.pybind.mark
    @pccm.static_function
    def point_to_voxel_static(self):
        return self.point_to_voxel_static_template(False)

    @pccm.pybind.mark
    @pccm.static_function
    def point_to_voxel_empty_mean_static(self):
        return self.point_to_voxel_static_template(True)

    @pccm.pybind.mark
    @pccm.member_function
    def point_to_voxel(self):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("clear_voxels", "bool", "true")
        code.raw(f"""
        tv::Tensor points_voxel_id = tv::empty({{points.dim(0)}}, tv::int64, -1);

        return point_to_voxel_static(points, voxels, indices, num_per_voxel, densehashdata, 
            points_voxel_id, tvarray2array(vsize), 
            tvarray2array(grid_size), tvarray2array(grid_stride), 
            tvarray2array(coors_range), clear_voxels);
        """)
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")

    @pccm.pybind.mark
    @pccm.member_function
    def point_to_voxel_empty_mean(self):
        code = pccm.FunctionCode()
        code.arg("points", "tv::Tensor")
        code.arg("clear_voxels", "bool", "true")
        code.raw(f"""
        tv::Tensor points_voxel_id = tv::empty({{points.dim(0)}}, tv::int64, -1);

        return point_to_voxel_empty_mean_static(points, voxels, indices, num_per_voxel, 
            densehashdata, points_voxel_id, tvarray2array(vsize), 
            tvarray2array(grid_size), tvarray2array(grid_stride), 
            tvarray2array(coors_range), clear_voxels);
        """)
        return code.ret("std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>")
