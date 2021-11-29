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

import os
from pathlib import Path
from typing import List
from cumm.constants import CUMM_CPU_ONLY_BUILD

import pccm
from cumm import dtypes
from cumm.common import (TensorView, TensorViewCPU, TensorViewHashKernel,
                         TensorViewKernel, TslRobinMap)
from spconv.csrc.sparse.cpu_core import OMPLib

if CUMM_CPU_ONLY_BUILD:
    _member_func = pccm.member_function
else:
    _member_func = pccm.cuda.member_function

def _dispatch_ints(code: pccm.FunctionCode, ints: List[int], var: str):
    for i, val in enumerate(ints):
        if i == 0:
            with code.if_(f"{var} == {val}"):
                yield val 
        else:
            with code.else_if_(f"{var} == {val}"):
                yield val 
    with code.else_():
        code.raw(f"""
        TV_THROW_RT_ERR("unknown val {var}, available: {ints}")
        """)

def _dispatch(code: pccm.FunctionCode, dts: List[dtypes.DType], var: str):
    for i, dtype in enumerate(dts):
        if i == 0:
            with code.if_(f"{var} == tv::DType({dtype.tv_dtype})"):
                yield dtype 
        else:
            with code.else_if_(f"{var} == tv::DType({dtype.tv_dtype})"):
                yield dtype 
    with code.else_():
        code.raw(f"""
        TV_THROW_RT_ERR("unknown dtype {var}, available: {dts}")
        """)

class HashTableKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, TensorViewHashKernel, TensorViewKernel)

    @pccm.cuda.cuda_global_function
    def insert_exist_keys_kernel(self):
        code = pccm.FunctionCode()
        code.targ("THashTableSplit")
        code.arg("table", "THashTableSplit")
        code.arg("key_ptr", "const typename THashTableSplit::key_type *__restrict__")
        code.arg("value_ptr", "const typename THashTableSplit::mapped_type *__restrict__")
        code.arg("is_empty_ptr", "uint8_t*")
        code.arg("size", "size_t")

        code.raw(f"""
        auto value_data = table.value_ptr();
        for (size_t i : tv::KernelLoopX<size_t>(size)){{
            auto key = key_ptr[i];
            auto offset = table.lookup_offset(key);
            is_empty_ptr[i] = offset == -1;
            if (offset != -1){{
                value_data[offset] = value_ptr[i];
            }}
        }}
        """)
        return code 

class HashTable(pccm.Class, pccm.pybind.PybindClassMixin):
    """a simple hashtable for both cpu and cuda.
    CPU implementation don't support parallel.
    both cpu and cuda only support 32/64bit key value.
    """
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, TslRobinMap)
        if CUMM_CPU_ONLY_BUILD:
            self.add_dependency(OMPLib)
        self.add_include("tensorview/parallel/all.h")
        self.add_member("keys_data, values_data", "tv::Tensor")
        self.add_pybind_member("key_itemsize_", "int", prop_name="key_itemsize", readwrite=False)
        self.add_pybind_member("value_itemsize_", "int", prop_name="value_itemsize", readwrite=False)

        self.add_pybind_member("is_cpu", "bool", readwrite=False)
        self.add_member("map_4_4", "tsl::robin_map<uint32_t, uint32_t>")
        self.add_member("map_4_8", "tsl::robin_map<uint32_t, uint64_t>")
        self.add_member("map_8_4", "tsl::robin_map<uint64_t, uint32_t>")
        self.add_member("map_8_8", "tsl::robin_map<uint64_t, uint64_t>")
        self.add_pybind_member("insert_count_", "int64_t", prop_name="insert_count", readwrite=False)

    @pccm.pybind.mark 
    @pccm.constructor
    def ctor(self):
        code = pccm.FunctionCode()
        code.arg("is_cpu", "bool")
        code.arg("key_itemsize, value_itemsize", "int")
        code.arg("keys_data", "tv::Tensor")
        code.arg("values_data", "tv::Tensor")
        code.arg("stream", "std::uintptr_t", "0")

        code.ctor_init("is_cpu", "is_cpu")
        code.ctor_init("keys_data", "keys_data")
        code.ctor_init("values_data", "values_data")
        code.ctor_init("key_itemsize_", "key_itemsize")
        code.ctor_init("value_itemsize_", "value_itemsize")

        code.ctor_init("insert_count_", "0")

        code.raw(f"""
        TV_ASSERT_RT_ERR(key_itemsize == 4 || key_itemsize == 8, "key_itemsize must be 4 or 8");
        TV_ASSERT_RT_ERR(value_itemsize == 4 || value_itemsize == 8, "value_itemsize must be 4 or 8");

        if (!is_cpu){{
            TV_ASSERT_RT_ERR(!keys_data.empty() && !values_data.empty(), "key and value must not empty");
            TV_ASSERT_RT_ERR(keys_data.dim(0) == values_data.dim(0), "key and value must have same size");
            TV_ASSERT_RT_ERR(key_itemsize == keys_data.itemsize(), "key_itemsize must equal to key_data");
            TV_ASSERT_RT_ERR(value_itemsize == values_data.itemsize(), "value_itemsize must equal to values_data");
            // clear cuda table here.
            clear(stream);
        }}
        """)
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_ASSERT_RT_ERR(is_cpu, \"spconv not built with CUDA\");")
        return code 

    @pccm.pybind.mark 
    @_member_func
    def clear(self):
        """ in this function, if values is empty, it will be assigned to zero.
        """
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(TensorViewHashKernel)
        code.arg("stream", "std::uintptr_t", "0")
        with code.if_("is_cpu"):
            code.raw(f"""
            if (is_cpu){{
                map_4_4.clear();
                map_4_8.clear();
                map_8_4.clear();
                map_8_8.clear();
                return;
            }}
            """)
        if not CUMM_CPU_ONLY_BUILD:
            with code.else_():
                code.raw(f"""
                auto custream = reinterpret_cast<cudaStream_t>(stream);
                """)
                for k_items in _dispatch_ints(code, [4, 8], "keys_data.itemsize()"):
                    code.raw(f"""
                    using K = tv::hash::itemsize_to_unsigned_t<{k_items}>;
                    constexpr K kEmptyKey = std::numeric_limits<K>::max();

                    K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
                    """)
                    for v_items in _dispatch_ints(code, [4, 8], "values_data.itemsize()"):
                        code.raw(f"""
                        using V = tv::hash::itemsize_to_unsigned_t<{v_items}>;
                        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
                        using table_t =
                            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                        kEmptyKey, false>;
                        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
                        tv::cuda::Launch launcher(table.size(), custream);
                        launcher(tv::hash::clear_table_split<table_t>, table);
                        """)
        return code 


    @pccm.pybind.mark 
    @_member_func
    def insert(self):
        """ in this function, if values is empty, it will be assigned to zero.
        """
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(TensorViewHashKernel)
        code.arg("keys", "tv::Tensor")
        code.arg("values", "tv::Tensor", "tv::Tensor()", pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("stream", "std::uintptr_t", "0")

        code.raw(f"""
        if (!is_cpu){{
            int64_t value_after_insert = keys.dim(0) + insert_count_;
            TV_ASSERT_RT_ERR(value_after_insert < keys_data.dim(0), "inserted count exceed maximum hash size");
            insert_count_ += keys.dim(0);
        }}
        auto N = keys.dim(0);
        TV_ASSERT_RT_ERR(keys.itemsize() == key_itemsize_, "keys itemsize not equal to", key_itemsize_);
        if (!values.empty()){{
            TV_ASSERT_RT_ERR(values.itemsize() == value_itemsize_, "values itemsize not equal to", value_itemsize_);
            TV_ASSERT_RT_ERR(keys.dim(0) == values.dim(0), "number of key and value must same");
        }}
        """)
        with code.if_("is_cpu"):
            map_name = "cpu_map"
            for k_type, v_type in self.cpu_map_storage_select("key_itemsize_", "value_itemsize_", map_name, code):
                code.raw(f"""
                auto k_ptr = reinterpret_cast<const {k_type}*>(keys.raw_data());
                if (values.empty()){{
                    for (size_t i = 0; i < N; ++i){{
                        {map_name}.insert({{k_ptr[i], {v_type}(0)}});
                    }}
                }}
                else{{
                    auto v_ptr = reinterpret_cast<const {v_type}*>(values.raw_data());
                    for (size_t i = 0; i < N; ++i){{
                        {map_name}.insert({{k_ptr[i], v_ptr[i]}});
                    }}
                }}
                """)
        if not CUMM_CPU_ONLY_BUILD:
            with code.else_():
                code.raw(f"""
                auto custream = reinterpret_cast<cudaStream_t>(stream);
                """)
                for k_items in _dispatch_ints(code, [4, 8], "keys_data.itemsize()"):
                    code.raw(f"""
                    using K = tv::hash::itemsize_to_unsigned_t<{k_items}>;
                    constexpr K kEmptyKey = std::numeric_limits<K>::max();

                    K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
                    const K* key_ptr = reinterpret_cast<const K*>(keys.raw_data());

                    """)
                    for v_items in _dispatch_ints(code, [4, 8], "values_data.itemsize()"):
                        code.raw(f"""
                        using V = tv::hash::itemsize_to_unsigned_t<{v_items}>;
                        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
                        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());

                        using table_t =
                            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                        kEmptyKey, false>;
                        tv::cuda::Launch launcher(N, custream);
                        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
                        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
                        """)
        else:
            code.raw(f"""
            TV_THROW_RT_ERR("spconv not compiled with cuda, don't support cuda");
            """)
        return code 

    @pccm.pybind.mark 
    @_member_func
    def query(self):
        """query keys, save to values, and save is_empty to is_empty
        """
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(TensorViewHashKernel)
        code.arg("keys", "tv::Tensor")
        code.arg("values", "tv::Tensor")
        code.arg("is_empty", "tv::Tensor")

        code.arg("stream", "std::uintptr_t")

        code.raw(f"""
        auto N = keys.dim(0);
        TV_ASSERT_RT_ERR(keys.itemsize() == key_itemsize_, "keys itemsize not equal to", key_itemsize_);
        TV_ASSERT_RT_ERR(values.itemsize() == value_itemsize_, "values itemsize not equal to", value_itemsize_);
        TV_ASSERT_RT_ERR(N == values.dim(0) && is_empty.dim(0) == N, "number of key and value must same");
        auto is_empty_ptr = is_empty.data_ptr<uint8_t>();
        """)
        with code.if_("is_cpu"):
            map_name = "cpu_map"
            # here it's safe to use omp in query.
            for k_type, v_type in self.cpu_map_storage_select("key_itemsize_", "value_itemsize_", map_name, code):
                code.raw(f"""
                auto k_ptr = reinterpret_cast<{k_type}*>(keys.raw_data());
                auto v_ptr = reinterpret_cast<{v_type}*>(values.raw_data());
                tv::kernel_1d_cpu(keys.device(), N, [&](size_t begin, size_t end, size_t step){{
                    bool emp;
                    for (size_t i = begin; i < end; i += step){{
                        auto iter = {map_name}.find(k_ptr[i]);
                        emp = iter == {map_name}.end();
                        if (!emp){{
                            v_ptr[i] = iter->second;
                        }}
                        is_empty_ptr[i] = uint8_t(emp);
                    }}
                }});
                """)
        if not CUMM_CPU_ONLY_BUILD:
            with code.else_():
                code.raw(f"""
                auto custream = reinterpret_cast<cudaStream_t>(stream);
                """)
                for k_items in _dispatch_ints(code, [4, 8], "keys_data.itemsize()"):
                    code.raw(f"""
                    using K = tv::hash::itemsize_to_unsigned_t<{k_items}>;
                    constexpr K kEmptyKey = std::numeric_limits<K>::max();
                    K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
                    K* key_ptr = reinterpret_cast<K*>(keys.raw_data());

                    """)
                    for v_items in _dispatch_ints(code, [4, 8], "values_data.itemsize()"):
                        code.raw(f"""
                        using V = tv::hash::itemsize_to_unsigned_t<{v_items}>;
                        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
                        V* value_ptr = reinterpret_cast<V*>(values.raw_data());
                        using table_t =
                            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                        kEmptyKey, false>;
                        tv::cuda::Launch launcher(N, custream);
                        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
                        launcher(tv::hash::query_split<table_t>, table, key_ptr, value_ptr, is_empty_ptr, size_t(N));
                        """)
        else:
            code.raw(f"""
            TV_THROW_RT_ERR("spconv not compiled with cuda, don't support cuda");
            """)
        return code 

    @pccm.pybind.mark 
    @_member_func
    def assign_arange_(self):
        """ this function assign "arange(NumItem)" to table values.
        useful in "unique-like" operations.
        unlike insert/query, this method only support i32/i64/u32/u64 for value.
        count must be u32/u64.
        """
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(TensorViewHashKernel)
        code.arg("count", "tv::Tensor")

        code.arg("stream", "std::uintptr_t", "0")
        with code.if_("is_cpu"):
            map_name = "cpu_map"
            for k_type, v_type in self.cpu_map_storage_select("key_itemsize_", "value_itemsize_", map_name, code):
                code.raw(f"""
                {v_type} index = 0;
                for (auto it = {map_name}.begin(); it != {map_name}.end(); ++it){{
                    it.value() = index;
                    ++index;
                }}

                """)
        if not CUMM_CPU_ONLY_BUILD:
            with code.else_():
                code.raw(f"""
                TV_ASSERT_RT_ERR(count.device() == 0, "count must be cuda");
                auto custream = reinterpret_cast<cudaStream_t>(stream);
                """)
                for k_items in _dispatch_ints(code, [4, 8], "keys_data.itemsize()"):
                    code.raw(f"""
                    using K = tv::hash::itemsize_to_unsigned_t<{k_items}>;
                    constexpr K kEmptyKey = std::numeric_limits<K>::max();
                    auto count_ptr = count.data_ptr<K>();

                    K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
                    """)
                    val_dtypes = [dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64]
                    for v_dtype in _dispatch(code, val_dtypes, "values_data.dtype()"):
                        code.raw(f"""
                        using V = {v_dtype};
                        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
                        using table_t =
                            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                        kEmptyKey, false>;
                        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
                        tv::cuda::Launch launcher(table.size(), custream);
                        launcher(tv::hash::assign_arange_split<table_t, K>, table, count_ptr);
                        """)
        else:
            code.raw(f"""
            TV_THROW_RT_ERR("spconv not compiled with cuda, don't support cuda");
            """)
        return code 

    @pccm.pybind.mark 
    @_member_func
    def size_cpu(self):
        """ this function can only be used to get cpu hash table size.
        """
        code = pccm.FunctionCode()
        code.raw(f"""
        int64_t res = -1;
        TV_ASSERT_RT_ERR(is_cpu, "size_cpu can only be used in cpu hash table");
        """)
        with code.if_("is_cpu"):
            map_name = "cpu_map"
            for _ in self.cpu_map_storage_select("key_itemsize_", "value_itemsize_", map_name, code):
                code.raw(f"""
                res = {map_name}.size();
                """)
        code.raw(f"return res;")
        return code.ret("int64_t")


    @pccm.pybind.mark 
    @_member_func
    def items(self):
        """get items.
        """
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(TensorViewHashKernel)
        code.arg("keys", "tv::Tensor")
        code.arg("values", "tv::Tensor")
        code.arg("count", "tv::Tensor")

        code.arg("stream", "std::uintptr_t")

        code.raw(f"""
        auto N = keys.dim(0);
        TV_ASSERT_RT_ERR(keys.itemsize() == key_itemsize_, "keys itemsize not equal to", key_itemsize_);
        TV_ASSERT_RT_ERR(values.itemsize() == value_itemsize_, "values itemsize not equal to", value_itemsize_);
        TV_ASSERT_RT_ERR(N == values.dim(0), "number of key and value must same");
        
        """)
        with code.if_("is_cpu"):
            map_name = "cpu_map"
            # here it's safe to use omp in query.
            for k_type, v_type in self.cpu_map_storage_select("key_itemsize_", "value_itemsize_", map_name, code):
                code.raw(f"""
                auto k_ptr = reinterpret_cast<{k_type}*>(keys.raw_data());
                auto v_ptr = reinterpret_cast<{v_type}*>(values.raw_data());
                {v_type} index = 0;
                for (auto it = {map_name}.begin(); it != {map_name}.end(); ++it){{
                    if (index >= N){{
                        break;
                    }}
                    k_ptr[index] = it->first;
                    v_ptr[index] = it->second;
                    ++index;
                }}
                """)
        if not CUMM_CPU_ONLY_BUILD:
            with code.else_():
                code.raw(f"""
                auto custream = reinterpret_cast<cudaStream_t>(stream);
                """)
                for k_items in _dispatch_ints(code, [4, 8], "keys_data.itemsize()"):
                    code.raw(f"""
                    using K = tv::hash::itemsize_to_unsigned_t<{k_items}>;
                    auto count_ptr = count.data_ptr<K>();

                    constexpr K kEmptyKey = std::numeric_limits<K>::max();
                    K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
                    K* key_ptr = reinterpret_cast<K*>(keys.raw_data());

                    """)
                    for v_items in _dispatch_ints(code, [4, 8], "values_data.itemsize()"):
                        code.raw(f"""
                        using V = tv::hash::itemsize_to_unsigned_t<{v_items}>;
                        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
                        V* value_ptr = reinterpret_cast<V*>(values.raw_data());
                        using table_t =
                            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                        kEmptyKey, false>;
                        tv::cuda::Launch launcher(N, custream);
                        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
                        launcher(tv::hash::iterate_table_split<table_t, K>, table, key_ptr, value_ptr, size_t(N), count_ptr);
                        """)
        else:
            code.raw(f"""
            TV_THROW_RT_ERR("spconv not compiled with cuda, don't support cuda");
            """)
        return code 

    @pccm.pybind.mark 
    @_member_func
    def insert_exist_keys(self):
        """insert v of given k if k exists. won't insert any new key.
        """
        code = pccm.FunctionCode()
        if not CUMM_CPU_ONLY_BUILD:
            code.add_dependency(TensorViewHashKernel, HashTableKernel)
        code.arg("keys", "tv::Tensor")
        code.arg("values", "tv::Tensor")
        code.arg("is_empty", "tv::Tensor")

        code.arg("stream", "std::uintptr_t")

        code.raw(f"""
        auto N = keys.dim(0);
        TV_ASSERT_RT_ERR(keys.itemsize() == key_itemsize_, "keys itemsize not equal to", key_itemsize_);
        TV_ASSERT_RT_ERR(values.itemsize() == value_itemsize_, "values itemsize not equal to", value_itemsize_);
        TV_ASSERT_RT_ERR(N == values.dim(0) && is_empty.dim(0) == N, "number of key and value must same");
        auto is_empty_ptr = is_empty.data_ptr<uint8_t>();
        """)
        with code.if_("is_cpu"):
            map_name = "cpu_map"
            # here it's safe to use omp in query.
            for k_type, v_type in self.cpu_map_storage_select("key_itemsize_", "value_itemsize_", map_name, code):
                code.raw(f"""
                auto k_ptr = reinterpret_cast<{k_type}*>(keys.raw_data());
                auto v_ptr = reinterpret_cast<{v_type}*>(values.raw_data());
                tv::kernel_1d_cpu(keys.device(), N, [&](size_t begin, size_t end, size_t step){{
                    bool emp;
                    for (size_t i = begin; i < end; i += step){{
                        auto iter = {map_name}.find(k_ptr[i]);
                        emp = iter == {map_name}.end();
                        if (!emp){{
                            iter.value() = v_ptr[i];
                        }}
                        is_empty_ptr[i] = uint8_t(emp);
                    }}
                }});
                """)
        if not CUMM_CPU_ONLY_BUILD:
            with code.else_():
                code.raw(f"""
                auto custream = reinterpret_cast<cudaStream_t>(stream);
                """)
                for k_items in _dispatch_ints(code, [4, 8], "keys_data.itemsize()"):
                    code.raw(f"""
                    using K = tv::hash::itemsize_to_unsigned_t<{k_items}>;
                    constexpr K kEmptyKey = std::numeric_limits<K>::max();
                    K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
                    const K* key_ptr = reinterpret_cast<const K*>(keys.raw_data());

                    """)
                    for v_items in _dispatch_ints(code, [4, 8], "values_data.itemsize()"):
                        code.raw(f"""
                        using V = tv::hash::itemsize_to_unsigned_t<{v_items}>;
                        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
                        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
                        using table_t =
                            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                                        kEmptyKey, false>;
                        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
                        tv::cuda::Launch launcher(N, custream);
                        launcher(insert_exist_keys_kernel<table_t>, table, key_ptr, value_ptr, is_empty_ptr, size_t(N));
                        """)
        else:
            code.raw(f"""
            TV_THROW_RT_ERR("spconv not compiled with cuda, don't support cuda");
            """)
        return code 

    def cpu_map_storage_select(self, k_itemsize: str, v_itemsize: str, res_var: str, code: pccm.FunctionCode):
        different_kvs = [(4, 4), (4, 8), (8, 4), (8, 8)]
        item_size_to_dtype = {
            4: "uint32_t",
            8: "uint64_t",
        }
        with code.block(""):
            code.raw("bool found = false;")
            for kit, vit in different_kvs:
                with code.if_(f"{k_itemsize} == {kit} && {v_itemsize} == {vit}"):
                    code.raw(f"auto& {res_var} = map_{kit}_{vit};")
                    yield item_size_to_dtype[kit], item_size_to_dtype[vit]
                    code.raw(f"found = true;")
            code.raw("TV_ASSERT_RT_ERR(found, \"suitable hash table not found.\");")
        

