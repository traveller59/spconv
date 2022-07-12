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

from typing import Optional
import torch 
from cumm import tensorview as tv 
from spconv.pytorch.cppcore import torch_tensor_to_tv, get_current_stream

from spconv.core_cc.csrc.hash.core import HashTable as _HashTable

_TORCH_DTYPE_TO_ITEMSIZE = {
    torch.int32: 4,
    torch.int64: 8,
    torch.float32: 4,
    torch.float64: 8,
}

class HashTable:
    """simple hash table for 32 and 64 bit data. support both cpu and cuda.
    for cuda, it's a fixed-size table, you must provide maximum size 
    (recommend 2 * num).
    key must be int32/int64.
    see spconv/pytorch/functional/sparse_add_hash_based, a real example
    that show how to use hash table to implement 
    sparse add (same shape, different indices)
    """
    def __init__(self, device: torch.device, key_dtype: torch.dtype, 
                value_dtype: torch.dtype, 
                max_size: int = -1) -> None:
        is_cpu = device.type == "cpu"
        self.is_cpu = is_cpu
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        key_data_tv = tv.Tensor()
        value_data_tv = tv.Tensor()
        if is_cpu:
            self.keys_data = None 
            self.values_data = None 
        else:
            assert max_size > 0, "you must provide max_size for fixed-size cuda hash table, usually *2 of num of keys"
            assert device is not None, "you must specify device for cuda hash table."
            self.keys_data = torch.empty([max_size], dtype=key_dtype, device=device)
            self.values_data = torch.empty([max_size], dtype=value_dtype, device=device)
            key_data_tv = torch_tensor_to_tv(self.keys_data)
            value_data_tv = torch_tensor_to_tv(self.values_data)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        self.key_itemsize = _TORCH_DTYPE_TO_ITEMSIZE[self.key_dtype]
        self.value_itemsize = _TORCH_DTYPE_TO_ITEMSIZE[self.value_dtype]
        self._valid_value_dtype_for_arange = set([torch.int32, torch.int64])

        self._table = _HashTable(is_cpu, self.key_itemsize, self.value_itemsize, key_data_tv, value_data_tv, stream)


    def insert(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None):
        """insert hash table by keys and values
        if values is None, only key is inserted, the value is undefined.
        """
        keys_tv = torch_tensor_to_tv(keys)
        values_tv = tv.Tensor()
        if values is not None:
            values_tv = torch_tensor_to_tv(values)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()

        return self._table.insert(keys_tv, values_tv, stream)

    def query(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None):
        """query value by keys, if values is not None, create a new one.
        return values and a uint8 tensor that whether query fail.
        """
        keys_tv = torch_tensor_to_tv(keys)
        if values is None:
            values = torch.empty([keys.shape[0]], dtype=self.value_dtype, device=keys.device)
        values_tv = torch_tensor_to_tv(values)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        is_empty = torch.empty([keys.shape[0]], dtype=torch.uint8, device=keys.device)
        is_empty_tv = torch_tensor_to_tv(is_empty)
        self._table.query(keys_tv, values_tv, is_empty_tv, stream)
        return values, is_empty > 0

    def insert_exist_keys(self, keys: torch.Tensor, values: torch.Tensor):
        """insert kv that k exists in table. return a uint8 tensor that
        whether insert fail.
        """
        keys_tv = torch_tensor_to_tv(keys)
        values_tv = torch_tensor_to_tv(values)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        is_empty = torch.empty([keys.shape[0]], dtype=torch.uint8, device=keys.device)
        is_empty_tv = torch_tensor_to_tv(is_empty)
        self._table.insert_exist_keys(keys_tv, values_tv, is_empty_tv, stream)
        return is_empty

    def assign_arange_(self):
        """iterate table, assign values with "arange" value.
        equivalent to 1. get key by items(), 2. use key and arange(key.shape[0]) to insert
        """
        count_tv = tv.Tensor()
        count = torch.Tensor()
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        else:
            assert self.value_dtype in self._valid_value_dtype_for_arange
        if not self.is_cpu:
            assert self.values_data is not None
            if self.key_itemsize == 4:
                count = torch.zeros([1], dtype=torch.int32, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint32)
            elif self.key_itemsize == 8:
                count = torch.zeros([1], dtype=torch.int64, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint64)
            else:
                raise NotImplementedError
        else:
            max_size = self._table.size_cpu()
            count = torch.tensor([max_size], dtype=torch.int64)

        self._table.assign_arange_(count_tv, stream)
        return count

    def items(self, max_size: int = -1):
        count_tv = tv.Tensor()
        count = torch.Tensor()
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        if not self.is_cpu:
            assert self.values_data is not None
            if self.key_itemsize == 4:
                count = torch.zeros([1], dtype=torch.int32, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint32)
            elif self.key_itemsize == 8:
                count = torch.zeros([1], dtype=torch.int64, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint64)
            else:
                raise NotImplementedError
        if not self.is_cpu:
            assert self.values_data is not None
            if max_size == -1:
                max_size = self.values_data.shape[0]
            keys = torch.empty([max_size], dtype=self.key_dtype, device=self.values_data.device)
            values = torch.empty([max_size], dtype=self.value_dtype, device=self.values_data.device)

        else:
            max_size = self._table.size_cpu()
            count = torch.tensor([max_size], dtype=torch.int64)
            keys = torch.empty([max_size], dtype=self.key_dtype)
            values = torch.empty([max_size], dtype=self.value_dtype)
        keys_tv = torch_tensor_to_tv(keys)
        values_tv = torch_tensor_to_tv(values)
        self._table.items(keys_tv, values_tv, count_tv, stream)
        return keys, values, count


def main():
    is_cpus = [True, False]
    max_size = 1000
    k_dtype = torch.int32 
    v_dtype = torch.int64
    for is_cpu in is_cpus:
        if is_cpu:
            dev = torch.device("cpu")
            table = HashTable(dev, k_dtype, v_dtype)
        else:
            dev = torch.device("cuda:0")

            table = HashTable(dev, k_dtype, v_dtype, max_size=max_size)

        keys = torch.tensor([5, 3, 7, 4, 6, 2, 10, 8], dtype=k_dtype, device=dev)
        values = torch.tensor([1, 6, 4, 77, 23, 756, 12, 12], dtype=v_dtype, device=dev)
        keys_query = torch.tensor([8, 10, 2, 6, 4, 7, 3, 5], dtype=k_dtype, device=dev)

        table.insert(keys, values)

        vq, _ = table.query(keys_query)
        print(vq)
        ks, vs, cnt = table.items()
        cnt_item = cnt.item()
        print(cnt, ks[:cnt_item], vs[:cnt_item])

        table.assign_arange_()
        ks, vs, cnt = table.items()
        cnt_item = cnt.item()
        print(cnt, ks[:cnt_item], vs[:cnt_item])

if __name__ == "__main__":
    main()