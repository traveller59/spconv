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

import torch 
from spconv.pytorch.hash import HashTable


def main():
    """Fixed-Size CUDA Hash Table:
    this hash table can't delete keys after insert, and can't resize.
    You need to pre-define a fixed-length of hash table, recommend 2x size
    of your key num.

    """
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

        print("----------Insert Exist Keys----------")
        is_empty = table.insert_exist_keys(keys, values)
        ks, vs, cnt = table.items()
        cnt_item = cnt.item()
        print(cnt, ks[:cnt_item], vs[:cnt_item])


if __name__ == "__main__":
    main()