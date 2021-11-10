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

import numpy as np
from cumm import tensorview as tv
from spconv.core_cc.csrc.sparse.all import SpconvOps
import pickle
import torch

from spconv.pytorch.cppcore import torch_tensor_to_tv


def main():
    with open("/home/yy/asd.pkl", "rb") as f:
        a_th = pickle.load(f)
    mask_argsort = torch.empty((1, a_th.shape[1]),
                               dtype=torch.int32,
                               device=a_th.device)

    a = a_th.cpu().numpy()[0]
    a_tv = torch_tensor_to_tv(a_th)
    mask_argsort_tv = torch_tensor_to_tv(mask_argsort)
    for i in range(10):
        a_tv_1 = a_tv.clone()
        SpconvOps.sort_1d_by_key(a_tv_1[0], mask_argsort_tv[0])


if __name__ == "__main__":
    main()
