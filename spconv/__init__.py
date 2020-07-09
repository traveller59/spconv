# Copyright 2019-2020 Yan Yan
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

import platform
from pathlib import Path

import numpy as np
import torch

from spconv import ops, utils
from spconv.conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                         SparseConvTranspose3d, SparseInverseConv2d,
                         SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.core import SparseConvTensor
from spconv.identity import Identity
from spconv.modules import SparseModule, SparseSequential
from spconv.ops import ConvAlgo
from spconv.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.tables import AddTable, ConcatTable, JoinTable

_LIB_FILE_NAME = "libspconv.so"
if platform.system() == "Windows":
    _LIB_FILE_NAME = "spconv.dll"
_LIB_PATH = str(Path(__file__).parent / _LIB_FILE_NAME)
torch.ops.load_library(_LIB_PATH)


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor.
    """
    def forward(self, x: SparseConvTensor):
        return x.dense()


class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer.
    """
    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x
