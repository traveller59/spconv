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

import pccm
from ccimport import compat
from cumm.common import TensorView

class OMPLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("tensorview/parallel/all.h")
        if compat.InWindows:
            self.build_meta.add_public_cflags("cl", "/openmp")
        else:
            self.build_meta.add_public_cflags("g++", "-fopenmp")
            self.build_meta.add_public_cflags("clang++", "-fopenmp")
            self.build_meta.add_ldflags("g++,clang++", "-fopenmp")
