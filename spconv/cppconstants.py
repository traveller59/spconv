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

import spconv.core_cc as _ext

if hasattr(_ext, "cumm"):
    CPU_ONLY_BUILD = False
else:
    CPU_ONLY_BUILD = True

from spconv.core_cc.csrc.utils.boxops import BoxOps
from spconv.core_cc.cumm.common import CompileInfo
HAS_BOOST = BoxOps.has_boost()

COMPILED_CUDA_ARCHS = set(CompileInfo.get_compiled_cuda_arch())
