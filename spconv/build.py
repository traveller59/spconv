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

from pathlib import Path

import pccm
from pccm.utils import project_is_editable, project_is_installed
from ccimport.compat import InWindows
from .constants import PACKAGE_NAME, PACKAGE_ROOT, DISABLE_JIT

if project_is_installed(PACKAGE_NAME) and project_is_editable(
        PACKAGE_NAME) and not DISABLE_JIT:
    from spconv.core import SHUFFLE_SIMT_PARAMS, SHUFFLE_VOLTA_PARAMS, SHUFFLE_TURING_PARAMS
    from spconv.core import IMPLGEMM_SIMT_PARAMS, IMPLGEMM_VOLTA_PARAMS, IMPLGEMM_TURING_PARAMS

    from cumm.gemm.main import GemmMainUnitTest
    from cumm.conv.main import ConvMainUnitTest
    from cumm.common import CompileInfo

    from spconv.csrc.sparse.all import SpconvOps
    from spconv.csrc.utils import BoxOps
    from spconv.csrc.hash.core import HashTable

    cu = GemmMainUnitTest(SHUFFLE_SIMT_PARAMS + SHUFFLE_VOLTA_PARAMS +
                          SHUFFLE_TURING_PARAMS)
    cu.namespace = "cumm.gemm.main"
    convcu = ConvMainUnitTest(IMPLGEMM_SIMT_PARAMS + IMPLGEMM_VOLTA_PARAMS +
                              IMPLGEMM_TURING_PARAMS)
    convcu.namespace = "cumm.conv.main"
    objects_folder = None
    if InWindows:
        # windows have command line limit, so we use objects_folder to reduce command size.
        objects_folder = "objects"
    pccm.builder.build_pybind([cu, convcu, SpconvOps(), BoxOps(), HashTable(), CompileInfo()],
                              PACKAGE_ROOT / "core_cc",
                              namespace_root=PACKAGE_ROOT,
                              objects_folder=objects_folder,
                              load_library=False)
