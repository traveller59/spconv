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
from typing import List

import pccm
from pccm.utils import project_is_editable, project_is_installed
from ccimport.compat import InWindows
from .constants import PACKAGE_NAME, PACKAGE_ROOT, DISABLE_JIT, SPCONV_INT8_DEBUG

if project_is_installed(PACKAGE_NAME) and project_is_editable(
        PACKAGE_NAME) and not DISABLE_JIT and not SPCONV_INT8_DEBUG:
    from spconv.core import SHUFFLE_SIMT_PARAMS, SHUFFLE_VOLTA_PARAMS, SHUFFLE_TURING_PARAMS, SHUFFLE_AMPERE_PARAMS
    from spconv.core import IMPLGEMM_SIMT_PARAMS, IMPLGEMM_VOLTA_PARAMS, IMPLGEMM_TURING_PARAMS, IMPLGEMM_AMPERE_PARAMS

    from cumm.gemm.main import GemmMainUnitTest
    from cumm.conv.main import ConvMainUnitTest
    from cumm.common import CompileInfo

    from spconv.csrc.sparse.all import SpconvOps
    from spconv.csrc.sparse.alloc import ExternalAllocator
    from spconv.csrc.utils import BoxOps, PointCloudCompress
    from spconv.csrc.hash.core import HashTable
    from spconv.csrc.sparse.convops import GemmTunerSimple, ExternalSpconvMatmul
    from spconv.csrc.sparse.convops import ConvTunerSimple, ConvGemmOps
    from spconv.csrc.sparse.convops import SimpleExternalSpconvMatmul
    from spconv.csrc.sparse.inference import InferenceOps

    all_shuffle = SHUFFLE_SIMT_PARAMS + SHUFFLE_VOLTA_PARAMS + SHUFFLE_TURING_PARAMS + SHUFFLE_AMPERE_PARAMS
    # all_shuffle = list(filter(lambda x: not x.is_nvrtc, all_shuffle))
    cu = GemmMainUnitTest(all_shuffle)
    cu.namespace = "cumm.gemm.main"
    all_imp = (IMPLGEMM_SIMT_PARAMS + IMPLGEMM_VOLTA_PARAMS +
               IMPLGEMM_TURING_PARAMS + IMPLGEMM_AMPERE_PARAMS)
    # all_imp = list(filter(lambda x: not x.is_nvrtc, all_imp))
    convcu = ConvMainUnitTest(all_imp)
    convcu.namespace = "cumm.conv.main"
    gemmtuner = GemmTunerSimple(cu)
    gemmtuner.namespace = "csrc.sparse.convops.gemmops"
    convtuner = ConvTunerSimple(convcu)
    convtuner.namespace = "csrc.sparse.convops.convops"
    convops = ConvGemmOps(gemmtuner, convtuner)
    convops.namespace = "csrc.sparse.convops.spops"

    cus = [
        cu, convcu, gemmtuner, convtuner,
        convops,
        SpconvOps(),
        BoxOps(),
        HashTable(),
        CompileInfo(),
        ExternalAllocator(),
        ExternalSpconvMatmul(),
        SimpleExternalSpconvMatmul(), # for debug, won't be included in release
        InferenceOps(),
        PointCloudCompress(),
    ]
    pccm.builder.build_pybind(cus,
                              PACKAGE_ROOT / "core_cc",
                              namespace_root=PACKAGE_ROOT,
                              load_library=False,
                              verbose=True)

    # cus_dev: List[pccm.Class] = [
    # ]
    # pccm.builder.build_pybind(cus_dev,
    #                           PACKAGE_ROOT / "core_cc_dev",
    #                           namespace_root=PACKAGE_ROOT,
    #                           load_library=False,
    #                           verbose=True)
