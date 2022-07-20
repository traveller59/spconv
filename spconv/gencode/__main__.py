import os

import fire
from cumm.common import CompileInfo
from cumm.conv.main import ConvMainUnitTest
from cumm.gemm.main import GemmMainUnitTest
from pccm.builder.pybind import gen_cmake
from spconv.core import (IMPLGEMM_SIMT_PARAMS, IMPLGEMM_TURING_PARAMS,
                         IMPLGEMM_VOLTA_PARAMS, SHUFFLE_SIMT_PARAMS,
                         SHUFFLE_TURING_PARAMS, SHUFFLE_VOLTA_PARAMS)
from spconv.csrc.hash.core import HashTable
from spconv.csrc.sparse.all import SpconvOps
from spconv.csrc.sparse.alloc import ExternalAllocator
from spconv.csrc.sparse.convops import (ConvGemmOps, ConvTunerSimple,
                                        ExternalSpconvMatmul, GemmTunerSimple,
                                        SimpleExternalSpconvMatmul)
from spconv.csrc.utils import BoxOps


def main(include: str,
         src: str,
         libname: str = "spconv",
         prefix: str = "spconvlib"):
    all_shuffle = SHUFFLE_SIMT_PARAMS + SHUFFLE_VOLTA_PARAMS + SHUFFLE_TURING_PARAMS
    all_shuffle = list(filter(lambda x: not x.is_nvrtc, all_shuffle))
    cu = GemmMainUnitTest(all_shuffle)
    cu.namespace = "cumm.gemm.main"
    all_imp = (IMPLGEMM_SIMT_PARAMS + IMPLGEMM_VOLTA_PARAMS +
               IMPLGEMM_TURING_PARAMS)
    # all_imp = IMPLGEMM_SIMT_PARAMS
    all_imp = list(filter(lambda x: not x.is_nvrtc, all_imp))
    convcu = ConvMainUnitTest(all_imp)
    convcu.namespace = "cumm.conv.main"
    gemmtuner = GemmTunerSimple(cu)
    gemmtuner.namespace = "csrc.sparse.convops.gemmops"
    convtuner = ConvTunerSimple(convcu)
    convtuner.namespace = "csrc.sparse.convops.convops"
    convops = ConvGemmOps(gemmtuner, convtuner)
    convops.namespace = "csrc.sparse.convops.spops"

    cus = [
        cu,
        convcu,
        gemmtuner,
        convtuner,
        convops,
        SpconvOps(),
        BoxOps(),
        HashTable(),
        CompileInfo(),
        ExternalAllocator(),
        ExternalSpconvMatmul(),
        SimpleExternalSpconvMatmul(),
    ]

    gen_cmake(libname, cus, include, src, namespace_prefix=prefix)


if __name__ == "__main__":
    fire.Fire(main)
