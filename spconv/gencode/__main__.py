import os

import fire
from cumm.common import CompileInfo
from cumm.conv.main import ConvMainUnitTest
from cumm.gemm.main import GemmMainUnitTest
from pccm.builder.pybind import gen_cmake
from spconv.core import (IMPLGEMM_SIMT_PARAMS, IMPLGEMM_TURING_PARAMS, IMPLGEMM_AMPERE_PARAMS,
                         IMPLGEMM_VOLTA_PARAMS, SHUFFLE_SIMT_PARAMS,
                         SHUFFLE_TURING_PARAMS, SHUFFLE_VOLTA_PARAMS, SHUFFLE_AMPERE_PARAMS)
from spconv.csrc.hash.core import HashTable
from spconv.csrc.sparse.all import SpconvOps
from spconv.csrc.sparse.alloc import ExternalAllocator, StaticAllocator
from spconv.csrc.sparse.convops import (ConvGemmOps, ConvTunerSimple,
                                        ExternalSpconvMatmul, GemmTunerSimple,
                                        SimpleExternalSpconvMatmul)
from spconv.csrc.utils import BoxOps
from cumm.gemm.algospec.core import (GemmAlgo, ShuffleStrideType)
from cumm.conv.bases import ConvLayout, ConvLayoutType, ConvOpType
from spconv.csrc.sparse.inference import InferenceOps


def main(include: str,
         src: str,
         libname: str = "spconv",
         prefix: str = "spconvlib",
         inference_only: bool = False):
    all_shuffle = SHUFFLE_SIMT_PARAMS + SHUFFLE_VOLTA_PARAMS + SHUFFLE_TURING_PARAMS + SHUFFLE_AMPERE_PARAMS
    all_shuffle = list(filter(lambda x: not x.is_nvrtc, all_shuffle))
    if inference_only:
        all_shuffle = list(filter(lambda x: x.shuffle_stride != ShuffleStrideType.ShuffleAB, all_shuffle))

    cu = GemmMainUnitTest(all_shuffle)
    cu.namespace = "cumm.gemm.main"
    all_imp = (IMPLGEMM_SIMT_PARAMS + IMPLGEMM_VOLTA_PARAMS +
               IMPLGEMM_TURING_PARAMS + IMPLGEMM_AMPERE_PARAMS)
    # keep all int8 kernels in libspconv
    for x in all_imp:
        if x.int8_inference:
            x.is_nvrtc = False 
    all_imp = list(filter(lambda x: (not x.is_nvrtc), all_imp))
    if inference_only:
        all_imp = list(filter(lambda x: x.op_type == ConvOpType.kForward, all_imp))
    convcu = ConvMainUnitTest(all_imp)
    convcu.namespace = "cumm.conv.main"
    gemmtuner = GemmTunerSimple(cu)
    gemmtuner.namespace = "spconv.csrc.sparse.convops.gemmops"
    convtuner = ConvTunerSimple(convcu)
    convtuner.namespace = "spconv.csrc.sparse.convops.convops"
    convops = ConvGemmOps(gemmtuner, convtuner)
    convops.namespace = "spconv.csrc.sparse.convops.spops"

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
        StaticAllocator(),
        InferenceOps(),
    ]

    gen_cmake(libname, cus, include, src, namespace_prefix=prefix)


if __name__ == "__main__":
    fire.Fire(main)
