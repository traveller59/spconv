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
from enum import Enum
from cumm.gemm.main import gen_shuffle_params, GemmAlgoParams
from cumm.gemm import kernel
from typing import List
from cumm.gemm.algospec.core import TensorOpParams
class ConvAlgo(Enum):
    Native = "Native"
    MaskImplicitGemm = "MaskImplicitGemm"
    MaskSplitImplicitGemm = "MaskSplitImplicitGemm"


class AlgoHint(Enum):
    NoHint = 0b000
    Fowrard = 0b001
    BackwardInput = 0b010
    BackwardWeight = 0b100

# we can't add more kernels here because build in github action is very slow.
# TODO two step build: build gemm kernels first, then bind for every python

SHUFFLE_SIMT_PARAMS: List[GemmAlgoParams] = [
    # *gen_shuffle_params(
    #     (64, 128, 32), (32, 64, 32), ["s8,s8,s32,s32,s32"],
    #     2, kernel.GemmAlgo.SimtDP4A, None),
    # *gen_shuffle_params(
    #     (128, 64, 32), (64, 32, 32), ["s8,s8,s32,s32,s32"],
    #     2, kernel.GemmAlgo.SimtDP4A, None),
    # *gen_shuffle_params(
    #     (128, 128, 32),
    #     (32, 64, 32), ["s8,s8,s32,s32,s32"], 2,
    #     kernel.GemmAlgo.SimtDP4A, None),
    # *gen_shuffle_params(
    #     (128, 128, 32),
    #     (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
    #     kernel.GemmAlgo.SimtDP4A, None),
    *gen_shuffle_params(
        (64, 64, 32), (32, 32, 32), ["s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.SimtDP4A, None),
    # *gen_shuffle_params(
    #     (64, 256, 8),
    #     (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # # *gen_shuffle_params(
    # #     (64, 256, 8),
    # #     (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (32, 128, 16),
    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (32, 512, 8),
    #     (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # # *gen_shuffle_params(
    # #     (128, 128, 8),
    # #     (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (128, 128, 8),
    #     (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (64, 128, 8),
    #     (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # # *gen_shuffle_params(
    # #     (64, 128, 8),
    # #     (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # # *gen_shuffle_params(
    # #     (128, 64, 8),
    # #     (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (128, 64, 8),
    #     (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (64, 64, 8),
    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (32, 64, 16),
    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (64, 32, 16),
    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (32, 32, 32),
    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # # fall back kernels if mat is misaligned for half
    # # *gen_shuffle_params(
    # #     (128, 128, 8),
    # #     (32, 64, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (32, 64, 32),
    #     (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (32, 32, 32),
    #     (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # # *gen_shuffle_params(
    # #     (64, 64, 16),
    # #     (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (64, 128, 16),
    #     (32, 64, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # *gen_shuffle_params(
    #     (64, 64, 8),
    #     (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
]

SHUFFLE_VOLTA_PARAMS: List[GemmAlgoParams] = [
    # *gen_shuffle_params(
    #     (64, 64, 32),
    #     (32, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    # # *gen_shuffle_params(
    # #     (128, 128, 32),
    # #     (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    # #     kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    # *gen_shuffle_params(
    #     (128, 256, 32),
    #     (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    # *gen_shuffle_params(
    #     (256, 128, 32),
    #     (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    # *gen_shuffle_params(
    #     (128, 64, 32),
    #     (64, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    # *gen_shuffle_params(
    #     (64, 128, 32),
    #     (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
]
# SHUFFLE_VOLTA_PARAMS = []
SHUFFLE_TURING_PARAMS: List[GemmAlgoParams] = [
    # *gen_shuffle_params(
    #     (64, 64, 32),
    #     (32, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (128, 128, 32),
    #     (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # # *gen_shuffle_params(
    # #     (128, 128, 32),
    # #     (64, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    # #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (64, 64, 64),
    #     (32, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (64, 128, 64),
    #     (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (128, 256, 32),
    #     (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (256, 128, 32),
    #     (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (128, 64, 32),
    #     (64, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (64, 128, 32),
    #     (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    # *gen_shuffle_params(
    #     (64, 64, 32), (32, 32, 32), ["s8,s8,s32,s32,s32"],
    #     2, kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    # *gen_shuffle_params(
    #     (128, 128, 32),
    #     (32, 64, 32), ["s8,s8,s32,s32,s32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    # # *gen_shuffle_params(
    # #     (128, 128, 32),
    # #     (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
    # #     kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    # *gen_shuffle_params(
    #     (128, 256, 32),
    #     (64, 64, 32), ["s8,s8,s32,s32,s32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    # *gen_shuffle_params(
    #     (256, 128, 32),
    #     (64, 64, 32), ["s8,s8,s32,s32,s32"], 2,
    #     kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    # *gen_shuffle_params(
    #     (128, 64, 32), (64, 32, 32), ["s8,s8,s32,s32,s32"],
    #     2, kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    # *gen_shuffle_params(
    #     (64, 128, 32), (32, 64, 32), ["s8,s8,s32,s32,s32"],
    #     2, kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
]
