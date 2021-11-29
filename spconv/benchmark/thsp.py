"""Benchmark torchsparse
"""
from spconv.benchmark.core import get_voxel_data

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from spconv.core import ConvAlgo
from cumm import dtypes
from spconv.test_utils import params_grid

_DTYPE_TO_TORCH_DTYPE = {
    dtypes.float32: torch.float32,
    dtypes.float16: torch.float16,
}

def bench_torchsparse_basic(dtype_str: str):
    dtype = dtypes.get_dtype_by_shortcut(dtype_str)
    if dtype not in _DTYPE_TO_TORCH_DTYPE:
        raise NotImplementedError("only support bench f32 and f16 for now")
    torch_dtype = _DTYPE_TO_TORCH_DTYPE[dtype]
