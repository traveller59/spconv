from spconv.benchmark.core import get_voxel_data, get_voxel_data_large


import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from cumm import tensorview as tv
from spconv.core import ConvAlgo
from cumm import dtypes
import spconv.pytorch as spconv
from spconv.test_utils import params_grid
import spconv as spconv_core
class Net(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, 3, bias=False, indice_key="c0",
                              algo=algo),

            spconv.SubMConv3d(64,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # spconv.SparseConv3d(64, 64, 2, 2, bias=False, indice_key="m0"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(64,
                              96,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            spconv.SubMConv3d(96,
                              96,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # spconv.SparseConv3d(96, 96, 2, 2, bias=False, indice_key="m1"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(96,
                              128,
                              3,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            spconv.SubMConv3d(128,
                              128,
                              3,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # spconv.SparseConv3d(128, 128, 2, 2, bias=False, indice_key="m2"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(128,
                              160,
                              3,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            spconv.SubMConv3d(160,
                              160,
                              3,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # spconv.SparseConv3d(160, 160, 2, 2, bias=False, indice_key="m3"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(160,
                              192,
                              3,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            spconv.SubMConv3d(192,
                              192,
                              3,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2, indice_key="m4", algo=pool_algo),
            # spconv.SparseConv3d(192, 192, 2, 2, bias=False, indice_key="m4"),
            spconv.SubMConv3d(192,
                              224,
                              3,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            spconv.SubMConv3d(224,
                              224,
                              3,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            # nn.BatchNorm1d(224),
            # nn.ReLU(),
            # spconv.SparseConv3d(224, 224, 2, 2, bias=False, indice_key="m5"),
            spconv.SparseMaxPool3d(2, 2, indice_key="m5", algo=pool_algo),
            spconv.SubMConv3d(224,
                              256,
                              3,
                              bias=False,
                              indice_key="c6",
                              algo=algo),
            spconv.SubMConv3d(256,
                              256,
                              3,
                              bias=False,
                              indice_key="c6",
                              algo=algo),

            # nn.BatchNorm1d(256),
            # nn.ReLU(),

            # spconv.SparseInverseConv3d(256, 128, 2, indice_key="m5", bias=False, algo=algo),
            # # # nn.BatchNorm1d(128),
            # # # nn.ReLU(),

            # spconv.SparseInverseConv3d(128, 64, 2, indice_key="m4", bias=False, algo=algo),
        )
        max_batch_size = 1
        self.shape = shape

    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size,
                                    enable_timer=enable_timer)
        return self.net(x)

_DTYPE_TO_TORCH_DTYPE = {
    dtypes.float32: torch.float32,
    dtypes.float16: torch.float16,
}

def bench_basic(dtype_str: str, is_large: bool = False):
    assert dtype_str in ["f16", "f32", "tf32"], "only support f16, f32, tf32"
    if dtype_str == "tf32":
        spconv_core.constants.SPCONV_ALLOW_TF32 = True
        dtype_str = "f32"

    dtype = dtypes.get_dtype_by_shortcut(dtype_str)
    if dtype not in _DTYPE_TO_TORCH_DTYPE:
        raise NotImplementedError("only support bench f32 and f16 for now")
    torch_dtype = _DTYPE_TO_TORCH_DTYPE[dtype]
    algos = [spconv.ConvAlgo.Native, spconv.ConvAlgo.MaskImplicitGemm, spconv.ConvAlgo.MaskSplitImplicitGemm]
    if is_large:
        (voxels, coors, spatial_shape) = get_voxel_data_large()
    else:
        (voxels, coors, spatial_shape) = get_voxel_data()
    name = "basic-L" if is_large else "basic"
    device = torch.device("cuda:0")
    for algo, in params_grid(algos):
        voxels_th = torch.from_numpy(voxels).to(device).to(torch_dtype)
        coors_th = torch.from_numpy(coors).to(device).int()
        voxels_th.requires_grad = True
        net = Net(spatial_shape, algo).to(device).train().to(torch_dtype)# .train()
        spconv.assign_name_for_sparse_modules(net)
        with torch.no_grad():
            out: spconv.SparseConvTensor = net(voxels_th, coors_th, 1)
        dout = np.random.uniform(-0.2, 0.2, out.features.shape).astype(np.float32)
        dout_t = torch.from_numpy(dout).to(device).to(torch_dtype)
        times = []
        with torch.no_grad():
            for i in range(100):
                with tv.measure_duration() as measure:
                    out_nograd = net(voxels_th, coors_th, 1, False)
                times.append(measure.duration)
        print(f"{name}[{dtype_str}|{algo}|forward]", np.mean(times[50:]))
        times = []

        for i in range(50):
            out = net(voxels_th, coors_th, 1)
            with tv.measure_duration() as measure:
                out.features.backward(dout_t)
            times.append(measure.duration)
        print(f"{name}[{dtype_str}|{algo}|backward]", np.mean(times[25:]))


def bench_large(dtype_str: str):
    return bench_basic(dtype_str, True)

if __name__ == "__main__":
    bench_basic("f16")