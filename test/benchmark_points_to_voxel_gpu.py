import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import spconv
from spconv.utils import VoxelGeneratorV3


def waymo_data(batch_size=1):
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    points = torch.from_numpy(data['pc']).cuda().float()
    voxel_size = torch.Tensor([0.1, 0.1, 0.1]).to(points.dtype).to(points.device)
    coors_range = torch.Tensor([-80, -80, -2, 80, 80, 6]).to(points.dtype).to(points.device)

    gen = VoxelGeneratorV3(voxel_size, coors_range)
    voxels, coors = gen.generate(points)
    N = coors.shape[0]
    batch_id = torch.zeros([N, 1], dtype=coors.dtype, device=coors.device)
    coors = torch.cat([batch_id, coors], dim=1)
    return voxels, coors, gen.grid_size


class Net(nn.Module):
    def __init__(self, shape, algo, device):
        super().__init__()
        self.device = device
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, 3, bias=False, indice_key="c0", algo=algo),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key="c0", algo=algo),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(64, 96, 3, bias=False, indice_key="c1", algo=algo),
            spconv.SubMConv3d(96, 96, 3, bias=False, indice_key="c1", algo=algo),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(96, 128, 3, bias=False, indice_key="c2", algo=algo),
            spconv.SubMConv3d(128, 128, 3, bias=False, indice_key="c2", algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(128, 160, 3, bias=False, indice_key="c3", algo=algo),
            spconv.SubMConv3d(160, 160, 3, bias=False, indice_key="c3", algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(160, 192, 3, bias=False, indice_key="c4", algo=algo),
            spconv.SubMConv3d(192, 192, 3, bias=False, indice_key="c4", algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(192, 224, 3, bias=False, indice_key="c5", algo=algo),
            spconv.SubMConv3d(224, 224, 3, bias=False, indice_key="c5", algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(224, 256, 3, bias=False, indice_key="c6", algo=algo),
            spconv.SubMConv3d(256, 256, 3, bias=False, indice_key="c6", algo=algo),
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        self.grid = torch.full([max_batch_size, *shape], -1,
                               dtype=torch.int32, device=self.device)
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)


def main():
    voxels, coors, spatial_shape = waymo_data()
    voxels_th, coors_th = voxels, coors
    algo = spconv.ConvAlgo.Native
    net = Net(spatial_shape[::-1], algo, voxels_th.device).cuda(device=voxels_th.device).eval().float()
    print(coors_th.shape)
    out = net(voxels_th, coors_th, 1)
    print(out.spatial_shape)
    times = []
    with torch.no_grad():
        for i in range(20):
            torch.cuda.synchronize()
            t = time.time()
            out = net(voxels_th, coors_th, 1)
            torch.cuda.synchronize()
            times.append(time.time() - t)
    # print((net.grid == -1).float().sum(), net.grid.numel())
    # print("spconv time", time.time() - t)
    print("spconv time", np.mean(times[10:]))


if __name__ == "__main__":
    main()
