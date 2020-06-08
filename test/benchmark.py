import torch 
import spconv 
import numpy as np 
from spconv.utils import VoxelGeneratorV2
from pathlib import Path 
from torch import nn
import time 

def waymo_data(batch_size=1):
    gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 1, 150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = data["pc"]
    data = gen.generate(pc)
    voxels = data["voxels"].reshape(-1, 3)
    coors = data["coordinates"]
    N = coors.shape[0]
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size

class Net(nn.Module):
    def __init__(self,
                 shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, 3, bias=False, indice_key="c0"),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key="c0"),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(64, 96, 3, bias=False, indice_key="c1"),
            spconv.SubMConv3d(96, 96, 3, bias=False, indice_key="c1"),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(96, 128, 3, bias=False, indice_key="c2"),
            spconv.SubMConv3d(128, 128, 3, bias=False, indice_key="c2"),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(128, 160, 3, bias=False, indice_key="c3"),
            spconv.SubMConv3d(160, 160, 3, bias=False, indice_key="c3"),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(160, 192, 3, bias=False, indice_key="c4"),
            spconv.SubMConv3d(192, 192, 3, bias=False, indice_key="c4"),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(192, 224, 3, bias=False, indice_key="c5"),
            spconv.SubMConv3d(224, 224, 3, bias=False, indice_key="c5"),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(224, 256, 3, bias=False, indice_key="c6"),
            spconv.SubMConv3d(256, 256, 3, bias=False, indice_key="c6"),
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        self.grid = torch.full([max_batch_size, *shape], -1, dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)


def main():
    voxels, coors, spatial_shape = waymo_data()
    voxels_th = torch.from_numpy(voxels).cuda().float()
    coors_th = torch.from_numpy(coors).cuda().int()
    net = Net(spatial_shape[::-1]).cuda().eval().float()
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