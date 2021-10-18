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

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from cumm import tensorview as tv 

import spconv.pytorch as spconv
from spconv.utils import Point2VoxelCPU3d
def waymo_data(batch_size=1):
    gen = Point2VoxelCPU3d([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3,
                           150000, 1)
    # gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 1,
    #                        150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = np.ascontiguousarray(data["pc"])
    print(pc.shape)
    voxels_tv, indices_tv, _ = gen.point_to_voxel(tv.from_numpy(pc))
    voxels = voxels_tv.numpy().reshape(-1, 3)
    coors = indices_tv.numpy()
    N = coors.shape[0]
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size


class Net(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, 3, bias=False, indice_key="c0",
                              algo=algo),
            # spconv.SubMConv3d(32,
            #                   32,
            #                   3,
            #                   bias=False,
            #                   indice_key="c0",
            #                   algo=algo),
            # # nn.BatchNorm1d(32),
            # # nn.ReLU(),
            # # spconv.SparseConv3d(64, 64, 2, 2, bias=False,
            # #                   algo=algo),
            # spconv.SubMConv3d(32, 64, 3, bias=False, indice_key="c0",
            #                   algo=algo),
            spconv.SubMConv3d(64,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),

            spconv.SparseMaxPool3d(2, 2),
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
            spconv.SparseMaxPool3d(2, 2),
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
            spconv.SparseMaxPool3d(2, 2),
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
            spconv.SparseMaxPool3d(2, 2),
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
            spconv.SparseMaxPool3d(2, 2),
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
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
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
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        self.grid = torch.full([max_batch_size, *shape], -1,
                               dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)

class Net2(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 256, 3, bias=False, indice_key="c0",
                              algo=algo),
            # spconv.SubMConv3d(32,
            #                   32,
            #                   3,
            #                   bias=False,
            #                   indice_key="c0",
            #                   algo=algo),
            # # nn.BatchNorm1d(32),
            # # nn.ReLU(),
            # # spconv.SparseConv3d(64, 64, 2, 2, bias=False,
            # #                   algo=algo),
            # spconv.SubMConv3d(32, 64, 3, bias=False, indice_key="c0",
            #                   algo=algo),
            spconv.SubMConv3d(256,
                              256,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
            spconv.SubMConv3d(256,
                              512,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            spconv.SubMConv3d(512,
                              512,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        self.grid = torch.full([max_batch_size, *shape], -1,
                               dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)


def main():
    import pickle 
    np.random.seed(50051)
    torch.manual_seed(50051)
    # voxels, coors, spatial_shape = waymo_data()
    # with open("/home/yy/test_spconv.pkl", "wb") as f:
    #     pickle.dump((voxels, coors, spatial_shape), f)
    with open("/home/yy/test_spconv.pkl", "rb") as f:
        (voxels, coors, spatial_shape) = pickle.load(f)
    print(spatial_shape)
    print(voxels.shape)
    # voxels = voxels[:100]
    # coors = coors[:100]
    voxels_th = torch.from_numpy(voxels).cuda().float()
    coors_th = torch.from_numpy(coors).cuda().int()
    voxels_th.requires_grad = True

    algo = spconv.ConvAlgo.Native
    net = Net(spatial_shape, algo).cuda().eval().float()
    print(coors_th.shape)
    out = net(voxels_th, coors_th, 1)
    print(out.spatial_shape)
    print(voxels.mean(),  voxels.max(), voxels.min())
    dout = np.random.uniform(-0.2, 0.2,
                                out.features.shape).astype(np.float32)
    dout_t = torch.from_numpy(dout).cuda()

    print(out.spatial_shape, out.features.mean(),  out.features.max(),  out.features.min())
    times = []
    with torch.no_grad():
        for i in range(20):
            print("------------")
            torch.cuda.synchronize()
            t = time.time()
            out_nograd = net(voxels_th, coors_th, 1)
            torch.cuda.synchronize()
            times.append(time.time() - t)
    print("spconv time", np.mean(times[10:]))
    times = []

    for i in range(10):
        out = net(voxels_th, coors_th, 1)
        print("------------")
        torch.cuda.synchronize()
        t = time.time()
        out.features.backward(dout_t)
        torch.cuda.synchronize()
        times.append(time.time() - t)

    # print((net.grid == -1).float().sum(), net.grid.numel())
    # print("spconv time", time.time() - t)
    print("spconv bw time", np.mean(times[5:]))


if __name__ == "__main__":
    main()
