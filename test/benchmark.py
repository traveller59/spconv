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
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
from spconv.utils import Point2VoxelCPU3d

# torch.backends.cudnn.enabled = False
def waymo_data(batch_size=1, num_features=-1):
    gen = Point2VoxelCPU3d([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3,
                           150000, 1)
    # gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 1,
    #                        150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = np.ascontiguousarray(data["pc"])
    voxels_tv, indices_tv, _ = gen.point_to_voxel(tv.from_numpy(pc))
    voxels = voxels_tv.numpy().reshape(-1, 3)

    if num_features > 0:
        voxels = np.zeros((voxels.shape[0], num_features), dtype=voxels.dtype)
    print(voxels.shape)
    coors = indices_tv.numpy()
    N = coors.shape[0]
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size

def waymo_data_large(batch_size=1):
    from spconv.utils import Point2VoxelGPU3d

    gen = Point2VoxelGPU3d([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3,
                           1600000, 1)
    # gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 1,
    #                        150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = np.ascontiguousarray(data["pc"])
    pcs = [pc]
    for i in range(7):
        pc2 = pc.copy()
        pc2[:, 1] += i + 1
        pcs.append(pc2)

    pc = np.concatenate(pcs)
    print(pc.shape)
    voxels_tv, indices_tv, _ = gen.point_to_voxel_hash(tv.from_numpy(pc).cuda())
    voxels = voxels_tv.cpu().numpy().reshape(-1, 3)
    coors = indices_tv.cpu().numpy()
    N = coors.shape[0]
    print("num voxels", N, gen.grid_size)
    # breakpoint()
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size


class Net(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
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

            # spconv.SubMConv3d(64, 64, 3, bias=False, indice_key="c0",
            #                   algo=algo),
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
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        # self.grid = torch.full([max_batch_size, *shape], -1,
        #                        dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size,
                                    # self.grid,
                                    enable_timer=enable_timer)
        return self.net(x)


class Net2(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3,
                              128,
                              3,
                              bias=False,
                              indice_key="c0",
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
            spconv.SubMConv3d(128,
                              128,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # spconv.SparseMaxPool3d(2, 2),
            # spconv.SubMConv3d(256,
            #                   512,
            #                   3,
            #                   bias=False,
            #                   indice_key="c1",
            #                   algo=algo),
            # spconv.SubMConv3d(512,
            #                   512,
            #                   3,
            #                   bias=False,
            #                   indice_key="c1",
            #                   algo=algo),
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


class Net_kv75(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, (3, 5, 5), bias=False, indice_key="c0",
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

            # spconv.SubMConv3d(64, 64, 3, bias=False, indice_key="c0",
            #                   algo=algo),
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
                              (3, 5, 5),
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # spconv.SparseConv3d(64, 64, 2, 2, bias=False, indice_key="m0"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(64,
                              96,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            spconv.SubMConv3d(96,
                              96,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # spconv.SparseConv3d(96, 96, 2, 2, bias=False, indice_key="m1"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(96,
                              128,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            spconv.SubMConv3d(128,
                              128,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # spconv.SparseConv3d(128, 128, 2, 2, bias=False, indice_key="m2"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(128,
                              160,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            spconv.SubMConv3d(160,
                              160,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # spconv.SparseConv3d(160, 160, 2, 2, bias=False, indice_key="m3"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(160,
                              192,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            spconv.SubMConv3d(192,
                              192,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2, indice_key="m4", algo=pool_algo),
            # spconv.SparseConv3d(192, 192, 2, 2, bias=False, indice_key="m4"),
            spconv.SubMConv3d(192,
                              224,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            spconv.SubMConv3d(224,
                              224,
                              (3, 5, 5),
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            # nn.BatchNorm1d(224),
            # nn.ReLU(),
            # spconv.SparseConv3d(224, 224, 2, 2, bias=False, indice_key="m5"),
            spconv.SparseMaxPool3d(2, 2, indice_key="m5", algo=pool_algo),
            spconv.SparseConv3d(224,
                              256,
                              (3, 5, 5),
                              padding=(1, 2, 2),
                              bias=False,
                              # indice_key="c6",
                              algo=algo),
            spconv.SubMConv3d(256,
                              256,
                              (3, 5, 5),
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
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        # self.grid = torch.full([max_batch_size, *shape], -1,
        #                        dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size,
                                    # self.grid,
                                    enable_timer=enable_timer)
        return self.net(x)


class Net_kv125(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, 5, bias=False, indice_key="c0",
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

            # spconv.SubMConv3d(64, 64, 3, bias=False, indice_key="c0",
            #                   algo=algo),
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
                              5,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # spconv.SparseConv3d(64, 64, 2, 2, bias=False, indice_key="m0"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(64,
                              96,
                              5,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            spconv.SubMConv3d(96,
                              96,
                              5,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # spconv.SparseConv3d(96, 96, 2, 2, bias=False, indice_key="m1"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(96,
                              128,
                              5,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            spconv.SubMConv3d(128,
                              128,
                              5,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # spconv.SparseConv3d(128, 128, 2, 2, bias=False, indice_key="m2"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(128,
                              160,
                              5,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            spconv.SubMConv3d(160,
                              160,
                              5,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # spconv.SparseConv3d(160, 160, 2, 2, bias=False, indice_key="m3"),
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(160,
                              192,
                              5,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            spconv.SubMConv3d(192,
                              192,
                              5,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2, indice_key="m4", algo=pool_algo),
            # spconv.SparseConv3d(192, 192, 2, 2, bias=False, indice_key="m4"),
            spconv.SubMConv3d(192,
                              224,
                              5,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            spconv.SubMConv3d(224,
                              224,
                              5,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            # nn.BatchNorm1d(224),
            # nn.ReLU(),
            # spconv.SparseConv3d(224, 224, 2, 2, bias=False, indice_key="m5"),
            spconv.SparseMaxPool3d(2, 2, indice_key="m5", algo=pool_algo),
            spconv.SubMConv3d(224,
                              256,
                              5,
                              bias=False,
                              indice_key="c6",
                              algo=algo),
            spconv.SubMConv3d(256,
                              256,
                              5,
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
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        # self.grid = torch.full([max_batch_size, *shape], -1,
        #                        dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size,
                                    # self.grid,
                                    enable_timer=enable_timer)
        return self.net(x)



class NetSm(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3,
                              8,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(8,
                              16,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(16,
                              32,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(32,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        self.grid = torch.full([max_batch_size, *shape], -1,
                               dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid, enable_timer=enable_timer)
        return self.net(x)

import numpy as np
from cumm import tensorview as tv
from spconv.core_cc.csrc.sparse.all import SpconvOps
import pickle
import torch

from spconv.pytorch.cppcore import torch_tensor_to_tv


def sort_bench():
    with open("/home/yy/asd.pkl", "rb") as f:
        a_th = pickle.load(f)
    mask_argsort = torch.empty((1, a_th.shape[1]),
                               dtype=torch.int32,
                               device=a_th.device)

    a = a_th.cpu().numpy()[0]
    a_tv = torch_tensor_to_tv(a_th)
    mask_argsort_tv = torch_tensor_to_tv(mask_argsort)
    for i in range(10):
        a_tv_1 = a_tv.clone()
        SpconvOps.sort_1d_by_key(a_tv_1[0], mask_argsort_tv[0])
import json
def waymo_data_large_debug(batch_size=1):
    gen = Point2VoxelCPU3d([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3,
                           1200000, 1)
    # gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 1,
    #                        150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = np.ascontiguousarray(data["pc"])
    pc2 = pc.copy()
    pc2[:, 1] += 1
    pc3 = pc.copy()
    pc3[:, 1] += 2
    pc4 = pc.copy()
    pc4[:, 1] += 3
    pc5 = pc.copy()
    pc5[:, 1] += 4
    pc6 = pc.copy()
    pc6[:, 1] += 5
    pc7 = pc.copy()
    pc7[:, 1] += 6
    pc8 = pc.copy()
    pc8[:, 1] += 7

    pc = np.concatenate([pc, pc2, pc3, pc4, pc5, pc6, pc7, pc8])
    print(pc.shape)
    voxels_tv, indices_tv, _ = gen.point_to_voxel(tv.from_numpy(pc))
    voxels = voxels_tv.numpy().reshape(-1, 3)
    coors = indices_tv.numpy()
    N = coors.shape[0]
    print("num voxels", N)
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size

def main():
    import pickle

    np.random.seed(50051)
    torch.manual_seed(50051)
    # voxels, coors, spatial_shape = waymo_data(num_features=3)
    # with open(Path(__file__).parent / "data" / "test_spconv.pkl", "rb") as f:
    #     (voxels, coors, spatial_shape) = pickle.load(f)
    voxels, coors, spatial_shape = waymo_data_large()
    # breakpoint()

    print(spatial_shape)
    print(voxels.shape)
    # voxels = voxels[:100]
    # coors = coors[:100]
    dtype = torch.float16
    device = torch.device("cuda:0")
    voxels_th = torch.from_numpy(voxels).to(device).to(dtype)
    coors_th = torch.from_numpy(coors).to(device).int()
    voxels_th.requires_grad = True
    algo = spconv.ConvAlgo.MaskImplicitGemm
    print("ALGO")
    # 3080 Laptop
    # MaskImpGemm: 11.2ms
    # MaskSplitImpGemm: 12.2ms
    # Native: 13.7ms
    # F32
    # MaskSplitImpGemm: 22ms
    # MaskImplicitGemm: 23.5ms
    # Native: 21.7ms
    # Pure Gemm
    # Native: 6.6ms
    # MaskImpGemm: 4.3ms
    # MaskSplitImpGemm: 4.0ms
    # F16 Bwd
    # MaskSplitImpGemm: 12.2ms
    # MaskImpGemm: 13.8ms
    # Native: 25.2ms

    # F32 Bwd
    # Native: 41.9ms
    # MaskImpGemm: 51.0ms
    # MaskSplitImpGemm: 41.1ms
    # algo = None
    net = Net(spatial_shape, algo).to(device).eval().to(dtype)# .train()
    # net.load_state_dict(net.state_dict())
    spconv.assign_name_for_sparse_modules(net)
    print(coors_th.shape)
    out = net(voxels_th, coors_th, 1)
    print(out.spatial_shape)
    print(voxels.mean(), voxels.max(), voxels.min())
    dout = np.random.uniform(-0.2, 0.2, out.features.shape).astype(np.float32)
    dout_t = torch.from_numpy(dout).to(device).to(dtype)

    print(out.spatial_shape, out.features.sum(1).mean(), out.features.max(),
          out.features.min())
    times = []
    show_metrics = False
    with torch.no_grad():
        for i in range(100):
            # print("------------")
            with tv.measure_duration() as measure:
                out_nograd = net(voxels_th, coors_th, 1, show_metrics)
            times.append(measure.duration)
            if show_metrics:
                timer = out_nograd._timer
                items = list(timer.get_all_pair_time().items())
                items.sort(key=lambda x: x[0])
                print("SUM TIME:",  sum([x[1] for x in items]))
                # print(json.dumps(dict(items), indent=2))
                inds_sum = 0
                gemm_sum = 0
                for k, v in items:
                    if "gen_pairs" in k:
                        inds_sum += v 
                for k, v in items:
                    if "gemm" in k:
                        gemm_sum += v 

                print("SUM GEN INDS:",  inds_sum, "GEMM:", gemm_sum)

    # state = net.state_dict()
    # state.pop("net.2.max_num_voxels_during_training")
    # net.load_state_dict(state)
    # breakpoint()
    # net = net.train()
    print("spconv time", np.mean(times[10:]))
    # times = []

    # for i in range(10):
    #     out = net(voxels_th, coors_th, 1)
    #     print("------------")
    #     torch.cuda.synchronize()
    #     t = time.time()
    #     out.features.backward(dout_t)
    #     torch.cuda.synchronize()
    #     times.append(time.time() - t)

    # # # print((net.grid == -1).float().sum(), net.grid.numel())
    # # # print("spconv time", time.time() - t)
    # print("spconv bw time", np.mean(times[5:]))


if __name__ == "__main__":
    main()
