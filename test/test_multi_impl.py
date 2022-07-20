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

"""Compare results between different algos:
CPU: simple gather-mm-scatter
Native: Fused gather-mm-scatter
ImplicitGemm: implicit gemm
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from cumm import tensorview as tv
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
import pickle
from spconv.test_utils import generate_sparse_data, params_grid


class Net(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 32, 3, bias=False, indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(32,
                              32,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # # nn.BatchNorm1d(32),
            # # nn.ReLU(),
            spconv.SubMConv3d(32, 64, 3, bias=False, indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(64,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # # nn.ReLU(),
            spconv.SparseConv3d(64, 64, 3, 2, 1, bias=False, indice_key="m0", algo=algo),
            # # spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
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
            spconv.SparseConv3d(96, 96, 2, 2, bias=False, indice_key="m1", algo=algo),
            # spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
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
            spconv.SparseMaxPool3d(2, 2, algo=pool_algo, indice_key="m3"),
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
            # nn.BatchNorm1d(256),
            # nn.ReLU(),

            spconv.SparseInverseConv3d(224, 128, 2, indice_key="m4", bias=False, algo=algo),
            # # nn.BatchNorm1d(128),
            # nn.ReLU(),

            spconv.SparseInverseConv3d(128, 64, 2, indice_key="m3", bias=False, algo=algo),
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size)
        return self.net(x)

class NetLight(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 32, 3, bias=False, indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(32,
                              32,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # # nn.BatchNorm1d(32),
            # # nn.ReLU(),
            spconv.SubMConv3d(32, 64, 3, bias=False, indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(64,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            # nn.BatchNorm1d(32),
            # # nn.ReLU(),
            spconv.SparseConv3d(64, 64, 3, 2, 1, bias=False, indice_key="m0", algo=algo),
            # # spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
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
            spconv.SparseConv3d(96, 96, 2, 2, bias=False, indice_key="m1", algo=algo),
            # spconv.SparseMaxPool3d(2, 2, algo=pool_algo),

            spconv.SparseInverseConv3d(96, 64, 2, indice_key="m1", bias=False, algo=algo),
            # # nn.BatchNorm1d(128),
            # nn.ReLU(),

            spconv.SparseInverseConv3d(64, 32, 3, indice_key="m0", bias=False, algo=algo),
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size)
        return self.net(x)


def _test_multi_impl(dtype: torch.dtype):
    # TODO pytorch 1.12 don't support cpu half mm, f**k pytorch
    # TODO remove or release this when tf32 op is ready
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    np.random.seed(50051)
    if dtype != torch.float16:
        with open(Path(__file__).parent / "data" / "test_spconv.pkl", "rb") as f:
            (voxels, coors, spatial_shape) = pickle.load(f)
    else:
        # CPU fp16 is very slow, so we use a small data here.
        spatial_shape = [19, 18, 17]
        sparse_dict = generate_sparse_data(spatial_shape, [1500] * 1, 3)

        voxels = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        coors = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
    device = torch.device("cuda:0")
    device_cpu = torch.device("cpu:0")

    voxels_th = torch.from_numpy(voxels).to(device_cpu).to(dtype)
    coors_th = torch.from_numpy(coors).to(device_cpu).int()
    voxels_th_cuda = torch.from_numpy(voxels).to(device).to(dtype)
    coors_th_cuda = torch.from_numpy(coors).to(device).int()
    net_cls = Net
    if dtype == torch.float16:
        # CPU fp16 is very slow, so we use a small network here.
        net_cls = NetLight
    # cpu 
    torch.manual_seed(50051)
    net_native_cpu = net_cls(spatial_shape, ConvAlgo.Native).to(device_cpu).to(dtype)
    # gpu_native 
    torch.manual_seed(50051)
    net_native_gpu = net_cls(spatial_shape, ConvAlgo.Native).to(device).to(dtype)
    
    torch.manual_seed(50051)
    net_imp_gpu = net_cls(spatial_shape, ConvAlgo.MaskImplicitGemm).to(device).to(dtype)
    
    torch.manual_seed(50051)
    net_simp_gpu = net_cls(spatial_shape, ConvAlgo.MaskSplitImplicitGemm).to(device).to(dtype)

    spconv.assign_name_for_sparse_modules(net_native_cpu)
    spconv.assign_name_for_sparse_modules(net_native_gpu)
    spconv.assign_name_for_sparse_modules(net_imp_gpu)
    spconv.assign_name_for_sparse_modules(net_simp_gpu)
    with torch.no_grad():
        out: torch.Tensor = net_native_cpu(voxels_th, coors_th, 1).dense()
    dout = np.random.uniform(-0.2, 0.2, out.shape).astype(np.float32)
    dout_t = torch.from_numpy(dout).to(device_cpu).to(dtype)
    dout_t_cu = torch.from_numpy(dout).to(device).to(dtype)

    t = time.time()
    print(1, time.time() - t)
    out_cpu = net_native_cpu(voxels_th, coors_th, 1).dense()
    if dtype != torch.float16:
        out_cpu.backward(dout_t)
    out = net_native_gpu(voxels_th_cuda, coors_th_cuda, 1).dense()
    print(2, time.time() - t)

    out.backward(dout_t_cu)
    out_imp = net_imp_gpu(voxels_th_cuda, coors_th_cuda, 1).dense()
    print(3, time.time() - t)

    out_imp.backward(dout_t_cu)
    out_simp = net_simp_gpu(voxels_th_cuda, coors_th_cuda, 1).dense()
    print(4, time.time() - t)

    out_simp.backward(dout_t_cu)
    with torch.no_grad():
        dense_cpu = out_cpu.cuda()
        dense_native = out
        dense_imp = out_imp
        dense_simp = out_simp

        error_native = torch.linalg.norm(dense_cpu - dense_native).cpu().item()
        error_imp = torch.linalg.norm(dense_cpu - dense_imp).cpu().item()
        error_simp = torch.linalg.norm(dense_cpu - dense_simp).cpu().item()
    print(5, time.time() - t)

    print("error_native", error_native)
    print("error_imp", error_imp)
    print("error_simp", error_simp)
    if dtype == torch.float32:
        assert error_native < 0.01
        assert error_imp < 0.01
        assert error_simp < 0.01
    else:
        assert error_native < 10
        assert error_imp < 10
        assert error_simp < 10


    cpu_params = dict(net_native_cpu.named_parameters())
    native_params = dict(net_native_gpu.named_parameters())
    imp_params = dict(net_imp_gpu.named_parameters())
    simp_params = dict(net_simp_gpu.named_parameters())

    for k, cpu_w in cpu_params.items():
        native_w = native_params[k]
        imp_w = imp_params[k]
        simp_w = simp_params[k]
        native_w_grad = native_w.grad.detach()
        imp_w_grad = imp_w.grad.detach()
        simp_w_grad = simp_w.grad.detach()
        if dtype != torch.float16:
            cpu_w_grad = cpu_w.grad.detach().cuda()
            error_native = torch.linalg.norm(native_w_grad - cpu_w_grad).cpu().item()
        error_imp = torch.linalg.norm(native_w_grad - imp_w_grad).cpu().item()
        error_simp = torch.linalg.norm(native_w_grad - simp_w_grad).cpu().item()
        print(k, error_imp, error_simp)
        assert error_imp < 1
        assert error_simp < 1

def test_multi_impl():
    _test_multi_impl(torch.float32)
    _test_multi_impl(torch.float16)


if __name__ == "__main__":
    test_multi_impl()
