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

"""Compare results between sparse and dense layers:
SparseConvXd
SparseConvTransposeXd
SparseMaxPoolXd
"""

import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
from spconv.test_utils import TestCase, generate_sparse_data, params_grid
from spconv.constants import ALL_WEIGHT_IS_KRSC, FILTER_HWIO

# we must disable tf32 to increase reference precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SparseConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.MaskSplitImplicitGemm):
        super().__init__()
        self.algo = algo
        layers = [
            spconv.SparseConv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding=padding,
                                dilation=dilation,
                                bias=False,
                                algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConv3d(out_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False,
                                    algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)  # .dense()

class Conv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding=padding,
                      dilation=dilation,
                      bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.Conv3d(out_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          padding=padding,
                          dilation=dilation,
                          bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()

class SparseDeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride, padding, dilation, algo):
        super().__init__()
        self.algo = algo
        layers = [
            spconv.SparseConvTranspose3d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding=padding,
                                         dilation=dilation,
                                         bias=False,
                                         algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConvTranspose3d(out_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=False,
                                             algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class DeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.ConvTranspose3d(out_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseMaxPoolTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation, algo):
        super().__init__()
        self.algo = algo
        layers = [
            spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation, algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation, algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()

class SparseGlobalMaxPoolTestTorch(nn.Module):
    def __init__(self, shape):
        super().__init__()
        layers = [
            spconv.SparseGlobalMaxPool()
        ]
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class MaxPool3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation):
        super().__init__()
        layers = [nn.MaxPool3d(kernel_size, stride, padding, dilation)]
        for i in range(1, num_layers):
            layers.append(nn.MaxPool3d(kernel_size, stride, padding, dilation))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()

def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(
        params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

def test_spconv3d():
    test_case = TestCase()
    np.random.seed(484)
    torch.manual_seed(48848)
    devices = ["cuda:0"]
    shapes = [[19, 18, 17]]
    batchsizes = [1, 2]

    in_channels = [32]
    out_channels = [32, 48, 64]
    ksizes = [2, 3]
    strides = [1, 2, 3]
    paddings = [0, 1, 2]
    dilations = [1, 2, 3]
    algos = [
        ConvAlgo.Native, ConvAlgo.MaskImplicitGemm,
        ConvAlgo.MaskSplitImplicitGemm
    ]
    algos = [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm, ConvAlgo.MaskSplitImplicitGemm]

    for dev, shape, bs, IC, OC, k, s, p, d, al in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, algos):
        if all([s > 1, d > 1]):
            continue  # don't support this.
        # print(dev, shape, bs, IC, OC, k, s, p, d)
        device = torch.device(dev)
        num_points = [1500] * bs
        dtype = torch.float32
        net = SparseConv3dTestTorch(1,
                                    3,
                                    shape,
                                    IC,
                                    OC,
                                    k,
                                    s,
                                    p,
                                    d,
                                    algo=al).to(device).to(dtype)
        net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                    d).to(device).to(dtype)

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device)
        features_t = torch.from_numpy(features).to(device).to(dtype)
        features_t.requires_grad = True
        features_dense_t = torch.from_numpy(features_dense).to(device).to(
            dtype)
        features_dense_t.requires_grad = True
        if net.algo == ConvAlgo.Native and not ALL_WEIGHT_IS_KRSC:
            if FILTER_HWIO:
                filters = np.random.uniform(-1, 1,
                                            size=[k, k, k, IC,
                                                    OC]).astype(np.float32)
            else:
                filters = np.random.uniform(-1, 1,
                                            size=[k, k, k, OC,
                                                    IC]).astype(np.float32)
            filters_t = torch.from_numpy(filters).to(device).to(dtype)
            if FILTER_HWIO:
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    4, 3, 0, 1, 2).contiguous()
            else:
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    3, 4, 0, 1, 2).contiguous()
        else:
            filters = np.random.uniform(-1, 1,
                                        size=[OC, k, k, k,
                                                IC]).astype(np.float32)
            filters_t = torch.from_numpy(filters).to(device).to(dtype)
            net_ref.net[0].weight.data[:] = filters_t.permute(
                0, 4, 1, 2, 3).contiguous()
        net.net[0].weight.data[:] = filters_t
        out_ref = net_ref(features_dense_t)
        out = net(features_t, indices_t, bs).dense()
        out_np = out.detach().cpu().numpy()
        out_ref_np = out_ref.detach().cpu().numpy()
        test_case.assertAllClose(out_np, out_ref_np, atol=1e-4)

        dout = np.random.uniform(-0.2, 0.2,
                                    out_ref.shape).astype(features.dtype)
        dout_t = torch.from_numpy(dout).to(device)
        out.backward(dout_t)
        out_ref.backward(dout_t)
        din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                            1).contiguous()
        din_sparse = gather_nd(din_dense, indices_t.long())
        din = features_t.grad.detach()

        din_np = din.cpu().numpy()
        din_sparse_np = din_sparse.cpu().numpy()
        for layer, layer_ref in zip(net.net, net_ref.net):
            dw = layer.weight.grad.detach().cpu().numpy()
            dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
            if net.algo == ConvAlgo.Native and not ALL_WEIGHT_IS_KRSC:
                if FILTER_HWIO:
                    dw = dw.transpose(4, 3, 0, 1, 2)
                else:
                    dw = dw.transpose(3, 4, 0, 1, 2)
            else:
                # OHWI -> OIHW
                dw = dw.transpose(0, 4, 1, 2, 3)

            test_case.assertAllClose(dw, dw_ref, atol=1e-4)
        test_case.assertAllClose(din_np, din_sparse_np, atol=1e-4)

def test_spdeconv3d():
    test_case = TestCase()

    np.random.seed(484)
    devices = ["cuda:0"]
    shapes = [[19, 18, 17]]
    batchsizes = [1, 2]

    in_channels = [64]
    out_channels = [32, 48, 64]
    ksizes = [2, 3]
    strides = [2, 3]
    paddings = [0, 1, 2]
    dilations = [1, 2, 3]

    algos = [
        ConvAlgo.Native, ConvAlgo.MaskImplicitGemm,
        ConvAlgo.MaskSplitImplicitGemm
    ]

    for dev, shape, bs, IC, OC, k, s, p, d, al in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, algos):
        if all([s > 1, d > 1]):
            continue  # don't support this.
        device = torch.device(dev)
        num_points = [1000] * bs
        dtype = torch.float32

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        net = SparseDeConv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                        d, al).to(device)
        net_ref = DeConv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                    d).to(device)

        if net.algo == ConvAlgo.Native and not ALL_WEIGHT_IS_KRSC:
            if FILTER_HWIO:
                filters = np.random.uniform(-1, 1,
                                            size=[k, k, k, IC,
                                                    OC]).astype(np.float32)
            else:
                filters = np.random.uniform(-1, 1,
                                            size=[k, k, k, OC,
                                                    IC]).astype(np.float32)
            filters_t = torch.from_numpy(filters).to(device).to(dtype)
            if FILTER_HWIO:
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    3, 4, 0, 1, 2).contiguous()
            else:
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    4, 3, 0, 1, 2).contiguous()
        else:
            filters = np.random.uniform(-1, 1,
                                        size=[OC, k, k, k,
                                                IC]).astype(np.float32)
            filters_t = torch.from_numpy(filters).to(device).to(dtype)
            net_ref.net[0].weight.data[:] = filters_t.permute(
                4, 0, 1, 2, 3).contiguous()
        net.net[0].weight.data[:] = filters_t

        indices_t = torch.from_numpy(indices).int().to(device)
        features_t = torch.from_numpy(features).to(device)
        features_t.requires_grad = True
        features_dense_t = torch.from_numpy(features_dense).to(device)
        features_dense_t.requires_grad = True
        filters_t = torch.from_numpy(filters).to(device)
        out_ref = net_ref(features_dense_t)
        out = net(features_t, indices_t, bs).dense()
        out_np = out.detach().cpu().numpy()
        out_ref_np = out_ref.detach().cpu().numpy()
        test_case.assertAllClose(out_np, out_ref_np, atol=1e-4)

        dout = np.random.uniform(-0.2, 0.2,
                                    out_ref.shape).astype(features.dtype)
        dout_t = torch.from_numpy(dout).to(device)
        out.backward(dout_t)
        out_ref.backward(dout_t)
        din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                            1).contiguous()
        din_sparse = gather_nd(din_dense, indices_t.long())
        din = features_t.grad.detach()
        din_np = din.cpu().numpy()
        din_sparse_np = din_sparse.cpu().numpy()
        test_case.assertAllClose(din_np, din_sparse_np, atol=1e-4)
        for layer, layer_ref in zip(net.net, net_ref.net):
            dw = layer.weight.grad.detach().cpu().numpy()
            dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
            if net.algo == ConvAlgo.Native and not ALL_WEIGHT_IS_KRSC:
                if FILTER_HWIO:
                    dw = dw.transpose(3, 4, 0, 1, 2)
                else:
                    dw = dw.transpose(4, 3, 0, 1, 2)
            else:
                # OHWI -> OIHW
                dw = dw.transpose(4, 0, 1, 2, 3)
            test_case.assertAllClose(dw, dw_ref, atol=1e-4)

def test_spmaxpool3d():
    test_case = TestCase()

    np.random.seed(485)
    devices = ["cuda:0"]
    shapes = [[19, 18, 17]]
    batchsizes = [1, 2]

    in_channels = [64]
    out_channels = [64]
    ksizes = [2, 3]
    strides = [1, 2, 3]
    paddings = [0, 1]
    dilations = [1, 2, 3]
    # ksizes = [2]
    # strides = [2]
    # paddings = [0]
    # dilations = [1]
    algos = [
        ConvAlgo.Native, ConvAlgo.MaskImplicitGemm,
        ConvAlgo.MaskSplitImplicitGemm
    ]


    for dev, shape, bs, IC, OC, k, s, p, d, al in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, algos):
        if all([s > 1, d > 1]):
            continue  # don't support this.
        device = torch.device(dev)
        num_points = [1000] * bs

        # when data contains negative, sparse maxpool is not equal to dense maxpool.
        sparse_dict = generate_sparse_data(shape,
                                            num_points,
                                            IC,
                                            data_range=[0.1, 1])

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device)
        features_t = torch.from_numpy(features).to(device)
        features_t.requires_grad = True
        features_dense_t = torch.from_numpy(features_dense).to(device)
        features_dense_t.requires_grad = True
        net = SparseMaxPoolTestTorch(1, 3, shape, k, s, p, d, al).to(device)
        net_ref = MaxPool3dTestTorch(1, 3, shape, k, s, p, d).to(device)

        out_ref = net_ref(features_dense_t)
        out = net(features_t, indices_t, bs)

        outids = out.indices
        outfeatures = out.features
        outids_dev = outids.float()
        out_dense = out.dense(channels_first=False)
        out = out_dense.permute(0, 4, 1, 2, 3).contiguous()
        out_np = out.detach().cpu().numpy()
        out_ref_np = out_ref.detach().cpu().numpy()
        test_case.assertAllClose(out_np, out_ref_np, atol=1e-4)

        dout_sparse = np.random.uniform(
            -0.2, 0.2, outfeatures.shape).astype(features.dtype)
        dout_sparse_t = torch.from_numpy(dout_sparse).to(device)
        dout_t = scatter_nd(outids.long(), dout_sparse_t,
                            list(out_dense.shape))
        dout_t = dout_t.permute(0, 4, 1, 2, 3).contiguous()
        out.backward(dout_t)
        out_ref.backward(dout_t)
        din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                            1).contiguous()
        din_sparse = gather_nd(din_dense, indices_t.long())
        din = features_t.grad.detach()

        din_np = din.cpu().numpy()
        din_sparse_np = din_sparse.cpu().numpy()
        test_case.assertAllClose(din_np, din_sparse_np, atol=1e-4)


def test_spglobalmaxpool3d():
    test_case = TestCase()

    np.random.seed(485)
    devices = ["cpu:0", "cuda:0"]
    shapes = [[19, 18, 17]]
    batchsizes = [1, 2]

    channels = [64]
    # ksizes = [2]
    # strides = [2]
    # paddings = [0]
    # dilations = [1]


    for dev, shape, bs, C in params_grid(
            devices, shapes, batchsizes, channels):
        device = torch.device(dev)
        num_points = [1000] * bs
        # when data contains negative, sparse maxpool is not equal to dense maxpool.
        sparse_dict = generate_sparse_data(shape,
                                            num_points,
                                            C,
                                            data_range=[0.1, 0.4])

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device)
        features_t = torch.from_numpy(features).to(device)
        features_t.requires_grad = True
        features_dense_t = torch.from_numpy(features_dense).to(device)
        features_dense_t.requires_grad = True
        net = SparseGlobalMaxPoolTestTorch(shape).to(device)
        net_ref = MaxPool3dTestTorch(1, 3, shape, shape, shape, 0, 1).to(device)

        out_ref = net_ref(features_dense_t)
        out = net(features_t, indices_t, bs)
        out_dense = out
        out_np = out.detach().cpu().numpy()
        out_ref_np = out_ref.detach().cpu().numpy()
        test_case.assertAllClose(out_np.reshape(-1), out_ref_np.reshape(-1), atol=1e-4)

        dout = np.random.uniform(
            -0.2, 0.2, out_dense.shape).astype(features.dtype)
        dout_t = torch.from_numpy(dout).to(device).view(bs, C, 1, 1, 1)
        out.backward(dout_t.reshape(bs, C))
        out_ref.backward(dout_t)
        din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                            1).contiguous()
        din_sparse = gather_nd(din_dense, indices_t.long())
        din = features_t.grad.detach()
        din_np = din.cpu().numpy()
        din_sparse_np = din_sparse.cpu().numpy()
        test_case.assertAllClose(din_np, din_sparse_np, atol=1e-4)

if __name__ == "__main__":
    test_spglobalmaxpool3d()
