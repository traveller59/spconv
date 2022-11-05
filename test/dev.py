import spconv.pytorch as spconv 
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
from spconv.test_utils import TestCase, generate_sparse_data, params_grid

import torch 
import numpy as np 
class SparseMaxPool2dTestTorch(torch.nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation, algo):
        super().__init__()
        self.algo = algo
        layers = [
            spconv.SparseMaxPool2d(kernel_size, stride, padding, dilation, algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseMaxPool2d(kernel_size, stride, padding, dilation, algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()
shapes = [[65536, 65536]]
batchsizes = [32]

in_channels = [32]
out_channels = [32]
ksizes = [2]
strides = [2]
paddings = [0]
dilations = [1]
algos = [
    # ConvAlgo.Native, 
    ConvAlgo.MaskImplicitGemm,
    # ConvAlgo.MaskSplitImplicitGemm
]
devices = ["cuda:0"]


for dev, shape, bs, IC, OC, k, s, p, d, al in params_grid(
        devices, shapes, batchsizes, in_channels, out_channels, ksizes,
        strides, paddings, dilations, algos):
    device = torch.device(dev)
    num_points = [1000] * bs
    print(1)
    sparse_dict = generate_sparse_data(shape,
                                        num_points,
                                        IC,
                                        with_dense=False,
                                        data_range=[0.1, 1],
                                        shape_scale = 64)
    print(2)
    net = SparseMaxPool2dTestTorch(1, 2, shape, k, s, p, d, al).to(device)
    features = np.ascontiguousarray(sparse_dict["features"]).astype(
        np.float32)
    indices = np.ascontiguousarray(
        sparse_dict["indices"][:, [2, 0, 1]]).astype(np.int32)
    print(indices.max(0))
    indices_t = torch.from_numpy(indices).int().to(device)
    features_t = torch.from_numpy(features).to(device)
    features_t.requires_grad = True

    out = net(features_t, indices_t, bs)
    print(out.indices.min(0))
