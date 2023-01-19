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

from __future__ import print_function

import argparse
import contextlib
import copy
from typing import Dict, Optional

import torch
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as qfx
import torch.cuda.amp
import torch.fx
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import spconv.pytorch as spconv
import spconv.pytorch.quantization as spconvq
from spconv.pytorch.quantization.core import quantize_per_tensor
from spconv.pytorch.quantization.fake_q import \
    get_default_spconv_qconfig_mapping
import spconv.pytorch.quantization.intrinsic.quantized as snniq

from spconv.pytorch.quantization.interpreter import NetworkInterpreter, register_node_handler, register_method_handler
import spconv.pytorch.quantization.intrinsic as snni
import spconv.pytorch.quantization.intrinsic.quantized as snniq
import spconv.pytorch.quantization.quantized as snnq
import spconv.pytorch.quantization.quantized.reference as snnqr
from spconv.pytorch.cppcore import torch_tensor_to_tv
import numpy as np 

import spconv.constants as spconvc
# enable trace mode here, or use environment variable SPCONV_FX_TRACE_MODE=1
spconvc.SPCONV_FX_TRACE_MODE = True 

@contextlib.contextmanager
def identity_ctx():
    yield

class SubMConvBNReLU(spconv.SparseSequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(SubMConvBNReLU, self).__init__(
            spconv.SubMConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

class SparseConvBNReLU(spconv.SparseSequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(SparseConvBNReLU, self).__init__(
            spconv.SparseConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

class SparseBasicBlock(spconv.SparseModule):
    """residual block that supported by spconv quantization.
    """
    expansion = 1
    def __init__(self,
                 in_planes, out_planes,
                 stride=1,
                 downsample=None):
        spconv.SparseModule.__init__(self)
        conv1 = spconv.SubMConv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        conv2 = spconv.SubMConv2d(out_planes, out_planes, 3, stride, 1, bias=False)

        norm1 = nn.BatchNorm1d(out_planes, momentum=0.1)
        norm2 = nn.BatchNorm1d(out_planes, momentum=0.1)

        self.conv1_bn_relu = spconv.SparseSequential(conv=conv1, bn=norm1, relu=nn.ReLU(inplace=True))
        self.conv2_bn = spconv.SparseSequential(conv=conv2, bn=norm2)

        self.relu = spconv.SparseReLU(inplace=True)
        self.downsample = downsample
        self.iden_for_fx_match = spconv.SparseIdentity()

    def forward(self, x: spconv.SparseConvTensor):
        identity = x
        # if self.training:
        #     assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        out = self.conv1_bn_relu(x)
        out = self.conv2_bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResidualNetPTQ(nn.Module):
    """pytorch currently don't support cuda int8 inference, so
    we build a pure sparse network here.
    """
    def __init__(self):
        super(ResidualNetPTQ, self).__init__()
        self.net = spconv.SparseSequential(
            SubMConvBNReLU(1, 32, 3),
            SparseBasicBlock(32, 32),
            SubMConvBNReLU(32, 64, 3),
            SparseConvBNReLU(64, 64, 2, 2), # 14x14
            SparseConvBNReLU(64, 64, 2, 2), # 7x7
            SparseConvBNReLU(64, 64, 3, 2, 1), # 4x4
            spconv.SparseConv2d(64, 10, 4, 4),
            # spconv.ToDense(),
        )
        # self.fc1 = nn.Linear(64 * 1 * 1, 128)
        # self.fc2 = nn.Linear(128, 10)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
    
    def forward(self, features: torch.Tensor, indices: torch.Tensor, batch_size: int):
        # x: [N, 28, 28, 1], must be NHWC tensor
        # x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 1))
        x_sp = spconv.SparseConvTensor(features, indices, [28, 28], batch_size)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x_sp = self.net(x_sp)
        # print(x_sp.shape)
        x = x_sp
        # x = torch.flatten(x, 1)
        # x = self.dequant(x)
        # output = F.log_softmax(x, dim=1)
        return x


def calibrate(args, model: torch.nn.Module, data_loader, device):
    model.eval()
    
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device)
            if args.sparse:
                data_sp = spconv.SparseConvTensor.from_dense(image.reshape(-1, 28, 28, 1))
                output = model(data_sp.features, data_sp.indices, data_sp.batch_size)
                # output = model(data_sp)
            else:
                output = model(image)

# add module handler
@register_node_handler(snni.SpconvReLUNd)
def _spconv_fused_relu(net, target: snni.SpconvReLUNd, args, kwargs, name: str):
    # add plugin here...
    print("add sparse conv plugin here...", target, name)
    return args[0]

@register_node_handler(snni.SpconvAddReLUNd)
def _spconv_fused_add_relu(net, target: snni.SpconvReLUNd, args, kwargs, name: str):
    # add plugin here...
    print("add sparse conv plugin here...", target, name)
    return args[0]

@register_node_handler(snniq.SparseConvReLU)
def _spconv_fused_q_relu(net, target: snniq.SparseConvReLU, args, kwargs, name: str):
    # add plugin here...
    print("add sparse conv plugin here...", target, name)
    return args[0]

@register_node_handler(snniq.SparseConvAddReLU)
def _spconv_fused_q_add_relu(net, target: snniq.SparseConvAddReLU, args, kwargs, name: str):
    # add fused conv-add-relu plugin here...
    inp0 = args[0]
    inp1 = args[1]
    print("add fused sparse conv add relu plugin here...", target, name)
    return args[0]

@register_node_handler(snnqr.SpConv)
def _spconv_r(net, target: snnqr.SpConv, args, kwargs, name: str):
    # add plugin here...
    input_scale = args[0].int8_scale 
    output_scale = target.scale
    q_weight = target.get_quantized_weight()
    w_scales = q_weight.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    bias_np = target.bias.detach().cpu().numpy()
    w = torch_tensor_to_tv(q_weight).cpu().numpy()
    # spconv int8 format
    channel_scale = (input_scale * w_scales) / output_scale
    bias_np = bias_np / output_scale

    print("add sparse conv plugin here...", target, name)
    return args[0]

@register_node_handler(snnq.SparseConv)
def _spconv_fused_q(net, target: snnq.SparseConv, args, kwargs, name: str):
    # add plugin here...
    print("add sparse conv plugin here...", target, name)
    return args[0]

@register_node_handler(spconv.SparseConvTensor)
def _get_sparse_conv_tensor(net, target: spconv.SparseConvTensor, args, kwargs, name: str):
    return spconv.SparseConvTensor(*args, **kwargs)

# add tensor method handler
@register_method_handler("replace_feature", spconv.SparseConvTensor)
def _replace_new_feature(net, target, args, kwargs, name: str):
    input: spconv.SparseConvTensor = args[0]
    if isinstance(input, spconv.SparseConvTensor):
        return input.replace_feature(*args[1:])
    else:
        raise NotImplementedError

@register_node_handler(quantize_per_tensor)
def _quantize_per_tensor(net, target, args, kwargs, name: str):
    inp: spconv.SparseConvTensor = args[0]
    scale = args[1].detach().cpu().numpy()
    zero_point = args[2]
    print("implement quantize here...", name, scale)
    # WARNING
    # we need to store scale to SparseConvTensor because pytorch dequantize don't 
    # have any argument
    inp.int8_scale = scale
    return inp


@register_method_handler("dequantize", spconv.SparseConvTensor)
def _dequantize(net, target, args, kwargs, name: str):
    inp: spconv.SparseConvTensor = args[0]

    assert inp.int8_scale is not None 
    print("implement dequantize here...", inp.int8_scale)

    return inp

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--sparse',
                        action='store_true',
                        default=True,
                        help='use sparse conv network instead of dense')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help='For mixed precision training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda and args.sparse else "cpu")
    qdevice = torch.device("cuda" if use_cuda and args.sparse else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = ResidualNetPTQ().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # here we remove norm to get sparse tensor with lots of zeros
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # here we remove norm to get sparse tensor with lots of zeros
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    model.eval()
    spconvq.prepare_spconv_torch_inference(True)
    # tensorrt only support symmetric quantization, per-tensor act and per-channel weight.
    qconfig_mapping = get_default_spconv_qconfig_mapping(is_qat=False)
    prepare_cfg = spconvq.get_spconv_prepare_custom_config()
    backend_cfg = spconvq.get_spconv_backend_config()
    # prepare: fuse your model, all patterns such as conv-bn-relu fuse to modules in torch.ao.quantization.intrinsic / spconv.pytorch.quantization.intrinsic
    # then add observers to fused model.
    prepared_model = qfx.prepare_fx(model, qconfig_mapping, (), backend_config=backend_cfg, prepare_custom_config=prepare_cfg)
    # calibrate: run model with some inputs
    calibrate(args, prepared_model, test_loader, qdevice)
    # convert (ptq): replace intrinsic blocks with quantized modules
    converted_model = qfx.convert_fx(prepared_model, qconfig_mapping=qconfig_mapping, backend_config=backend_cfg)
    converted_model = spconvq.transform_qdq(converted_model)
    # test converted ptq model with int8 kernel
    converted_model = spconvq.remove_conv_add_dq(converted_model)
    # use trt ITensor as input here...
    # input is same as converted_model inputs
    # here we just use torch tensor. we can actually use any input here.
    ft = torch.zeros([500, 1], dtype=torch.float32, device=device)
    ind = torch.zeros([500, 3], dtype=torch.int32, device=device)

    interp = NetworkInterpreter(None, converted_model, [ft, ind, 1])
    # get converted outputs from interp
    outputs = interp.run()

if __name__ == '__main__':
    main()
