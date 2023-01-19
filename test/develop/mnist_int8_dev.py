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
import torch.nn.functional as F
import torch.optim as optim
from torch.ao.quantization import (DeQuantStub, QuantStub,
                                   get_default_qconfig_mapping)
from torch.ao.quantization.fx._lower_to_native_backend import \
    STATIC_LOWER_FUSED_MODULE_MAP, STATIC_LOWER_MODULE_MAP
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import spconv.pytorch as spconv
import spconv.pytorch.quantization as spconvq
from spconv.pytorch.quantization import get_default_spconv_trt_ptq_qconfig
from spconv.pytorch.quantization.backend_cfg import \
    SPCONV_STATIC_LOWER_FUSED_MODULE_MAP, SPCONV_STATIC_LOWER_MODULE_MAP
from spconv.pytorch.quantization.core import quantize_per_tensor
from spconv.pytorch.quantization.fake_q import \
    get_default_spconv_qconfig_mapping
from spconv.pytorch.quantization.intrinsic.modules import SpconvBnAddReLUNd, SpconvAddReLUNd
import spconv.pytorch.quantization.intrinsic.quantized as snniq

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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.iden_for_fx_match = nn.Identity()

    def forward(self, x: spconv.SparseConvTensor):
        identity = self.iden_for_fx_match(x.features)
        out = self.conv1_bn_relu(x)
        out = self.conv2_bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(self.relu(out.features + identity))
        return out

class SparseBasicBlock1(spconv.SparseModule):
    """residual block that supported by spconv quantization.
    """
    expansion = 1
    def __init__(self,
                 in_planes, out_planes,
                 stride=1,
                 downsample=None):
        spconv.SparseModule.__init__(self)
        self.conv1 = spconv.SubMConv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.conv2 = spconv.SubMConv2d(out_planes, out_planes, 3, stride, 1, bias=False)

        self.norm1 = nn.BatchNorm1d(out_planes, momentum=0.1)
        self.norm2 = nn.BatchNorm1d(out_planes, momentum=0.1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.iden_for_fx_match = nn.Identity()

    def forward(self, x: spconv.SparseConvTensor):
        identity = self.iden_for_fx_match(x.features)
        # if self.training:
        #     assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        out = self.conv1(x)
        out = out.replace_feature(self.relu1(self.norm1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out = out.replace_feature(self.relu2(out.features + identity))
        return out

class SparseBasicBlock2(spconv.SparseModule):
    """residual block that supported by spconv quantization.
    """
    expansion = 1
    def __init__(self,
                 in_planes, out_planes,
                 stride=1,
                 downsample=None):
        spconv.SparseModule.__init__(self)
        self.conv1 = spconv.SubMConv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.conv2 = spconv.SubMConv2d(out_planes, out_planes, 3, stride, 1, bias=False)

        self.norm1 = spconv.SparseBatchNorm(out_planes, momentum=0.1)
        self.norm2 = spconv.SparseBatchNorm(out_planes, momentum=0.1)

        self.relu1 = spconv.SparseReLU(inplace=True)
        self.relu2 = spconv.SparseReLU(inplace=True)
        self.downsample = downsample
        self.iden_for_fx_match = spconv.SparseIdentity()

    def forward(self, x: spconv.SparseConvTensor):
        identity = self.iden_for_fx_match(x)
        # if self.training:
        #     assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        out = self.conv1(x)
        out = self.relu1(self.norm1(out))
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu2(out + identity)
        return out

class SparseBasicBlock3(spconv.SparseModule):
    """residual block that supported by spconv quantization.
    """
    expansion = 1
    def __init__(self,
                 in_planes, out_planes,
                 stride=1,
                 downsample=None):
        spconv.SparseModule.__init__(self)
        self.conv1 = spconv.SubMConv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        conv2 = spconv.SubMConv2d(out_planes, out_planes, 3, stride, 1, bias=False)

        self.norm1 = spconv.SparseBatchNorm(out_planes, momentum=0.1)
        norm2 = spconv.SparseBatchNorm(out_planes, momentum=0.1)
        self.residual_conv = SpconvAddReLUNd(conv2, spconv.SparseReLU(inplace=True))
        self.relu1 = spconv.SparseReLU(inplace=True)
        # self.relu2 = spconv.SparseReLU(inplace=True)
        self.downsample = downsample
        self.iden_for_fx_match = spconv.SparseIdentity()

    def forward(self, x: spconv.SparseConvTensor):
        identity = self.iden_for_fx_match(x)
        # if self.training:
        #     assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        out = self.conv1(x)
        out = self.relu1(self.norm1(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.residual_conv(out, identity)
        return out

class SparseBasicBlock4(spconv.SparseModule):
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = spconv.SparseSequential(
            SubMConvBNReLU(1, 32, 3),
            SubMConvBNReLU(32, 64, 3),
            SparseConvBNReLU(64, 64, 2, 2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(14 * 14 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x_sp: spconv.SparseConvTensor):
    # def forward(self, features: torch.Tensor, indices: torch.Tensor, batch_size: int):
        # x: [N, 28, 28, 1], must be NHWC tensor
        # x = self.quant(x)
        # x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 1))
        # x_sp = spconv.SparseConvTensor(features, indices, [28, 28], batch_size)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.net(x_sp)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.dequant(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()
        self.net = spconv.SparseSequential(
            SubMConvBNReLU(1, 32, 3),
            SubMConvBNReLU(32, 64, 3),
            SparseConvBNReLU(64, 64, 2, 2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(14 * 14 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, features: torch.Tensor, indices: torch.Tensor, batch_size: int):
        # x: [N, 28, 28, 1], must be NHWC tensor
        x = self.quant(features)
        # x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 1))
        x_sp = spconv.SparseConvTensor(features, indices, [28, 28], batch_size)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.net(x_sp)
        x = torch.flatten(x, 1)
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dequant(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetPTQ(nn.Module):
    """pytorch currently don't support cuda int8 inference, so
    we build a pure sparse network here.
    """
    def __init__(self):
        super(NetPTQ, self).__init__()
        self.net = spconv.SparseSequential(
            SubMConvBNReLU(1, 32, 3),
            SubMConvBNReLU(32, 64, 3),
            SparseConvBNReLU(64, 64, 2, 2), # 14x14
            SparseConvBNReLU(64, 64, 2, 2), # 7x7
            SparseConvBNReLU(64, 64, 3, 2, 1), # 4x4
            spconv.SparseConv2d(64, 10, 4, 4),
            spconv.ToDense(),
        )
        # self.fc1 = nn.Linear(64 * 1 * 1, 128)
        # self.fc2 = nn.Linear(128, 10)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, features: torch.Tensor, indices: torch.Tensor, batch_size: int):
        # x: [N, 28, 28, 1], must be NHWC tensor
        features = self.quant(features)
        # x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 1))
        x_sp = spconv.SparseConvTensor(features, indices, [28, 28], batch_size)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x_sp = self.net(x_sp)
        # print(x_sp.shape)
        x = x_sp
        x = torch.flatten(x, 1)
        x = self.dequant(x)
        output = F.log_softmax(x, dim=1)
        return output

class ResidualNetPTQ(nn.Module):
    """pytorch currently don't support cuda int8 inference, so
    we build a pure sparse network here.
    """
    def __init__(self):
        super(ResidualNetPTQ, self).__init__()
        self.net = spconv.SparseSequential(
            SubMConvBNReLU(1, 32, 3),
            # SubMConvBNReLU(32, 32, 3),
            SparseBasicBlock4(32, 32),
            SubMConvBNReLU(32, 64, 3),
            SparseConvBNReLU(64, 64, 2, 2), # 14x14
            SparseConvBNReLU(64, 64, 2, 2), # 7x7
            SparseConvBNReLU(64, 64, 3, 2, 1), # 4x4
            spconv.SparseConv2d(64, 10, 4, 4),
            spconv.ToDense(),
        )
        # self.fc1 = nn.Linear(64 * 1 * 1, 128)
        # self.fc2 = nn.Linear(128, 10)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, features: torch.Tensor, indices: torch.Tensor, batch_size: int):
        # x: [N, 28, 28, 1], must be NHWC tensor
        features = self.quant(features)
        # x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 1))
        x_sp = spconv.SparseConvTensor(features, indices, [28, 28], batch_size)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x_sp = self.net(x_sp)
        # print(x_sp.shape)
        x = x_sp
        x = torch.flatten(x, 1)
        x = self.dequant(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetDense(nn.Module):
    def __init__(self):
        super(NetDense, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.iden = spconv.SparseIdentity()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)

        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.iden(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dequant(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    amp_ctx = contextlib.nullcontext()
    if args.fp16:
        amp_ctx = torch.cuda.amp.autocast()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with amp_ctx:
            if args.sparse:
                data_sp = spconv.SparseConvTensor.from_dense(data.reshape(-1, 28, 28, 1))
                # output = model(data_sp)
                output = model(data_sp.features, data_sp.indices, data_sp.batch_size)
            else:
                output = model(data)

            loss = F.nll_loss(output, target)
            scale = 1.0
            if args.fp16:
                assert loss.dtype is torch.float32
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                # scaler.unscale_(optim)

                # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                # You may use the same value for max_norm here as you would without gradient scaling.
                # torch.nn.utils.clip_grad_norm_(models[0].net.parameters(), max_norm=0.1)

                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                scale = scaler.get_scale()
            else:
                loss.backward()
                optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    amp_ctx = contextlib.nullcontext()
    if args.fp16:
        amp_ctx = torch.cuda.amp.autocast()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            with amp_ctx:
                if args.sparse:
                    data_sp = spconv.SparseConvTensor.from_dense(data.reshape(-1, 28, 28, 1))
                    # output = model(data_sp)
                    output = model(data_sp.features, data_sp.indices, data_sp.batch_size)
                else:
                    output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


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
    if args.sparse:
        model = ResidualNetPTQ().to(device)
    else:
        model = NetDense().to(device)

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

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    model.eval()
    if not args.sparse:
        model = model.cpu()

    model_qat = copy.deepcopy(model)
    STATIC_LOWER_FUSED_MODULE_MAP.update(SPCONV_STATIC_LOWER_FUSED_MODULE_MAP)
    STATIC_LOWER_MODULE_MAP.update(SPCONV_STATIC_LOWER_MODULE_MAP)

    # tensorrt only support symmetric quantization, per-tensor act and per-channel weight.
    qconfig_mapping = get_default_spconv_qconfig_mapping(False)
    prepare_cfg = spconvq.get_spconv_prepare_custom_config()
    backend_cfg = spconvq.get_spconv_backend_config()
    # convert_cfg = spconvq.get_spconv_convert_custom_config()
    # prepare: fuse your model, all patterns such as conv-bn-relu fuse to modules in torch.ao.quantization.intrinsic / spconv.pytorch.quantization.intrinsic
    # then add observers to fused model.
    prepared_model = qfx.prepare_fx(model, qconfig_mapping, (), backend_config=backend_cfg, prepare_custom_config=prepare_cfg)
    # print(prepared_model)
    # breakpoint()

    # print(prepared_model)
    # calibrate: run model with some inputs
    calibrate(args, prepared_model, test_loader, qdevice)
    # convert (ptq): replace intrinsic blocks with quantized modules
    converted_model = qfx.convert_fx(prepared_model, qconfig_mapping=qconfig_mapping, backend_config=backend_cfg)
    converted_model = spconvq.transform_qdq(converted_model)
    # test converted ptq model with int8 kernel
    spconvq.remove_conv_add_dq(converted_model)

    print(converted_model)
    breakpoint()

    test(args, converted_model, qdevice, test_loader)
    # do qat
    # qconfig_mapping_qat = get_default_spconv_qconfig_mapping(True)
    # prepared_model_qat = qfx.prepare_qat_fx(model_qat, qconfig_mapping_qat, (), backend_config=backend_cfg, prepare_custom_config=prepare_cfg)
    # # converted_model = qfx.convert_fx(prepared_model_qat, qconfig_mapping=qconfig_mapping_qat, backend_config=backend_cfg)
    # # breakpoint()
    # print(prepared_model_qat)
    # train(args, prepared_model_qat, qdevice, train_loader, optimizer, 1)
    # converted_model = qfx.convert_fx(prepared_model_qat, qconfig_mapping=qconfig_mapping_qat, backend_config=backend_cfg)
    # converted_model = transform_qdq(converted_model)
    # test(args, converted_model, qdevice, test_loader)
    # # [type(m) for m in prepared_model_qat.modules()]
    # # model.qconfig = get_default_spconv_trt_ptq_qconfig()
    # # prepare_custom_config_dict = spconvq.get_prepare_custom_config()
    # # convert_custom_config_dict = spconvq.get_convert_custom_config()
    # # torch.ao.quantization.prepare(model, inplace=True)
    # # print('Post Training Quantization Prepare: Inserting Observers')
    # # print('\n ConvBnReLUBlock:After observer insertion \n\n', model.net[0])
    # # test(args, model, device, test_loader)
    # print(converted_model)
    # you will see some nvrtc compile log here, which means int8 kernel is used.
    breakpoint()

if __name__ == '__main__':
    main()
