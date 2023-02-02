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

"""
This example shows how to write custom fx2trt like tool to convert
pytorch model to tensorrt.
"""

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

from torch.fx import Tracer
import tensorrt as trt 

from spconv.pytorch.quantization.interpreter import NetworkInterpreter, register_node_handler, register_method_handler
from spconv.pytorch.cppcore import torch_tensor_to_tv
import numpy as np 
import spconv.constants as spconvc
import torch.nn.functional as F

def _simple_repr(x):
    return f"Tensor[{x.shape}|{x.dtype}]"
# add verbose for ITensor
trt.ITensor.__repr__ = _simple_repr

class NetDense(nn.Module):
    def __init__(self):
        super(NetDense, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.conv_pool = nn.Conv2d(64, 64, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv_pool(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.training:
            x = F.log_softmax(x, dim=1)
        return x

def _activation(net, x, act_type, alpha=None, beta=None, name=None):
    layer = net.add_activation(x, act_type)
    if alpha is not None:
        layer.alpha = alpha
    if beta is not None:
        layer.beta = beta
    output = layer.get_output(0)
    if name is not None:
        output.name = name
        layer.name = name
    return output

def _trt_reshape(net, inp, shape, name):
    layer = net.add_shuffle(inp)
    layer.reshape_dims = shape
    output = layer.get_output(0)
    layer.name = name
    output.name = name
    return output

# add module handler
@register_node_handler(nn.Conv2d)
def _conv2d(net, target: nn.Conv2d, args, kwargs, name: str):
    x = args[0]
    bias = target.bias
    if target.bias is None:
        bias = None 
    else:
        bias = target.bias.detach().cpu().numpy()
    weight = target.weight.detach().cpu().numpy()
    
    O, I_groups, *ksize = weight.shape
    I = I_groups * target.groups
    stride = target.stride
    padding = target.padding
    dilation = target.dilation
    weight_qdq = None
    if not isinstance(weight, np.ndarray):
        weight_qdq = weight
        weight = trt.Weights()
    else:
        weight = trt.Weights(weight)
    if bias is None:
        bias = trt.Weights()
    else:
        bias = trt.Weights(bias)
    layer = net.add_convolution_nd(x, O, tuple(ksize), weight, bias)
    if weight_qdq is not None:
        # in explicit quantization, we need this
        layer.set_input(1, weight_qdq)
    layer.stride_nd = tuple(stride)
    layer.padding_nd = tuple(padding)
    layer.dilation_nd = tuple(dilation)
    layer.num_groups = target.groups
    output = layer.get_output(0)
    output.name = name
    layer.name = name
    return output
    
@register_node_handler(F.relu)
def _relu(net, target: nn.Conv2d, args, kwargs, name: str):
    return _activation(net, args[0], trt.ActivationType.RELU, name=name)


@register_node_handler(nn.Dropout)
@register_node_handler(nn.Dropout1d)
@register_node_handler(nn.Dropout2d)
@register_node_handler(nn.Dropout3d)
def _identity_single(net, target, args, kwargs, name: str):
    return args[0]

@register_node_handler(torch.flatten)
def _flatten(net, target, args, kwargs, name: str):
    start_dim = args[1]
    x = args[0]
    return _trt_reshape(net, x, [*x.shape[:start_dim], int(np.prod(x.shape[start_dim:]))], name)

def _dot(net, x, y, transpose_x=False, transpose_y=False, name=None):
    mode_x = trt.MatrixOperation.NONE
    if transpose_x:
        mode_x = trt.MatrixOperation.TRANSPOSE
    mode_y = trt.MatrixOperation.NONE
    if transpose_y:
        mode_y = trt.MatrixOperation.TRANSPOSE
    layer = net.add_matrix_multiply(x, mode_x, y, mode_y)

    output = layer.get_output(0)
    assert name is not None

    output.name = name
    layer.name = name
    return output

def _constant(net, array, name):
    array = np.array(array)
    layer = net.add_constant(array.shape, trt.Weights(array.reshape(-1)))
    out = layer.get_output(0)
    layer.name = name
    out.name = name
    return out

@register_node_handler(nn.Linear)
def _linear(net, target: nn.Linear, args, kwargs, name: str):
    x = args[0]
    bias = target.bias
    if target.bias is None:
        bias = None 
    else:
        bias = target.bias.detach().cpu().numpy()
    weight = target.weight.detach().cpu().numpy()
    weight_trt = _constant(net, weight, name + "/weight")
    res = _dot(net, x, weight_trt, transpose_y=True, name=name)
    if bias is not None:
        bias_trt = _constant(net, bias.reshape(1, -1), name + "/bias")
        layer = net.add_elementwise(res, bias_trt, trt.ElementWiseOperation.SUM)
        res = layer.get_output(0)
        add_name = name + "/add"
        res.name = add_name
        layer.name = add_name
    return res 

def main():
    model = NetDense()
    model = model.eval()
    tc = Tracer()
    graph_trace = tc.trace(model)
    gm = torch.fx.GraphModule(tc.root, graph_trace)
    import tensorrt as trt 
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # try:
    #     import pycuda.autoprimaryctx
    # except ModuleNotFoundError:
    #     import pycuda.autoinit
    with trt.Runtime(TRT_LOGGER) as rt:
        with trt.Builder(TRT_LOGGER) as builder:
            with builder.create_network(True) as network:
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 30
                input_tensor = network.add_input(name="inp", dtype=trt.float32, shape=[1, 1, 28, 28])
                interp = NetworkInterpreter(network, gm, [input_tensor], verbose=True)
                # get converted outputs from interp
                outputs = interp.run()
                network.mark_output(tensor=outputs[0])
                plan = builder.build_serialized_network(network, config)
                engine = rt.deserialize_cuda_engine(plan)

if __name__ == '__main__':
    main()
