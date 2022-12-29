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

"""simple fuse
see https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html
"""

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from cumm import tensorview as tv
from spconv.core import ConvAlgo
import torch.fx
import spconv.pytorch as spconv
import copy 
import pickle
from spconv.pytorch.conv import SparseConvolution
from spconv.pytorch import functional as Fsp


def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim+1] + [i+1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    # OIDHW
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0,] + [i+2 for i in range(NDim)] + [1,]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)

def fuse_bn(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_act_net(conv, act):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)
    if isinstance(act, torch.nn.ReLU):
        fused_conv.act_type = tv.gemm.Activation.ReLU
    elif isinstance(act, torch.nn.Sigmoid):
        fused_conv.act_type = tv.gemm.Activation.Sigmoid
    elif isinstance(act, torch.nn.LeakyReLU):
        fused_conv.act_type = tv.gemm.Activation.LeakyReLU
        fused_conv.act_alpha = act.negative_slope
    else:
        raise NotImplementedError
    return fused_conv

def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: torch.fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)

def fuse(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    model = copy.deepcopy(model)
    # The first step of most FX passes is to symbolically trace our model to
    # obtain a `GraphModule`. This is a representation of our original model
    # that is functionally identical to our original model, except that we now
    # also have a graph representation of our forward pass.
    fx_model = model
    modules = dict(fx_model.named_modules())
    # The primary representation for working with FX are the `Graph` and the
    # `Node`. Each `GraphModule` has a `Graph` associated with it - this
    # `Graph` is also what generates `GraphModule.code`.
    # The `Graph` itself is represented as a list of `Node` objects. Thus, to
    # iterate through all of the operations in our graph, we iterate over each
    # `Node` in our `Graph`.
    for node in fx_model.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.

        if node.op != 'call_module': # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        # print(node.target, node.args, node.args[0].args)
        if isinstance(modules[node.target], torch.nn.BatchNorm1d):
            if node.args[0].target in modules and isinstance(modules[node.args[0].target], SparseConvolution):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_conv = fuse_bn(conv, bn)
                assert isinstance(fused_conv, SparseConvolution)
                replace_node_module(node.args[0], modules, fused_conv)
                modules_to_fuse = [node.args[0].target, node.target]
                try:
                    if isinstance(modules[node.next.target], torch.nn.ReLU):
                        modules_to_fuse.append(node.next.target)
                except Exception as e:
                    pass
                # As we've folded the batch nor into the conv, we need to replace all uses
                # of the batch norm with the conv.
                node.replace_all_uses_with(node.args[0])
                # Now that all uses of the batch norm have been replaced, we can
                # safely remove the batch norm.
                fx_model.graph.erase_node(node)
                
    fx_model.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    fx_model.recompile()
    return fx_model


def fuse_act(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    model = copy.deepcopy(model)
    # The first step of most FX passes is to symbolically trace our model to
    # obtain a `GraphModule`. This is a representation of our original model
    # that is functionally identical to our original model, except that we now
    # also have a graph representation of our forward pass.
    fx_model = model
    modules = dict(fx_model.named_modules())
    # The primary representation for working with FX are the `Graph` and the
    # `Node`. Each `GraphModule` has a `Graph` associated with it - this
    # `Graph` is also what generates `GraphModule.code`.
    # The `Graph` itself is represented as a list of `Node` objects. Thus, to
    # iterate through all of the operations in our graph, we iterate over each
    # `Node` in our `Graph`.
    for node in fx_model.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.

        if node.op != 'call_module': # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        # print(node.target, node.args, node.args[0].args)
        if isinstance(modules[node.target], (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid)):
            if node.args[0].target in modules and isinstance(modules[node.args[0].target], SparseConvolution):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                act = modules[node.target]
                fused_conv = fuse_act_net(conv, act)
                assert isinstance(fused_conv, SparseConvolution)
                replace_node_module(node.args[0], modules, fused_conv)
                # As we've folded the batch nor into the conv, we need to replace all uses
                # of the batch norm with the conv.
                node.replace_all_uses_with(node.args[0])
                # Now that all uses of the batch norm have been replaced, we can
                # safely remove the batch norm.
                fx_model.graph.erase_node(node)
                
    fx_model.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    fx_model.recompile()
    return fx_model


class Net(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(16, 64, 3, bias=False, indice_key="c0",
                              algo=algo),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64, 2, 2, bias=False, indice_key="m0", algo=algo),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64,
                              96,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            spconv.SubMConv3d(96,
                              96,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            spconv.SparseConv3d(96, 96, 2, 2, bias=False, indice_key="m1", algo=algo),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            spconv.SubMConv3d(96,
                              128,
                              3,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SubMConv3d(128,
                              128,
                              3,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SparseConv3d(128, 128, 2, 2, bias=False, indice_key="m2", algo=algo),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            spconv.SubMConv3d(128,
                              160,
                              3,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            spconv.SubMConv3d(160,
                              160,
                              3,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            spconv.SparseConv3d(160, 160, 2, 2, bias=False, indice_key="m3", algo=algo),
            nn.BatchNorm1d(160),
            nn.ReLU(),

            spconv.SubMConv3d(160,
                              192,
                              3,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            nn.BatchNorm1d(192),
            nn.ReLU(),

            spconv.SubMConv3d(192,
                              192,
                              3,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            spconv.SparseConv3d(192, 192, 2, 2, bias=False, indice_key="m4", algo=algo),
            nn.BatchNorm1d(192),
            nn.ReLU(),

            spconv.SubMConv3d(192,
                              224,
                              3,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            nn.BatchNorm1d(224),
            nn.ReLU(),

            spconv.SubMConv3d(224,
                              224,
                              3,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            nn.BatchNorm1d(224),
            nn.ReLU(),
            spconv.SparseConv3d(224, 224, 2, 2, bias=False, indice_key="m5", algo=algo),
            nn.BatchNorm1d(224),
            nn.ReLU(),

            spconv.SubMConv3d(224,
                              256,
                              3,
                              bias=False,
                              indice_key="c6",
                              algo=algo),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            spconv.SubMConv3d(256,
                              256,
                              3,
                              bias=False,
                              indice_key="c6",
                              algo=algo),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            spconv.SparseInverseConv3d(256,
                                       128,
                                       2,
                                       indice_key="m5",
                                       bias=False,
                                       algo=algo),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SparseInverseConv3d(128,
                                       64,
                                       2,
                                       indice_key="m4",
                                       bias=False,
                                       algo=algo),
            nn.BatchNorm1d(64),
            nn.ReLU(),

        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        # self.grid = None
        self.shape = shape
        for n in self.net.modules():
            if isinstance(n, nn.BatchNorm1d):
                n.bias.data.uniform_(-0.1, 0.1)

    def forward(self, features, coors, batch_size, vx_num=None):
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size, voxel_num=vx_num)
        return self.net(x)

def _set_enable_int8_test_inplace(simple_module: torch.fx.GraphModule, enable: bool):
    for m in simple_module.modules():
        if isinstance(m, SparseConvolution):
            if m.in_channels % 32 == 0 and m.out_channels % 32 == 0:
                m.enable_int8_test_mode = enable


class MyTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name):
        is_custom_leaf_module = isinstance(m, SparseConvolution)
        return super().is_leaf_module(m, module_qualified_name) or is_custom_leaf_module

def main():
    # run this file with SPCONV_FX_TRACE_MODE=1
    torch.manual_seed(50051)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with open(Path(__file__).parent.parent / "test" / "data" / "test_spconv.pkl", "rb") as f:
        (voxels, coors, spatial_shape) = pickle.load(f)
    voxels = np.random.uniform(-1, 1, size=[voxels.shape[0], 16]).astype(np.float32)
    np.random.seed(50051)
    device = torch.device("cuda:0")
    device_cpu = torch.device("cpu:0")
    dtype = torch.float32
    net = Net(spatial_shape, ConvAlgo.MaskImplicitGemm).cuda().eval().to(dtype)
    tracer = MyTracer()
    graph_trace = tracer.trace(net)
    net_fused = torch.fx.GraphModule(tracer.root, graph_trace)
    net_fused = fuse(net_fused)
    net_fused = fuse_act(net_fused)
    print(net_fused)
    voxels_th = torch.from_numpy(voxels).to(device_cpu).to(dtype)
    coors_th = torch.from_numpy(coors).to(device_cpu).int()
    voxels_th_cuda = torch.from_numpy(voxels).to(device).to(dtype)
    coors_th_cuda = torch.from_numpy(coors).to(device).int()

    out_ref = net(voxels_th_cuda, coors_th_cuda, 1)
    print("-------------fused------------")
    out_fused = net_fused(voxels_th_cuda, coors_th_cuda, 1)
    res = Fsp.sparse_add_hash_based(out_ref, out_fused.minus())
    print(torch.linalg.norm(res.features))
    _set_enable_int8_test_inplace(net_fused, True)
    qvoxels_cuda = voxels_th_cuda.to(torch.int8)
    
    out_int8 = net_fused(qvoxels_cuda, coors_th_cuda, 1)

if __name__ == "__main__":
    main()