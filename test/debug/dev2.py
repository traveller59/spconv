from collections import OrderedDict
import contextlib
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.ao.quantization.fx.match_utils import (
    MatchAllNode,
)
from torch.ao.quantization.quantize_fx import (
    fuse_fx,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
    get_fbgemm_backend_config
)
from torch.ao.quantization import get_default_qconfig_mapping

import torch.ao.quantization.quantize_fx as qfx

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3)
        self.iden = nn.Conv2d(3, 3, 3)
        # self.iden2 = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        y = x
        y = self.iden(x)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.add(x, y)
        x = self.relu(x)
        return x


m = M().eval()

def fuse_conv_bn_relu(is_qat, relu, add_pattern):
    _, bn_pattern, _ = add_pattern
    bn, conv = bn_pattern
    return conv

def conv_bn_res_relu_root_node_getter(pattern):
    relu, add_pattern = pattern
    _, bn_pattern, _ = add_pattern
    bn, conv = bn_pattern
    return conv

def conv_bn_res_relu_extra_inputs_getter(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, bn_pattern, extra_input = add_pattern
    bn, conv = bn_pattern
    return [extra_input]
# for pytorch <= 1.13
# conv_bn_res_relu_config = BackendPatternConfig((nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
#     .set_fuser_method(fuse_conv_bn_relu) \
#     ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
#     ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter)
fbgemm_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)
# for pytorch master
conv_bn_res_relu_config = BackendPatternConfig() \
    ._set_pattern_complex_format((nn.ReLU, (torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
    .set_fuser_method(fuse_conv_bn_relu) \
    ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
    ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter) \
    .set_dtype_configs(fbgemm_weighted_op_int8_dtype_config)

backend_config = get_fbgemm_backend_config()# .set_backend_pattern_config(conv_bn_res_relu_config)
# m = fuse_fx(m, backend_config=backend_config)
qmapping = get_default_qconfig_mapping()
prepared_model = qfx.prepare_fx(m, qmapping, (), backend_config=backend_config)
prepared_model.print_readable()
converted_model = qfx.convert_fx(prepared_model, qconfig_mapping=qmapping, backend_config=backend_config)

converted_model.print_readable()