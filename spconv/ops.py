# Copyright 2019 Yan Yan
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

from enum import Enum

import torch

import spconv


class ConvAlgo(Enum):
    Native = 0  # small memory cost, faster when number of points is large.
    Batch = 1  # high memory cost, faster when number of points is small (< 50000)
    BatchGemmGather = 2  # high memory cost, faster when number of points medium


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                           output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[
            i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(indices,
                     batch_size,
                     spatial_shape,
                     ksize=3,
                     stride=1,
                     padding=0,
                     dilation=1,
                     out_padding=0,
                     subm=False,
                     transpose=False,
                     grid=None,
                     use_hash=False):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                               padding, dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation)
    else:
        out_shape = spatial_shape
    if grid is None:
        res = torch.ops.spconv.get_indice_pairs(indices, batch_size, out_shape,
                                                spatial_shape, ksize, stride,
                                                padding, dilation, out_padding,
                                                int(subm), int(transpose),
                                                int(use_hash))
        return res
    else:
        if ndim == 2:
            get_indice_pairs_func = torch.ops.spconv.get_indice_pairs_grid_2d
        elif ndim == 3:
            get_indice_pairs_func = torch.ops.spconv.get_indice_pairs_grid_3d
        else:
            raise NotImplementedError
        return get_indice_pairs_func(indices, grid, batch_size, out_shape,
                                     spatial_shape, ksize, stride, padding,
                                     dilation, out_padding, int(subm),
                                     int(transpose), int(use_hash))


def indice_conv(features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                inverse=False,
                subm=False,
                algo=ConvAlgo.Native.value):
    return torch.ops.spconv.indice_conv(features, filters, indice_pairs,
                                        indice_pair_num, num_activate_out,
                                        int(inverse), int(subm), algo)


def fused_indice_conv(features, filters, bias, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm):
    return torch.ops.spconv.fused_indice_conv_bn(features, filters, bias,
                                                 indice_pairs, indice_pair_num,
                                                 num_activate_out,
                                                 int(inverse), int(subm))


def indice_conv_backward(features,
                         filters,
                         out_bp,
                         indice_pairs,
                         indice_pair_num,
                         inverse=False,
                         subm=False,
                         algo=ConvAlgo.Native.value):
    return torch.ops.spconv.indice_conv_backward(features, filters, out_bp,
                                                 indice_pairs, indice_pair_num,
                                                 int(inverse), int(subm), algo)


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    return torch.ops.spconv.indice_maxpool(features, indice_pairs,
                                           indice_pair_num, num_activate_out)


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs,
                            indice_pair_num):
    return torch.ops.spconv.indice_maxpool_backward(features, out_features,
                                                    out_bp, indice_pairs,
                                                    indice_pair_num)


def nms(boxes, scores, pre_max_size, post_max_size, thresh, eps):
    res = torch.ops.spconv.nms(boxes, scores, pre_max_size, post_max_size,
                               thresh, eps)
    return res


def pillar_scatter(features, coors, shape):
    if features.dtype == torch.float32:
        return torch.ops.spconv.pillar_scatter_float(features, coors, shape)
    elif features.dtype == torch.half:
        return torch.ops.spconv.pillar_scatter_half(features, coors, shape)
    else:
        raise NotImplementedError
