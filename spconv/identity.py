# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import Module


class Identity(Module):
    def forward(self, input):
        return input

    def input_spatial_size(self, out_size):
        return out_size
