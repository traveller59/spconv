from spconv.pytorch.modules import SparseModule
from spconv.pytorch.conv import SparseConvolution
from spconv.pytorch.core import SparseConvTensor

import torch 

class ConvBatchNormAddAct(torch.nn.Module):
    """for simple int8 residual op fusion, we can use this module to handle add.
    """
    def __init__(self, conv: SparseConvolution, bn: torch.nn.BatchNorm1d, act: torch.nn.ReLU) -> None:
        super().__init__()

        self.conv = conv 
        self.bn = bn 
        self.act = act 

    def forward(self, x: SparseConvTensor, x_add: SparseConvTensor):
        x = self.conv(x)
        x = x.replace_feature(self.bn(x.features))
        return self.act(x.replace_feature(x.features + x_add.features))

