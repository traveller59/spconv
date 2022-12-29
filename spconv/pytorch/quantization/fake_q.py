
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize, fused_wt_fake_quant_range_neg_127_to_127
from spconv.pytorch.core import SparseConvTensor
import torch 
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver

class SparseFusedMovingAvgObsFakeQuantize(FusedMovingAvgObsFakeQuantize):
    def forward(self, input:SparseConvTensor):
        # add lines to support spconv
        x = input.features
        res_features = super().forward(x)
        return input.replace_feature(res_features)


default_symmetric_spconv_qat_qconfig = QConfig(
    activation=SparseFusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       reduce_range=False,
                                                       eps=2 ** -12),
    weight=fused_wt_fake_quant_range_neg_127_to_127)
