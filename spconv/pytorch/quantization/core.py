from typing import Union, List, Dict 

import torch 

from spconv.pytorch.core import SparseConvTensor

def quantize_per_tensor(ten: Union[Union[SparseConvTensor, torch.Tensor], List[Union[SparseConvTensor, torch.Tensor]]], scale, zero_point, dtype):
    if isinstance(ten, (list, tuple)):
        res = []
        for i, v in enumerate(ten):
            if isinstance(v, SparseConvTensor):
                res.append(v.replace_feature(torch.quantize_per_tensor(v.features, scale[i], zero_point[i], dtype)))
            else:
                res.append(torch.quantize_per_tensor(v, scale[i], zero_point[i], dtype))
        return res 
    else:
        if isinstance(ten, SparseConvTensor):
            return ten.replace_feature(torch.quantize_per_tensor(ten.features, scale, zero_point, dtype))
        else:
            return torch.quantize_per_tensor(ten, scale, zero_point, dtype)
    