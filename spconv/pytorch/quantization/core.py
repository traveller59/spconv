from typing import Union, List, Dict 

import torch 

from spconv.pytorch.core import SparseConvTensor
from cumm import tensorview as tv 
from spconv.pytorch.cppcore import get_current_stream, torch_tensor_to_tv


def quantize_per_tensor(ten: Union[Union[SparseConvTensor, torch.Tensor], List[Union[SparseConvTensor, torch.Tensor]]], scale, zero_point, dtype):
    # with tv.measure_and_print("quantize_per_tensor", stream=get_current_stream()):
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

def quantized_add(x: torch.Tensor, y: torch.Tensor, scale, zero_point):
    x_detach = torch.zeros(size=x.shape, dtype=torch.int8, device=x.device)
    y_detach = torch.zeros(size=y.shape, dtype=torch.int8, device=y.device)
    torch_tensor_to_tv(x_detach).copy_(torch_tensor_to_tv(x))
    torch_tensor_to_tv(y_detach).copy_(torch_tensor_to_tv(y))
    res = (x_detach.to(torch.float32) * x.q_scale() + y_detach.to(torch.float32) * y.q_scale()) / scale
    res = torch.clip(torch.round(res), -128, 127).to(torch.int8)
    res_q = torch._empty_affine_quantized(size=res.shape, dtype=torch.qint8, scale=scale, zero_point=zero_point, device=x.device)
    torch_tensor_to_tv(res_q, tv.int8).copy_(torch_tensor_to_tv(res))
    return res_q


