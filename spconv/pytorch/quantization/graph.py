import torch.fx
import torch 
from torch import nn 
from typing import Dict, Optional
from spconv.pytorch.quantization.core import quantize_per_tensor, quantized_add
import spconv.pytorch.quantization.intrinsic.quantized as snniq

def is_dequantize_node(node):
    return isinstance(node, torch.fx.Node) and node.op == "call_method" and node.target == "dequantize"

def _get_module(node: torch.fx.Node, modules: Dict[str, nn.Module]) -> Optional[nn.Module]:
    """
    Return the `torch.nn.Module` that corresponds to the specified node's target.
    If no such node exists, return None.
    """
    if node.op == "call_module" and str(node.target) in modules:
        return modules[str(node.target)]
    else:
        return None

def remove_conv_add_dq(model: torch.fx.graph_module.GraphModule):
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        if (n.op == "call_module" and type(_get_module(n, modules)) == snniq.SparseConvAddReLU):
            # check second input, if it's dequantized, remove that dequantize node
            arg1 = n.args[1]
            if is_dequantize_node(arg1):
                dq_node = arg1
                assert(isinstance(dq_node, torch.fx.Node))
                dn_input = dq_node.args[0]
                n.replace_input_with(dq_node, dn_input)
    model.graph.eliminate_dead_code()
    model.recompile()
    model.graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.
    return model
    
def transform_qdq(m: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """torch.quantize_per_tensor don't support SparseConvTensor, so we
    use a custom one by fx transform.
    """
    for node in m.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.quantize_per_tensor:
                node.target = quantize_per_tensor
            if node.target == torch.ops.quantized.add:
                node.target = quantized_add
    m.graph.eliminate_dead_code()
    m.recompile()
    m.graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.
    return m
