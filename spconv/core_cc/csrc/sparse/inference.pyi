from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
from cumm.tensorview.gemm import Activation
class InferenceOps:
    @staticmethod
    def bias_add_act_inplace(out: Tensor, bias: Tensor, act_type: Activation =  Activation.None_, alpha: float = 0.0, beta: float = 0.0, stream: int = 0) -> None: 
        """
        Args:
            out: 
            bias: 
            act_type: 
            alpha: 
            beta: 
            stream: 
        """
        ...
    @staticmethod
    def bias_add_inplace(out: Tensor, bias: Tensor, stream: int = 0) -> None: 
        """
        Args:
            out: 
            bias: 
            stream: 
        """
        ...
    @staticmethod
    def activation_inplace(out: Tensor, act_type: Activation, alpha: float, beta: float, stream: int = 0) -> None: 
        """
        Args:
            out: 
            act_type: 
            alpha: 
            beta: 
            stream: 
        """
        ...
