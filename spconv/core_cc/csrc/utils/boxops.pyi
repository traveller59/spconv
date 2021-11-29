from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class BoxOps:
    @staticmethod
    def has_boost() -> bool: ...
    @staticmethod
    def non_max_suppression_cpu(boxes: Tensor, order: Tensor, thresh: float, eps: float = 0) -> List[int]: 
        """
        Args:
            boxes: 
            order: 
            thresh: 
            eps: 
        """
        ...
