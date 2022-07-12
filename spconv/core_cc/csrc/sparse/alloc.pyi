from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class ExternalAllocator:
    def zeros(self, name: str, shape: List[int], dtype: int, device: int) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            dtype: 
            device: 
        """
        ...
    def empty(self, name: str, shape: List[int], dtype: int, device: int) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            dtype: 
            device: 
        """
        ...
    def full_int(self, name: str, shape: List[int], value: int, dtype: int, device: int) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            value: 
            dtype: 
            device: 
        """
        ...
    def full_float(self, name: str, shape: List[int], value: float, dtype: int, device: int) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            value: 
            dtype: 
            device: 
        """
        ...
    def free(self, ten: Tensor) -> None: 
        """
        Args:
            ten: 
        """
        ...
    def free_noexcept(self, ten: Tensor) -> None: 
        """
        Args:
            ten: 
        """
        ...
