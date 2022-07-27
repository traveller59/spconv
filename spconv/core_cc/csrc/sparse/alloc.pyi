from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class ExternalAllocator:
    def zeros(self, name: str, shape: List[int], dtype: int, device: int, stream: int = 0, is_temp_memory: bool = False) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            dtype: 
            device: 
            stream: 
            is_temp_memory: 
        """
        ...
    def empty(self, name: str, shape: List[int], dtype: int, device: int, stream: int = 0, is_temp_memory: bool = False) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            dtype: 
            device: 
            stream: 
            is_temp_memory: 
        """
        ...
    def full_int(self, name: str, shape: List[int], value: int, dtype: int, device: int, stream: int = 0, is_temp_memory: bool = False) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            value: 
            dtype: 
            device: 
            stream: 
            is_temp_memory: 
        """
        ...
    def full_float(self, name: str, shape: List[int], value: float, dtype: int, device: int, stream: int = 0, is_temp_memory: bool = False) -> Tensor: 
        """
        Args:
            name: 
            shape: 
            value: 
            dtype: 
            device: 
            stream: 
            is_temp_memory: 
        """
        ...
    def get_tensor_by_name(self, name: str) -> Tensor: 
        """
        Args:
            name: 
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
