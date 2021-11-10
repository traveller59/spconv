from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
class CUDAEvent:
    def __init__(self, name: str) -> None: 
        """
        Args:
            name: 
        """
        ...
    def record(self, stream: int = 0) -> None: 
        """
        Args:
            stream: 
        """
        ...
    def sync(self) -> None: ...
    @staticmethod
    def duration(start: "CUDAEvent", stop: "CUDAEvent") -> float: 
        """
        Args:
            start: 
            stop: 
        """
        ...
class CUDAKernelTimer:
    enable: bool
    def __init__(self, enable: bool = True) -> None: 
        """
        Args:
            enable: 
        """
        ...
    def push(self, name: str) -> None: 
        """
        Args:
            name: 
        """
        ...
    def pop(self) -> None: ...
    def record(self, name: str, stream: int = 0) -> None: 
        """
        Args:
            name: 
            stream: 
        """
        ...
    def insert_pair(self, name: str, start: str, stop: str) -> None: 
        """
        Args:
            name: 
            start: 
            stop: 
        """
        ...
    def get_all_pair_duration(self) -> Dict[str, float]: ...
    def sync(self) -> None: ...
