from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
class CompileInfo:
    @staticmethod
    def get_compiled_cuda_arch() -> List[Tuple[int, int]]: ...
    @staticmethod
    def get_compiled_gemm_cuda_arch() -> List[Tuple[int, int]]: ...
    @staticmethod
    def arch_is_compiled(arch: Tuple[int, int]) -> bool: 
        """
        Args:
            arch: 
        """
        ...
    @staticmethod
    def arch_is_compiled_gemm(arch: Tuple[int, int]) -> bool: 
        """
        Args:
            arch: 
        """
        ...
