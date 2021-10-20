from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class ScatterAll:
    def __init__(self) -> None: ...
    @staticmethod
    def get_all_scatter_params() -> List[Tuple[int, int, int, int]]: ...
    def supported_scatter(self, tile_m: int, tile_k_bytes: int, bytes_per_access: int, num_threads: int, channel_size: int, dtype: int) -> bool: 
        """
        Args:
            tile_m: 
            tile_k_bytes: 
            bytes_per_access: 
            num_threads: 
            channel_size: 
            dtype: 
        """
        ...
    @staticmethod
    def stream_synchronize(stream: int = 0) -> None: 
        """
        Args:
            stream: 
        """
        ...
    def scatter(self, output: Tensor, input: Tensor, indices: Tensor, tile_m: int, tile_k_bytes: int, bytes_per_access: int, num_threads: int, stream: int = 0) -> None: 
        """
        Args:
            output: 
            input: 
            indices: 
            tile_m: 
            tile_k_bytes: 
            bytes_per_access: 
            num_threads: 
            stream: 
        """
        ...
    def scatter2(self, output: Tensor, input: Tensor, indices: Tensor, size: int, stream: int = 0) -> None: 
        """
        Args:
            output: 
            input: 
            indices: 
            size: 
            stream: 
        """
        ...
class GatherAll:
    def __init__(self) -> None: ...
    @staticmethod
    def get_all_gather_params() -> List[Tuple[int, int, int, int]]: ...
    @staticmethod
    def supported(bytes_per_access: int, channel_size: int, dtype: int) -> bool: 
        """
        Args:
            bytes_per_access: 
            channel_size: 
            dtype: 
        """
        ...
    @staticmethod
    def stream_synchronize(stream: int = 0) -> None: 
        """
        Args:
            stream: 
        """
        ...
    def gather(self, output: Tensor, input: Tensor, indices: Tensor, tile_m: int, tile_k_bytes: int, bytes_per_access: int, num_threads: int, stream: int = 0) -> None: 
        """
        Args:
            output: 
            input: 
            indices: 
            tile_m: 
            tile_k_bytes: 
            bytes_per_access: 
            num_threads: 
            stream: 
        """
        ...
    def gather2(self, output: Tensor, input: Tensor, indices: Tensor, size: int, stream: int = 0) -> None: 
        """
        Args:
            output: 
            input: 
            indices: 
            size: 
            stream: 
        """
        ...
