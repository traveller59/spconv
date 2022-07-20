from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview.gemm import GemmAlgoDesp
from cumm.tensorview.gemm import GemmParams
class GemmMainUnitTest:
    @staticmethod
    def get_all_algo_desp() -> List[GemmAlgoDesp]: ...
    @staticmethod
    def extract_mnk(a_shape: List[int], b_shape: List[int], trans_a: bool, trans_b: bool, trans_c: bool, shuffle_type: int = 0, a_inds_shape: List[int] =  [], b_inds_shape: List[int] =  [], c_inds_shape: List[int] =  []) -> Tuple[int, int, int]: 
        """
        Args:
            a_shape: 
            b_shape: 
            trans_a: 
            trans_b: 
            trans_c: 
            shuffle_type: 
            a_inds_shape: 
            b_inds_shape: 
            c_inds_shape: 
        """
        ...
    @staticmethod
    def align_to_power2(val: int) -> int: 
        """
        Args:
            val: 
        """
        ...
    @staticmethod
    def device_synchronize() -> None: ...
    @staticmethod
    def stream_synchronize(stream: int) -> None: 
        """
        Args:
            stream: 
        """
        ...
    @staticmethod
    def simple_select_tile_shape(m: int, n: int, k: int, tile_ms: List[int], tile_ns: List[int], tile_ks: List[int], tile_shape_to_algos: Dict[int, List[int]], large_k_first: bool) -> List[int]: 
        """
        Args:
            m: 
            n: 
            k: 
            tile_ms: 
            tile_ns: 
            tile_ks: 
            tile_shape_to_algos: 
            large_k_first: 
        """
        ...
    @staticmethod
    def matmul2(params: GemmParams) -> None: 
        """
        Args:
            params: 
        """
        ...
