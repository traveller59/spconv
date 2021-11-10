from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
from cumm.tensorview import CUDAKernelTimer
class GemmAlgoDesp:
    dtype_a: int
    dtype_b: int
    dtype_c: int
    tile_shape: Tuple[int, int, int]
    warp_tile_shape: Tuple[int, int, int]
    num_stage: int
    dacc: int
    dcomp: int
    algo: str
    tensorop: List[int]
    split_k_serial_: int
    split_k_parallel_: int
    shuffle_type: str
    element_per_access_a: int
    element_per_access_b: int
    element_per_access_c: int
    access_per_vector: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def split_k_serial(self) -> bool: ...
    @split_k_serial.setter
    def split_k_serial(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def split_k_parallel(self) -> bool: ...
    @split_k_parallel.setter
    def split_k_parallel(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    def check_valid(self) -> None: ...
    @property
    def trans_a(self) -> bool: ...
    @trans_a.setter
    def trans_a(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def trans_b(self) -> bool: ...
    @trans_b.setter
    def trans_b(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def trans_c(self) -> bool: ...
    @trans_c.setter
    def trans_c(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    def query_workspace_size(self, m: int, n: int, k: int, split_k_slices: int) -> int: 
        """
        Args:
            m: 
            n: 
            k: 
            split_k_slices: 
        """
        ...
    def supported(self, m: int, n: int, k: int) -> bool: 
        """
        Args:
            m: 
            n: 
            k: 
        """
        ...
    def supported_ldx(self, lda: int, ldb: int, ldc: int) -> bool: 
        """
        Args:
            lda: 
            ldb: 
            ldc: 
        """
        ...
class GemmParams:
    algo_desp: GemmAlgoDesp
    split_k_slices: int
    workspace: Tensor =  Tensor()
    a_inds: Tensor =  Tensor()
    b_inds: Tensor =  Tensor()
    c_inds: Tensor =  Tensor()
    alpha: float
    beta: float
    stream: int
    timer: CUDAKernelTimer
    def __init__(self, timer: CUDAKernelTimer =  CUDAKernelTimer(False)) -> None: 
        """
        Args:
            timer: 
        """
        ...
    def check_valid(self) -> None: ...
    @property
    def a(self) -> Tensor: ...
    @a.setter
    def a(self, val: Tensor) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def b(self) -> Tensor: ...
    @b.setter
    def b(self, val: Tensor) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def c(self) -> Tensor: ...
    @c.setter
    def c(self, val: Tensor) -> None: 
        """
        Args:
            val: 
        """
        ...
class GemmMainUnitTest:
    @staticmethod
    def get_all_algo_desp() -> List[GemmAlgoDesp]: ...
    @staticmethod
    def extract_mnk(a_shape: List[int], b_shape: List[int], trans_a: bool, trans_b: bool, trans_c: bool, shuffle_type: str = "NS", a_inds_shape: List[int] =  [], b_inds_shape: List[int] =  [], c_inds_shape: List[int] =  []) -> Tuple[int, int, int]: 
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
