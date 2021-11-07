from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from ...cumm.gemm.main import GemmAlgoDesp
from cumm.tensorview import Tensor
class ConvAlgoDesp(GemmAlgoDesp):
    ndim: int
    op_type: int
    iter_algo: int
    layout_i: int
    layout_w: int
    layout_o: int
    interleave_i: int
    interleave_w: int
    interleave_o: int
    mask_sparse: bool
    increment_k_first: bool
    def __init__(self, ndim: int, op_type: int) -> None: 
        """
        Args:
            ndim: 
            op_type: 
        """
        ...
    def __repr__(self) -> str: ...
    @staticmethod
    def conv_iwo_012_to_abc(op_type: int) -> List[int]: 
        """
        Args:
            op_type: 
        """
        ...
    @staticmethod
    def gemm_abc_012_to_iwo(op_type: int) -> List[int]: 
        """
        Args:
            op_type: 
        """
        ...
    @property
    def dtype_input(self) -> int: ...
    @property
    def dtype_weight(self) -> int: ...
    @property
    def dtype_output(self) -> int: ...
    def supported(self, m: int, n: int, k: int, C: int, K: int, mask_width: int) -> bool: 
        """
        Args:
            m: 
            n: 
            k: 
            C: 
            K: 
            mask_width: 
        """
        ...
    def query_conv_workspace_size(self, m: int, n: int, k: int, split_k_slices: int, kv: int) -> int: 
        """
        Args:
            m: 
            n: 
            k: 
            split_k_slices: 
            kv: 
        """
        ...
    def supported_ldx_conv(self, ldi: int, ldw: int, ldo: int) -> bool: 
        """
        Args:
            ldi: 
            ldw: 
            ldo: 
        """
        ...
class ConvParams:
    conv_algo_desp: Any
    input: Tensor
    weight: Tensor
    output: Tensor
    split_k_slices: int
    padding: List[int]
    stride: List[int]
    dilation: List[int]
    alpha: float
    beta: float
    mask_width: int
    mask_filter: int
    reverse_mask: bool
    verbose: bool
    workspace: Tensor =  Tensor()
    mask: Tensor =  Tensor()
    mask_argsort: Tensor =  Tensor()
    indices: Tensor =  Tensor()
    mask_output: Tensor =  Tensor()
    stream: int
    def __init__(self, ndim: int, op_type: int) -> None: 
        """
        Args:
            ndim: 
            op_type: 
        """
        ...
class ConvMainUnitTest:
    @staticmethod
    def extract_mnk(op_type: int, N: int, C: int, K: int, kernel_volume: int, in_prod: int, out_prod: int, mask_sparse: bool) -> List[int]: 
        """
        Args:
            op_type: 
            N: 
            C: 
            K: 
            kernel_volume: 
            in_prod: 
            out_prod: 
            mask_sparse: 
        """
        ...
    @staticmethod
    def implicit_gemm2(params: ConvParams) -> None: 
        """
        Args:
            params: 
        """
        ...
    @staticmethod
    def get_all_conv_algo_desp() -> List[ConvAlgoDesp]: ...
