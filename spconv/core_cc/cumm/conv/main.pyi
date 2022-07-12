from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview.gemm import ConvParams
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
