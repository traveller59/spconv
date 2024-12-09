from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue, enum
from cumm.tensorview.gemm import GemmAlgoDesp
from cumm.tensorview.gemm import ConvAlgoDesp
from cumm.tensorview import Tensor
class GemmTuneResult:
    algo_desp: GemmAlgoDesp
    arch: Tuple[int, int]
    splitk: int
    def is_valid(self) -> bool: ...
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, algo_desp: GemmAlgoDesp, arch: Tuple[int, int], splitk: int) -> None: 
        """
        Args:
            algo_desp: 
            arch: 
            splitk: 
        """
        ...
class ConvTuneResult:
    algo_desp: ConvAlgoDesp
    arch: Tuple[int, int]
    splitk: int
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, algo_desp: ConvAlgoDesp, arch: Tuple[int, int], splitk: int) -> None: 
        """
        Args:
            algo_desp: 
            arch: 
            splitk: 
        """
        ...
    def is_valid(self) -> bool: ...
class ExternalSpconvMatmul:
    def indice_conv_init_gemm(self, features_n: str, filters_n: str, all_weight_is_krsc: bool, is_kc_not_ck: bool, kv_center: int, out_channel: int, stream_int: int = 0) -> Tensor: 
        """
        Args:
            features_n: 
            filters_n: 
            all_weight_is_krsc: 
            is_kc_not_ck: 
            kv_center: 
            out_channel: 
            stream_int: 
        """
        ...
    def indice_conv_cpu_gemm(self, inp_buffer_n: str, out_buffer_n: str, filters_n: str, all_weight_is_krsc: bool, is_kc_not_ck: bool, nhot: int, index: int) -> None: 
        """
        Args:
            inp_buffer_n: 
            out_buffer_n: 
            filters_n: 
            all_weight_is_krsc: 
            is_kc_not_ck: 
            nhot: 
            index: 
        """
        ...
    def indice_conv_bwd_init_gemm(self, features_n: str, filters_n: str, out_bp_n: str, dfilters_n: str, all_weight_is_krsc: bool, is_kc_not_ck: bool, kv_center: int, stream_int: int = 0) -> Tensor: 
        """
        Args:
            features_n: 
            filters_n: 
            out_bp_n: 
            dfilters_n: 
            all_weight_is_krsc: 
            is_kc_not_ck: 
            kv_center: 
            stream_int: 
        """
        ...
    def indice_conv_bwd_cpu_gemm(self, inp_buffer_n: str, out_buffer_n: str, filters_n: str, dfilters_n: str, all_weight_is_krsc: bool, is_kc_not_ck: bool, nhot: int, index: int) -> None: 
        """
        Args:
            inp_buffer_n: 
            out_buffer_n: 
            filters_n: 
            dfilters_n: 
            all_weight_is_krsc: 
            is_kc_not_ck: 
            nhot: 
            index: 
        """
        ...
class SimpleExternalSpconvMatmul(ExternalSpconvMatmul):
    def __init__(self, alloc) -> None: 
        """
        Args:
            alloc: 
        """
        ...
