from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview.gemm import GemmAlgoDesp
from cumm.tensorview import Tensor
from cumm.tensorview.gemm import NVRTCParams
from spconv.core_cc.csrc.sparse.convops import GemmTuneResult
from cumm.tensorview import CUDAKernelTimer
from cumm.tensorview.gemm import Activation
class GemmTunerSimple:
    def __init__(self, desps: List[GemmAlgoDesp]) -> None: 
        """
        Args:
            desps: 
        """
        ...
    @staticmethod
    def get_available_algo_str_from_arch(arch: Tuple[int, int]) -> List[str]: 
        """
        Args:
            arch: 
        """
        ...
    def get_all_available(self, a: Tensor, b: Tensor, c: Tensor, trans_a: bool, trans_b: bool, trans_c: bool, arch: Tuple[int, int], shuffle_type: int, use_tf32: bool = True) -> List[GemmAlgoDesp]: 
        """
        Args:
            a: 
            b: 
            c: 
            trans_a: 
            trans_b: 
            trans_c: 
            arch: 
            shuffle_type: 
            use_tf32: 
        """
        ...
    def cached_get_nvrtc_params(self, desp: GemmAlgoDesp, arch: Tuple[int, int], stream_int: int) -> NVRTCParams: 
        """
        Args:
            desp: 
            arch: 
            stream_int: 
        """
        ...
    def tune_and_cache(self, a: Tensor, b: Tensor, c: Tensor, trans_a: bool, trans_b: bool, trans_c: bool, arch: Tuple[int, int], shuffle_type: int, a_inds: Tensor, b_inds: Tensor, c_inds: Tensor, hint: int = 0, alpha: float = 1.0, beta: float = 0.0, stream_int: int = 0, num_run: int = 5, use_tf32: bool = True) -> Tuple[GemmTuneResult, float]: 
        """
        Args:
            a: 
            b: 
            c: 
            trans_a: 
            trans_b: 
            trans_c: 
            arch: 
            shuffle_type: 
            a_inds: 
            b_inds: 
            c_inds: 
            hint: 
            alpha: 
            beta: 
            stream_int: 
            num_run: 
            use_tf32: 
        """
        ...
    def get_tuned_algo(self, a_dtype: int, b_dtype: int, c_dtype: int, a_shape: List[int], b_shape: List[int], c_shape: List[int], trans_a: bool, trans_b: bool, trans_c: bool, arch: Tuple[int, int], shuffle_type: int, a_inds_shape: List[int], b_inds_shape: List[int], c_inds_shape: List[int], hint: int = 0) -> Tuple[Any, bool]: 
        """
        Args:
            a_dtype: 
            b_dtype: 
            c_dtype: 
            a_shape: 
            b_shape: 
            c_shape: 
            trans_a: 
            trans_b: 
            trans_c: 
            arch: 
            shuffle_type: 
            a_inds_shape: 
            b_inds_shape: 
            c_inds_shape: 
            hint: 
        """
        ...
    def run_with_tuned_result(self, profile_res, a: Tensor, b: Tensor, c: Tensor, trans_a: bool, trans_b: bool, trans_c: bool, arch: Tuple[int, int], stream_int: int, shuffle_type: int, a_inds: Tensor, b_inds: Tensor, c_inds: Tensor, hint: int = 0, alpha: float = 1.0, beta: float = 0.0, workspace: Tensor =  Tensor(), timer: CUDAKernelTimer =  CUDAKernelTimer(False), force_nvrtc: bool = False, bias: Tensor =  Tensor(), act_alpha: float = 0.0, act_beta: float = 0.0, act_type: Activation =  Activation.None_) -> None: 
        """
        Args:
            profile_res: 
            a: 
            b: 
            c: 
            trans_a: 
            trans_b: 
            trans_c: 
            arch: 
            stream_int: 
            shuffle_type: 
            a_inds: 
            b_inds: 
            c_inds: 
            hint: 
            alpha: 
            beta: 
            workspace: 
            timer: 
            force_nvrtc: 
            bias: 
            act_alpha: 
            act_beta: 
            act_type: 
        """
        ...
