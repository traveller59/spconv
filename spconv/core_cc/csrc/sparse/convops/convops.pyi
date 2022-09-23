from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview.gemm import ConvAlgoDesp
from cumm.tensorview import Tensor
from cumm.tensorview.gemm import NVRTCParams
from spconv.core_cc.csrc.sparse.convops import ConvTuneResult
from cumm.tensorview import CUDAKernelTimer
from cumm.tensorview.gemm import Activation
class ConvTunerSimple:
    def __init__(self, desps: List[ConvAlgoDesp]) -> None: 
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
    def get_all_available(self, inp: Tensor, weight: Tensor, out: Tensor, layout_i: int, layout_w: int, layout_o: int, interleave_i: int, interleave_w: int, interleave_o: int, arch: Tuple[int, int], op_type: int, mask_width: int, auto_fp32_accum: bool, fp32_accum: bool, use_tf32: bool = True) -> List[ConvAlgoDesp]: 
        """
        Args:
            inp: 
            weight: 
            out: 
            layout_i: 
            layout_w: 
            layout_o: 
            interleave_i: 
            interleave_w: 
            interleave_o: 
            arch: 
            op_type: 
            mask_width: 
            auto_fp32_accum: 
            fp32_accum: 
            use_tf32: 
        """
        ...
    def cached_get_nvrtc_params(self, desp: ConvAlgoDesp, arch: Tuple[int, int], stream_int: int) -> NVRTCParams: 
        """
        Args:
            desp: 
            arch: 
            stream_int: 
        """
        ...
    def tune_and_cache(self, op_type: int, inp: Tensor, weight: Tensor, output: Tensor, layout_i: int, layout_w: int, layout_o: int, interleave_i: int, interleave_w: int, interleave_o: int, arch: Tuple[int, int], mask: Tensor, mask_argsort: Tensor, indices: Tensor, reverse_mask: bool, mask_filter: int = 0xffffffff, mask_width: int = -1, mask_output: Tensor =  Tensor(), alpha: float = 1.0, beta: float = 0.0, stream_int: int = 0, auto_fp32_accum: bool = True, fp32_accum: bool = False, num_run: int = 5, use_tf32: bool = True) -> Tuple[ConvTuneResult, float]: 
        """
        Args:
            op_type: 
            inp: 
            weight: 
            output: 
            layout_i: 
            layout_w: 
            layout_o: 
            interleave_i: 
            interleave_w: 
            interleave_o: 
            arch: 
            mask: 
            mask_argsort: 
            indices: 
            reverse_mask: 
            mask_filter: 
            mask_width: 
            mask_output: 
            alpha: 
            beta: 
            stream_int: 
            auto_fp32_accum: 
            fp32_accum: 
            num_run: 
            use_tf32: 
        """
        ...
    def get_tuned_algo(self, op_type: int, i_dtype: int, w_dtype: int, o_dtype: int, k: int, c: int, arch: Tuple[int, int], mask_width: int = -1) -> Tuple[Any, bool]: 
        """
        Args:
            op_type: 
            i_dtype: 
            w_dtype: 
            o_dtype: 
            k: 
            c: 
            arch: 
            mask_width: 
        """
        ...
    def run_with_tuned_result(self, profile_res, op_type: int, inp: Tensor, weight: Tensor, output: Tensor, mask: Tensor, mask_argsort: Tensor, mask_output: Tensor, indices: Tensor, reverse_mask: bool, mask_filter: int = 0xffffffff, mask_width: int = -1, alpha: float = 1.0, beta: float = 0.0, stream_int: int = 0, workspace: Tensor =  Tensor(), verbose: bool = False, timer: CUDAKernelTimer =  CUDAKernelTimer(false), force_nvrtc: bool = False, bias: Tensor =  Tensor(), act_alpha: float = 0.0, act_beta: float = 0.0, act_type: Activation =  Activation.None_) -> None: 
        """
        Args:
            profile_res: 
            op_type: 
            inp: 
            weight: 
            output: 
            mask: 
            mask_argsort: 
            mask_output: 
            indices: 
            reverse_mask: 
            mask_filter: 
            mask_width: 
            alpha: 
            beta: 
            stream_int: 
            workspace: 
            verbose: 
            timer: 
            force_nvrtc: 
            bias: 
            act_alpha: 
            act_beta: 
            act_type: 
        """
        ...
    def query_workspace_size(self, desp: ConvAlgoDesp, splitk: int, op_type: int, N: int, C: int, K: int, kv: int) -> int: 
        """
        Args:
            desp: 
            splitk: 
            op_type: 
            N: 
            C: 
            K: 
            kv: 
        """
        ...
