from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class SpconvOps:
    @staticmethod
    def generate_conv_inds_stage1(indices: Tensor, indice_pairs: Tensor, indice_pairs_uniq: Tensor, indice_num_per_loc: Tensor, batch_size: int, output_dims: List[int], input_dims: List[int], ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool = False, stream_int: int = 0) -> None: 
        """
        Args:
            indices: 
            indice_pairs: 
            indice_pairs_uniq: 
            indice_num_per_loc: 
            batch_size: 
            output_dims: 
            input_dims: 
            ksize: 
            stride: 
            padding: 
            dilation: 
            transposed: 
            stream_int: 
        """
        ...
    @staticmethod
    def generate_conv_inds_stage1_5(indice_pairs_uniq: Tensor, ndim: int, uniq_size: int, stream_int: int = 0) -> int: 
        """
        Args:
            indice_pairs_uniq: 
            ndim: 
            uniq_size: 
            stream_int: 
        """
        ...
    @staticmethod
    def generate_conv_inds_stage2(indices: Tensor, hashdata: Tensor, indice_pairs: Tensor, indice_pairs_uniq: Tensor, out_inds: Tensor, num_out_act: int, batch_size: int, output_dims: List[int], input_dims: List[int], ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool = False, stream_int: int = 0) -> int: 
        """
        Args:
            indices: 
            hashdata: 
            indice_pairs: 
            indice_pairs_uniq: 
            out_inds: 
            num_out_act: 
            batch_size: 
            output_dims: 
            input_dims: 
            ksize: 
            stride: 
            padding: 
            dilation: 
            transposed: 
            stream_int: 
        """
        ...
    @staticmethod
    def generate_subm_conv_inds(indices: Tensor, hashdata: Tensor, indice_pairs: Tensor, out_inds: Tensor, indice_num_per_loc: Tensor, batch_size: int, input_dims: List[int], ksize: List[int], dilation: List[int], indice_pair_mask: Tensor =  Tensor(), backward: bool = False, stream_int: int =  0) -> int: 
        """
        Args:
            indices: 
            hashdata: 
            indice_pairs: 
            out_inds: 
            indice_num_per_loc: 
            batch_size: 
            input_dims: 
            ksize: 
            dilation: 
            indice_pair_mask: 
            backward: 
            stream_int: 
        """
        ...
    @staticmethod
    def maxpool_forward(out: Tensor, inp: Tensor, out_inds: Tensor, in_inds: Tensor, stream: int = 0) -> None: 
        """
        Args:
            out: 
            inp: 
            out_inds: 
            in_inds: 
            stream: 
        """
        ...
    @staticmethod
    def maxpool_backward(out: Tensor, inp: Tensor, dout: Tensor, dinp: Tensor, out_inds: Tensor, in_inds: Tensor, stream: int = 0) -> None: 
        """
        Args:
            out: 
            inp: 
            dout: 
            dinp: 
            out_inds: 
            in_inds: 
            stream: 
        """
        ...
    @staticmethod
    def sort_1d_by_key(data: Tensor) -> Tensor: 
        """
        Args:
            data: 
        """
        ...
