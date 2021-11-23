from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class Point2Voxel:
    hashdata: Tensor
    point_indice_data: Tensor
    voxels: Tensor
    indices: Tensor
    num_per_voxel: Tensor
    @property
    def grid_size(self) -> List[int]: ...
    def __init__(self, vsize_xyz: List[float], coors_range_xyz: List[float], num_point_features: int, max_num_voxels: int, max_num_points_per_voxel: int) -> None: 
        """
        Args:
            vsize_xyz: 
            coors_range_xyz: 
            num_point_features: 
            max_num_voxels: 
            max_num_points_per_voxel: 
        """
        ...
    def point_to_voxel_hash(self, points: Tensor, clear_voxels: bool = True, empty_mean: bool = False, stream_int: int = 0) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            clear_voxels: 
            empty_mean: 
            stream_int: 
        """
        ...
    @staticmethod
    def point_to_voxel_hash_static(points: Tensor, voxels: Tensor, indices: Tensor, num_per_voxel: Tensor, hashdata: Tensor, point_indice_data: Tensor, points_voxel_id: Tensor, vsize: List[float], grid_size: List[int], grid_stride: List[int], coors_range: List[float], clear_voxels: bool = True, empty_mean: bool = False, stream_int: int = 0) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            voxels: 
            indices: 
            num_per_voxel: 
            hashdata: 
            point_indice_data: 
            points_voxel_id: 
            vsize: 
            grid_size: 
            grid_stride: 
            coors_range: 
            clear_voxels: 
            empty_mean: 
            stream_int: 
        """
        ...
