from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class Point2VoxelCPU:
    densehashdata: Tensor
    voxels: Tensor
    indices: Tensor
    num_per_voxel: Tensor
    @property
    def grid_size(self) -> List[int]: ...
    @staticmethod
    def calc_meta_data(vsize_xyz: List[float], coors_range_xyz: List[float]) -> Tuple[List[float], List[int], List[int], List[float]]: 
        """
        Args:
            vsize_xyz: 
            coors_range_xyz: 
        """
        ...
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
    @staticmethod
    def point_to_voxel_static(points: Tensor, voxels: Tensor, indices: Tensor, num_per_voxel: Tensor, densehashdata: Tensor, points_voxel_id: Tensor, vsize: List[float], grid_size: List[int], grid_stride: List[int], coors_range: List[float], clear_voxels: bool = True) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            voxels: 
            indices: 
            num_per_voxel: 
            densehashdata: 
            points_voxel_id: 
            vsize: 
            grid_size: 
            grid_stride: 
            coors_range: 
            clear_voxels: 
        """
        ...
    @staticmethod
    def point_to_voxel_empty_mean_static(points: Tensor, voxels: Tensor, indices: Tensor, num_per_voxel: Tensor, densehashdata: Tensor, points_voxel_id: Tensor, vsize: List[float], grid_size: List[int], grid_stride: List[int], coors_range: List[float], clear_voxels: bool = True) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            voxels: 
            indices: 
            num_per_voxel: 
            densehashdata: 
            points_voxel_id: 
            vsize: 
            grid_size: 
            grid_stride: 
            coors_range: 
            clear_voxels: 
        """
        ...
    def point_to_voxel(self, points: Tensor, clear_voxels: bool = True) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            clear_voxels: 
        """
        ...
    def point_to_voxel_empty_mean(self, points: Tensor, clear_voxels: bool = True) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            clear_voxels: 
        """
        ...
