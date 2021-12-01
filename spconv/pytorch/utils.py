# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union
import torch
from cumm import tensorview as tv

from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.pytorch.cppcore import torch_tensor_to_tv, get_current_stream


class PointToVoxel(object):
    """WARNING: you MUST construct PointToVoxel AFTER set device.
    """
    def __init__(self,
                 vsize_xyz: List[float],
                 coors_range_xyz: List[float],
                 num_point_features: int,
                 max_num_voxels: int,
                 max_num_points_per_voxel: int,
                 device: torch.device = torch.device("cpu:0")):
        self.ndim = len(vsize_xyz)

        self.device = device
        vsize, grid_size, grid_stride, coors_range = SpconvOps.calc_point2voxel_meta_data(
            vsize_xyz, coors_range_xyz)
        self.num_point_features = num_point_features
        self.max_num_voxels = max_num_voxels
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.vsize = vsize
        self.grid_size = grid_size
        self.grid_stride = grid_stride
        self.coors_range = coors_range

        self.voxels = torch.zeros(
            [max_num_voxels, max_num_points_per_voxel, num_point_features],
            dtype=torch.float32,
            device=device)
        self.indices = torch.zeros([max_num_voxels, self.ndim],
                                   dtype=torch.int32,
                                   device=device)
        self.num_per_voxel = torch.zeros([max_num_voxels],
                                         dtype=torch.int32,
                                         device=device)
        if device.type == "cpu":
            self.hashdata = torch.full(grid_size,
                                       -1,
                                       dtype=torch.int32,
                                       device=device)
            self.point_indice_data = torch.Tensor()
        else:
            self.hashdata = torch.empty([1, 2],
                                        dtype=torch.int64,
                                        device=device)
            self.point_indice_data = torch.empty([1],
                                                 dtype=torch.int64,
                                                 device=device)

    def __call__(self,
                 pc: torch.Tensor,
                 clear_voxels: bool = True,
                 empty_mean: bool = False):
        """generate voxels/indices/num_point_per_voxel/pc_voxel_ids from 
        point cloud.
        This function don't return pc_voxel_id for backward compatility.
        pc_voxel_id will be added in spconv 2.2.
        Args:
            pc: [N, 3+] point cloud.
            clear_voxels: if True, call zero on voxels
            empty_mean: if True, full empty location of voxels with mean.
        Returns:
            voxels: voxels
            indices: quantized coords
            num_per_voxel: number of points in a voxel
        """

        res = self.generate_voxel_with_id(pc, clear_voxels, empty_mean)
        return res[0], res[1], res[2]

    def generate_voxel_with_id(self,
                 pc: torch.Tensor,
                 clear_voxels: bool = True,
                 empty_mean: bool = False):
        """generate voxels/indices/num_point_per_voxel/pc_voxel_ids from 
        point cloud.
        Args:
            pc: [N, 3+] point cloud.
            clear_voxels: if True, call zero on voxels
            empty_mean: if True, full empty location of voxels with mean.
        Returns:
            voxels: voxels
            indices: quantized coords
            num_per_voxel: number of points in a voxel
            pc_voxel_id: voxel id for every point. if not exists, -1.
        """
        assert pc.device.type == self.device.type, "your pc device is wrong"
        expected_hash_data_num = pc.shape[0] * 2
        with torch.no_grad():
            pc_voxel_id = torch.empty([pc.shape[0]],
                                    dtype=torch.int64,
                                    device=self.device)
            pc_voxel_id_tv = torch_tensor_to_tv(pc_voxel_id)

            if self.device.type != "cpu":
                hashdata = torch.empty([expected_hash_data_num, 2],
                                            dtype=torch.int64,
                                            device=pc.device)

                point_indice_data = torch.empty([pc.shape[0]],
                                            dtype=torch.int64,
                                            device=pc.device)

                pc_tv = torch_tensor_to_tv(pc)
                stream = get_current_stream()
                voxels_tv = torch_tensor_to_tv(self.voxels)
                indices_tv = torch_tensor_to_tv(self.indices)
                num_per_voxel_tv = torch_tensor_to_tv(self.num_per_voxel)
                hashdata_tv = torch_tensor_to_tv(
                    hashdata,
                    dtype=tv.custom128,
                    shape=[hashdata.shape[0]])
                point_indice_data_tv = torch_tensor_to_tv(point_indice_data)
                with torch.cuda.device(pc.device):
                    res = SpconvOps.point2voxel_cuda(
                        pc_tv, voxels_tv, indices_tv, num_per_voxel_tv,
                        hashdata_tv, point_indice_data_tv, pc_voxel_id_tv, self.vsize,
                        self.grid_size, self.grid_stride, self.coors_range,
                        empty_mean, clear_voxels, stream)
                num_voxels = res[0].shape[0]
            else:
                pc_tv = torch_tensor_to_tv(pc)
                voxels_tv = torch_tensor_to_tv(self.voxels)
                indices_tv = torch_tensor_to_tv(self.indices)
                num_per_voxel_tv = torch_tensor_to_tv(self.num_per_voxel)
                hashdata_tv = torch_tensor_to_tv(self.hashdata, dtype=tv.int32)
                res = SpconvOps.point2voxel_cpu(pc_tv, voxels_tv, indices_tv,
                                                num_per_voxel_tv, hashdata_tv,
                                                pc_voxel_id_tv,
                                                self.vsize, self.grid_size,
                                                self.grid_stride,
                                                self.coors_range, empty_mean,
                                                clear_voxels)
                num_voxels = res[0].shape[0]

            return (self.voxels[:num_voxels].clone(), self.indices[:num_voxels].clone(),
                    self.num_per_voxel[:num_voxels].clone(), pc_voxel_id)


def gather_features_by_pc_voxel_id(seg_res_features: torch.Tensor, pc_voxel_id: torch.Tensor, invalid_value: Union[int, float] = 0):
    """This function is used to gather segmentation result to match origin pc.
    """
    if seg_res_features.device != pc_voxel_id.device:
        pc_voxel_id = pc_voxel_id.to(seg_res_features.device)
    res_feature_shape = (pc_voxel_id.shape[0], *seg_res_features.shape[1:])
    if invalid_value == 0:
        res = torch.zeros(res_feature_shape, dtype=seg_res_features.dtype, device=seg_res_features.device)
    else:
        res = torch.full(res_feature_shape, invalid_value, dtype=seg_res_features.dtype, device=seg_res_features.device)
    pc_voxel_id_valid = pc_voxel_id != -1
    pc_voxel_id_valid_ids = torch.nonzero(pc_voxel_id_valid).view(-1)
    seg_res_features_valid = seg_res_features[pc_voxel_id[pc_voxel_id_valid_ids]]
    res[pc_voxel_id_valid_ids] = seg_res_features_valid
    return res 

