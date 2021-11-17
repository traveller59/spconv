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

import numpy as np

from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d
from spconv.pytorch.utils import PointToVoxel
import torch


def main():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    gen = Point2VoxelCPU3d(vsize_xyz=[0.1, 0.1, 0.1],
                           coors_range_xyz=[-80, -80, -2, 80, 80, 6],
                           num_point_features=3,
                           max_num_voxels=5000,
                           max_num_points_per_voxel=5)

    pc = np.random.uniform(-10, 10, size=[1000, 3])
    pc_tv = tv.from_numpy(pc)
    # generate voxels, note that voxels_tv reference to a persistent buffer in generator,
    # so we can't run it in multi-thread.
    voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel(pc_tv)
    voxels_np = voxels_tv.numpy_view()
    indices_np = indices_tv.numpy_view()
    num_p_in_vx_np = num_p_in_vx_tv.numpy_view()
    print(f"------Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])
    # run voxel gen and FILL MEAN VALUE to voxel remain
    voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_empty_mean(
        pc_tv)
    voxels_np = voxels_tv.numpy_view()
    indices_np = indices_tv.numpy_view()
    num_p_in_vx_np = num_p_in_vx_tv.numpy_view()
    print("------Voxels with mean filled-------")
    print(voxels_np[0])


def main_point_with_features():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    gen = Point2VoxelCPU3d(
        vsize_xyz=[0.1, 0.1, 0.1],
        coors_range_xyz=[-80, -80, -2, 80, 80, 6],
        num_point_features=
        4,  # here num_point_features must equal to pc.shape[1]
        max_num_voxels=5000,
        max_num_points_per_voxel=5)

    pc = np.random.uniform(-10, 10, size=[1000, 3])
    other_pc_feature = np.random.uniform(-1, 1, size=[1000, 1])
    pc_with_feature = np.concatenate([pc, other_pc_feature], axis=1)
    pc_tv = tv.from_numpy(pc_with_feature)
    # generate voxels, note that voxels_tv reference to a persistent buffer in generator,
    # so we can't run it in multi-thread.
    voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel(pc_tv)
    voxels_np = voxels_tv.numpy_view()
    indices_np = indices_tv.numpy_view()
    num_p_in_vx_np = num_p_in_vx_tv.numpy_view()
    print(f"------Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])
    # run voxel gen and FILL MEAN VALUE to voxel remain
    voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_empty_mean(
        pc_tv)
    voxels_np = voxels_tv.numpy_view()
    indices_np = indices_tv.numpy_view()
    num_p_in_vx_np = num_p_in_vx_tv.numpy_view()
    print("------Voxels with mean filled-------")
    print(voxels_np[0])


def main_pytorch_voxel_gen():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    gen = PointToVoxel(vsize_xyz=[0.1, 0.1, 0.1],
                       coors_range_xyz=[-80, -80, -2, 80, 80, 6],
                       num_point_features=3,
                       max_num_voxels=5000,
                       max_num_points_per_voxel=5)

    pc = np.random.uniform(-10, 10, size=[1000, 3])
    pc_th = torch.from_numpy(pc)
    voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
    voxels_np = voxels_th.numpy()
    indices_np = indices_th.numpy()
    num_p_in_vx_np = num_p_in_vx_th.numpy()
    print(f"------Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])
    # run voxel gen and FILL MEAN VALUE to voxel remain
    voxels_tv, indices_tv, num_p_in_vx_tv = gen(pc_th, empty_mean=True)
    voxels_np = voxels_tv.numpy()
    indices_np = indices_tv.numpy()
    num_p_in_vx_np = num_p_in_vx_tv.numpy()
    print("------Voxels with mean filled-------")
    print(voxels_np[0])


def main_pytorch_voxel_gen_cuda():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    device = torch.device("cuda:0")
    gen = PointToVoxel(vsize_xyz=[0.1, 0.1, 0.1],
                       coors_range_xyz=[-80, -80, -2, 80, 80, 6],
                       num_point_features=3,
                       max_num_voxels=5000,
                       max_num_points_per_voxel=5,
                       device=device)

    pc = np.random.uniform(-10, 10, size=[1000, 3]).astype(np.float32)
    pc_th = torch.from_numpy(pc).to(device)
    voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
    voxels_np = voxels_th.cpu().numpy()
    indices_np = indices_th.cpu().numpy()
    num_p_in_vx_np = num_p_in_vx_th.cpu().numpy()
    print(f"------Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])
    # run voxel gen and FILL MEAN VALUE to voxel remain
    voxels_tv, indices_tv, num_p_in_vx_tv = gen(pc_th, empty_mean=True)
    voxels_np = voxels_tv.cpu().numpy()
    indices_np = indices_tv.cpu().numpy()
    num_p_in_vx_np = num_p_in_vx_tv.cpu().numpy()
    print("------Voxels with mean filled-------")
    print(voxels_np[0])


if __name__ == "__main__":
    main()
    main_point_with_features()
    main_pytorch_voxel_gen()
    if torch.cuda.is_available():
        main_pytorch_voxel_gen_cuda()
