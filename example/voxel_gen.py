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
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
import torch

def main_pytorch_voxel_gen():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    gen = PointToVoxel(vsize_xyz=[0.1, 0.1, 0.1],
                       coors_range_xyz=[-80, -80, -6, 80, 80, 6],
                       num_point_features=3,
                       max_num_voxels=5000,
                       max_num_points_per_voxel=5)

    pc = np.random.uniform(-4, 4, size=[1000, 3])
    pc_th = torch.from_numpy(pc)
    voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
    voxels_np = voxels_th.numpy()
    indices_np = indices_th.numpy()
    num_p_in_vx_np = num_p_in_vx_th.numpy()
    print(f"------Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])
    # run voxel gen and FILL MEAN VALUE to voxel remain
    voxels_th, indices_th, num_p_in_vx_th = gen(pc_th, empty_mean=True)
    voxels_np = voxels_th.numpy()
    indices_np = indices_th.numpy()
    num_p_in_vx_np = num_p_in_vx_th.numpy()
    print("------Voxels with mean filled-------")
    print(voxels_np[0])
    voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
    print("------Voxel ids for every point-------")
    print(pc_voxel_id[:10])



def main_pytorch_voxel_gen_cuda():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    pc = np.random.uniform(-2, 8, size=[1000, 3]).astype(np.float32)

    for device in [torch.device("cuda:0"), torch.device("cpu:0")]:
        gen = PointToVoxel(vsize_xyz=[0.25, 0.25, 0.25],
                        coors_range_xyz=[0, 0, 0, 10, 10, 10],
                        num_point_features=3,
                        max_num_voxels=5000,
                        max_num_points_per_voxel=5,
                        device=device)

        pc_th = torch.from_numpy(pc).to(device)
        voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
        voxels_np = voxels_th.cpu().numpy()
        indices_np = indices_th.cpu().numpy()
        num_p_in_vx_np = num_p_in_vx_th.cpu().numpy()
        print(f"------{device} Raw Voxels {voxels_np.shape[0]}-------")
        print(voxels_np[0])
        # run voxel gen and FILL MEAN VALUE to voxel remain
        voxels_tv, indices_tv, num_p_in_vx_tv = gen(pc_th, empty_mean=True)
        voxels_np = voxels_tv.cpu().numpy()
        indices_np = indices_tv.cpu().numpy()
        num_p_in_vx_np = num_p_in_vx_tv.cpu().numpy()
        print(f"------{device} Voxels with mean filled-------")
        print(voxels_np[0])
        voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
        print(f"------{device} Reconstruct Indices From Voxel ids for every point-------")
        indices_th_float = indices_th.float()
        # we gather indices by voxel_id to see correctness of voxel id.
        indices_th_voxel_id = gather_features_by_pc_voxel_id(indices_th_float, pc_voxel_id)
        indices_th_voxel_id_np = indices_th_voxel_id[:10].cpu().numpy()
        print(pc[:10])
        print(indices_th_voxel_id_np[:, ::-1] / 4)

def main_gather_features_by_pc_voxel_id():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    device = torch.device("cuda:0")
    gen = PointToVoxel(vsize_xyz=[0.25, 0.25, 0.25],
                       coors_range_xyz=[-10, -10, -10, 10, 10, 10],
                       num_point_features=3,
                       max_num_voxels=2000,
                       max_num_points_per_voxel=5,
                       device=device)

    pc = np.random.uniform(-8, 8, size=[5000, 3]).astype(np.float32)
    pc_th = torch.from_numpy(pc).to(device)

    voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
    res_features_from_seg = torch.zeros((voxels_th.shape[0], 128), dtype=torch.float32, device=device)
    
    pc_features = gather_features_by_pc_voxel_id(res_features_from_seg, pc_voxel_id)
    print(pc.shape, pc_features.shape)

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

def main_cuda():
    np.random.seed(50051)
    from spconv.utils import Point2VoxelGPU3d

    # voxel gen source code: spconv/csrc/sparse/pointops.py
    gen = Point2VoxelGPU3d(vsize_xyz=[0.1, 0.1, 0.1],
                           coors_range_xyz=[-80, -80, -2, 80, 80, 6],
                           num_point_features=3,
                           max_num_voxels=5000,
                           max_num_points_per_voxel=5)

    pc = np.random.uniform(-10, 10, size=[100000, 3]).astype(np.float32)
    pc_tv = tv.from_numpy(pc).cuda()
    # generate voxels, note that voxels_tv reference to a persistent buffer in generator,
    # so we can't run it in multi-thread.
    voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_tv)
    voxels_np = voxels_tv.cpu().numpy()
    indices_np = indices_tv.cpu().numpy()
    num_p_in_vx_np = num_p_in_vx_tv.cpu().numpy()
    print(f"------CUDA Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])


if __name__ == "__main__":
    main()
    main_point_with_features()
    main_pytorch_voxel_gen()
    if torch.cuda.is_available():
        main_cuda()
        main_pytorch_voxel_gen_cuda()
        main_gather_features_by_pc_voxel_id()
