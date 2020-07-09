import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import spconv
from spconv.utils import VoxelGeneratorV2, VoxelGeneratorV3


def waymo_data_gpu(batch_size=1):
    print('gpu with total points available per voxel')
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    points = torch.from_numpy(data['pc']).cuda().float()
    voxel_size = torch.Tensor([0.1, 0.1,
                               0.1]).to(points.dtype).to(points.device)
    coors_range = torch.Tensor([-80, -80, -2, 80, 80,
                                6]).to(points.dtype).to(points.device)

    gen = VoxelGeneratorV3(voxel_size,
                           coors_range,
                           max_points=200000,
                           num_features=points.shape[1],
                           dtype=points.dtype,
                           device=points.device)
    voxels, coors = gen.generate(points)

    times = []
    with torch.no_grad():
        for i in range(200):
            torch.cuda.synchronize()
            t = time.time()
            voxels, coors = gen.generate(points)
            torch.cuda.synchronize()
            times.append(time.time() - t)
    print("voxelization time", np.mean(times[100:]))

    N = coors.shape[0]
    batch_id = torch.zeros([N, 1], dtype=coors.dtype, device=coors.device)
    coors = torch.cat([batch_id, coors], dim=1)
    return voxels, coors, gen.grid_size


def waymo_data_cpu(max_points_per_voxel=1, batch_size=1):
    print('cpu with %d max points per voxel' % max_points_per_voxel)
    gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6],
                           max_points_per_voxel, 150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = data["pc"]
    data = gen.generate(pc)

    times = []
    with torch.no_grad():
        for i in range(200):
            torch.cuda.synchronize()
            t = time.time()
            data = gen.generate(pc)
            torch.cuda.synchronize()
            times.append(time.time() - t)
    print("voxelization time", np.mean(times[100:]))

    voxels = data["voxels"].reshape(-1, 3)
    coors = data["coordinates"]
    N = coors.shape[0]
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size


def get_index(coor, grid_size):
    index = coor[0]
    for c, g in zip(coor[1:], grid_size):
        index = index * g + c
    return index


def main():
    voxels_gpu, coors_gpu, grid_size_gpu = waymo_data_gpu()
    voxels_cpu, coors_cpu, grid_size_cpu = waymo_data_cpu(1)
    waymo_data_cpu(10)
    waymo_data_cpu(40)

    print('...')

    grid_size_gpu = grid_size_gpu[::-1]
    grid_size_cpu = grid_size_cpu[::-1]

    assert len(grid_size_gpu) == len(grid_size_cpu), "mismatch grid size"
    assert grid_size_gpu[0] == grid_size_cpu[0], "mismatch grid size"
    assert grid_size_gpu[1] == grid_size_cpu[1], "mismatch grid size"
    assert grid_size_gpu[2] == grid_size_cpu[2], "mismatch grid size"

    assert coors_gpu.shape[0] == coors_cpu.shape[0], "mismatch coors shape"

    index2voxel = dict()
    for coor, voxel in zip(coors_gpu, voxels_gpu):
        index = get_index(coor, grid_size_gpu).item()
        index2voxel[index] = voxel[:3].cpu()

    for coor, voxel in zip(coors_cpu, voxels_cpu):
        index = get_index(coor, grid_size_cpu).item()
        assert index in index2voxel, "mismatch index: " + str(index)
        assert (index2voxel.pop(index) - voxel[:3]).abs().max() < 0.1, \
                    "voxel diff should be smaller than voxel_size 0.1"

    print('Perfect GPU Voxelization!!!')


if __name__ == "__main__":
    main()
