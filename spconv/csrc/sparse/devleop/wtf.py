#!/home/yy/library/anaconda3/bin/python
import sys
from pathlib import Path 
import ctypes
# _cudart = ctypes.CDLL('libcudart.so')

print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))



from spconv import tensorview as tv 

from spconv.sparse import build
import numpy as np 
from pathlib import Path 
from spconv.spconv_ops_cc.sparse.all.ops import Point2Voxel
from spconv.spconv_ops_cc.sparse.all import SpconvOps

import time 

def main():
    data = np.load("/home/yy/OneDrive/dev/spconv/test/data/benchmark-pc.npz")["pc"].astype(np.float32)
    print(data.shape, data.dtype)
    p2v = Point2Voxel([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3, 150000, 1)
    gs = p2v.grid_size # zyx
    print(gs)
    # return
    data_tv = tv.from_numpy(data).cuda()
    for i in range(6):
        t = time.time()

        voxels, indices, num_per_voxel = p2v.point_to_voxel_hash(data_tv)   
        
        print(time.time() - t)
    voxels, indices, num_per_voxel = p2v.point_to_voxel_hash(data_tv)   
    print(voxels.shape, gs)
    gs_xyz = gs
    indices_np = indices.cpu().numpy()
    # indices_offset = indices_np[:, 0] * gs_xyz[1] * gs_xyz[2] + indices_np[:, 1] * gs_xyz[2] + indices_np[:, 2]
    # uq = np.unique(indices_offset)
    # print(uq.shape, indices_offset.shape, gs_xyz)
    # return 
    ksize = [3] * 3 
    kv = int(np.prod(ksize))
    indices_with_bs = np.zeros((indices_np.shape[0], 4), dtype=np.int32)
    indices_with_bs[:, 1:] = indices_np
    print(indices_with_bs.mean(), indices_with_bs.max(), indices_with_bs.min())

    indices = tv.from_numpy(indices_with_bs).cuda()
    out_indices = tv.zeros([indices.dim(0) * kv, 4], tv.int32, 0)
    indice_num_per_loc = tv.zeros([kv], tv.int32, 0)


    points = voxels.view([-1, 3])
    hashdata = tv.zeros([points.dim(0) * kv * 2], tv.custom64, 0)
    hashdata_subm = tv.zeros([points.dim(0) * 2], tv.custom64, 0)

    indice_pairs = tv.full([2, kv, indices.dim(0)], -1, tv.int32, 0)
    indice_pairs_uniq = tv.zeros([indice_pairs.size // 2 + 1], tv.int32, 0)

    # for i in range(10):
    #     indice_pairs.fill_int_(-1)
    #     np.random.shuffle(indices_with_bs)
    #     indices = tv.from_numpy(indices_with_bs).cuda()

    #     indice_num_per_loc.zero_()
    #     out_act = SpconvOps.generate_conv_inds(indices, hashdata, indice_pairs,
    #         indice_pairs_uniq, out_indices, indice_num_per_loc, 
    #         1, gs, gs, [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1])
    #     indice_num_per_loc.zero_()
    #     out_act = SpconvOps.generate_subm_conv_inds(indices, hashdata_subm, indice_pairs,
    #         out_indices, indice_num_per_loc, 
    #         1, gs, ksize, [1, 1, 1])
    #     indice_num_per_loc_cpu = indice_num_per_loc.cpu().numpy()
    #     indice_pairs_cpu = indice_pairs.cpu().numpy()
    #     indice_pairs_cpu_flat = indice_pairs_cpu.reshape(-1)
    #     uq, count = np.unique(indice_pairs_cpu_flat, return_counts=True)
    #     print(out_act, indice_pairs_cpu.shape, indice_pairs_cpu.mean(), indice_num_per_loc_cpu.tolist())
    #     print(indice_pairs_cpu[:, 13, :2])
    #     print(uq, count)

if __name__ == "__main__":

    main()