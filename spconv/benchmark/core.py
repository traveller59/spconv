import requests
import fire 
import pickle 
from io import BytesIO
import numpy as np
from spconv.constants import PACKAGE_ROOT
from spconv.utils import Point2VoxelCPU3d, Point2VoxelGPU3d
from cumm import tensorview as tv

RAW_TEST_DATA_PATH = "https://raw.githubusercontent.com/traveller59/spconv/v2.1.10/test/data/test_spconv.pkl"
RAW_PC_PATH = "https://raw.githubusercontent.com/traveller59/spconv/v2.1.10/test/data/benchmark-pc.npz"

def get_voxel_data():
    editable_test_data_path = PACKAGE_ROOT.parent / "test/data/test_spconv.pkl"
    if editable_test_data_path.exists():
        with editable_test_data_path.open("rb") as f:
            return pickle.load(f)
    ff = BytesIO()
    with requests.get(RAW_TEST_DATA_PATH, stream=True) as req:
        req.raise_for_status()
        for chunk in req.iter_content(chunk_size=8192): 
            ff.write(chunk)
    ff.seek(0) 
    (voxels, coors, spatial_shape) = pickle.load(ff)
    return voxels, coors, spatial_shape

def get_pc_data():
    editable_test_data_path = PACKAGE_ROOT.parent / "test/data/benchmark-pc.npz"
    if editable_test_data_path.exists():
        pc = np.load(str(editable_test_data_path))["pc"]
        return pc 
    ff = BytesIO()
    with requests.get(RAW_PC_PATH, stream=True) as req:
        req.raise_for_status()
        for chunk in req.iter_content(chunk_size=8192): 
            ff.write(chunk)
    ff.seek(0) 
    pc = np.load(ff)["pc"]
    return pc

def get_voxel_data_large():
    pc = get_pc_data()
    gen = Point2VoxelGPU3d([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3,
                           1600000, 1)

    pcs = [pc]
    for i in range(7):
        pc2 = pc.copy()
        pc2[:, 1] += i + 1
        pcs.append(pc2)

    pc = np.concatenate(pcs)
    voxels_tv, indices_tv, _ = gen.point_to_voxel_hash(tv.from_numpy(pc).cuda())
    voxels = voxels_tv.cpu().numpy().reshape(-1, 3)
    coors = indices_tv.cpu().numpy()
    N = coors.shape[0]
    # breakpoint()
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size


if __name__ == "__main__":
    pc = get_pc_data()
    print(pc[:10])