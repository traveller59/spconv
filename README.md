# SpConv: PyTorch Spatially Sparse Convolution Library

[![Build Status](https://github.com/traveller59/spconv/workflows/build/badge.svg)](https://github.com/traveller59/spconv/actions?query=workflow%3Abuild)

This is a spatially sparse convolution library like [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) but faster and easy to read. This library provide sparse convolution/transposed, submanifold convolution, inverse convolution and sparse maxpool.


2020-5-2, we add ConcatTable, JoinTable, AddTable, and Identity function to build ResNet and Unet in this version of spconv.


## Docker:

```docker pull scrin/dev-spconv```, contains python 3.8, cuda 10.1, fish shell, newest pytorch and tensorflow.

## Install on Ubuntu 16.04/18.04

* if you are using pytorch 1.4+ and encounter "nvcc fatal: unknown -Wall", you need to go to torch package dir and remove flags contains "-Wall" in INTERFACE_COMPILE_OPTIONS in Caffe2Targets.cmake. This problem can't be fixed in this project (to avoid this, I need to remove all torch dependency in cuda sources and drop half support).

0. Use ```git clone xxx.git --recursive``` to clone this repo.

1. Install boost headers to your system include path, you can use either ```sudo apt-get install libboost-all-dev``` or download compressed files from boost official website and copy headers to include path.

2. Download cmake >= 3.13.2, then add cmake executables to PATH.

3. Ensure you have installed pytorch 1.0+ in your environment, run ```python setup.py bdist_wheel``` (don't use ```python setup.py install```).

4. Run ```cd ./dist```, use pip to install generated whl file.

## Install on Windows 10 (Not supported for now)

## Compare with SparseConvNet

### Features

* SparseConvNet's Sparse Convolution don't support padding and dilation, spconv support this.

* spconv only contains sparse convolutions, the batchnorm and activations can directly use layers from torch.nn, SparseConvNet contains lots of their own implementation of layers such as batchnorm and activations.

### Speed

* spconv is faster than SparseConvNet due to gpu indice generation and gather-gemm-scatter algorithm. SparseConvNet use hand-written gemm which is slow.

## Usage

### SparseConvTensor

```Python
features = # your features with shape [N, numPlanes]
indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
batch_size = # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
print(x.sparity) # helper function to check sparity. 
```

### Sparse Convolution

```Python
import spconv
from torch import nn
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3), # just like nn.Conv3d but don't support group and all([d > 1, s > 1])
            nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # when use submanifold convolutions, their indices can be shared to save indices generation time.
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(64, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)# .dense()
```

### Inverse Convolution

Inverse sparse convolution means "inv" of sparse convolution. the output of inverse convolution contains same indices as input of sparse convolution.

Inverse convolution usually used in semantic segmentation.

```Python
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, 2, indice_key="cp0"),
            spconv.SparseInverseConv3d(64, 32, 3, indice_key="cp0"), # need provide kernel size to create weight
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)
```

### Utility functions

* convert point cloud to voxel

```Python

voxel_generator = spconv.utils.VoxelGenerator(
    voxel_size=[0.1, 0.1, 0.1], 
    point_cloud_range=[-50, -50, -3, 50, 50, 1],
    max_num_points=30,
    max_voxels=40000
)

points = # [N, 3+] tensor.
voxels, coords, num_points_per_voxel = voxel_generator.generate(points)
```

## Implementation Details

This implementation use gather-gemm-scatter framework to do sparse convolution.

## Projects using spconv:

* [second.pytorch](https://github.com/traveller59/second.pytorch): Point Cloud Object Detection in KITTI Dataset.

## Authors

* **Yan Yan** - *Initial work* - [traveller59](https://github.com/traveller59)

* **Bo Li** - *gpu indice generation idea, owner of patent of the sparse conv gpu indice generation algorithm (don't include subm)* - [prclibo](https://github.com/prclibo)

## Third party libraries

* [CUDPP](https://github.com/cudpp/cudpp): A cuda library. contains a cuda hash implementation.

* [robin-map](https://github.com/Tessil/robin-map): A fast c++ hash library. almost 2x faster than std::unordered_map in this project.

* [pybind11](https://github.com/pybind/pybind11): A head-only python c++ binding library.

* [prettyprint](https://github.com/louisdx/cxx-prettyprint): A head-only library for container print.

## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

The [CUDPP](https://github.com/cudpp/cudpp) hash code is licensed under BSD License.

The [robin-map](https://github.com/Tessil/robin-map) code is licensed under MIT license.
