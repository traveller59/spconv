<!--
 Copyright 2021 Yan Yan
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Usage

## Concept

* Sparse Conv Tensor: like hybird [torch.sparse_coo_tensor](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) but only have two difference: 1. SparseConvTensor only have one dense dim, 2. indice of SparseConvTensor is transposed. see torch doc for more details.

* Sparse Convolution: equivalent to perform dense convolution when you convert SparseConvTensor to dense. Sparse Convolution only run calculation on valid data.

* Submanifold Convolution (SubMConv): like Sparse Convolution but indices keeps same. imagine that you copy same spatial structure to output, then iterate them, get input coordinates by conv rule, finally apply convolution **ONLY** in these output coordinates.



## SparseConvTensor

* features: ```[N, num_channels]``` tensor.

* indices: ```[N, (batch_idx + x + y + z)]``` coordinate tensor with batch axis. note that the coordinates xyz order MUST match spatial shape and conv params such as kernel size

```Python
features = # your features with shape [N, num_channels]
indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
batch_size = # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
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

voxel generator in spconv generate indices in **ZYX** order, the params format are **XYZ**.

voxel generator in spconv takes a ```tv.Tensor``` return a ```tv.Tensor```, this tensor reference to a **permanent** storage in generator.


```Python
from spconv.utils import Point2VoxelCPU3d
# this generator generate ZYX indices.
gen = Point2VoxelCPU3d(
    vsize_xyz=[0.1, 0.1, 0.1], 
    coors_range_xyz=[-80, -80, -2, 80, 80, 6], 
    num_point_features=3, 
    max_num_voxels=5000, 
    max_num_points_per_voxel=5)
pc = np.random.uniform(-10, 10, size=[1000, 3])
pc_tv = tv.from_numpy(pc)
voxels, coords, num_points_per_voxel = gen.generate(pc_tv)

# get numpy
voxels_np = voxels.numpy_view() # no copy, but become invalid if generator is destroyed.
voxels_np = voxels.numpy() # will perform copy
```
