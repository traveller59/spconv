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

## Short API description

```Python
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from torch import nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable
```

| Layer APIs                         | Common Usage             |            Dense Version    |Note    |
|----------------------------------- |:------------------------:|----------------------------:|----------------------------:| 
| ```spconv.SparseConv3d```          | Downsample               | ```nn.Conv3d```             | Use ```indice_key``` to save data for inverse |
| ```spconv.SubMConv3d```            | Convolution              | N/A                         | Use ```indice_key``` to save data for reuse |
| ```spconv.SparseInverseConv3d```   | Upsample                 |  N/A                        | Use pre-saved ```indice_key``` to upsample |
| ```spconv.SparseConvTranspose3d``` | Upsample (don't use this)|  ```nn.ConvTranspose3d```   | VERY SLOW and CAN'T RECOVER ORIGIN POINT CLOUD |
| ```spconv.SparseMaxPool3d```       | Downsample               |  ```nn.MaxPool3d```         | Use ```indice_key``` to save data for inverse |
| ```spconv.SparseSequential```       | Container               |  ```nn.Sequential```         | support layers above and ```nn.ReLU, nn.BatchNorm, ...```|


| Functional APIs                    | Usage                    |
|----------------------------------- |:------------------------:|
| ```Fsp.sparse_add```               | Add sparse tensors with same shape and different indices    |

| Input APIs                         | Usage                    |
|----------------------------------- |:------------------------:|
| ```PointToVoxel```                 | point cloud to voxels    |

| Misc APIs                         | Usage                    |
|----------------------------------- |:------------------------:|
| ```HashTable```                   | hash table, one-slot     |

| Layer APIs                         | [torchsparse](https://github.com/mit-han-lab/torchsparse)             |    [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)             |   
|----------------------------------- |:------------------------:|:------------------------:|
| ```spconv.SparseConv3d```          | ```Conv3d(stride!=1, transpose=False)```               |```MinkowskiConvolution(stride!=1)```| 
| ```spconv.SubMConv3d```            | ```Conv3d(stride=1, transpose=False)```              | ```MinkowskiConvolution(stride=1)```| 
| ```spconv.SparseInverseConv3d```   | ```Conv3d(stride!=1, transpose=True)```                 |```MinkowskiConvolutionTranspose```| 
| ```spconv.SparseConvTranspose3d``` | N/A                |```MinkowskiConvolutionTranspose```| 
| ```spconv.SparseMaxPool3d```       | N/A               | ```MinkowskiMaxPooling```| 


## Concept

* Sparse Conv Tensor: like hybird [torch.sparse_coo_tensor](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) but only have two difference: 1. SparseConvTensor only have one dense dim, 2. indice of SparseConvTensor is transposed. see torch doc for more details.

* Sparse Convolution: equivalent to perform dense convolution when you convert SparseConvTensor to dense. Sparse Convolution only run calculation on valid data.

* Submanifold Convolution (SubMConv): like Sparse Convolution but indices keeps same. imagine that you copy same spatial structure to output, then iterate them, get input coordinates by conv rule, finally apply convolution **ONLY** in these output coordinates.



## SparseConvTensor

* features: ```[N, num_channels]``` tensor.

* indices: ```[N, (batch_idx + x + y + z)]``` coordinate tensor with batch axis. note that the coordinates xyz order MUST match spatial shape and conv params such as kernel size

```Python
import spconv.pytorch as spconv
features = # your features with shape [N, num_channels]
indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
batch_size = # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
```


### Sparse Convolution

```Python
import spconv.pytorch as spconv
from torch import nn
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3), # just like nn.Conv3d but don't support group
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

**WARNING** ```SparseInverseConv``` isn't equivalent to ```SparseConvTranspose```. SparseConvTranspose is equivalent to ```ConvTranspose``` in pytorch, but SparseInverseConv isn't.

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

### Sparse Add

In sematic segmentation network, we may use conv1x3, 3x1 and 3x3 in a block, but it's impossible to sum result from these layers because regular add requires same indices.

spconv >= 2.1.17 provide a operation to add sparse tensors with different indices (shape must same), but with limits:

```Python
from spconv.pytorch import functional as Fsp
res_1x3 = conv1x3(x)
res_3x1 = conv3x1(x)
# WRONG
# because we can't "inverse" this operation
wrong_usage_cant_inverse = Fsp.sparse_add(res_1x3, res_3x1)

# CORRECT
# res_3x3 already contains all indices of res_1x3 and res_3x1, 
# so output spatial structure isn't changed, we can "inverse" back.
res_3x3 = conv3x3(x)
correct = Fsp.sparse_add(res_1x3, res_3x1, res_3x3)
```

If you use a network without ```SparseInverseConv```, limits above aren't exists, the only drawback of ```sparse_add``` is that it run slower than simple aligned add.

### Fast Mixed Percision Training

see example/mnist_sparse. we support ```torch.cuda.amp```.

### Utility functions

* convert point cloud to voxel

voxel generator in spconv generate indices in **ZYX** order, the params format are **XYZ**.

generated indices don't include batch axis, you need to add it by yourself.

see examples/voxel_gen.py for examples.

```Python
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
# this generator generate ZYX indices.
gen = PointToVoxel(
    vsize_xyz=[0.1, 0.1, 0.1], 
    coors_range_xyz=[-80, -80, -2, 80, 80, 6], 
    num_point_features=3, 
    max_num_voxels=5000, 
    max_num_points_per_voxel=5)
pc = np.random.uniform(-10, 10, size=[1000, 3])
pc_th = torch.from_numpy(pc)
voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)
```

If you want to get label for every point of your pc, you need to use another function to get pc_voxel_id and gather features from sematic segmentation result:
```Python
voxels, coords, num_points_per_voxel, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
seg_features = YourSegNet(...)
# if voxel id is invalid (point out of range, or no space left in a voxel)
# features will be zero.
point_features = gather_features_by_pc_voxel_id(seg_features, pc_voxel_id)
```