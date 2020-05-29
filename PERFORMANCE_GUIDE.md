## Performance Guide

### 1. Regular sparse conv is very slow

Regular sparse convolution will greatly increase the number of active points. for 3x3x3 3D convolution, we can get at most 27x active points, which means next convolution will perform 27x slower!

This problem can be solved by using submanifold convolution (SubMConv3d). This kind of sparse convolution doesn't generate new active points.

**NEVER** use SparseConv3d except downsample data, **NEVER** use SparseConv3dTranspose, use SparseInverseConv3d instead.

### 2. Large Spatial Shape cost too much GPU memory

Our implementation use dense map to generate indices in GPU for sparse convolution, which means if your spatial shape is ```[batchSize=4, 1600, 1600, 40]```, it will cost ~2GB GPU memory.

To solve this problem, you can use CPU algorithm (hash map) for first layer that has large shape, then convert generated indices to GPU and use GPU algorithm for downsampled data.

Another way is use cuda hash. Unfortunately this library isn't stable enough, it should only be used when the spatial shape is very large.

### 3. Stacked submanifold convolution can share same indice data

When you using stacked subm convolution, there is no need to generate indice data again, but this can't be done automatically. you need to specify a unique key ```indice_key="c0"``` and use it for all stacked subm convolution.

### 4. Different convolution algorithm may lead to different performance

There are three kind of algorithm: ```Native```, ```Batch```, ```BatchGemmGather```. 

* ```Native```: should be used for all submanifold convolutions. should be used when there are too much active points.

* ```Batch```: **cost more GPU memory** should be used when number of active points is small.

* ```BatchGemmGather```: **cost more GPU memory** can be used for regular convolution.