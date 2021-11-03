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

# SpConv: PyTorch Spatially Sparse Convolution Library

[![Build Status](https://github.com/traveller59/spconv/workflows/build/badge.svg)](https://github.com/traveller59/spconv/actions?query=workflow%3Abuild)

[Spconv 1.x code](https://github.com/traveller59/spconv/tree/v1.2.1). We won't provide any support for spconv 1.x since it's deprecated. use spconv 2.x if possible.

## Breaking changes in Spconv 2.x

* ```spconv.xxx``` move to ```spconv.pytorch.xxx```, change all ```import spconv``` to ```import spconv.pytorch as spconv``` and ```from spconv.xxx import``` to ```from spconv.pytorch.xxx import```.
* ```use_hash``` in Sparse Convolution is removed, we only use hash table in 2.x.
* ```x.features = F.relu(x)``` now raise error. use ```x = x.replace_feature(F.relu(x.features))``` instead.
* weight layout has been changed to RSKC (native algorithm) or KRSC (implicit gemm), no longer RSCK (spconv 1.x). RS is kernel size, C is input channel, K is output channel.
* all util ops are removed (pillar scatter/nms/...)
* VoxelGenerator has been replaced by Point2VoxelGPU[1-4]d/Point2VoxelCPU[1-4]d.
* spconv 2.x don't support CPU for now

* test spconv 1.x model in spconv 2.x: set environment variable before run program. Linux: ```export SPCONV_FILTER_HWIO="1"```, Windows powershell: ```$Env:SPCONV_FILTER_HWIO = "1"```

## Upcoming release Spconv 2.1.0 (Delay to 11.7.2021, sorry):

**Status**: CPU build is ready. implicit gemm is ready. working on implicit-gemm-style indice generation for standard conv/pool, and implicit-gemm-style maxpool op.

* implicit gemm algorithm, greatly faster than native algorithm when using float16 (tested in RTX 3080 Laptop).
* simple CPU support and CPU-only build
* add pytorch cpu/cuda voxel generator
* fix a bug of mixed precision training.

## News in Spconv 2.0.0

* training/inference speed is increased (+50~80% for float32)
* support int8/tensor core
* doesn't depend on pytorch binary. 
* since spconv 2.x doesn't depend on pytorch binary (never in future), it's impossible to support torch.jit/libtorch inference.

Spconv 2.1.0 vs 1.x speed:

|                | 1080Ti Spconv 1.x F32 | 1080Ti Spconv 2.0 F32 | 3080M* Spconv 2.1 F16  |
| -------------- |:---------------------:| ---------------------:| ----------:|
| 27x128x128 Fwd | 11ms                  | 5.4ms                 | 1.4ms      |

\* 3080M (Laptop) ~= 3070 Desktop

## Usage

Firstly you need to use ```import spconv.pytorch as spconv``` in spconv 2.x.

Then see docs/USAGE.md.


## Install

You need to install python >= 3.7 first to use spconv 2.x.

You need to install CUDA toolkit first before using prebuilt binaries or build from source.

You need at least CUDA 10.2 to build and run spconv 2.x. We won't offer any support for CUDA < 10.2.

### Prebuilt

We offer python 3.7-3.10 and 11.1/11.4 prebuilt binaries for linux (manylinux) and windows 10/11.

CUDA 10.2 support will be added in version 2.0.2.

We will offer prebuilts for CUDA versions supported by latest pytorch release. For example, pytorch 1.9 support cuda 10.2 and 11.1, so we support them too.

For Linux users, you need to install pip >= 20.3 first to install prebuilt.

```pip install spconv-cu111``` for CUDA 11.1

```pip install spconv-cu114``` for CUDA 11.4

**NOTE** It's safe to have different minor cuda version between system and conda (pytorch). for example, you can use spconv-cu114 with anaconda version of pytorch cuda 11.1 in a OS with CUDA 11.2 installed.

### Build from source

You need to rebuild ```cumm``` first if you are build along a CUDA version that not provided in prebuilts.

#### Linux

1. install build-essential, install CUDA
2. run ```export SPCONV_DISABLE_JIT="1"```
3. run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

#### Windows 10/11

1. install visual studio 2019 or newer. make sure C++ development package is installed. install CUDA
2. set [powershell script execution policy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1)
3. start a new powershell, run ```tools/msvc_setup.ps1```
4. run ```$Env:SPCONV_DISABLE_JIT = "1"```
5. run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

## TODO in Spconv 2.x
- [ ] Ampere (A100 / RTX 3000 series) feature support (work in progress)
- [ ] torch QAT support (work in progress)
- [ ] TensorRT (torch.fx based)
- [ ] Build C++ only package
- [ ] JIT compilation for CUDA kernels
- [ ] Document (low priority)
- [ ] CPU support (low priority)

## Note

The work is done when the author is an employee at Tusimple.

## LICENSE

Apache 2.0
