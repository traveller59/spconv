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

[pypi-download]: https://img.shields.io/pypi/dm/spconv-cu114
[pypi-url]: https://pypi.org/project/spconv-cu114/
[pypi-image]: https://badge.fury.io/py/spconv-cu114.svg

# SpConv: Spatially Sparse Convolution Library
[![Build Status](https://github.com/traveller59/spconv/workflows/build/badge.svg)](https://github.com/traveller59/spconv/actions?query=workflow%3Abuild) [![PyPI Version][pypi-image]][pypi-url] [![pypi monthly download][pypi-download]][pypi-url]


```spconv``` is a project that provide heavily-optimized sparse convolution implementation with tensor core support.

[Spconv 1.x code](https://github.com/traveller59/spconv/tree/v1.2.1). We won't provide any support for spconv 1.x since it's deprecated. use spconv 2.x if possible. <!--remove this message in spconv 2.2-->

**WARNING** spconv < 2.1.4 users need to upgrade your version to 2.1.4, it fix a serious bug in SparseInverseConvXd.

## Breaking changes in Spconv 2.x

Spconv 1.x users **NEED READ [THIS](docs/SPCONV_2_BREAKING_CHANGEs.md)** before using spconv 2.x.

## Spconv 2.1 vs Spconv 1.x

* spconv now can be installed by **pip**. see install section in readme for more details. Users don't need to build manually anymore!
* Microsoft Windows support (only windows 10 has been tested).
* fp32 (not tf32) training/inference speed is increased (+50~80%)
* fp16 training/inference speed is greatly increased when your layer support tensor core (channel size must be multiple of 8).
* int8 op is ready, but we still need some time to figure out how to run int8 in pytorch.
* doesn't depend on pytorch binary. 
* since spconv 2.x doesn't depend on pytorch binary (never in future), it's impossible to support torch.jit/libtorch inference.

Spconv 2.1 vs 1.x speed:

|                | 1080Ti Spconv 1.x F32 | 1080Ti Spconv 2.0 F32 | 3080M* Spconv 2.1 F16  |
| -------------- |:---------------------:| ---------------------:| ----------:|
| 27x128x128 Fwd | 11ms                  | 5.4ms                 | 1.4ms      |

\* 3080M (Laptop) ~= 3070 Desktop


<!--
TODO Spconv vs [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) vs [torchsparse](https://github.com/mit-han-lab/torchsparse)
-->

## Usage

Firstly you need to use ```import spconv.pytorch as spconv``` in spconv 2.x.

Then see [this](docs/USAGE.md).

Don't forget to check [performance guide](docs/PERFORMANCE_GUIDE.md).

## Install

You need to install python >= 3.7 first to use spconv 2.x.

You need to install CUDA toolkit first before using prebuilt binaries or build from source.

You need at least CUDA 10.2 to build and run spconv 2.x. We won't offer any support for CUDA < 10.2.

### Prebuilt

We offer python 3.7-3.10 and cuda 10.2/11.1/11.3/11.4 prebuilt binaries for linux (manylinux) and windows 10/11.

We will provide prebuilts for CUDA versions supported by latest pytorch release. For example, pytorch 1.10 provide cuda 10.2 and 11.3 prebuilts, so we provide them too.

For Linux users, you need to install pip >= 20.3 first to install prebuilt.

CUDA 11.1 will be removed in spconv 2.2 because pytorch 1.10 don't provide prebuilts for it.

```pip install spconv``` for CPU only (**Linux Only**). you should only use this for debug usage, the performance isn't optimized due to manylinux limit (no omp support).

```pip install spconv-cu102``` for CUDA 10.2

```pip install spconv-cu111``` for CUDA 11.1

```pip install spconv-cu113``` for CUDA 11.3 (**Linux Only**)

```pip install spconv-cu114``` for CUDA 11.4

**NOTE** It's safe to have different **minor** cuda version between system and conda (pytorch) **in Linux**. for example, you can use spconv-cu114 with anaconda version of pytorch cuda 11.1 in a OS with CUDA 11.2 installed.

**NOTE** In Linux, you can install spconv-cuxxx without install CUDA to system! only suitable NVIDIA driver is required. for CUDA 11, we need driver >= 450.82.

### Build from source for development (JIT, recommend)

The c++ code will be built automatically when you change c++ code in project.

For NVIDIA Embedded Platforms, you need to specify cuda arch before build: ```export CUMM_CUDA_ARCH_LIST="7.2"``` for xavier.

#### Linux
0. uninstall spconv and cumm installed by pip
1. install build-essential, install CUDA
2. ```git clone https://github.com/FindDefinition/cumm```, ```cd ./cumm```, ```pip install -e .```
3. ```git clone https://github.com/traveller59/spconv```, ```cd ./spconv```, ```pip install -e .```
4. in python, ```import spconv``` and wait for build finish.

#### Windows
0. uninstall spconv and cumm installed by pip
1. install visual studio 2019 or newer. make sure C++ development component is installed. install CUDA
2. set [powershell script execution policy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1)
3. start a new powershell, run ```tools/msvc_setup.ps1```
4. ```git clone https://github.com/FindDefinition/cumm```, ```cd ./cumm```, ```pip install -e .```
5. ```git clone https://github.com/traveller59/spconv```, ```cd ./spconv```, ```pip install -e .```
6. in python, ```import spconv``` and wait for build finish.

### Build wheel from source (not recommend, this is done in CI.)

You need to rebuild ```cumm``` first if you are build along a CUDA version that not provided in prebuilts.

#### Linux

1. install build-essential, install CUDA
2. run ```export SPCONV_DISABLE_JIT="1"```
3. run ```pip install pccm cumm wheel```
4. run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

#### Windows

1. install visual studio 2019 or newer. make sure C++ development component is installed. install CUDA
2. set [powershell script execution policy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1)
3. start a new powershell, run ```tools/msvc_setup.ps1```
4. run ```$Env:SPCONV_DISABLE_JIT = "1"```
5. run ```pip install pccm cumm wheel```
6. run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

## Roadmap for Spconv 2.2-2.3: 
* TensorFormat32 support for faster fp32 training when you use NVIDIA Geforce RTX 30x0/Tesla A100/Quadro RTX Ax000 (2.2)
* change implicit gemm weight layout from KRSC to RSKC to make sure we can use native algorithm with implicit gemm weight. (2.2)
* documents (2.2)
* Ampere feature support (2.3)
* pytorch int8 inference, and QAT support (2.3)

## TODO in Spconv 2.x
- [ ] Ampere (A100 / RTX 3000 series) feature support (work in progress)
- [ ] torch QAT support (work in progress)
- [ ] TensorRT (torch.fx based)
- [ ] Build C++ only package
- [ ] JIT compilation for CUDA kernels
- [ ] Document (low priority)

## Note

The work is done when the author is an employee at [Tusimple](https://www.tusimple.com/).

## LICENSE

Apache 2.0
