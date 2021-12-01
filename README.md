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
[pypi-ver-cpu]: https://img.shields.io/pypi/v/spconv
[pypi-ver-114]: https://img.shields.io/pypi/v/spconv-cu114
[pypi-ver-111]: https://img.shields.io/pypi/v/spconv-cu111
[pypi-ver-113]: https://img.shields.io/pypi/v/spconv-cu113
[pypi-ver-102]: https://img.shields.io/pypi/v/spconv-cu102

[pypi-url-111]: https://pypi.org/project/spconv-cu111/
[pypi-download-111]: https://img.shields.io/pypi/dm/spconv-cu111
[pypi-url-113]: https://pypi.org/project/spconv-cu113/
[pypi-download-113]: https://img.shields.io/pypi/dm/spconv-cu113
[pypi-url-102]: https://pypi.org/project/spconv-cu102/
[pypi-download-102]: https://img.shields.io/pypi/dm/spconv-cu102
[pypi-url-114]: https://pypi.org/project/spconv-cu114/
[pypi-download-114]: https://img.shields.io/pypi/dm/spconv-cu114
[pypi-url-cpu]: https://pypi.org/project/spconv/
[pypi-download-cpu]: https://img.shields.io/pypi/dm/spconv

# SpConv: Spatially Sparse Convolution Library
[![Build Status](https://github.com/traveller59/spconv/workflows/build/badge.svg)](https://github.com/traveller59/spconv/actions?query=workflow%3Abuild) 


|                | PyPI   | Install  |Downloads  |
| -------------- |:---------------------:| ---------------------:| ---------------------:| 
| CPU (Linux Only) | [![PyPI Version][pypi-ver-cpu]][pypi-url-cpu] | ```pip install spconv``` | [![pypi monthly download][pypi-download-cpu]][pypi-url-cpu] | 
| CUDA 10.2 | [![PyPI Version][pypi-ver-102]][pypi-url-102] | ```pip install spconv-cu102``` | [![pypi monthly download][pypi-download-102]][pypi-url-102] | 
| CUDA 11.1 | [![PyPI Version][pypi-ver-111]][pypi-url-111] | ```pip install spconv-cu111```| [![pypi monthly download][pypi-download-111]][pypi-url-111]| 
| CUDA 11.3 (Linux Only) | [![PyPI Version][pypi-ver-113]][pypi-url-113] | ```pip install spconv-cu113```| [![pypi monthly download][pypi-download-113]][pypi-url-113]| 
| CUDA 11.4 | [![PyPI Version][pypi-ver-114]][pypi-url-114] | ```pip install spconv-cu114```| [![pypi monthly download][pypi-download-114]][pypi-url-114]|

```spconv``` is a project that provide heavily-optimized sparse convolution implementation with tensor core support. check [benchmark](docs/BENCHMARK.md) to see how fast spconv 2.x runs.

[Spconv 1.x code](https://github.com/traveller59/spconv/tree/v1.2.1). We won't provide any support for spconv 1.x since it's deprecated. use spconv 2.x if possible. <!--remove this message in spconv 2.2-->

Check [spconv 2.x algorithm introduction](docs/spconv2_algo.pdf) to understand sparse convolution algorithm in spconv 2.x!

**WARNING** spconv < 2.1.18 users need to upgrade your version to 2.1.18, it fix a bug in conv weight init which cause std of inited weight too large, and a bug in PointToVoxel.

## Breaking changes in Spconv 2.x

Spconv 1.x users **NEED READ [THIS](docs/SPCONV_2_BREAKING_CHANGEs.md)** before using spconv 2.x.

## Spconv 2.1 vs Spconv 1.x

* spconv now can be installed by **pip**. see install section in readme for more details. Users don't need to build manually anymore!
* Microsoft Windows support (only windows 10 has been tested).
* fp32 (not tf32) training/inference speed is increased (+50~80%)
* fp16 training/inference speed is greatly increased when your layer support tensor core (channel size must be multiple of 8).
* int8 op is ready, but we still need some time to figure out how to run int8 in pytorch.
* [doesn't depend on pytorch binary](docs/FAQ.md#What-does-no-dependency-on-pytorch-mean), but you may need at least pytorch >= 1.5.0 to run spconv 2.x.
* since spconv 2.x doesn't depend on pytorch binary (never in future), it's impossible to support torch.jit/libtorch inference.

## Spconv 2.x Development and Roadmap

Spconv 2.2 development has started. See [this issue](https://github.com/traveller59/spconv/issues/380) for more details.

See [dev plan](docs/SPCONV_DEVELOP_PLAN.md). A complete guide of spconv development will be released soon.

## Usage

Firstly you need to use ```import spconv.pytorch as spconv``` in spconv 2.x.

Then see [this](docs/USAGE.md).

Don't forget to check [performance guide](docs/PERFORMANCE_GUIDE.md).

## Install

You need to install python >= 3.6 (>=3.7 for windows) first to use spconv 2.x.

You need to install CUDA toolkit first before using prebuilt binaries or build from source.

You need at least CUDA 10.2 to build and run spconv 2.x. We won't offer any support for CUDA < 10.2.

### Prebuilt

We offer python 3.6-3.10 and cuda 10.2/11.1/11.3/11.4 prebuilt binaries for linux (manylinux).

We offer python 3.7-3.10 and cuda 10.2/11.1/11.4 prebuilt binaries for windows 10/11.

We will provide prebuilts for CUDA versions supported by latest pytorch release. For example, pytorch 1.10 provide cuda 10.2 and 11.3 prebuilts, so we provide them too.

For Linux users, you need to install pip >= 20.3 first to install prebuilt.

CUDA 11.1 will be removed in spconv 2.2 because pytorch 1.10 don't provide prebuilts for it.

```pip install spconv``` for CPU only (**Linux Only**). you should only use this for debug usage, the performance isn't optimized due to manylinux limit (no omp support).

```pip install spconv-cu102``` for CUDA 10.2

```pip install spconv-cu111``` for CUDA 11.1

```pip install spconv-cu113``` for CUDA 11.3 (**Linux Only**)

```pip install spconv-cu114``` for CUDA 11.4

**NOTE** It's safe to have different **minor** cuda version between system and conda (pytorch) in **CUDA >= 11.0** because of [CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility). For example, you can use spconv-cu114 with anaconda version of pytorch cuda 11.1 in a OS with CUDA 11.2 installed.

For CUDA 10, we don't know whether ```spconv-cu102``` works with CUDA 10.0 and 10.1. Users can have a try.

**NOTE** In Linux, you can install spconv-cuxxx without install CUDA to system! only suitable NVIDIA driver is required. for CUDA 11, we need driver >= 450.82.

#### Prebuilt GPU Support Matrix

See [this page](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to check supported GPU names by arch.

| CUDA version | GPU Arch List  |
| -------------- |:---------------------:|
| 10.2 | 50,52,60,61,70,75     | 
| 11.x       | 52,60,61,70,75,80,86     | 
| 12.x       | 60,61,70,75,80,86,90     | 

### Build from source for development (JIT, recommend)

The c++ code will be built automatically when you change c++ code in project.

For NVIDIA Embedded Platforms, you need to specify cuda arch before build: ```export CUMM_CUDA_ARCH_LIST="7.2"``` for xavier, ```export CUMM_CUDA_ARCH_LIST="6.2"``` for TX2, ```export CUMM_CUDA_ARCH_LIST="8.7"``` for orin.

You need to remove ```cumm``` in ```requires``` section in pyproject.toml after install editable ```cumm``` and before install spconv due to pyproject limit (can't find editable installed ```cumm```).

You need to ensure ```pip list | grep spconv``` and ```pip list | grep cumm``` show nothing before install editable spconv/cumm.

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

## Know issues

* Spconv 2.x F16 runs slow in A100. 

## Note

The work is done when the author is an employee at [Tusimple](https://www.tusimple.com/).

## LICENSE

Apache 2.0
