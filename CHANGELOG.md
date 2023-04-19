# Changelog
## [2.3.6] - 2023-04-19
### Fixed 
- Fix a CI bug that cpu cumm and spconv use different gcc compiler, must be same.

## [2.3.5] - 2023-03-24
### Fixed 
- use a flag to enable large kernel algo (need time to compile at runtime)

## [2.3.4] - 2023-03-23
### Added 
- Add SparseGlobalMaxPool and SparseGlobalAvgPool for training only. libspconv don't support it.

## [2.3.3] - 2023-02-02
### Fixed 
- Fix int8 nvrtc error when use prebuilt
- Fix int8 kernel when run on turing GPU

## [2.3.2] - 2023-01-20
### Changed 
- change version


## [2.3.1] - 2023-01-20
### Changed 
- change version

## [2.3.0] - 2023-01-19
### Added 
- Add int8 quantization support
- Add large kernel support for implicit gemm (kv <= 128)

## [2.2.6] - 2022-11-06
### Fixed 
- CI fail because of pypi temporary shutdown. assign a new version and run again.

## [2.2.5] - 2022-11-05
### Fixed 
- Fix overflow when shape is too large

## [2.2.4] - 2022-10-13
### Added 
- Add prebuilt for CUDA 11.8 (RTX 4090 and H100) and CUDA 11.6.
### Fixed 
- Fix small bugs

## [2.2.3] - 2022-9-28
### Fixed 
- Fix missing .contiguous for input feature
- Add some debug msg if points vanished.

## [2.2.2] - 2022-9-25
### Fixed 
- Fix CI problem: main function too long and cause OOM in CI vm.

## [2.2.1] - 2022-9-25
### Fixed 
- Fix build problem
- Fix nvrtc problem

## [2.2.0] - 2022-9-24
### Added 
- Add Ampere support. faster fp16, faster tf32 and greatly faster int8 kernels in Ampere GPUs.
- Add pure c++ code generation (libspconv.so) for deploy (or train in another deeplearning framework)
- Add NVRTC support for all gemm kernels. if your GPU architecture isn't compiled in prebuilt, spconv will use slightly slower (10-20us overhead for every kernel launch) NVRTC kernels. 

### Fixed
- Fix launch fail in maxpool if too much voxels 

### Changed
- all weight layout will be KRSC, don't support old spconv 1.x weights anymore.
- previous gemm ops in ops.py now move to c++ by default (controlled by spconv.constants.SPCONV_CPP_GEMM)

### Removed
- drop python 3.6 support.
- pascal and kepler architecture is removed in CUDA 12 prebuilt.

## [2.1.22] - 2022-6-11
### Fixed
- Fix thrust problem by adding -fvisibility=hidden

## [2.1.22] - 2022-4-14
### Added
- add full nvrtc support
- add support for large spatial shape and batch size. if detect large shape, we use int64 instead of int32 when hashing.

## [2.1.21] - 2021-12-9
### Added
- add sm_37
- add fp16 kernels witl fp32 accumulator (run slower, but can avoid nan if channel size is too large)
- add SPCONV_BWD_SPLITK env to control splitk candidates.

## [2.1.20] - 2021-12-6
### Added
- Add fp16 conv simt kernels for mixed-training in pascal or older GPUS. WARNING: not optimized for TESLA P100 which has 2x throughput in half.

## [2.1.19] - 2021-12-3
### Fixed
- Fix wrong arch assert in all kernels for old GPUs to make spconv work in sm_50 GPUs

## [2.1.18] - 2021-11-29
### Fixed
- Fix a small bug of spatial_shape.
- Fix a bug in PointToVoxel, we must always return a clone instead of a view.

## [2.1.17] - 2021-11-29
### Fixed
- Fix a bug in sparse add.
- Fix a serious bug in conv weight init.
### Added
- Add more wrong usage check
- Add insert_exist_keys for hash table

## [2.1.16] - 2021-11-28
### Fixed
- Fix strange compile problem in windows

## [2.1.15] - 2021-11-28
### Fixed
- Fix missing pccm.Class in setup.py

## [2.1.14] - 2021-11-28
### Added 
- Add hash table
- update cumm version
- Add AddTableMisaligned for sptensors with same shape but different indices.
### Fixed
- Fix a bug already fixed in 2.1.10 but introduced in 2.1.12 again.

## [2.1.13] - 2021-?-?
### Added 
- Add some ops from spconv 1.x, see spconv.utils for more details.
- Add some debug tool for users to attach more info in issue.

## [2.1.12] - 2021-11-23
### Added 
- Add a method for voxel generator to get pc_voxel_id, which is usually used in semantic segmentation
### Fixed
- Fix a bug in cuda voxel generater when max_voxels is smaller than real number of voxels

## [2.1.11] - 2021-11-22
### Fixed
- Fixed a bug Volta kernels (TITAN V, Tesla V100), backward weight kernels use f16 as accumulator. we should use f32.
- Fixed a corner case when user use kernel size = 1x1 but stride != 1.
- Fixed a corner case when input feature is non-contiguous when maxpool.

## [2.1.10] - 2021-11-19
### Fixed
- Fixed a bug in utils.PointToVoxel, shouldn't get cuda stream in cpu code

## [2.1.9] - 2021-11-18
### Removed
- Remove a wrong assert

## [2.1.8] - 2021-11-15
### Added
- Add support for pytorch 1.5

## [2.1.7] - 2021-11-11
### Fixed
- Fix a bug when net have inverse and run inference in eval mode.

## [2.1.6] - 2021-11-10
### Fixed
- Fix missing -fopenmp in linker for CPU only
### Removed
- remove stale comment sending in CI

## [2.1.5] - 2021-11-10
### Added
- Add cuda profile tool
- Add python 36 support
### Changed
- Format all code
### Removed
- remove a unnecessary device sync and slightly improve performance.

## [2.1.4] - 2021-11-10
### Fixed
- Fix a bug of SparseInverseConv3d

## [2.1.3] - 2021-11-08
### Fixed
- Fix a bug of CPU only package

## [2.1.2] - 2021-11-06
### Fixed
- Fix a bug of python 3.7

## [2.1.0] - 2021-10-31
### Added
- add implicit gemm algorithm for all kind of convolution with kernel volume <= 32. this algorithm is very fast with float16.
- add pytorch wrapper for voxel generator
- add CPU support and CPU-only build.

## [2.0.2] - 2021-10-26
### Fixed
- Fix a serious bug that do nothing with non-spconv layers in SparseSequential
- Fix a bug of ProxyableClassMeta

## [2.0.0] - 2021-10-16
### Changed
- Change build system from cmake to pccm.
- Change pytorch python code to spconv.pytorch
- Rewrite All c++ code.

## [1.2.1] - 2020-06-04
### Changed
- The subm indice pair generation speed is greatly increased by two tricks: 1. most subm conv use only kernelsize=3, so we can unroll loops to get 100% performance increase. 2. subm indice pairs have a property: indicePairs[0, i] = indicePairs[1, kernelVolume - i - 1], so we can get another 100% performance increase. 


## [1.2.0] - 2020-05-28
### Added
- add batch gemm support. small performance increasement but more gpu memory usage. you can use algo=spconv.ConvAlgo.Batch to use it.

### Changed
- replace most of 'functor' with c++14 dispatch in c++ code.

### Fixed
- change gather/scatterAdd kernel parameter to support large points.
