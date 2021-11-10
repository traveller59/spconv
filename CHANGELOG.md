# Changelog

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
