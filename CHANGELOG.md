# Changelog

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
