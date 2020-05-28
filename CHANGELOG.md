# Changelog

## [1.2.0] - 2020-05-28
### Added
- add batch gemm support. small performance increasement but more gpu memory usage. you can use algo=spconv.ConvAlgo.Batch to use it.

### Changed
- replace most of 'functor' with c++14 dispatch in c++ code.

### Fixed
- change gather/scatterAdd kernel parameter to support large points.
