#include <spconv/point2voxel_ops.h>
//#include <spconv/point2voxel.cu.h>

namespace spconv {

int64_t pointsToVoxel(torch::Tensor points, torch::Tensor indexes,
                      torch::Tensor pointIndex, torch::Tensor grids,
                      torch::Tensor numPointsPerGrid, torch::Tensor voxels,
                      torch::Tensor coors, std::vector<int64_t> gridShape,
                      const int64_t ndim) {
  if (points.device().type() == torch::kCPU) {
    TV_THROW_INVALID_ARG("not support cpu currently");
  }
#ifdef TV_CUDA
  else if (points.device().type() == torch::kCUDA) {
    scatter_point_to_grid_cuda(points, indexes, grids, numPointsPerGrid,
                               pointIndex, gridShape, ndim);
  }
#endif
  else {
    TV_THROW_INVALID_ARG("unknown device type");
  }
  auto res = torch::_unique(pointIndex);
  auto pointIndexUnique = std::get<0>(res);
  auto num_voxel = pointIndexUnique.size(0) - 1;
  if (points.device().type() == torch::kCPU) {
    TV_THROW_INVALID_ARG("not support cpu currently");
  }
#ifdef TV_CUDA
  else if (points.device().type() == torch::kCUDA) {
    gather_point_from_grid_cuda(grids, numPointsPerGrid, pointIndex,
                                pointIndexUnique, voxels, coors, gridShape,
                                ndim);
  }
#endif
  else {
    TV_THROW_INVALID_ARG("unknown device type");
  }
  return num_voxel;
}

} // namespace spconv
