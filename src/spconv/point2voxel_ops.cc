#include <spconv/point2voxel_ops.h>
//#include <spconv/point2voxel.cu.h>

namespace spconv {

std::vector<torch::Tensor>
pointsToVoxel(torch::Tensor points, torch::Tensor indexes,
                std::vector<int64_t> gridShape,
                const int64_t ndim,
                const int64_t gridVolume) {
  auto device = points.device().type();
  auto num_points = points.size(0);
  auto num_features = points.size(1);
  auto pointIndexUnique = torch::full(
      {num_points + 1}, std::numeric_limits<int>::max(),
      torch::dtype(torch::kInt32).device(points.device()));
  auto grids = torch::zeros({gridVolume, num_features},
      torch::dtype(points.dtype()).device(points.device()));
  auto numPointsPerGrid = torch::zeros({gridVolume},
      torch::dtype(torch::kInt32).device(points.device()));
  if (points.device().type() == torch::kCPU) {
    TV_THROW_INVALID_ARG("not support cpu currently");
  }
#ifdef TV_CUDA
  else if (points.device().type() == torch::kCUDA) {
    scatter_point_to_grid_cuda(points, indexes, grids,
        numPointsPerGrid, pointIndexUnique, gridShape, ndim);
  }
#endif
  else {
    TV_THROW_INVALID_ARG("unknown device type");
  }
  auto res = torch::_unique(pointIndexUnique);
  pointIndexUnique = std::get<0>(res);
  auto num_voxel = pointIndexUnique.size(0) - 1;
  auto voxels = torch::zeros({num_voxel, num_features},
      torch::dtype(points.dtype()).device(points.device()));
  auto coors = torch::zeros({num_voxel, ndim},
      torch::dtype(torch::kInt32).device(points.device()));
  if (points.device().type() == torch::kCPU) {
    TV_THROW_INVALID_ARG("not support cpu currently");
  }
#ifdef TV_CUDA
  else if (points.device().type() == torch::kCUDA) {
    gather_point_from_grid_cuda(grids, numPointsPerGrid,
        pointIndexUnique, voxels, coors, gridShape, ndim);
  }
#endif
  else {
    TV_THROW_INVALID_ARG("unknown device type");
  }
  return {voxels, coors};
}

} // namespace spconv
