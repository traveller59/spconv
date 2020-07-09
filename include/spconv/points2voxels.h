#pragma once

#include <tensorview/tensorview.h>
#include <torch/script.h>

namespace spconv {

void scatter_point_to_grid_cuda(torch::Tensor points, torch::Tensor indexes,
                                torch::Tensor grids,
                                torch::Tensor numPointsPerGrid,
                                torch::Tensor pointIndex,
                                std::vector<int64_t> gridShape, const int ndim);

void gather_point_from_grid_cuda(torch::Tensor grids,
                                 torch::Tensor numPointsPerGrid,
                                 torch::Tensor pointIndex,
                                 torch::Tensor pointIndexUnique,
                                 torch::Tensor voxels, torch::Tensor coors,
                                 std::vector<int64_t> gridShape,
                                 const int ndim);

} // namespace spconv
