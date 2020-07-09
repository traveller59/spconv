
#include <ATen/ATen.h>
#include <spconv/point2voxel.cu.h>
//#include <spconv/point2voxel.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/mp_helper.h>
#include <tensorview/tensor.h>
#include <tensorview/tensorview.h>
#include <tensorview/torch_utils.h>

namespace spconv {

void scatter_point_to_grid_cuda(torch::Tensor points, torch::Tensor indexes,
                                torch::Tensor grids,
                                torch::Tensor numPointsPerGrid,
                                torch::Tensor pointIndex,
                                std::vector<int64_t> gridShape,
                                const int ndim) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto num_points = points.size(0);
  auto num_features = points.size(1);
  tv::dispatch_torch<int32_t>(pointIndex.scalar_type(), [&](auto IndexValue) {
    using Index = decltype(IndexValue);
    tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
      constexpr int NDim = decltype(I)::value;
      tv::SimpleVector<Index, NDim> gs(gridShape.begin(), gridShape.end());
      scatterPointToGridKernel<Index, NDim>
          <<<tv::cuda::getBlocks(num_points), tv::cuda::CUDA_NUM_THREADS, 0,
             stream>>>(tv::torch2tv<float>(points),
                       tv::torch2tv<Index>(indexes), tv::torch2tv<float>(grids),
                       tv::torch2tv<Index>(numPointsPerGrid),
                       tv::torch2tv<Index>(pointIndex), gs);
      TV_CHECK_CUDA_ERR_V2("scatterPointToGridKernel failed");
#ifdef TV_LOG_KERNEL_INFO
      cudaFuncAttributes attr;
      checkCudaErrors(
          cudaFuncGetAttributes(&attr, scatterPointToGridKernel<Index, NDim>));
      tv::ssprint("scatterPointToGridKernel<", tv::type_s<Index>, NDim, ">",
                  attr.numRegs);
#endif
    });
  });
}

void gather_point_from_grid_cuda(torch::Tensor grids,
                                 torch::Tensor numPointsPerGrid,
                                 torch::Tensor pointIndex,
                                 torch::Tensor pointIndexUnique,
                                 torch::Tensor voxels, torch::Tensor coors,
                                 std::vector<int64_t> gridShape,
                                 const int ndim) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto num_voxel = voxels.size(0);
  auto num_max_points = pointIndex.size(0) - 1;
  auto grid_volume = grids.size(0);
  tv::dispatch_torch<int32_t>(
      pointIndexUnique.scalar_type(), [&](auto IndexValue) {
        using Index = decltype(IndexValue);
        tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
          constexpr int NDim = decltype(I)::value;
          tv::SimpleVector<Index, NDim> gs(gridShape.begin(), gridShape.end());

          resetPointIndexKernel<Index>
              <<<tv::cuda::getBlocks(num_max_points),
                 tv::cuda::CUDA_NUM_THREADS, 0, stream>>>(
                  tv::torch2tv<Index>(pointIndex), grid_volume);
          TV_CHECK_CUDA_ERR_V2("resetPointIndexKernel failed");
#ifdef TV_LOG_KERNEL_INFO
          cudaFuncAttributes attr0;
          checkCudaErrors(cudaFuncGetAttributes(
              &attr0, resetPointIndexKernel<Index, NDim>));
          tv::ssprint("resetPointIndexKernel<", tv::type_s<Index>, NDim, ">",
                      attr0.numRegs);
#endif

          gatherPointFromGridKernel<Index, NDim>
              <<<tv::cuda::getBlocks(num_voxel), tv::cuda::CUDA_NUM_THREADS, 0,
                 stream>>>(tv::torch2tv<float>(grids),
                           tv::torch2tv<Index>(numPointsPerGrid),
                           tv::torch2tv<Index>(pointIndexUnique),
                           tv::torch2tv<float>(voxels),
                           tv::torch2tv<Index>(coors), gs);
          TV_CHECK_CUDA_ERR_V2("gatherPointFromGridKernel failed");
#ifdef TV_LOG_KERNEL_INFO
          cudaFuncAttributes attr1;
          checkCudaErrors(cudaFuncGetAttributes(
              &attr1, gatherPointFromGridKernel<Index, NDim>));
          tv::ssprint("gatherPointFromGridKernel<", tv::type_s<Index>, NDim,
                      ">", attr1.numRegs);
#endif

          resetGridKernel<Index><<<tv::cuda::getBlocks(num_voxel),
                                   tv::cuda::CUDA_NUM_THREADS, 0, stream>>>(
              tv::torch2tv<float>(grids), tv::torch2tv<Index>(numPointsPerGrid),
              tv::torch2tv<Index>(pointIndexUnique));
          TV_CHECK_CUDA_ERR_V2("resetGridKernel failed");
#ifdef TV_LOG_KERNEL_INFO
          cudaFuncAttributes attr2;
          checkCudaErrors(
              cudaFuncGetAttributes(&attr2, resetGridKernel<Index, NDim>));
          tv::ssprint("resetGridKernel<", tv::type_s<Index>, NDim, ">",
                      attr2.numRegs);
#endif
        });
      });
}

} // namespace spconv
