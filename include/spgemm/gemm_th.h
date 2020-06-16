#pragma once
#include <cuda_runtime_api.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>

namespace spconv {
void cutlass_mm_out(torch::Tensor c, torch::Tensor a, torch::Tensor b);
void cutlass_mm_out(cudaStream_t stream, torch::Tensor c, torch::Tensor a,
                    torch::Tensor b);

} // namespace spconv