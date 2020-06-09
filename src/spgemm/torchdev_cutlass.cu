#define TV_CUDA
#include <cutlass/gemm/device/gemm.h>
#include <spgemm/gemm.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/kernel_utils.h>
#include <tensorview/tensor.h>
#include <tensorview/torch_utils.h>
#include <torch/script.h>
#include <utility/timer.h>

int main() {
  auto M = 100000;
  auto N = 128;
  auto K = 128;
  auto a =
      torch::rand({M, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto b = torch::rand({K, N}, a.options());
  auto c = torch::zeros({a.size(0), b.size(1)}, a.options());
  auto c2 = torch::zeros({a.size(0), b.size(1)}, a.options());
  torch::mm_out(c, a, b);
  auto status = spconv::cutlassGemm<float, false, false, false>(
      0, M, N, K, 1.0, a.data_ptr<float>(), a.size(1), b.data_ptr<float>(),
      b.size(1), 0.0, c2.data_ptr<float>(), c2.size(1));
  auto err = torch::norm(c2 - c);
  tv::ssprint(status, "linalg norm", err);
  tv::ssprint((c.view({-1}) == 0).sum());
  auto timer = spconv::CudaContextTimer<>();
  for (int i = 0; i < 10; ++i) {
    torch::mm_out(c, a, b);
    tv::ssprint("mm", timer.report() / 1000.0);
    spconv::cutlassGemm<float, false, false, false>(
        0, M, N, K, 1.0, a.data_ptr<float>(), a.size(1), b.data_ptr<float>(),
        b.size(1), 0.0, c2.data_ptr<float>(), c2.size(1));
    tv::ssprint("cutlass_mm", timer.report() / 1000.0);
  }

  return 0;
}