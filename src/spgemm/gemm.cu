#include <spgemm/gemm.h>
#include <spgemm/gemm_th.h>
namespace spconv {
template <typename T>
using determine_half_t =
    std::conditional_t<std::is_same<T, at::Half>::value, cutlass::half_t, T>;

void cutlass_mm_out(cudaStream_t stream, torch::Tensor c, torch::Tensor a,
                    torch::Tensor b) {
  TV_ASSERT_RT_ERR(c.dtype() == a.dtype() && c.dtype() == b.dtype(),
                   "dtype must be same");
  TV_ASSERT_RT_ERR(c.is_contiguous() && b.is_contiguous() && a.is_contiguous(),
                   "error");
  auto M = a.size(0);
  auto K = a.size(1);
  auto N = b.size(1);
  TV_ASSERT_RT_ERR(b.size(0) == K && c.size(0) == M && c.size(1) == N, "error");
  tv::dispatch_torch<float, at::Half>(c.scalar_type(), [&](auto I) {
    using T = decltype(I);
    using HalfT = determine_half_t<T>;
    auto status = cutlassGemm<HalfT, false, false, false>(
        stream, M, N, K, HalfT(1.0), reinterpret_cast<HalfT *>(a.data_ptr<T>()),
        a.size(1), reinterpret_cast<HalfT *>(b.data_ptr<T>()), b.size(1),
        HalfT(0.0), reinterpret_cast<HalfT *>(c.data_ptr<T>()), c.size(1));
    TV_ASSERT_RT_ERR(status == cudaSuccess, "error");
  });
}

void cutlass_mm_out(torch::Tensor c, torch::Tensor a, torch::Tensor b) {
  auto stream = at::cuda::getCurrentCUDAStream();
  return cutlass_mm_out(stream, c, a, b);
}

} // namespace spconv