#include <ATen/ATen.h>
#include <spconv/cublas_gemm.h>

namespace spconv {
template <>
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

template <>
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda,
                           const __half *B, int ldb, const __half *beta,
                           __half *C, int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

template <>
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const at::Half *alpha, const at::Half *A, int lda,
                           const at::Half *B, int ldb, const at::Half *beta,
                           at::Half *C, int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k,
                     reinterpret_cast<const __half *>(alpha),
                     reinterpret_cast<const __half *>(A), lda,
                     reinterpret_cast<const __half *>(B), ldb,
                     reinterpret_cast<const __half *>(beta),
                     reinterpret_cast<__half *>(C), ldc);
}

template <>
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta,
                           double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}


} // namespace spconv