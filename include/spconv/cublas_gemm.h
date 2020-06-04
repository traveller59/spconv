#pragma once
#include <cublas_v2.h>
#include <tensorview/tensorview.h>

namespace spconv {

template <class T>
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const T *alpha, const T *A, int lda, const T *B,
                           int ldb, const T *beta, T *C, int ldc);

template <class T>
cublasStatus_t cublasTgemmRow(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const T *alpha, const T *A, int lda, const T *B,
                              int ldb, const T *beta, T *C, int ldc) {
  return cublasTgemm<T>(handle, transb, transa, n, m, k, alpha, B, ldb, A, lda,
                        beta, C, ldc);
}

template <class T> inline T constant_scalar(float data) { return T(data); }

template <class T>
cublasStatus_t gemm(cublasHandle_t handle, bool transa, bool transb,
                    const tv::TensorView<T> A, const tv::TensorView<T> B,
                    tv::TensorView<T> C) {
  TV_ASSERT_RT_ERR(A.ndim() == 2, "error");
  TV_ASSERT_RT_ERR(B.ndim() == 2, "error");
  auto transa_cublas = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transb_cublas = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transa ? A.dim(1) : A.dim(0);
  int n = transb ? B.dim(0) : B.dim(1);
  int ka = transa ? A.dim(0) : A.dim(1);
  int kb = transb ? B.dim(1) : B.dim(0);
  int lda = transa ? m : ka;
  int ldb = transb ? ka : n;
  int ldc = n;
  TV_ASSERT_RT_ERR(ka == kb, "error");
  T alpha = constant_scalar<T>(1);
  T beta = constant_scalar<T>(0);
  return cublasTgemmRow<T>(handle, transa_cublas, transb_cublas, m, n, ka,
                           &alpha, A.data(), lda, B.data(), ldb, &beta,
                           C.data(), ldc);
}

} // namespace spconv
