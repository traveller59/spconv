#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <type_traits>
namespace spconv {

template <typename T>
using determine_acc_t =
    std::conditional_t<std::is_same<T, cutlass::half_t>::value, float, T>;

template <typename T, bool TransA, bool TransB, bool TransC>
cudaError_t cutlassGemm(cudaStream_t s, int M, int N, int K, T alpha,
                        T const *A, int lda, T const *B, int ldb, T beta, T *C,
                        int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions including the following example for single-precision GEMM.
  // Typical values are used as default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`
  using TAcc = determine_acc_t<T>;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;
  using LayoutA = std::conditional_t<TransA, ColumnMajor, RowMajor>;
  using LayoutB = std::conditional_t<TransB, ColumnMajor, RowMajor>;
  using LayoutC = std::conditional_t<TransC, ColumnMajor, RowMajor>;

  using CutlassGemm = cutlass::gemm::device::Gemm<T, // Data-type of A matrix
                                                  LayoutA, // Layout of A matrix
                                                  T, // Data-type of B matrix
                                                  LayoutB, // Layout of B matrix
                                                  T, // Data-type of C matrix
                                                  LayoutC,
                                                  TAcc>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible in host code and passed to kernels by value. These may
  // include pointers, strides, scalars, and other arguments needed by Gemm and
  // its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible arguments to kernels and (2.) minimized
  // initialization overhead on kernel entry.
  //
  typename CutlassGemm::Arguments args(
      {M, N, K}, // Gemm Problem dimensions
      {A, lda},  // Tensor-ref for source matrix A
      {B, ldb},  // Tensor-ref for source matrix B
      {C, ldc},  // Tensor-ref for source matrix C
      {C, ldc},  // Tensor-ref for destination matrix D (may be different memory
                 // than source C matrix)
      {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //

  cutlass::Status status = gemm_operator(args, nullptr, s);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

} // namespace spconv
