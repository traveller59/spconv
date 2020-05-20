#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#if CUDART_VERSION >= 4000
#define CUDA_DEVICE_SYNCHRONIZE() cudaDeviceSynchronize();
#else
#define CUDA_DEVICE_SYNCHRONIZE() cudaThreadSynchronize();
#endif

#define CUDA_SAFE_CALL_NO_SYNC(call)                                           \
  {                                                                            \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_SAFE_CALL(call) CUDA_SAFE_CALL_NO_SYNC(call);

//! Check for CUDA error
#ifdef _DEBUG
#define CUDA_CHECK_ERROR(errorMessage)                                         \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    err = CUDA_DEVICE_SYNCHRONIZE();                                           \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#else
#define CUDA_CHECK_ERROR(errorMessage)                                         \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#endif

#endif