
/*
BSD License

For SparseConvNet software

Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define TACC double

template <typename T, int32_t K, int32_t V>
__global__ void
dConvolution_KMxKN_forwardA(T *inFeatures, T *outFeatures, T *w,
                            int32_t *rulesIn, int32_t *rulesOut, int32_t nHot,
                            int32_t input_nPlanes, int32_t input_stride,
                            int32_t output_nPlanes, int32_t output_stride) {
  // nHot must be a multiple of K!!

  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,nGroups) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  int32_t M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  int32_t n = blockIdx.y;
  int32_t g = blockIdx.z;
  inFeatures += g * input_nPlanes;
  outFeatures += n * K + g * output_nPlanes;
  w += n * K + g * input_nPlanes * output_nPlanes;

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  int32_t R0[V];
  int32_t R1[V];
  const int32_t tx = threadIdx.x;
  int32_t ty[V];
#pragma unroll
  for (int32_t v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int32_t m = 0; m < M; m++) {
// Read w
#pragma unroll
    for (int32_t v = 0; v < V; v++)
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (int32_t s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        R0[v] = rulesIn[s + ty[v]];
        R1[v] = rulesOut[s + ty[v]];
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
        O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int32_t k = 0; k < K; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++)
          O[v] += I[ty[v]][k] * W[k][tx];

#pragma unroll
      for (int32_t v = 0; v < V; v++)
        O[v] += outFeatures[R1[v] * output_stride + tx];
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        outFeatures[R1[v] * output_stride + tx] = O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}
template <typename T, int32_t K, int32_t V>
__global__ void
dConvolution_KMxKN_forwardB(T *inFeatures, T *outFeatures, T *w,
                            int32_t *rulesIn, int32_t *rulesOut, int32_t nHot,
                            int32_t input_nPlanes, int32_t input_stride,
                            int32_t output_nPlanes, int32_t output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,nGroups) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  int32_t M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  int32_t n = blockIdx.y;
  int32_t g = blockIdx.z;
  inFeatures += g * input_nPlanes;
  outFeatures += n * K + g * output_nPlanes;
  w += n * K + g * input_nPlanes * output_nPlanes;

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  int32_t R0[V];
  int32_t R1[V];
  const int32_t tx = threadIdx.x;
  int32_t ty[V];
#pragma unroll
  for (int32_t v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int32_t m = 0; m < M; m++) {
// Read w
#pragma unroll
    for (int32_t v = 0; v < V; v++)
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (int32_t s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (s + ty[v] < nHot) {
          R0[v] = rulesIn[s + ty[v]];
          R1[v] = rulesOut[s + ty[v]];
        }
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (s + ty[v] < nHot)
          I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
        O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int32_t k = 0; k < K; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++)
          O[v] += I[ty[v]][k] * W[k][tx];

#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (s + ty[v] < nHot)
          O[v] += outFeatures[R1[v] * output_stride + tx];
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (s + ty[v] < nHot)
          outFeatures[R1[v] * output_stride + tx] = O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      int32_t o = (nHot / K) * K;                                              \
      if (o >= K)                                                              \
        dConvolution_KMxKN_forwardA<T, K, V>                                   \
            <<<dim3(std::min(o / K, (int32_t)512), output_nPlanes / K,         \
                    nGroups),                                                  \
               dim3(K, K / V), 0, s>>>(                                        \
                inFeatures, outFeatures, w, rulesIn, rulesOut, o,              \
                input_nPlanes, input_stride, output_nPlanes, output_stride);   \
      if (nHot > o)                                                            \
        dConvolution_KMxKN_forwardB<T, K, V>                                   \
            <<<dim3(1, output_nPlanes / K, nGroups), dim3(K, K / V), 0, s>>>(  \
                inFeatures, outFeatures, w, rulesIn + o, rulesOut + o,         \
                nHot - o, input_nPlanes, input_stride, output_nPlanes,         \
                output_stride);                                                \
      return;                                                                  \
    }                                                                          \
  }
template <typename T>
void dConvolution_forward(cudaStream_t s, T *inFeatures, T *outFeatures, T *w,
                          int32_t *rulesIn, int32_t *rulesOut, int32_t nHot,
                          int32_t input_nPlanes, int32_t input_stride,
                          int32_t output_nPlanes, int32_t output_stride,
                          int32_t nGroups) {
  FOO(T, 64, 16)
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
template <>
void dConvolution_forward<double>(cudaStream_t s, double *inFeatures,
                                  double *outFeatures, double *w,
                                  int32_t *rulesIn, int32_t *rulesOut,
                                  int32_t nHot, int32_t input_nPlanes,
                                  int32_t input_stride, int32_t output_nPlanes,
                                  int32_t output_stride, int32_t nGroups) {
  FOO(double, 32, 8)
  FOO(double, 16, 4)
  FOO(double, 8, 2)
  assert(false);
}
#undef FOO
// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,nGroups)
template <typename T, int32_t K, int32_t V>
__global__ void dConvolution_KMxKN_backward_dW_A(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *w, T *dw,
    int32_t *rulesIn, int32_t *rulesOut, int32_t nHot, int32_t input_nPlanes,
    int32_t input_stride, int32_t output_nPlanes, int32_t output_stride) {
  // M = gridDim.y == input_nPlanes / K
  int32_t N = output_nPlanes / K;
  int32_t m = blockIdx.y;
  int32_t g = blockIdx.z;
  inFeatures += m * K + g * input_nPlanes;
  dInFeatures += m * K + g * input_nPlanes;
  dOutFeatures += g * output_nPlanes;
  w += m * K * output_nPlanes + g * input_nPlanes * output_nPlanes;
  dw += m * K * output_nPlanes + g * input_nPlanes * output_nPlanes;
  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  int32_t R0[V];
  int32_t R1[V];
  const int32_t tx = threadIdx.x;
  int32_t ty[V];
#pragma unroll
  for (int32_t v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);
  for (int32_t n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int32_t v = 0; v < V; v++) {
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }
    for (int32_t s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        R0[v] = rulesIn[s + ty[v]];
        R1[v] = rulesOut[s + ty[v]];
        dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
        dO[ty[v]][tx] = dOutFeatures[R1[v] * output_stride + tx];
      }
      __syncthreads();
#pragma unroll
      for (int32_t k = 0; k < K; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++) {
          dI[v] += dO[ty[v]][k] * W[tx][k];
          dW[v] += I[k][ty[v]] * dO[k][tx];
        }
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        dI[v] += dInFeatures[R0[v] * input_stride + tx];
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        dInFeatures[R0[v] * input_stride + tx] = dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int32_t v = 0; v < V; v++)
      atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}
// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,nGroups)
template <typename T, int32_t K, int32_t V>
__global__ void dConvolution_KMxKN_backward_dW_B(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *w, T *dw,
    int32_t *rulesIn, int32_t *rulesOut, int32_t nHot, int32_t input_nPlanes,
    int32_t input_stride, int32_t output_nPlanes, int32_t output_stride) {
  // M = gridDim.y == input_nPlanes / K
  int32_t N = output_nPlanes / K;
  int32_t m = blockIdx.y;
  int32_t g = blockIdx.z;
  inFeatures += m * K + g * input_nPlanes;
  dInFeatures += m * K + g * input_nPlanes;
  dOutFeatures += g * output_nPlanes;
  w += m * K * output_nPlanes + g * input_nPlanes * output_nPlanes;
  dw += m * K * output_nPlanes + g * input_nPlanes * output_nPlanes;
  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  int32_t R0[V];
  int32_t R1[V];
  const int32_t tx = threadIdx.x;
  int32_t ty[V];
#pragma unroll
  for (int32_t v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);
  for (int32_t n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int32_t v = 0; v < V; v++) {
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }
    for (int32_t s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (s + ty[v] < nHot) {
          R0[v] = rulesIn[s + ty[v]];
          R1[v] = rulesOut[s + ty[v]];
        }
        dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (s + ty[v] < nHot) {
          I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
          dO[ty[v]][tx] = dOutFeatures[R1[v] * output_stride + tx];
        } else {
          I[ty[v]][tx] = 0;
          dO[ty[v]][tx] = 0;
        }
      __syncthreads();
#pragma unroll
      for (int32_t k = 0; k < K; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++) {
          dI[v] += dO[ty[v]][k] * W[tx][k];
          dW[v] += I[k][ty[v]] * dO[k][tx];
        }
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (s + ty[v] < nHot)
          dI[v] += dInFeatures[R0[v] * input_stride + tx];
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (s + ty[v] < nHot)
          dInFeatures[R0[v] * input_stride + tx] = dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int32_t v = 0; v < V; v++)
      atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}
#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      int32_t o = (nHot / K) * K;                                              \
      if (o >= K)                                                              \
        dConvolution_KMxKN_backward_dW_A<T, K, V>                              \
            <<<dim3(std::min(o / K, (int32_t)512), input_nPlanes / K,          \
                    nGroups),                                                  \
               dim3(K, K / V), 0, s>>>(inFeatures, dInFeatures, dOutFeatures,  \
                                       w, dw, rulesIn, rulesOut, o,            \
                                       input_nPlanes, input_stride,            \
                                       output_nPlanes, output_stride);         \
      if (nHot > o)                                                            \
        dConvolution_KMxKN_backward_dW_B<T, K, V>                              \
            <<<dim3(1, input_nPlanes / K, nGroups), dim3(K, K / V), 0, s>>>(   \
                inFeatures, dInFeatures, dOutFeatures, w, dw, rulesIn + o,     \
                rulesOut + o, nHot - o, input_nPlanes, input_stride,           \
                output_nPlanes, output_stride);                                \
      return;                                                                  \
    }                                                                          \
  }
template <typename T>
void dConvolution_backward_dW(cudaStream_t s, T *inFeatures, T *dInFeatures,
                              T *dOutFeatures, T *w, T *dw, int32_t *rulesIn,
                              int32_t *rulesOut, int32_t nHot,
                              int32_t input_nPlanes, int32_t input_stride,
                              int32_t output_nPlanes, int32_t output_stride,
                              int32_t nGroups) {
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
#undef FOO
template <typename T, int32_t K, int32_t V>
__global__ void
dConvolution_KMxKN_forward2(T *inFeatures, T *outFeatures, T *w,
                            int32_t *rulesIn, int32_t *rulesOut, int32_t nHot,
                            int32_t input_nPlanes, int32_t input_stride,
                            int32_t output_nPlanes, int32_t output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,nGroups) Volkov-blocks
  // K is a multiple of V,
  // nHot x input_nplanes<=KM -> nHot x output_nPlanes<=KN
  // - parallel over N,nHot - loop over M
  int32_t M = (input_nPlanes + K - 1) / K;
  // N = gridDim.y ~ output_nPlanes/K
  int32_t n = blockIdx.y;
  int32_t g = blockIdx.z;
  inFeatures += g * input_nPlanes;
  outFeatures += n * K + g * output_nPlanes;
  w += n * K + g * input_nPlanes * output_nPlanes;
  int32_t KO = min(K, output_nPlanes - K * n);
  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  __shared__ int32_t R[K * 2];
  const int32_t tx = threadIdx.x;
  int32_t ty[V];
#pragma unroll
  for (int32_t v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);
  for (int32_t m = 0; m < M; m++) {
    int32_t KI = min(K, input_nPlanes - K * m);
// Read w
#pragma unroll
    for (int32_t v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
        W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
    for (int32_t s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
// Read rules for K input/output pairs
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (ty[v] < 1) {
          if (s + tx < nHot) {
            R[2 * tx] = rulesIn[s + tx];
            R[2 * tx + 1] = rulesOut[s + tx];
          }
          // R[q] = rules[2 * s + q];
        }
      }
      __syncthreads();
// Read input, reset O[]
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (tx < KI and s + ty[v] < nHot)
          I[ty[v]][tx] = inFeatures[R[2 * ty[v]] * input_stride + tx];
        O[v] = 0;
      }
      __syncthreads();
#pragma unroll
      for (int32_t k = 0; k < KI; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++)
          O[v] += I[ty[v]][k] * W[k][tx];
      __syncthreads();
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (tx < KO and s + ty[v] < nHot)
          outFeatures[R[2 * ty[v] + 1] * output_stride + tx] += O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}
// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,nGroups)
template <typename T, int32_t K, int32_t V>
__global__ void dConvolution_KMxKN_backward_dW2(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *w, T *dw,
    int32_t *rulesIn, int32_t *rulesOut, int32_t nHot, int32_t input_nPlanes,
    int32_t input_stride, int32_t output_nPlanes, int32_t output_stride) {
  // M = gridDim.y == input_nPlanes / K
  int32_t N = (output_nPlanes + K - 1) / K;
  int32_t m = blockIdx.y;
  int32_t g = blockIdx.z;
  inFeatures += m * K + g * input_nPlanes;
  dInFeatures += m * K + g * input_nPlanes;
  dOutFeatures += g * output_nPlanes;
  w += m * K * output_nPlanes + g * input_nPlanes * output_nPlanes;
  dw += m * K * output_nPlanes + g * input_nPlanes * output_nPlanes;
  int32_t KI = min(K, input_nPlanes - K * m);
  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  __shared__ int32_t R[K * 2];
  const int32_t tx = threadIdx.x;
  int32_t ty[V];
#pragma unroll
  for (int32_t v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);
  for (int32_t n = 0; n < N; n++) {
    int32_t KO = min(K, output_nPlanes - K * n);
// Read w, reset dW
#pragma unroll
    for (int32_t v = 0; v < V; v++) {
      if (ty[v] < KI and tx < KO)
        W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }
    for (int32_t s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
// Read rules for K input/output pairs, reset dI[]
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (ty[v] < 1) {
          if (s + tx < nHot) {
            R[2 * tx] = rulesIn[s + tx];
            R[2 * tx + 1] = rulesOut[s + tx];
          }
          // R[q] = rules[2 * s + q];
        }
        dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int32_t v = 0; v < V; v++) {
        if (tx < KI and s + ty[v] < nHot)
          I[ty[v]][tx] = inFeatures[R[2 * ty[v]] * input_stride + tx];
        else
          I[ty[v]][tx] = 0;
        if (tx < KO and s + ty[v] < nHot)
          dO[ty[v]][tx] = dOutFeatures[R[2 * ty[v] + 1] * output_stride + tx];
        else
          dO[ty[v]][tx] = 0;
      }
      __syncthreads();
#pragma unroll
      for (int32_t k = 0; k < KO; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++)
          dI[v] += dO[ty[v]][k] * W[tx][k];
#pragma unroll
      for (int32_t k = 0; k < K; k++)
#pragma unroll
        for (int32_t v = 0; v < V; v++)
          dW[v] += I[k][ty[v]] * dO[k][tx];
      __syncthreads();
#pragma unroll
      for (int32_t v = 0; v < V; v++)
        if (tx < KI and s + ty[v] < nHot)
          dInFeatures[R[2 * ty[v]] * input_stride + tx] += dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int32_t v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
        atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}
template <typename T>
void dConvolution_forward2(cudaStream_t s, T *inFeatures, T *outFeatures, T *w,
                           int32_t *rulesIn, int32_t *rulesOut, int32_t nHot,
                           int32_t input_nPlanes, int32_t input_stride,
                           int32_t output_nPlanes, int32_t output_stride,
                           int32_t nGroups) {
  int32_t c = input_nPlanes * output_nPlanes * nGroups;
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int32_t K = 16;
    const int32_t V = 4;
    dConvolution_KMxKN_forward2<T, K, V>
        <<<dim3(128, (output_nPlanes + K - 1) / K, nGroups), dim3(K, K / V), 0,
           s>>>(inFeatures, outFeatures, w, rulesIn, rulesOut, nHot,
                input_nPlanes, input_stride, output_nPlanes, output_stride);

  } else {
    dConvolution_forward(s, inFeatures, outFeatures, w, rulesIn, rulesOut, nHot,
                         input_nPlanes, input_stride, output_nPlanes,
                         output_stride, nGroups);
  }
}
template <typename T>
void dConvolution_backward_dW2(cudaStream_t s, T *inFeatures, T *dInFeatures,
                               T *dOutFeatures, T *w, T *dw, int32_t *rulesIn,
                               int32_t *rulesOut, int32_t nHot,
                               int32_t input_nPlanes, int32_t input_stride,
                               int32_t output_nPlanes, int32_t output_stride,
                               int32_t nGroups) {
  int32_t c = input_nPlanes * output_nPlanes * nGroups;
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int32_t K = 16;
    const int32_t V = 4;
    dConvolution_KMxKN_backward_dW2<T, K, V>
        <<<dim3(128, (input_nPlanes + K - 1) / K, nGroups), dim3(K, K / V), 0,
           s>>>(inFeatures, dInFeatures, dOutFeatures, w, dw, rulesIn, rulesOut,
                nHot, input_nPlanes, input_stride, output_nPlanes,
                output_stride);
  } else {
    dConvolution_backward_dW(s, inFeatures, dInFeatures, dOutFeatures, w, dw,
                             rulesIn, rulesOut, nHot, input_nPlanes,
                             input_stride, output_nPlanes, output_stride,
                             nGroups);
  }
}
#undef TACC