// ------------------------------------------------------------------
// Deformable Convolutional Networks
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Modified from MATLAB Faster R-CNN
// (https://github.com/shaoqingren/faster_rcnn)
// ------------------------------------------------------------------
#include <cuda_runtime.h>
#include <iostream>
#include <spconv/nms_gpu.h>
#include <vector>

#define CUDA_CHECK(condition)                                                  \
  /* Code block avoids redefinition of cudaError_t error */                    \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      std::cout << cudaGetErrorString(error) << std::endl;                     \
    }                                                                          \
  } while (0)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename DType>
__device__ inline DType devIoU(DType const *const a, DType const *const b) {
  DType left = max(a[0], b[0]), right = min(a[2], b[2]);
  DType top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  DType width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  DType interS = width * height;
  DType Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  DType Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

template <typename DType, int BLOCK_THREADS>
__global__ void nms_kernel(const int n_boxes, const DType nms_overlap_thresh,
                           const DType *dev_boxes,
                           unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size = min(n_boxes - row_start * BLOCK_THREADS, BLOCK_THREADS);
  const int col_size = min(n_boxes - col_start * BLOCK_THREADS, BLOCK_THREADS);

  __shared__ DType block_boxes[BLOCK_THREADS * 5];
  if (threadIdx.x < col_size) {
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      block_boxes[threadIdx.x * 5 + i] =
          dev_boxes[(BLOCK_THREADS * col_start + threadIdx.x) * 5 + i];
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = BLOCK_THREADS * row_start + threadIdx.x;
    const DType *cur_box = dev_boxes + cur_box_idx * 5;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (int i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, BLOCK_THREADS);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

template <typename DType, int BLOCK_THREADS>
int _nms_gpu(int *keep_out, const DType *boxes_host, int boxes_num,
             int boxes_dim, DType nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  DType *boxes_dev = NULL;
  unsigned long long *mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, BLOCK_THREADS);

  CUDA_CHECK(cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(DType)));
  CUDA_CHECK(cudaMemcpy(boxes_dev, boxes_host,
                        boxes_num * boxes_dim * sizeof(DType),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, BLOCK_THREADS), DIVUP(boxes_num, BLOCK_THREADS));
  dim3 threads(BLOCK_THREADS);
  nms_kernel<DType, BLOCK_THREADS>
      <<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / BLOCK_THREADS;
    int inblock = i % BLOCK_THREADS;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
  return num_to_keep;
}

// template<>
template int _nms_gpu<float, threadsPerBlock>(int *keep_out,
                                              const float *boxes_host,
                                              int boxes_num, int boxes_dim,
                                              float nms_overlap_thresh,
                                              int device_id);
// template<>
template int _nms_gpu<double, threadsPerBlock>(int *keep_out,
                                               const double *boxes_host,
                                               int boxes_num, int boxes_dim,
                                               double nms_overlap_thresh,
                                               int device_id);