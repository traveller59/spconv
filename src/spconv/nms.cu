// ------------------------------------------------------------------
// Deformable Convolutional Networks
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Modified from MATLAB Faster R-CNN
// (https://github.com/shaoqingren/faster_rcnn)
// ------------------------------------------------------------------

#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <spconv/reordering.cu.h>
#include <spconv/reordering.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/kernel_utils.h>
#include <tensorview/mp_helper.h>
#include <tensorview/tensorview.h>
#include <type_traits>
#include <utility/timer.h>

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
