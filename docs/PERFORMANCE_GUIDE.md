<!--
 Copyright 2021 Yan Yan
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Spconv 2.x Performance Guide

## Short Guide

* If you train without Tensor Core (i.e. FP32 training), set all ```algo``` in convolution/maxpool to ```ConvAlgo.Native``` manually.
* If your GPU support Tensor Core, use FP16 (mixed precision training) if possible. 
* If you train with mixed precision training (use Tensor Core), you don't need to set algorithm manually.
* Currently fast algorithm only support kernel volume (prod of kernel size) <= 32, so don't use large kernel size.
* make sure your channel size is multiple of 8 when using fp16. multiple of 32 is better.

Network Benchmark without batchnorm (F32/F16) in RTX 3080 Laptop GPU

| F32/F16 | Spconv 1.x | Native| Implicit Gemm | Implicit Gemm Split Mask  |
| -------------- |:---------------------:|---------------------:|---------------------:| ---------------------:|
| Forward | 43ms     | 29ms/23ms    | 30ms/15ms      | 30ms/19ms      |
| Backward | 80ms    | 47ms/32ms    | 56ms/15ms      | 45ms/14ms      |

## Algorithm Overview

### Native Explicit (deprecated and removed in spconv 2.x)

native algorithm (explicit, no fused) is standard gather-gemm-scatter algorithm. Assume we compute 3x3 conv, We can split it to 9 of 1x1 conv which can be computed by matmul, then sum them to get final result.
For sparse convolution, we also do split-gemm-sum to calculate conv, but we need to collect data first because it's sparse.

### Native

Fused version of above algorithm. 1.5x-2x faster than non-fused version.

### Implicit Gemm

```Native``` algorithm do minimal mma (matrix multiply add), but it need to serialize IO. The pipeline of ```Native``` is gather-gemm-scatter-gather-gemm-scatter-...

```Implicit Gemm``` fuse all calculation to one kernel and perform overlapped gather-mma-scatter to save a lot of time. 

![Image Overlapped Gemm](https://raw.githubusercontent.com/NVIDIA/cutlass/master/media/images/software-pipeline.png)

In my test, ```Implicit Gemm``` is almost 2x faster than ```Native```.

### Implicit Gemm Split Mask

TODO

In my test, ```Implicit Gemm Split Mask``` is slightly faster than ```Implicit Gemm```, but the indice generation is greatly slower, so currently we use ```Implicit Gemm``` by default.