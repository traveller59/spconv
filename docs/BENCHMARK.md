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

## Simple Benchmark

### Network Benchmark without batchnorm (F32/F16) in RTX 3080 Laptop GPU 150W

Network Code: test/benchmark.py

| F32/F16 | Spconv 1.x F32 (1080Ti) | Native| Implicit Gemm | Implicit Gemm Split Mask  |
| -------------- |:---------------------:|---------------------:|---------------------:| ---------------------:|
| Forward | 43ms     | 21.7ms/13.7ms    | 23.5ms/11.2ms      | 22ms/12.2ms      |
| Backward | 80ms    | 41.9ms/25.2ms    | 51.0ms/13.8ms      | 41.1ms/12.2ms      |

| F16 Forward | Native| Implicit Gemm | Implicit Gemm Split Mask  |
| -------------- |:---------------------:|---------------------:| ---------------------:|
| RTX 3080 Laptop 150W@1755MHz | 13.7ms     | 11.2ms    | 12.2ms      |
| RTX A6000 | 19.1ms    |  11.7ms   | 14.0ms      |
| TESLA V100 | 17.9ms    |  11.4ms   | 13.4ms      |
| A100 | 23.8ms    |  12.4ms   | 15.1ms      |

| F16 Backward | Native| Implicit Gemm | Implicit Gemm Split Mask  |
| -------------- |:---------------------:|---------------------:| ---------------------:|
| RTX 3080 Laptop 150W@1755MHz | 25.2ms     | 13.8ms    | 12.2ms      |
| RTX A6000       | 28.1ms     | 9.2ms     | 8.9ms      |
| TESLA V100 | 33.9ms    |  12.2ms   | 12.9ms      |
| A100 | 37.6ms    |  12.2ms   | 13.9ms      |

### Network Gemm Kernel Benchmark FP16 in RTX 3080 Laptop GPU

Network Code: test/benchmark.py

The network/input/profile code is same as above table.

This table only profile **fp16 gemm kernels** without output tensor create/clear overhead. this table show the performance upper bound of our algorithm.

| F16 |  Native| Implicit Gemm | Implicit Gemm Split Mask  |
| -------------- |:---------------------:|---------------------:| ---------------------:|
| Forward | 8.0ms    | 4.3ms      | 4.0ms      |

We can see that the implicit gemm is very fast, gemm only use 4.3ms/11.2ms in network forward. we can achieve better performance in TensorRT + Pure C++.

**NOTE** 
When you want to benchmark network in your laptop, don't forget to close all apps except terminals! Other apps will consume GPU resource and make kernels run slower.


## Comparsion with [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [torchsparse](https://github.com/mit-han-lab/torchsparse)

TODO