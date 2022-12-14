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

### Network Benchmark without batchnorm (TF32/F16) in Different GPUs

Basic (120k voxels): ```python -m spconv.benchmark bench_basic f16``` and ```python -m spconv.benchmark bench_basic tf32```

| GPUs | F16-Forward | F16-Backward | TF32-Forward  | TF32-Backward |
| -------------- |:---------------------:|---------------------:|---------------------:| ---------------------:|
| T4 | 18.74     | 25.51    | N/A      | N/A      |
| RTX 3080 Laptop (150W) | 8.2    | 11.51    | 15.04      | 26.90      |
| A100 | 13.02    | 12.43    | 12.35      | 14.93      |
| RTX 3090 | 11.84    | 11.84    | 13.23      | 15.79      |
| RTX A6000 | 11.11    | 8.97    | 12.30      | 12.79      |
| TESLA V100-32G | 15.55    | 14.90    | N/A       | N/A      |
| TESLA V100-16G | 10.61    | 13.91    | N/A       | N/A      |
| RTX 4090 | 7.37    | 6.87    | 8.76      | 9.16      |

Large (900k voxels): ```python -m spconv.benchmark bench_large f16``` and ```python -m spconv.benchmark bench_large tf32```

| GPUs | F16-Forward | F16-Backward | TF32-Forward  | TF32-Backward |
| -------------- |:---------------------:|---------------------:|---------------------:| ---------------------:|
| T4 | 128.7     | 203.3    | N/A      | N/A      |
| RTX 3080 Laptop (150W) | 43.15    | 74.57    | 84.65      | 165.19      |
| A100 | 19.85    | 31.24    | 29.58      | 55.63      |
| RTX 3090 | 27.83    | 40.45    | 44.51      | 73.17      |
| RTX A6000 | 28.62    | 39.86    | 45.43      | 74.11      |
| TESLA V100-32G | 50.37    | 72.99   | N/A      | N/A      |
| TESLA V100-16G | 38.65    | 61.47    | N/A       | N/A      |
| RTX 4090 | 13.93  | 22.92    | 22.58      | 41.49      |


**NOTE** 
When you want to benchmark network in your laptop, don't forget to close all apps except terminals! Other apps will consume GPU resource and make kernels run slower.


