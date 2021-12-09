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

* If you train without Tensor Core (i.e. FP32 training or FP16 training for Pascal or older GPUS), set all ```algo``` in convolution/maxpool to ```ConvAlgo.Native``` manually. Default Algorithm is ```ConvAlgo.MaskImplicitGemm```, which is **SLOWER** than ```ConvAlgo.Native``` when use float32. this will be fixed in spconv 2.2.
* If your GPU support Tensor Core, use FP16 (mixed precision training) if possible. 
* If you train with mixed precision training (use Tensor Core), you don't need to set algorithm manually.
* Currently fast algorithm only support kernel volume (prod of kernel size) <= 32, so don't use large kernel size.
* make sure your channel size is multiple of 8 when using fp16. multiple of 32 is better.
* spconv 2.x in Windows 10 is 1.5x~2x slower than Linux. use Linux if possible.

See [benchmark](BENCHMARK.md) for more performance details of different algorithms.
