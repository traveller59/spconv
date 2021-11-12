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

# Breaking changes in Spconv 2.x for spconv 1.x users

* ```spconv.xxx``` move to ```spconv.pytorch.xxx```, change all ```import spconv``` to ```import spconv.pytorch as spconv``` and ```from spconv.xxx import``` to ```from spconv.pytorch.xxx import```.
* ```use_hash``` and ```fused_bn``` in Sparse Convolution is removed, we only use hash table in 2.x.
* ```x.features = F.relu(x.features)``` now raise error. use ```x = x.replace_feature(F.relu(x.features))``` instead.
* weight layout has been changed to RSKC (native algorithm) or KRSC (implicit gemm), no longer RSCK (spconv 1.x). RS is kernel size, C is input channel, K is output channel.
* all util ops are removed (pillar scatter/nms/rbbox_iou...)
* VoxelGenerator has been replaced by ```spconv.pytorch.utils.PointToVoxel``` (torch API) or Point2VoxelGPU[1-4]d/Point2VoxelCPU[1-4]d (tv.Tensor API).
* spconv < 2.1 don't support CPU. spconv 2.1+ support cpu for debug usage.

* test spconv 1.x model in spconv 2.x: Firstly set environment variable before run program, Then set all ```algo``` in conv/pool to ```ConvAlgo.Native```. Linux: ```export SPCONV_FILTER_HWIO="1"```, Windows powershell: ```$Env:SPCONV_FILTER_HWIO = "1"```. **WARNING** test spconv 1.x model don't support implicit gemm algorithm.
