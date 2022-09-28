<!--
 Copyright 2022 Yan Yan
 
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

# Common Problems

## the provided PTX was compiled with an unsupported toolchain

Update your GPU driver or downgrad your spconv/cumm cuda version.

## CUDA kernel launch blocks must be positive, but got N= 0

Your coordinates generate nothing with some conv params. Modify your conv params to make sure all input points have at least one output point.

Example:

Conv Params:
```spatial shape=[8, 200, 200],ksize=3,stride=2,padding=[0, 1, 1],dilation=1```
Coordinates:
```
[[0, 7, 153, 142]]
```

The convolution in z axis will drop ALL points in z == 7. change the padding-z to solve this problem.

