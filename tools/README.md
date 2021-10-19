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

## How to debug manylinux build

```Bash
docker run --rm -it -e PLAT=manylinux2014_x86_64 -v `pwd`:/io -v $HOME:/myhome scrin/manylinux2014-cuda:cu114-devel bash
/io/tools/build-wheels.sh

```

## Windows C++ Tips

* cuda attributes such as ```__device__``` must put before return type. when you see ```warning: __declspec attributes ignored```, this means ```__device__``` is ignored because you put it after return type, then cause error.

