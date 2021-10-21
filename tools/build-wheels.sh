#!/bin/bash
# Copyright 2021 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e -u -x

function repair_wheel {
    wheel="$1"
    outpath="$2"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w "$outpath"
    fi
}
gcc -v
export SPCONV_DISABLE_JIT="1"
export CUMM_CUDA_ARCH_LIST="all"
# export SPCONV_PYTHON_LIST="3.7;3.8;3.9;3.10"
# Compile wheels, we only support 3.6-3.10.
# "/opt/python/cp36-cp36m/bin/pip" wheel /io/ --no-deps -w /io/wheelhouse_tmp

for PYVER in ${SPCONV_PYTHON_LIST//;/ }
do
    PYVER2=`echo "$PYVER" | sed 's/\.//'`
    PYVER_CP="cp$PYVER2-cp$PYVER2"
    if [ "$PYVER2" = "36" ]; then
        PYVER_CP="cp$PYVER2-cp${PYVER2}m"
    fi
    if [ "$PYVER2" = "37" ]; then
        PYVER_CP="cp$PYVER2-cp${PYVER2}m"
    fi
    "/opt/python/$PYVER_CP/bin/pip" wheel /io/  -v --no-deps -w /io/wheelhouse_tmp
done

# Bundle external shared libraries into the wheels
for whl in /io/wheelhouse_tmp/*.whl; do
    repair_wheel "$whl" /io/dist
done

rm -rf /io/wheelhouse_tmp