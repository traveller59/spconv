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

import pccm
from cumm.common import TensorView
from cumm.constants import CUMM_CPU_ONLY_BUILD
from spconv.csrc.sparse.cpu_core import OMPLib
from typing import List


class GatherCPU(pccm.Class):
    def __init__(self):
        super().__init__()
        if CUMM_CPU_ONLY_BUILD:
            self.add_dependency(OMPLib)
        self.add_dependency(TensorView)
        self.add_include("tensorview/parallel/all.h")

    @pccm.static_function
    def gather(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("inds", "tv::Tensor")

        code.raw(f"""
        // tv::check_shape(inds, {{out.dim(0)}});

        auto nhot = inds.dim(0);
        int channel = in.dim(1);
        tv::dispatch<float, double>(out.dtype(), [&](auto I){{
            auto indices_data = inds.data_ptr<const int>();
            using T = TV_DECLTYPE(I);
            T *buffer_data = out.data_ptr<T>();
            const T *features_data = in.data_ptr<const T>();
            tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){{
                for (int i = begin; i < end; i += step) {{
                    std::memcpy(buffer_data + i * channel,
                                features_data + indices_data[i] * channel,
                                sizeof(T) * channel);
                }}
            }});
        }});
        """)
        return code

    @pccm.static_function
    def scatter_add(self):
        code = pccm.FunctionCode()
        code.arg("out", "tv::Tensor")
        code.arg("in", "tv::Tensor")
        code.arg("inds", "tv::Tensor")
        code.raw(f"""
        // tv::check_shape(inds, {{in.dim(0)}});
        auto nhot = inds.dim(0);
        int channel = in.dim(1);
        tv::dispatch<float, double>(out.dtype(), [&](auto I){{
            using T = TV_DECLTYPE(I);
            auto indices_data = inds.data_ptr<const int>();
            const T *buffer_data = in.data_ptr<const T>();
            T *features_data = out.data_ptr<T>();
            const T *buf = in.data_ptr<const T>();
            T *out_ptr = out.data_ptr<T>();
            tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){{
                for (int i = begin; i < end; i += step) {{
                    buf = buffer_data + i * channel;
                    out_ptr = features_data + indices_data[i] * channel;
                    for (int j = 0; j < channel; ++j) {{
                        out_ptr[j] = out_ptr[j] + buf[j];
                    }}
                }}
            }});
        }});
        """)
        return code
