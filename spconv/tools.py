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

from typing import Dict
from spconv.cppconstants import CPU_ONLY_BUILD
import contextlib
from spconv.utils import nullcontext
if not CPU_ONLY_BUILD:
    from cumm.tensorview import CUDAKernelTimer as _CUDAKernelTimer


class CUDAKernelTimer:
    def __init__(self, enable: bool = True) -> None:
        self.enable = enable and not CPU_ONLY_BUILD
        if self.enable:
            self._timer = _CUDAKernelTimer(enable)
        else:
            self._timer = None

    @contextlib.contextmanager
    def _namespace(self, name: str):
        assert self._timer is not None
        self._timer.push(name)
        try:
            yield
        finally:
            self._timer.pop()

    @contextlib.contextmanager
    def _record(self, name: str, stream: int = 0):
        assert self._timer is not None
        self._timer.push(name)
        try:
            self._timer.insert_pair("", "start", "stop")
            self._timer.record("start", stream)
            yield
            self._timer.record("stop", stream)
        finally:
            self._timer.pop()

    def namespace(self, name: str):
        if self.enable:
            return self._namespace(name)
        else:
            return nullcontext()

    def record(self, name: str, stream: int = 0):
        if self.enable:
            return self._record(name, stream)
        else:
            return nullcontext()

    def get_all_pair_time(self) -> Dict[str, float]:
        if self.enable:
            assert self._timer is not None
            return self._timer.get_all_pair_duration()
        else:
            return {}

    @staticmethod
    def collect_by_name(name: str, res: Dict[str, float]):
        filtered_res: Dict[str, float] = {}
        for k, v in res.items():
            k_split = k.split(".")
            if name in k_split:
                filtered_res[k] = v
        return filtered_res
