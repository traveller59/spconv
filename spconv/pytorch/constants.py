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

import torch
try:
    remove_plus = torch.__version__.find("+")
    remove_dotdev = torch.__version__.find(".dev")

    PYTORCH_VERSION = torch.__version__
    if remove_plus != -1:
        PYTORCH_VERSION = torch.__version__[:remove_plus]
    if remove_dotdev != -1:
        PYTORCH_VERSION = torch.__version__[:remove_dotdev]

    PYTORCH_VERSION = list(map(int, PYTORCH_VERSION.split(".")))
except:
    # for unknown errors, just set a version
    PYTORCH_VERSION = [1, 8, 0]


if PYTORCH_VERSION >= [1, 6, 0]:
    TORCH_HAS_AMP = True
else:
    TORCH_HAS_AMP = False

def is_amp_enabled():
    if TORCH_HAS_AMP:
        return torch.is_autocast_enabled()
    else:
        return False 