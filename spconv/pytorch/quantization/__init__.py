# Copyright 2022 Yan Yan
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

from .backend_cfg import (get_spconv_backend_config,
                          get_spconv_prepare_custom_config,
                          get_spconv_convert_custom_config,
                          prepare_spconv_torch_inference)
from .fake_q import (get_default_spconv_trt_ptq_qconfig,
                     get_default_spconv_trt_qat_qconfig,
                     get_default_spconv_qconfig_mapping)
from .qmapping import (get_spconv_fmod_to_qat_mapping,
                       get_spconv_qat_to_static_mapping)
from .core import quantize_per_tensor

from .graph import remove_conv_add_dq, transform_qdq