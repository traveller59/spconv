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

import pickle 
from pathlib import Path 

from spconv.constants import SPCONV_DEBUG_SAVE_PATH

def spconv_save_debug_data(data):
    if SPCONV_DEBUG_SAVE_PATH:
        try:
            save_path = Path(SPCONV_DEBUG_SAVE_PATH)
            assert save_path.parent.exists(), "parent of SPCONV_DEBUG_SAVE_PATH must exist"
            with save_path.open("wb") as f:
                pickle.dump(data, f)
            print((f"spconv save debug data to {SPCONV_DEBUG_SAVE_PATH}, "
                    "you can submit issue with log and this debug data attached."))
        except Exception as e:
            print((f"spconv try to save debug data to {SPCONV_DEBUG_SAVE_PATH}, "
                    f"but failed with exception {e}. please check your SPCONV_DEBUG_SAVE_PATH"))

    else:
        print((f"SPCONV_DEBUG_SAVE_PATH not found, "
                "you can specify SPCONV_DEBUG_SAVE_PATH as debug data save path "
                "to save debug data which can be attached in a issue."))
