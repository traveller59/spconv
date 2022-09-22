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

from spconv.core_cc.csrc.utils.pcc import PointCloudCompress
from pathlib import Path 
import numpy as np 
from cumm import tensorview as tv 

def main():
    data = np.load(Path(__file__).parent.parent / "test" / "data" / "benchmark-pc.npz")
    pc = np.ascontiguousarray(data["pc"]).astype(np.float32)
    pc_encoded, order = PointCloudCompress.encode_with_order(tv.from_numpy(pc), tv.Tensor(), 0.01, 0.01, 0.01, PointCloudCompress.EncodeType.XYZ_8, True)
    print(pc.nbytes, pc_encoded.bytesize())
    pc_recover = PointCloudCompress.decode(pc_encoded)
    pc_reorder = pc[order.numpy_view()]
    error = pc_recover.numpy_view()[:, :3] - pc_reorder[:, :3]
    print("ERROR", np.abs(error).max())

if __name__ == "__main__":
    main()