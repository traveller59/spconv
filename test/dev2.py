from cumm import tensorview as tv 

from cumm.tensorview import tvio
import numpy as np
from pathlib import Path


def main():
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    with open(Path(__file__).parent / "data" / "benchmark-pc.jarr", "wb") as f:
        f.write(tvio.dumps_jsonarray({
            "pc": data
        }).tobytes())
if __name__ == "__main__":
    main()