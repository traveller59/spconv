from .basic import bench_basic

import fire

def bench_me_basic(dtype_str: str):
    from spconv.benchmark.me import bench_me_basic
    return bench_me_basic(dtype_str)

def bench_torchsparse_basic(dtype_str: str):
    from spconv.benchmark.thsp import bench_torchsparse_basic
    return bench_torchsparse_basic(dtype_str)

if __name__ == "__main__":
    fire.Fire()
