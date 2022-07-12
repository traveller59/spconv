from spconv.pytorch.cppcore import TorchAllocator
print(1)

from spconv.core_cc.csrc.sparse.all import SpconvOps
import torch 
print(2)
if __name__ == "__main__":
    alloc = TorchAllocator(torch.device("cuda:0"))

    SpconvOps.test_allocator(alloc)
