import torch
import time


def main():

    arr = torch.randint(0, 130000, size=[130000]).to(torch.int32).cuda()
    arr2 = torch.randint(0, 130000, size=[130000]).to(torch.int32).cuda()

    torch.cuda.synchronize()
    ar = torch.arange(arr.shape[0]).cuda()

    t = time.time()
    for i in range(10):

        xx, indices = arr.sort()
        # thh = torch.empty_like(indices)
        xx2, indices2 = arr2.sort()

        # thh[indices] = ar
        torch.cuda.synchronize()
        print(time.time() - t)
        t = time.time()
    # print(indices[:10], thh[:10])
    a = torch.rand(130000, 27 * 32).cuda().float()
    b = torch.rand(27 * 32, 32).cuda().float()
    c = torch.rand(130000, 32).cuda().float()
    for i in range(10):
        torch.cuda.synchronize()
        t = time.time()
        torch.mm(a, b, out=c)
        # thh[indices] = ar
        torch.cuda.synchronize()
        print(time.time() - t)


if __name__ == "__main__":
    main()
