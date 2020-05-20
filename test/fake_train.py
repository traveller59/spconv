import time
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import distributed, nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import spconv
from spconv.test_utils import generate_sparse_data


class FakeSparseDataset(Dataset):
    def __len__(self):
        return 500

    def __getitem__(self, idx):
        data_ranges = {
            0: [-1, 1],
            1: [0, 2],
            2: [-2, 0],
            3: [-2, -2],
        }
        l = np.random.randint(0, 4, size=[2])

        data = generate_sparse_data([16, 64, 64], [16 * 64 * 64 // 2],
                                    3,
                                    data_range=data_ranges[l[0]],
                                    with_dense=False)
        data2 = generate_sparse_data([16, 64, 64], [16 * 64 * 64 // 2],
                                     3,
                                     data_range=data_ranges[l[1]],
                                     with_dense=False)

        features = np.ascontiguousarray(data["features"]).astype(np.float32)
        indices = np.ascontiguousarray(
            data["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features2 = np.ascontiguousarray(data2["features"]).astype(np.float32)
        indices2 = np.ascontiguousarray(
            data2["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features = np.ascontiguousarray(np.concatenate([features, features2]))
        indices = np.ascontiguousarray(np.concatenate([indices, indices2]))
        return features, indices, l


class FakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3,
                              8,
                              3,
                              indice_key="subm1",
                              padding=1,
                              use_hash=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            spconv.SparseConv3d(8, 16, 3, stride=2, padding=1, use_hash=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            spconv.SubMConv3d(16,
                              16,
                              3,
                              indice_key="subm2",
                              padding=1,
                              use_hash=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            spconv.SparseConv3d(16, 32, 3, stride=2, padding=1,
                                use_hash=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            spconv.SubMConv3d(32,
                              32,
                              3,
                              indice_key="subm3",
                              padding=1,
                              use_hash=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            spconv.SparseConv3d(32, 64, 3, stride=2, padding=1,
                                use_hash=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64,
                              64,
                              3,
                              indice_key="subm4",
                              padding=1,
                              use_hash=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.ToDense()  # [64, 2, 8, 8]
        )
        self.linear = nn.Linear(64 * 2 * 8 * 8, 4)

    def forward(self, features, indices):
        indices = indices.int()
        x = spconv.SparseConvTensor(features, indices, [16, 64, 64], 2)
        x = self.net(x)
        x = x.view(2, -1)
        x = self.linear(x)
        return x


def run():
    np.random.seed(50051)
    ds = FakeSparseDataset()
    device = torch.device('cuda')
    model = FakeClassifier()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in tqdm.tqdm(list(range(100))):
        # for j in range(4):
        #     features, indices, label = ds[(i * 4 + j) % len(ds)]

        features, indices, label = ds[i % len(ds)]
        features_t = torch.from_numpy(features)
        indices_t = torch.from_numpy(indices)
        features_t = features_t.to(device)
        indices_t = indices_t.to(device)
        target = torch.from_numpy(label).to(device)
        output = model(features_t, indices_t)
        # print(output.shape)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def dev():
    ds = FakeSparseDataset()
    for i in range(10):
        features, indices, label = ds[i]
        print(indices[:10])
    features_t = torch.from_numpy(features.astype(np.float32)).cuda()
    indices_t = torch.from_numpy(indices.astype(np.int32)).cuda()
    net = FakeClassifier().cuda()
    net(features_t, indices_t)


def main():
    run()


if __name__ == "__main__":
    fire.Fire(main)
