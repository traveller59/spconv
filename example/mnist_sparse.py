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

from __future__ import print_function
import argparse
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import contextlib
import torch.cuda.amp


@contextlib.contextmanager
def identity_ctx():
    yield


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = spconv.SparseSequential(
            nn.BatchNorm1d(1),
            spconv.SubMConv2d(1, 32, 3, 1),
            nn.ReLU(),
            spconv.SubMConv2d(32, 64, 3, 1),
            nn.ReLU(),
            spconv.SparseMaxPool2d(2, 2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(14 * 14 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x: torch.Tensor):
        # x: [N, 28, 28, 1], must be NHWC tensor
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 1))
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.net(x_sp)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    amp_ctx = contextlib.nullcontext()
    if args.fp16:
        amp_ctx = torch.cuda.amp.autocast()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with amp_ctx:
            output = model(data)
            loss = F.nll_loss(output, target)
            scale = 1.0
            if args.fp16:
                assert loss.dtype is torch.float32
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                # scaler.unscale_(optim)

                # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                # You may use the same value for max_norm here as you would without gradient scaling.
                # torch.nn.utils.clip_grad_norm_(models[0].net.parameters(), max_norm=0.1)

                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                scale = scaler.get_scale()
            else:
                loss.backward()
                optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    amp_ctx = contextlib.nullcontext()
    if args.fp16:
        amp_ctx = torch.cuda.amp.autocast()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            with amp_ctx:

                output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help='For mixed precision training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # here we remove norm to get sparse tensor with lots of zeros
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # here we remove norm to get sparse tensor with lots of zeros
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
