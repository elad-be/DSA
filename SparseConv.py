import torch.nn as nn
import torch


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride= stride,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels),
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x, mask):
        x = x * mask
        x = self.conv(x)
        normalizer = 1 / (self.sparsity(mask) + 1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.relu(x)

        mask = self.max_pool(mask)

        return x, mask

