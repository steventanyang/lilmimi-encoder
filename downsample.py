import torch.nn.functional as F
from torch import nn


class ConvDownsample(nn.Module):

    def __init__(self, dimension=512, stride=2):
        super().__init__()

        kernel_size = 2 * stride
        self.left_padding = kernel_size - stride

        self.conv = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0), mode="replicate")
        return self.conv(x)
