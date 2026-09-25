import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()

        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()

        hidden_channels = channels // 2

        self.network = nn.Sequential(
            nn.ELU(),
            CausalConv1d(
                channels,
                hidden_channels,
                kernel_size=3,
                dilation=dilation,
            ),
            nn.ELU(),
            CausalConv1d(
                hidden_channels,
                channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return x + self.network(x)


class SimpleSEANetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # converts raw waveform : [B, 1, T] -> [B, 1024, ~T/960]
        self.network = nn.Sequential(
            CausalConv1d(1, 64, kernel_size=7),

            ResidualBlock(64, dilation=1),
            nn.ELU(),
            CausalConv1d(64, 128, kernel_size=7, stride=4),

            ResidualBlock(128, dilation=1),
            nn.ELU(),
            CausalConv1d(128, 256, kernel_size=10, stride=5),

            ResidualBlock(256, dilation=1),
            nn.ELU(),
            CausalConv1d(256, 512, kernel_size=12, stride=6),

            ResidualBlock(512, dilation=1),
            nn.ELU(),
            CausalConv1d(512, 1024, kernel_size=16, stride=8),
        )

    def forward(self, waveform):
        return self.network(waveform)
