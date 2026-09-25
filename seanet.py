import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        use_weight_norm=True,
    ):
        super().__init__()

        effective_kernel_size = dilation * (kernel_size - 1) + 1
        self.left_padding = effective_kernel_size - stride

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
        )

        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, use_weight_norm=True):
        super().__init__()

        hidden_channels = channels // 2

        self.network = nn.Sequential(
            nn.ELU(),
            CausalConv1d(
                channels,
                hidden_channels,
                kernel_size=3,
                dilation=dilation,
                use_weight_norm=use_weight_norm,
            ),
            nn.ELU(),
            CausalConv1d(
                hidden_channels,
                channels,
                kernel_size=1,
                use_weight_norm=use_weight_norm,
            ),
        )

    def forward(self, x):
        return x + self.network(x)


class SimpleSEANetEncoder(nn.Module):
    strides = (4, 5, 6, 8)
    hop_length = math.prod(strides)

    def __init__(self, use_weight_norm=True):
        super().__init__()

        wn = use_weight_norm

        # converts raw waveform : [B, 1, T] -> [B, 512, T/960]
        self.network = nn.Sequential(
            CausalConv1d(1, 64, kernel_size=7, use_weight_norm=wn),

            ResidualBlock(64, dilation=1, use_weight_norm=wn),
            nn.ELU(),
            CausalConv1d(64, 128, kernel_size=8, stride=4, use_weight_norm=wn),

            ResidualBlock(128, dilation=1, use_weight_norm=wn),
            nn.ELU(),
            CausalConv1d(128, 256, kernel_size=10, stride=5, use_weight_norm=wn),

            ResidualBlock(256, dilation=1, use_weight_norm=wn),
            nn.ELU(),
            CausalConv1d(256, 512, kernel_size=12, stride=6, use_weight_norm=wn),

            ResidualBlock(512, dilation=1, use_weight_norm=wn),
            nn.ELU(),
            CausalConv1d(512, 1024, kernel_size=16, stride=8, use_weight_norm=wn),

            # project down to the latent dimension the transformer/quantizer expect
            nn.ELU(),
            CausalConv1d(1024, 512, kernel_size=3, use_weight_norm=wn),
        )

    def forward(self, waveform):
        extra = -waveform.shape[-1] % self.hop_length
        waveform = F.pad(waveform, (0, extra))
        return self.network(waveform)
