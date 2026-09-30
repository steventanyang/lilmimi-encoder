import math

import torch.nn.functional as F
from torch import nn

from conv import CausalConv1d, ResidualBlock


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
        # A trailing partial frame would be dropped by the downsampling
        # rather than contributing, so it is padded out to a whole one.
        extra = -waveform.shape[-1] % self.hop_length
        waveform = F.pad(waveform, (0, extra))

        return self.network(waveform)
