import math

from torch import nn

from conv import CausalConv1d, CausalConvTranspose1d, ResidualBlock


class SimpleSEANetDecoder(nn.Module):
    """
    Mirror of SimpleSEANetEncoder: [B, 512, T] -> [B, 1, T * 960].

    The encoder interleaves conv, block, activation; going the other way
    the activation comes first and the block follows the upsampling.
    """

    strides = (8, 6, 5, 4)
    hop_length = math.prod(strides)

    def __init__(self, use_weight_norm=True):
        super().__init__()

        wn = use_weight_norm

        self.network = nn.Sequential(
            CausalConv1d(512, 1024, kernel_size=7, use_weight_norm=wn),

            nn.ELU(),
            CausalConvTranspose1d(1024, 512, kernel_size=16, stride=8, use_weight_norm=wn),
            ResidualBlock(512, dilation=1, use_weight_norm=wn),

            nn.ELU(),
            CausalConvTranspose1d(512, 256, kernel_size=12, stride=6, use_weight_norm=wn),
            ResidualBlock(256, dilation=1, use_weight_norm=wn),

            nn.ELU(),
            CausalConvTranspose1d(256, 128, kernel_size=10, stride=5, use_weight_norm=wn),
            ResidualBlock(128, dilation=1, use_weight_norm=wn),

            nn.ELU(),
            CausalConvTranspose1d(128, 64, kernel_size=8, stride=4, use_weight_norm=wn),
            ResidualBlock(64, dilation=1, use_weight_norm=wn),

            # back down to a single audio channel
            nn.ELU(),
            CausalConv1d(64, 1, kernel_size=3, use_weight_norm=wn),
        )

    def forward(self, latents):
        return self.network(latents)
