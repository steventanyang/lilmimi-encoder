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
        bias=True,
        use_weight_norm=True,
        pad_mode="constant",
    ):
        super().__init__()

        # Padding by kernel - stride makes the output exactly input / stride,
        # with every output frame depending only on samples at or before it.
        effective_kernel_size = dilation * (kernel_size - 1) + 1
        self.left_padding = effective_kernel_size - stride
        self.pad_mode = pad_mode

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            padding=0,
        )

        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0), mode=self.pad_mode)

        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        bias=True,
        use_weight_norm=True,
    ):
        super().__init__()

        # A transposed conv spreads each input frame over kernel_size outputs
        # at stride spacing, so the last kernel - stride outputs depend on
        # input that has not arrived. Causality means dropping them.
        self.right_trim = kernel_size - stride

        conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            padding=0,
        )

        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x):
        x = self.conv(x)

        return x[..., : x.shape[-1] - self.right_trim]


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
