import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class StreamingConv(nn.Module):

    def __init__(self):
        super().__init__()

        self.streaming = False
        self.carried = None

    def reset_stream_state(self, streaming: bool):
        self.streaming = streaming
        self.carried = None


class CausalConv1d(StreamingConv):
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
        if not self.streaming:
            return self.conv(F.pad(x, (self.left_padding, 0), mode=self.pad_mode))

        if self.carried is None:
            x = F.pad(x, (self.left_padding, 0), mode=self.pad_mode)
        else:
            x = torch.cat([self.carried, x], dim=-1)

        if self.left_padding:
            self.carried = x[..., x.shape[-1] - self.left_padding:].clone()

        return self.conv(x)


class CausalConvTranspose1d(StreamingConv):
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
        y = self.conv(x)

        if not self.right_trim:
            return y

        finished = y[..., : y.shape[-1] - self.right_trim]
        unfinished = y[..., y.shape[-1] - self.right_trim:]

        if not self.streaming:
            return finished

        if self.carried is not None:
            finished = finished.clone()
            finished[..., : self.right_trim] += self.carried

        bias = self.conv.bias

        if bias is None:
            self.carried = unfinished.clone()
        else:
            self.carried = unfinished - bias[None, :, None]

        return finished


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
