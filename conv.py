import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from streaming import StreamingModule


class StreamingConv(StreamingModule):

    def __init__(self):
        super().__init__()

        self.carried = None

    def reset_stream_state(self, streaming: bool):
        super().reset_stream_state(streaming)
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

        self.effective_kernel_size = dilation * (kernel_size - 1) + 1
        self.left_padding = self.effective_kernel_size - stride
        self.stride = stride
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
        if x.shape[-1] == 0:
            return x.new_zeros(x.shape[0], self.conv.out_channels, 0)

        if not self.streaming:
            self.carried = None

        if self.carried is None:
            x = F.pad(x, (self.left_padding, 0), mode=self.pad_mode)
        else:
            x = torch.cat([self.carried, x], dim=-1)

        frames = (x.shape[-1] - self.effective_kernel_size) // self.stride + 1

        if frames < 1:
            self.carried = x.detach().clone()

            return x.new_zeros(x.shape[0], self.conv.out_channels, 0)

        self.carried = x[..., frames * self.stride:].detach().clone()

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
        if x.shape[-1] == 0:
            return x.new_zeros(x.shape[0], self.conv.out_channels, 0)

        y = self.conv(x)

        if not self.right_trim:
            return y

        finished = y[..., : y.shape[-1] - self.right_trim]
        unfinished = y[..., y.shape[-1] - self.right_trim:]

        if not self.streaming:
            self.carried = None

        if self.carried is not None:
            if finished.shape[-1] < self.right_trim:
                raise ValueError(
                    f"chunk of {x.shape[-1]} frames is too short to stream "
                    f"through a kernel {self.conv.kernel_size[0]} stride "
                    f"{self.conv.stride[0]} transposed convolution: it finishes "
                    f"{finished.shape[-1]} samples but owes {self.right_trim} "
                    f"to the previous chunk"
                )

            finished = finished.clone()
            finished[..., : self.right_trim] += self.carried

        bias = self.conv.bias

        if bias is None:
            self.carried = unfinished.detach().clone()
        else:
            self.carried = (unfinished - bias[None, :, None]).detach()

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
