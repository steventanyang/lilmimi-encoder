from torch import nn


class ConvUpsample(nn.Module):
    """
    Mirror of ConvDownsample. Kyutai make this one channel-wise, so each
    of the 512 channels is upsampled by its own kernel.
    """

    def __init__(self, dimension=512, stride=2):
        super().__init__()

        kernel_size = 2 * stride

        # A transposed convolution cannot pad its input into the past, so
        # causality means dropping the tail it writes into the future.
        self.right_trim = kernel_size - stride

        self.conv = nn.ConvTranspose1d(
            dimension,
            dimension,
            kernel_size=kernel_size,
            stride=stride,
            groups=dimension,
            bias=False,
        )

    def forward(self, x):
        x = self.conv(x)
        return x[..., : x.shape[-1] - self.right_trim]
