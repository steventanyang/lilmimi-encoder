"""
Moving between the SEANet frame rate and Mimi's token rate.

The encoder runs at 25 Hz and the quantizer at 12.5 Hz, so one stage has
to halve the sequence and its mirror has to double it again.
"""

from conv import CausalConv1d, CausalConvTranspose1d


class ConvDownsample(CausalConv1d):
    """
    25 Hz to 12.5 Hz. Kernel twice the stride, so each output frame sees
    the pair it replaces plus the boundary between them. Kyutai repeat the
    first sample rather than pad with zeros here.
    """

    def __init__(self, dimension=512, stride=2):
        super().__init__(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            bias=False,
            use_weight_norm=False,
            pad_mode="replicate",
        )


class ConvUpsample(CausalConvTranspose1d):
    """
    12.5 Hz back to 25 Hz. Unlike the downsample this one is channel-wise:
    512 groups, so every channel is upsampled by its own kernel.
    """

    def __init__(self, dimension=512, stride=2):
        super().__init__(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
            use_weight_norm=False,
        )
