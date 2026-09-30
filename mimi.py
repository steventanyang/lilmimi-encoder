import torch
import torch.nn.functional as F
from torch import nn

from quantizer import SplitResidualVectorEncoder
from resample import ConvDownsample, ConvUpsample
from seanet import SimpleSEANetEncoder
from seanet_decoder import SimpleSEANetDecoder
from transformer import MimiTransformer


class MimiEncoder(nn.Module):
    """
    Waveform to Moshi's audio tokens.
    """

    def __init__(self, use_weight_norm: bool = False):
        super().__init__()

        self.seanet = SimpleSEANetEncoder(use_weight_norm=use_weight_norm)
        self.transformer = MimiTransformer()
        self.downsample = ConvDownsample()
        self.quantizer = SplitResidualVectorEncoder()

    @property
    def frame_size(self) -> int:
        return self.seanet.hop_length * self.downsample.conv.stride[0]

    def forward(
        self,
        waveform: torch.Tensor,
        num_codebooks: int | None = None,
    ) -> torch.Tensor:
        """
        waveform: [B, 1, samples] at 24 kHz
        returns: [B, num_codebooks, samples / 1920]
        For Moshi: 8 codebooks at 12.5 Hz.
        """
        # A trailing partial frame would be dropped by the downsample,
        # so it is padded out to a whole one.
        remainder = -waveform.shape[-1] % self.frame_size
        waveform = F.pad(waveform, (0, remainder))

        # [B, 1, S] -> [B, 512, S / 960], i.e. 25 Hz
        latents = self.seanet(waveform)

        latents = self.transformer(latents)
        # 25 Hz -> 12.5 Hz
        latents = self.downsample(latents)

        return self.quantizer.encode(latents, num_codebooks=num_codebooks)


class MimiDecoder(nn.Module):
    """
    Moshi's audio tokens back to a waveform.

    Every stage undoes one of the encoder's, in reverse order.
    """

    def __init__(self, use_weight_norm: bool = False):
        super().__init__()

        self.quantizer = SplitResidualVectorEncoder()
        self.upsample = ConvUpsample()
        self.transformer = MimiTransformer()
        self.seanet = SimpleSEANetDecoder(use_weight_norm=use_weight_norm)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, K, T]
        returns: [B, 1, T * 1920] at 24 kHz
        """
        # [B, K, T] -> [B, 512, T], still 12.5 Hz
        latents = self.quantizer.decode(codes)

        # 12.5 Hz -> 25 Hz
        latents = self.upsample(latents)

        latents = self.transformer(latents)

        # [B, 512, T] -> [B, 1, T * 960]
        return self.seanet(latents)
