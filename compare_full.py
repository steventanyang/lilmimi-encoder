"""Check the assembled MimiEncoder and MimiDecoder against Kyutai's reference Mimi
model: same waveform in, identical codes out, identical waveform back."""

import sys
import types
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from checkpoint import load_mimi_decoder, load_mimi_encoder

MOSHI_REPO_DIR = Path.home() / "projects/moshi/moshi"
HF_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_CHECKPOINT = "tokenizer-e351c8d8-checkpoint125.safetensors"


SEANET_KWARGS = {
    "channels": 1, "dimension": 512, "causal": True, "n_filters": 64,
    "n_residual_layers": 1, "activation": "ELU", "compress": 2,
    "dilation_base": 2, "disable_norm_outer_blocks": 0, "kernel_size": 7,
    "residual_kernel_size": 3, "last_kernel_size": 3, "norm": "none",
    "pad_mode": "constant", "ratios": [8, 6, 5, 4], "true_skip": True,
}
TRANSFORMER_KWARGS = {
    "d_model": 512, "num_heads": 8, "num_layers": 8, "causal": True,
    "layer_scale": 0.01, "context": 250, "conv_layout": True,
    "max_period": 10000, "gating": "none", "norm": "layer_norm",
    "positional_embedding": "rope", "dim_feedforward": 2048,
    "input_dimension": 512, "output_dimensions": [512],
}
QUANTIZER_KWARGS = {
    "dimension": 256, "n_q": 32, "bins": 2048,
    "input_dimension": 512, "output_dimension": 512,
}


def build_reference_mimi(checkpoint_path):
    # moshi/__init__.py and moshi/models/__init__.py pull in the LM stack
    # (sentencepiece, etc.), so register empty stand-ins for those two and
    # let the rest import normally
    for name in ("moshi", "moshi.models"):
        package = types.ModuleType(name)
        package.__path__ = [str(MOSHI_REPO_DIR / name.replace(".", "/"))]
        sys.modules[name] = package

    from safetensors.torch import load_file
    from moshi.models.compression import MimiModel
    from moshi.modules.seanet import SEANetDecoder, SEANetEncoder
    from moshi.modules.transformer import ProjectedTransformer
    from moshi.quantization import SplitResidualVectorQuantizer

    encoder = SEANetEncoder(**SEANET_KWARGS)
    model = MimiModel(
        encoder,
        SEANetDecoder(**SEANET_KWARGS),
        SplitResidualVectorQuantizer(**QUANTIZER_KWARGS),
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        encoder_frame_rate=24000 / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=ProjectedTransformer(**TRANSFORMER_KWARGS),
        decoder_transformer=ProjectedTransformer(**TRANSFORMER_KWARGS),
    )
    model.load_state_dict(load_file(checkpoint_path))
    model.set_num_codebooks(8)

    return model.eval()


def main():
    checkpoint_path = hf_hub_download(HF_REPO, MIMI_CHECKPOINT)

    reference = build_reference_mimi(checkpoint_path)

    encoder = load_mimi_encoder(checkpoint_path)
    decoder = load_mimi_decoder(checkpoint_path)

    torch.manual_seed(0)
    t = torch.arange(24000 * 2) / 24000
    inputs = {
        "noise 1s": 0.1 * torch.randn(1, 1, 24000),
        "noise 2s": 0.1 * torch.randn(2, 1, 48000),
        "sine 440Hz": 0.5 * torch.sin(2 * torch.pi * 440 * t).view(1, 1, -1),
        "silence": torch.zeros(1, 1, 24000),
    }

    everything_matched = True
    with torch.no_grad():
        for name, waveform in inputs.items():
            expected = reference.encode(waveform)
            actual = encoder(waveform, num_codebooks=8)

            codes_equal = torch.equal(expected, actual)

            # Feed both decoders the same codes, so this isolates decode.
            expected_audio = reference.decode(actual)
            actual_audio = decoder(actual)

            drift = (expected_audio - actual_audio).abs().max().item()
            audio_close = drift < 1e-4

            everything_matched &= codes_equal and audio_close

            print(
                f"{name:>11}: codes {tuple(actual.shape)} "
                f"{'OK' if codes_equal else 'MISMATCH'}   "
                f"audio {tuple(actual_audio.shape)} "
                f"max |diff| {drift:.2e} {'OK' if audio_close else 'MISMATCH'}"
            )

    sys.exit(0 if everything_matched else 1)


if __name__ == "__main__":
    main()
