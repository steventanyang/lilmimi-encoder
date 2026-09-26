"""Load Kyutai's pretrained Mimi weights into our simplified SEANet encoder and
quantizer, and check they produce the same outputs as Moshi's reference implementation."""

import sys
import types
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from quantizer import SplitResidualVectorEncoder
from seanet import SimpleSEANetEncoder

MOSHI_REPO_DIR = Path.home() / "projects/moshi/moshi"
HF_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_CHECKPOINT = "tokenizer-e351c8d8-checkpoint125.safetensors"

MIMI_SEANET_KWARGS = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}

MIMI_QUANTIZER_KWARGS = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": 512,
    "output_dimension": 512,
}


def import_reference_modules():
    # moshi/__init__.py pulls in the whole LM stack (sentencepiece, etc.), so register
    # empty stand-in packages and import only the modules we compare against
    for name in ("moshi", "moshi.modules", "moshi.utils", "moshi.quantization"):
        package = types.ModuleType(name)
        package.__path__ = [str(MOSHI_REPO_DIR / name.replace(".", "/"))]
        sys.modules[name] = package
    from moshi.modules.seanet import SEANetEncoder
    from moshi.quantization.vq import SplitResidualVectorQuantizer

    return SEANetEncoder, SplitResidualVectorQuantizer


def load_state(checkpoint_path, prefix):
    with safe_open(checkpoint_path, framework="pt") as f:
        return {
            key.removeprefix(prefix): f.get_tensor(key).float()
            for key in f.keys()
            if key.startswith(prefix)
        }


def to_simple_encoder_key(mimi_key):
    # "model.1.block.1.conv.conv.weight" -> "network.1.network.1.conv.weight"
    return (
        mimi_key.replace("model.", "network.", 1)
        .replace(".block.", ".network.")
        .replace(".conv.conv.", ".conv.")
    )


def to_simple_quantizer_state(mimi_state):
    # "rvq_rest.vq.layers.3._codebook.embedding_sum" -> "acoustic_encoder.codebooks.3.embedding_sum"
    # output_proj is only used on the decode side, which we don't have
    return {
        key.replace("rvq_first.", "semantic_encoder.")
        .replace("rvq_rest.", "acoustic_encoder.")
        .replace(".vq.layers.", ".codebooks.")
        .replace("._codebook.", "."): value
        for key, value in mimi_state.items()
        if ".output_proj." not in key
    }


def report(name, ok, detail):
    print(f"{name:>22}: {detail}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    checkpoint_path = hf_hub_download(HF_REPO, MIMI_CHECKPOINT)
    encoder_state = load_state(checkpoint_path, "encoder.")
    quantizer_state = load_state(checkpoint_path, "quantizer.")
    print(
        f"loaded {len(encoder_state)} encoder + {len(quantizer_state)} quantizer "
        f"tensors from {MIMI_CHECKPOINT}"
    )

    SEANetEncoder, SplitResidualVectorQuantizer = import_reference_modules()

    ref_encoder = SEANetEncoder(**MIMI_SEANET_KWARGS).eval()
    ref_encoder.load_state_dict(encoder_state)
    # checkpoint weights have weight norm already folded in, so load into plain convs
    our_encoder = SimpleSEANetEncoder(use_weight_norm=False).eval()
    our_encoder.load_state_dict(
        {to_simple_encoder_key(k): v for k, v in encoder_state.items()}
    )

    ref_quantizer = SplitResidualVectorQuantizer(**MIMI_QUANTIZER_KWARGS).eval()
    ref_quantizer.load_state_dict(quantizer_state)
    our_quantizer = SplitResidualVectorEncoder().eval()
    our_quantizer.load_state_dict(to_simple_quantizer_state(quantizer_state))

    torch.manual_seed(0)
    t = torch.arange(24000 * 2) / 24000
    waveforms = {
        "noise": 0.1 * torch.randn(2, 1, 48000),
        "sine 440Hz": 0.5 * torch.sin(2 * torch.pi * 440 * t).view(1, 1, -1),
    }

    all_ok = True
    with torch.no_grad():
        print("\nSEANet encoder")
        latents = {"random latents": torch.randn(2, 512, 50)}
        for name, waveform in waveforms.items():
            expected = ref_encoder(waveform)
            actual = our_encoder(waveform)
            max_diff = (expected - actual).abs().max().item()
            ok = torch.allclose(expected, actual, atol=1e-4, rtol=1e-4)
            all_ok &= report(
                name, ok, f"shape {tuple(actual.shape)}  max |diff| {max_diff:.2e}"
            )
            latents[f"SEANet({name})"] = expected

        # real Mimi feeds the quantizer transformer + downsampled latents, which we
        # don't have yet; RVQ codes only depend on the input vectors, so random and
        # SEANet latents still exercise every codebook
        print("\nquantizer codes (32 codebooks)")
        for name, latent in latents.items():
            expected = ref_quantizer.encode(latent)
            actual = our_quantizer.encode(latent)
            matching = (expected == actual).float().mean().item()
            ok = torch.equal(expected, actual)
            all_ok &= report(
                name, ok, f"shape {tuple(actual.shape)}  {matching:.2%} codes match"
            )

            # Moshi runs with 8 codebooks; RVQ is sequential, so they're the first 8
            actual_8 = our_quantizer.encode(latent, num_codebooks=8)
            all_ok &= report(
                f"{name} [8]", torch.equal(expected[:, :8], actual_8),
                f"shape {tuple(actual_8.shape)}",
            )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
