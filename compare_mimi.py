"""Load Kyutai's pretrained Mimi SEANet encoder weights into SimpleSEANetEncoder
and check that it produces the same output as Moshi's reference implementation."""

import sys
import types
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

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


def import_reference_encoder():
    # moshi/__init__.py pulls in the whole LM stack (sentencepiece, etc.), so register
    # empty stand-in packages and import only the SEANet module and its conv deps
    for name in ("moshi", "moshi.modules", "moshi.utils"):
        package = types.ModuleType(name)
        package.__path__ = [str(MOSHI_REPO_DIR / name.replace(".", "/"))]
        sys.modules[name] = package
    from moshi.modules.seanet import SEANetEncoder

    return SEANetEncoder


def load_encoder_state(checkpoint_path):
    # Mimi keys look like "encoder.model.1.block.1.conv.conv.weight"
    with safe_open(checkpoint_path, framework="pt") as f:
        return {
            key.removeprefix("encoder."): f.get_tensor(key).float()
            for key in f.keys()
            if key.startswith("encoder.")
        }


def to_simple_key(mimi_key):
    # "model.1.block.1.conv.conv.weight" -> "network.1.network.1.conv.weight"
    return (
        mimi_key.replace("model.", "network.", 1)
        .replace(".block.", ".network.")
        .replace(".conv.conv.", ".conv.")
    )


def main():
    checkpoint_path = hf_hub_download(HF_REPO, MIMI_CHECKPOINT)
    mimi_state = load_encoder_state(checkpoint_path)
    print(f"loaded {len(mimi_state)} encoder tensors from {MIMI_CHECKPOINT}")

    SEANetEncoder = import_reference_encoder()
    reference = SEANetEncoder(**MIMI_SEANET_KWARGS).eval()
    reference.load_state_dict(mimi_state)

    # checkpoint weights have weight norm already folded in, so load into plain convs
    ours = SimpleSEANetEncoder(use_weight_norm=False).eval()
    ours.load_state_dict({to_simple_key(k): v for k, v in mimi_state.items()})

    torch.manual_seed(0)
    t = torch.arange(24000 * 2) / 24000
    inputs = {
        "noise": 0.1 * torch.randn(2, 1, 48000),
        "sine 440Hz": 0.5 * torch.sin(2 * torch.pi * 440 * t).view(1, 1, -1),
    }

    all_close = True
    with torch.no_grad():
        for name, waveform in inputs.items():
            expected = reference(waveform)
            actual = ours(waveform)
            max_diff = (expected - actual).abs().max().item()
            close = torch.allclose(expected, actual, atol=1e-4, rtol=1e-4)
            all_close &= close
            print(
                f"{name:>11}: shape {tuple(actual.shape)}  "
                f"max |diff| {max_diff:.2e}  {'OK' if close else 'MISMATCH'}"
            )

    sys.exit(0 if all_close else 1)


if __name__ == "__main__":
    main()
