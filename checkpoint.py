"""
Loading Kyutai's pretrained Mimi weights into this implementation.

Everything here is about their checkpoint's naming, not about the model,
so the modules themselves stay free of it.
"""

import torch
from safetensors import safe_open

from mimi import MimiDecoder, MimiEncoder

# Mimi keeps the feed-forward flat on the layer and uses torch's own
# attention names.
TRANSFORMER_RENAMES = {
    ".norm1.": ".ln1.",
    ".norm2.": ".ln2.",
    ".self_attn.": ".attn.",
    ".linear1.": ".mlp.up.",
    ".linear2.": ".mlp.down.",
}


def convert_transformer_key(key: str) -> str:
    """
    "layers.0.self_attn.out_proj.weight" -> "layers.0.attn.out_proj.weight"
    """
    for old, new in TRANSFORMER_RENAMES.items():
        key = key.replace(old, new)

    return key


def convert_seanet_key(key: str) -> str:
    """
    "model.1.block.1.conv.conv.weight" -> "network.1.network.1.conv.weight"

    The reference wraps every convolution in a norm layer and a streaming
    layer, which this implementation folds into one module.
    """
    return (
        key.replace("model.", "network.", 1)
        .replace(".block.", ".network.")
        .replace(".conv.conv.", ".conv.")
        .replace(".convtr.convtr.", ".conv.")
    )


def convert_quantizer_key(key: str) -> str:
    """
    Shared by both directions: the codebooks are the same objects.
    """
    return (
        key.replace("rvq_first.", "semantic_encoder.", 1)
        .replace("rvq_rest.", "acoustic_encoder.", 1)
        .replace("vq.layers.", "codebooks.")
        .replace("._codebook.", ".")
    )


def convert_encoder_key(key: str) -> str | None:
    """
    Rename one checkpoint key onto MimiEncoder, or return None when it
    belongs to the decode path.
    """
    if key.startswith("encoder."):
        return "seanet." + convert_seanet_key(key.removeprefix("encoder."))

    if key.startswith("encoder_transformer.transformer."):
        inner = key.removeprefix("encoder_transformer.transformer.")
        return "transformer." + convert_transformer_key(inner)

    if key.startswith("downsample."):
        return "downsample.conv.weight"

    if key.startswith("quantizer."):
        return convert_quantizer_key(key)

    return None


def convert_decoder_key(key: str) -> str | None:
    """
    Rename one checkpoint key onto MimiDecoder, or return None when it
    belongs to the encode path.
    """
    if key.startswith("decoder."):
        return "seanet." + convert_seanet_key(key.removeprefix("decoder."))

    if key.startswith("decoder_transformer.transformer."):
        inner = key.removeprefix("decoder_transformer.transformer.")
        return "transformer." + convert_transformer_key(inner)

    if key.startswith("upsample."):
        return "upsample.conv.weight"

    if key.startswith("quantizer."):
        return convert_quantizer_key(key)

    return None


def _load(model, checkpoint_path, convert, dtype):
    with safe_open(checkpoint_path, framework="pt") as f:
        state = {}

        for key in f.keys():
            converted = convert(key)

            if converted is not None:
                state[converted] = f.get_tensor(key).float()

    # The quantizer serves both directions, so each side sees projections
    # it has no use for. Strict loading still fails on anything the model
    # expects and the checkpoint did not supply.
    wanted = set(model.state_dict())
    state = {key: value for key, value in state.items() if key in wanted}

    # Load at full precision: load_state_dict copies into the existing
    # parameters, so the cast has to happen to the module afterwards.
    model.load_state_dict(state, strict=True)

    return model.to(dtype).eval()


def load_mimi_encoder(checkpoint_path, dtype=torch.float32) -> MimiEncoder:
    """
    Build an encoder holding Kyutai's pretrained Mimi weights.

    The checkpoint ships in bfloat16. Widening to float32 is the safe
    default; on a GPU, loading it back as bfloat16 halves memory traffic.
    """
    # Checkpoint convolutions already have weight norm folded in.
    return _load(MimiEncoder(use_weight_norm=False), checkpoint_path,
                 convert_encoder_key, dtype)


def load_mimi_decoder(checkpoint_path, dtype=torch.float32) -> MimiDecoder:
    """Build a decoder holding Kyutai's pretrained Mimi weights."""
    return _load(MimiDecoder(use_weight_norm=False), checkpoint_path,
                 convert_decoder_key, dtype)
