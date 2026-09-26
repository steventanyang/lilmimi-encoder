import math

import torch
import torch.nn.functional as F
from torch import nn


def rotary_embedding(
    x: torch.Tensor,
    offset: int = 0,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """
    x: [B, H, T, head_dim]
    returns: [B, H, T, head_dim]
    """
    *_, time, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    pair = torch.arange(head_dim // 2, device=x.device, dtype=torch.float32)
    freqs = torch.exp(pair * (-math.log(max_period) * 2 / head_dim))

    positions = offset + torch.arange(
        time,
        device=x.device,
        dtype=torch.float32,
    )
    angles = freqs * positions[:, None]

    cos = angles.cos()
    sin = angles.sin()

    # Channels are interleaved as [r0, i0, r1, i1, ...].
    pairs = x.reshape(*x.shape[:-1], head_dim // 2, 2).float()
    real, imag = pairs[..., 0], pairs[..., 1]

    rotated = torch.stack(
        [
            real * cos - imag * sin,
            real * sin + imag * cos,
        ],
        dim=-1,
    )

    return rotated.reshape(x.shape).to(x.dtype)


class LayerScale(nn.Module):

    def __init__(self, dim: int = 512, init: float = 0.01):
        super().__init__()

        # Checkpoint key: layer_scale_N.scale
        self.scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        return x * self.scale


class MultiheadAttention(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        context: int = 250,
        max_period: float = 10_000.0,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context = context
        self.max_period = max_period

        # Checkpoint key: self_attn.in_proj_weight, [3 * D, D]
        # Q, K, V stacked in that order along dim 0. No bias.
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * d_model, d_model)
        )

        # Checkpoint key: self_attn.out_proj.weight
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        batch, time, _ = x.shape

        # [B, T, 3D] -> [3, B, H, T, head_dim], Q/K/V stacked first.
        qkv = F.linear(x, self.in_proj_weight)
        qkv = qkv.reshape(batch, time, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Position enters through Q and K only; V carries content.
        q = rotary_embedding(q, max_period=self.max_period)
        k = rotary_embedding(k, max_period=self.max_period)

        positions = torch.arange(time, device=x.device)
        delta = positions[:, None] - positions[None, :]
        can_attend = (delta >= 0) & (delta < self.context)

        attended = F.scaled_dot_product_attention(q, k, v, can_attend)

        # [B, H, T, head_dim] -> [B, T, D]
        attended = attended.transpose(1, 2).reshape(batch, time, self.d_model)

        return self.out_proj(attended)


class MLP(nn.Module):
    def __init__(self, d_model: int = 512, dim_feedforward: int = 2048):
        super().__init__()

        self.up = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        context: int = 250,
        max_period: float = 10_000.0,
        layer_scale_init: float = 0.01,
    ):
        super().__init__()

        # LayerNorm is the only place biases survive in this model.
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            context=context,
            max_period=max_period,
        )
        self.layer_scale_1 = LayerScale(d_model, layer_scale_init)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_feedforward)
        self.layer_scale_2 = LayerScale(d_model, layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        x = x + self.layer_scale_1(self.attn(self.ln1(x)))
        x = x + self.layer_scale_2(self.mlp(self.ln2(x)))

        return x


class MimiTransformer(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        context: int = 250,
        max_period: float = 10_000.0,
        layer_scale_init: float = 0.01,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    max_period=max_period,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T]
        returns: [B, D, T]
        """
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        return x.transpose(1, 2)


def remap_mimi_keys(state: dict) -> dict:
    renames = {
        ".norm1.": ".ln1.",
        ".norm2.": ".ln2.",
        ".self_attn.": ".attn.",
        ".linear1.": ".mlp.up.",
        ".linear2.": ".mlp.down.",
    }

    remapped = {}

    for key, value in state.items():
        for old, new in renames.items():
            key = key.replace(old, new)
        remapped[key] = value

    return remapped
