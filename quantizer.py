import torch
import torch.nn.functional as F
from torch import nn


class Codebook(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        codebook_size: int = 2048,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon

        self.register_buffer(
            "_initialized",
            torch.tensor([False], dtype=torch.float32),
        )

        self.register_buffer(
            "cluster_usage",
            torch.ones(codebook_size),
        )

        self.register_buffer(
            "embedding_sum",
            torch.zeros(codebook_size, dim),
        )

    @property
    def embedding(self) -> torch.Tensor:
        usage = self.cluster_usage.clamp(min=self.epsilon)
        return self.embedding_sum / usage[:, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., dim]
        returns: [...]
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected dimension {self.dim}, got {x.shape[-1]}"
            )

        shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.dim)

        centroids = self.embedding.to(
            device=x.device,
            dtype=x.dtype,
        )

        x_squared = flat_x.square().sum(dim=-1, keepdim=True)
        e_squared = centroids.square().sum(dim=-1).unsqueeze(0)
        cross_term = flat_x @ centroids.transpose(0, 1)

        distances = x_squared - 2 * cross_term + e_squared
        indices = distances.argmin(dim=-1)

        return indices.reshape(shape)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [...]
        returns: [..., dim]
        """
        return F.embedding(codes, self.embedding)


class ResidualVectorEncoder(nn.Module):
    def __init__(
        self,
        n_q: int,
        input_dimension: int = 512,
        dimension: int = 256,
        bins: int = 2048,
    ):
        super().__init__()

        self.n_q = n_q
        self.input_dimension = input_dimension
        self.dimension = dimension

        self.input_proj = nn.Conv1d(
            input_dimension,
            dimension,
            kernel_size=1,
            bias=False,
        )

        self.codebooks = nn.ModuleList(
            [
                Codebook(
                    dim=dimension,
                    codebook_size=bins,
                )
                for _ in range(n_q)
            ]
        )

    def encode(
        self,
        x: torch.Tensor,
        n_q: int | None = None,
    ) -> torch.Tensor:
        """
        x: [B, input_dimension, T]
        returns: [B, n_q, T]
        """
        if n_q is None:
            n_q = self.n_q

        if not 1 <= n_q <= self.n_q:
            raise ValueError(
                f"n_q must be between 1 and {self.n_q}, got {n_q}"
            )

        x = self.input_proj(x)
        residual = x

        all_codes = []

        for level, codebook in enumerate(self.codebooks[:n_q]):
            # [B, D, T] -> [B, T, D]
            residual_vectors = residual.transpose(1, 2)

            # [B, T]
            codes = codebook.encode(residual_vectors)
            all_codes.append(codes)

            # Nothing reads the residual the last level leaves.
            if level + 1 == n_q:
                break

            # Required internally:
            # [B, T] -> [B, T, D] -> [B, D, T]
            selected_centroids = codebook.decode(codes)
            selected_centroids = selected_centroids.transpose(1, 2)

            residual = residual - selected_centroids

        return torch.stack(all_codes, dim=1)


class SplitResidualVectorEncoder(nn.Module):
    def __init__(
        self,
        n_q: int = 32,
        n_q_semantic: int = 1,
        input_dimension: int = 512,
        dimension: int = 256,
        bins: int = 2048,
    ):
        super().__init__()

        if n_q <= n_q_semantic:
            raise ValueError("n_q must be greater than n_q_semantic")

        self.n_q = n_q
        self.n_q_semantic = n_q_semantic

        self.semantic_encoder = ResidualVectorEncoder(
            n_q=n_q_semantic,
            input_dimension=input_dimension,
            dimension=dimension,
            bins=bins,
        )

        self.acoustic_encoder = ResidualVectorEncoder(
            n_q=n_q - n_q_semantic,
            input_dimension=input_dimension,
            dimension=dimension,
            bins=bins,
        )

    def encode(
        self,
        x: torch.Tensor,
        num_codebooks: int | None = None,
    ) -> torch.Tensor:
        """
        x: [B, 512, T]
        returns: [B, K, T]
        """
        if num_codebooks is None:
            num_codebooks = self.n_q

        if not self.n_q_semantic <= num_codebooks <= self.n_q:
            raise ValueError(
                f"num_codebooks must be between "
                f"{self.n_q_semantic} and {self.n_q}"
            )

        semantic_codes = self.semantic_encoder.encode(x)

        num_acoustic = num_codebooks - self.n_q_semantic

        if num_acoustic == 0:
            return semantic_codes

        acoustic_codes = self.acoustic_encoder.encode(x, n_q=num_acoustic)

        return torch.cat(
            [semantic_codes, acoustic_codes],
            dim=1,
        )