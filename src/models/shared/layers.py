"""Small building blocks reused across detectors."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions, added to the projected input."""

    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.size(1) > self.encoding.size(1):
            raise ValueError(
                f"sequence length {values.size(1)} exceeds positional limit "
                f"{self.encoding.size(1)}"
            )
        return self.dropout(values + self.encoding[:, : values.size(1)].to(values.dtype))


class LearnedPositionalEncoding(nn.Module):
    """Learned positions, added to the projected input."""

    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(1, max_length, d_model))
        nn.init.trunc_normal_(self.embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.size(1) > self.embedding.size(1):
            raise ValueError(
                f"sequence length {values.size(1)} exceeds positional limit "
                f"{self.embedding.size(1)}"
            )
        return self.dropout(values + self.embedding[:, : values.size(1)])


POSITIONAL_ENCODINGS = {
    "sinusoidal": SinusoidalPositionalEncoding,
    "learned": LearnedPositionalEncoding,
}


def get_positional_encoding(
    positional_encoding: str, d_model: int, max_length: int, dropout: float
) -> nn.Module:
    if positional_encoding not in POSITIONAL_ENCODINGS:
        raise ValueError(
            f"unknown positional_encoding {positional_encoding!r}; "
            f"available: {sorted(POSITIONAL_ENCODINGS)}"
        )
    return POSITIONAL_ENCODINGS[positional_encoding](d_model, max_length, dropout)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a feed-forward block.

    Queries carry their own residual stream; ``memory`` is read-only.
    """

    def __init__(self, d_model: int, n_heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalised_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(queries), normalised_memory, normalised_memory, need_weights=False
        )
        queries = queries + self.attention_dropout(attended)
        return queries + self.feedforward(self.ff_norm(queries))


class MultiHorizonPredictionHeads(nn.Module):
    """One independent MLP head per forecast horizon; no weight sharing."""

    def __init__(
        self, horizon_count: int, d_model: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.horizon_count = horizon_count
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(horizon_count)
        )

    def forward(self, horizon_features: torch.Tensor) -> torch.Tensor:
        predictions = [head(horizon_features[:, index]) for index, head in enumerate(self.heads)]
        return torch.stack(predictions, dim=1)

    def get_config(self) -> dict[str, Any]:
        first = self.heads[0]
        return {
            "type": "MultiHorizonPredictionHeads",
            "horizon_count": self.horizon_count,
            "d_model": first[1].in_features,
            "hidden_dim": first[1].out_features,
            "output_dim": first[4].out_features,
        }
