"""Horizon decoders: turn encoder memory into one latent per forecast horizon."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import CrossAttentionBlock


class BaseHorizonDecoder(nn.Module):
    """Maps memory ``[B, M, D]`` to per-horizon features ``[B, K, D]``."""

    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError


class HorizonQueryDecoder(BaseHorizonDecoder):
    """One learned query per horizon, cross-attending to the whole memory.

    Each horizon can therefore read a different part of the context, instead of
    sharing a single summary vector.
    """

    def __init__(
        self,
        horizon_count: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.horizon_count = horizon_count
        self.d_model = d_model
        self.queries = nn.Parameter(torch.empty(horizon_count, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, n_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(memory.size(0), -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.output_norm(queries)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "HorizonQueryDecoder",
            "horizon_count": self.horizon_count,
            "d_model": self.d_model,
            "num_layers": len(self.blocks),
        }


class LastStateDecoder(BaseHorizonDecoder):
    """Broadcast the final memory state to every horizon.

    Carries no parameters: horizons are differentiated only by the prediction
    heads downstream. This is the minimal forecasting reference point.
    """

    def __init__(self, horizon_count: int, **kwargs) -> None:
        super().__init__()
        self.horizon_count = horizon_count

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return memory[:, -1:].expand(-1, self.horizon_count, -1)

    def get_config(self) -> dict[str, Any]:
        return {"type": "LastStateDecoder", "horizon_count": self.horizon_count}


DECODERS = {
    "HorizonQueryDecoder": HorizonQueryDecoder,
    "LastStateDecoder": LastStateDecoder,
}


def get(decoder_type: str):
    if decoder_type not in DECODERS:
        raise ValueError(f"unknown decoder {decoder_type!r}; available: {sorted(DECODERS)}")
    return DECODERS[decoder_type]
