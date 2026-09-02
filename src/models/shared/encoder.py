"""Context encoders. Pick one with :func:`get`."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import get_positional_encoding


class BaseEncoder(nn.Module):
    """Maps a context window ``[B, L, F]`` to hidden states ``[B, L, D]``."""

    def get_out_chan(self) -> int:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError


class CausalTransformerEncoder(BaseEncoder):
    """Small Transformer over the time axis.

    Causal masking keeps each ``h_t`` a valid state at time ``t``. It is not a
    leakage safeguard here: the forecast targets lie outside the context window,
    so no future value is reachable either way.
    """

    def __init__(
        self,
        in_chan: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        positional_encoding: str,
        max_length: int,
        causal: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_chan = in_chan
        self.d_model = d_model
        self.causal = causal
        self.input_projection = nn.Linear(in_chan, d_model)
        self.position = get_positional_encoding(positional_encoding, d_model, max_length, dropout)
        self.positional_encoding = positional_encoding
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.scale = d_model**0.5

    def get_out_chan(self) -> int:
        return self.d_model

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.position(self.input_projection(context) * self.scale)
        mask = None
        if self.causal:
            length = hidden.size(1)
            mask = torch.triu(
                torch.ones(length, length, device=hidden.device, dtype=torch.bool), diagonal=1
            )
        return self.encoder(hidden, mask=mask)

    def get_config(self) -> dict[str, Any]:
        layer = self.encoder.layers[0]
        return {
            "type": "CausalTransformerEncoder",
            "in_chan": self.in_chan,
            "d_model": self.d_model,
            "n_heads": layer.self_attn.num_heads,
            "num_layers": len(self.encoder.layers),
            "feedforward_dim": layer.linear1.out_features,
            "positional_encoding": self.positional_encoding,
            "causal": self.causal,
        }


ENCODERS = {
    "CausalTransformerEncoder": CausalTransformerEncoder,
}


def get(encoder_type: str):
    if encoder_type not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder_type!r}; available: {sorted(ENCODERS)}")
    return ENCODERS[encoder_type]
