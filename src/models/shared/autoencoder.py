"""Reconstruction branch and the projection that keeps it aligned with forecasting."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DenoisingAutoencoderBranch(nn.Module):
    """Rebuild the whole window from the final encoder state alone.

    Only ``h_L`` enters the bottleneck, and the decoder expands that single
    vector back to ``[B, L, C]``. Reconstructing ``L`` frames from one
    bottlenecked state forces the branch to encode the window's shape rather
    than copy it. Inputs are corrupted during training; the loss is computed
    against the clean window.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        decoder_hidden_dim: int,
        context_length: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, context_length * output_dim),
        )

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.bottleneck(memory[:, -1])
        reconstruction = self.decoder(latent).reshape(
            memory.size(0), self.context_length, self.output_dim
        )
        return reconstruction, latent

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "DenoisingAutoencoderBranch",
            "bottleneck_dim": self.bottleneck_dim,
            "decoder_hidden_dim": self.decoder[0].out_features,
            "context_length": self.context_length,
            "output_dim": self.output_dim,
        }


class LatentAlignment(nn.Module):
    """Project the autoencoder bottleneck into the prediction feature space.

    The projection exists only so a consistency loss can pull the two branches
    together in representation space. The result is never added to the
    prediction path, which keeps the two anomaly evidences separately computed.
    """

    def __init__(self, bottleneck_dim: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(bottleneck_dim, d_model), nn.LayerNorm(d_model))

    def forward(self, reconstruction_latent: torch.Tensor) -> torch.Tensor:
        return self.projection(reconstruction_latent)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "LatentAlignment",
            "bottleneck_dim": self.projection[0].in_features,
            "d_model": self.projection[0].out_features,
        }
