"""Contracts every detector in this repository follows.

A detector reads one context window and returns a :class:`DetectorOutput`.
Losses and anomaly scores are computed outside the model, so a model never
sees a label and never decides a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class DetectorOutput:
    """What every detector returns.

    Attributes:
        prediction: Forecast for the scored response channels, ``[B, K, C]``.
        reconstruction: Reconstructed context window ``[B, L, C]``, or ``None``
            for forecasting-only detectors.
        prediction_features: Per-horizon latent that fed the prediction heads,
            ``[B, K, D]``. Used by auxiliary losses.
        reconstruction_latent: Autoencoder bottleneck ``[B, b]``, or ``None``.
        aligned_reconstruction: ``reconstruction_latent`` projected into the
            prediction space ``[B, D]``, or ``None``. Used by the consistency
            loss; it is never injected into the prediction path.
        auxiliary: Model-specific extras that the training loop may read.
    """

    prediction: torch.Tensor
    reconstruction: torch.Tensor | None = None
    prediction_features: torch.Tensor | None = None
    reconstruction_latent: torch.Tensor | None = None
    aligned_reconstruction: torch.Tensor | None = None
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)


class BaseDetector(nn.Module):
    """Base class for every detector.

    Subclasses build their submodules in ``init_modules`` and describe
    themselves through ``get_config``, so a run can be reconstructed from the
    serialised configuration alone.
    """

    def init_modules(self) -> None:
        raise NotImplementedError

    def forward(self, context: torch.Tensor) -> DetectorOutput:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def check_context(self, context: torch.Tensor, context_length: int, input_dim: int) -> None:
        """Reject a batch whose shape does not match this model's contract."""
        if context.ndim != 3:
            raise ValueError(f"context must be [B, L, F], received {tuple(context.shape)}")
        if context.size(1) != context_length:
            raise ValueError(
                f"context length must be {context_length}, received {context.size(1)}"
            )
        if context.size(2) != input_dim:
            raise ValueError(
                f"input feature dimension must be {input_dim}, received {context.size(2)}"
            )
