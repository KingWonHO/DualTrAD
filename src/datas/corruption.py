"""Input corruption used only while training the reconstruction branch."""

from __future__ import annotations

from typing import Any

import torch


class ContextCorruption:
    """Mask and/or noise the context the model sees.

    The clean window is kept separately and is what the reconstruction loss is
    scored against, which is what makes the branch *denoising* rather than a
    copy. Both probabilities default to zero, so a forecasting-only run sees the
    untouched window.
    """

    def __init__(self, mask_probability: float = 0.0, gaussian_std: float = 0.0) -> None:
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0, 1]")
        if gaussian_std < 0.0:
            raise ValueError("gaussian_std must be non-negative")
        self.mask_probability = mask_probability
        self.gaussian_std = gaussian_std

    @property
    def enabled(self) -> bool:
        return self.mask_probability > 0.0 or self.gaussian_std > 0.0

    def __call__(self, context: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return context
        corrupted = context
        if self.mask_probability > 0.0:
            keep = torch.rand_like(corrupted) >= self.mask_probability
            corrupted = corrupted * keep
        if self.gaussian_std > 0.0:
            corrupted = corrupted + torch.randn_like(corrupted) * self.gaussian_std
        return corrupted

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "ContextCorruption",
            "mask_probability": self.mask_probability,
            "gaussian_std": self.gaussian_std,
        }
