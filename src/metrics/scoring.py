"""From model residuals to one anomaly score per vehicle.

The pipeline is deliberately split so that no step can see a test label:

1. ``window_residuals``   - per-window residuals, model output only.
2. ``EvidenceNormalizer`` - scales fitted on *normal calibration* data only.
3. ``fuse``               - combine the normalised evidences.
4. ``aggregate_vehicles`` - one score per vehicle.

Thresholding lives in :mod:`src.metrics.decision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def window_residuals(
    output,
    target: torch.Tensor,
    clean_context: torch.Tensor,
    prediction_error: str = "squared",
    derivative_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-window residuals.

    The forecasting residual keeps its channel axis: ``[B, K, C]`` when there
    are several channels, ``[B, K]`` when there is one. Averaging raw errors
    over channels first would let whichever channel has the largest scale
    dominate the score, so the channel axis survives until each channel has
    been divided by its own calibration scale.
    """
    difference = output.prediction - target
    if prediction_error == "squared":
        per_channel = difference.square()
    elif prediction_error == "absolute":
        per_channel = difference.abs()
    else:
        raise ValueError(f"unknown prediction_error {prediction_error!r}")
    forecast = per_channel.squeeze(-1) if per_channel.size(-1) == 1 else per_channel

    reconstruction = None
    if output.reconstruction is not None:
        value = (output.reconstruction - clean_context).abs().mean(dim=(1, 2))
        derivative = (
            (output.reconstruction[:, 1:] - output.reconstruction[:, :-1])
            - (clean_context[:, 1:] - clean_context[:, :-1])
        ).abs().mean(dim=(1, 2))
        reconstruction = value + derivative_weight * derivative
    return forecast, reconstruction


@dataclass(frozen=True)
class EvidenceNormalizer:
    """Puts both evidences on a common, unit-referenced scale.

    Every scale is a high quantile of the residual over *normal calibration*
    windows, so a value near ``1`` sits at the edge of the normal range no
    matter which channel or horizon it came from.
    """

    forecast_scale: np.ndarray
    reconstruction_scale: float | None
    epsilon: float

    @classmethod
    def fit(
        cls,
        forecast_residual: np.ndarray,
        reconstruction_residual: np.ndarray | None,
        quantile: float = 0.99,
        epsilon: float = 1e-8,
    ) -> "EvidenceNormalizer":
        if forecast_residual.ndim not in (2, 3) or len(forecast_residual) == 0:
            raise ValueError("forecast_residual must be a non-empty [N,K] or [N,K,C] array")
        forecast_scale = np.maximum(
            np.quantile(forecast_residual, quantile, axis=0), epsilon
        ).astype(np.float64)
        reconstruction_scale = None
        if reconstruction_residual is not None:
            reconstruction_scale = max(
                float(np.quantile(reconstruction_residual, quantile)), epsilon
            )
        return cls(forecast_scale, reconstruction_scale, epsilon)

    def transform(
        self, forecast_residual: np.ndarray, reconstruction_residual: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        scale = self.forecast_scale
        normalised = forecast_residual / scale.reshape((1,) + scale.shape)
        if normalised.ndim == 3:
            # Average over channels *after* normalisation. With one channel this
            # step is the identity, so the single-channel formula is unchanged.
            normalised = normalised.mean(axis=2)
        forecast_evidence = np.max(normalised, axis=1)

        reconstruction_evidence = None
        if reconstruction_residual is not None:
            if self.reconstruction_scale is None:
                raise ValueError("reconstruction scale was never fitted")
            reconstruction_evidence = reconstruction_residual / self.reconstruction_scale
        return forecast_evidence, reconstruction_evidence

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "EvidenceNormalizer",
            "forecast_scale": self.forecast_scale.tolist(),
            "reconstruction_scale": self.reconstruction_scale,
            "epsilon": self.epsilon,
        }


def fuse(
    forecast: np.ndarray,
    reconstruction: np.ndarray | None,
    mode: str = "geometric_mean",
    alpha: float = 0.7,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Combine the two evidences into one window score.

    ``geometric_mean`` requires *both* evidences to be elevated before the
    score rises, which suppresses windows where only one branch reacts.
    ``forecast_only`` is the control that ignores the reconstruction branch.
    """
    if mode == "forecast_only":
        return forecast
    if reconstruction is None:
        raise ValueError(f"fusion mode {mode!r} needs reconstruction evidence")
    forecast = np.maximum(forecast, 0.0)
    reconstruction = np.maximum(reconstruction, 0.0)
    if mode == "geometric_mean":
        return np.sqrt(np.maximum(forecast * reconstruction, epsilon))
    if mode == "weighted_sum":
        return alpha * forecast + (1.0 - alpha) * reconstruction
    if mode == "max":
        return np.maximum(forecast, reconstruction)
    raise ValueError(f"unknown fusion mode {mode!r}")


def aggregate_vehicles(
    window_scores: np.ndarray, vehicle_indices: np.ndarray, quantile: float = 0.99
) -> dict[int, float]:
    """One score per vehicle: a high quantile of that vehicle's windows.

    A quantile rather than the maximum keeps the decision sensitive to a
    localised fault without letting one outlying window decide it.
    """
    scores: dict[int, float] = {}
    for vehicle in np.unique(vehicle_indices):
        selected = window_scores[vehicle_indices == vehicle]
        if len(selected):
            scores[int(vehicle)] = float(np.quantile(selected, quantile))
    return scores
