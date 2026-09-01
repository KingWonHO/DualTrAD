"""Evaluation.

The order of operations is the point of this module. Test labels enter exactly
once, in :func:`evaluate`, after the threshold has already been fixed on
calibration data. Nothing upstream of that call can see them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..metrics import (
    EvidenceNormalizer,
    Threshold,
    aggregate_vehicles,
    average_precision,
    confusion,
    fuse,
    normal_quantile_threshold,
    roc_auc,
    window_residuals,
)


@dataclass
class WindowEvidence:
    """Residuals for every window of one split, plus where each came from."""

    forecast_residual: np.ndarray            # [N, K] or [N, K, C]
    reconstruction_residual: np.ndarray | None  # [N]
    vehicle_indices: np.ndarray             # [N]

    def __len__(self) -> int:
        return len(self.vehicle_indices)


@torch.no_grad()
def collect_evidence(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prediction_error: str = "squared",
    derivative_weight: float = 0.0,
) -> WindowEvidence:
    """Run the model over a split and keep only its residuals."""
    model.eval()
    forecast, reconstruction, vehicles = [], [], []
    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        output = model(context)
        f, r = window_residuals(
            output, target, context, prediction_error, derivative_weight
        )
        forecast.append(f.float().cpu().numpy())
        if r is not None:
            reconstruction.append(r.float().cpu().numpy())
        vehicles.append(batch["vehicle_index"].numpy())
    return WindowEvidence(
        forecast_residual=np.concatenate(forecast, axis=0),
        reconstruction_residual=np.concatenate(reconstruction) if reconstruction else None,
        vehicle_indices=np.concatenate(vehicles),
    )


def vehicle_scores(
    evidence: WindowEvidence,
    normalizer: EvidenceNormalizer,
    fusion: str,
    vehicle_quantile: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce window residuals to one score per vehicle, in vehicle order."""
    forecast, reconstruction = normalizer.transform(
        evidence.forecast_residual, evidence.reconstruction_residual
    )
    window = fuse(forecast, reconstruction, mode=fusion)
    grouped = aggregate_vehicles(window, evidence.vehicle_indices, vehicle_quantile)
    order = sorted(grouped)
    return np.asarray([grouped[v] for v in order]), np.asarray(order)


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: Threshold
) -> dict[str, Any]:
    """The first and only place a test label is read.

    AUROC and AUPR are threshold-free, so they are unaffected by which decision
    rule produced ``threshold``; the confusion counts are not.
    """
    metrics = confusion(labels, threshold.apply(scores))
    metrics["auroc"] = roc_auc(labels, scores)
    metrics["aupr"] = average_precision(labels, scores)
    metrics["decision_rule"] = threshold.get_config()
    return metrics


def fit_threshold(
    calibration_scores: np.ndarray, calibration_labels: np.ndarray, target_fpr: float = 0.10
) -> Threshold:
    """Fix the operating point on calibration vehicles, before any test data."""
    return normal_quantile_threshold(calibration_scores, calibration_labels, target_fpr)
