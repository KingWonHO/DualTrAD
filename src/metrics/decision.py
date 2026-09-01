"""Turning vehicle scores into vehicle decisions.

Two rules are provided. They differ only in where the threshold comes from,
never in the scores themselves, so both sit on the same ROC curve and neither
can change AUROC or AUPR.

``snippet_tail``  reads the threshold from the *window* score distribution and
                  applies it to a *vehicle* aggregate. Those are different
                  distributions: a vehicle score is a mean of high window
                  scores, while the window tail sits far above it. On this
                  corpus the mismatch is severe enough that the threshold can
                  exceed every calibration vehicle, in which case nothing is
                  ever flagged.

``normal_quantile`` reads the threshold from the distribution it is applied to
                  -- the normal calibration *vehicles* -- at a stated target
                  false-positive rate. It consumes no abnormal labels, so the
                  operating point becomes a specification rather than a
                  quantity tuned on scarce positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np


def top_count(fraction: float, total: int) -> int:
    """``fraction * total`` rounded half-up, clamped to ``[1, total]``.

    Decimal rounding, not Python's banker's rounding, so that e.g.
    ``0.15 * 10`` is exactly ``1.5`` and rounds to ``2``.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    value = Decimal(str(fraction)) * Decimal(int(total))
    return min(max(int(value.to_integral_value(rounding=ROUND_HALF_UP)), 1), total)


@dataclass(frozen=True)
class Threshold:
    value: float
    rule: str
    target_fpr: float | None = None

    def apply(self, vehicle_scores: np.ndarray) -> np.ndarray:
        """Strictly greater than: a score exactly at the threshold is normal."""
        return (np.asarray(vehicle_scores, dtype=np.float64) > self.value).astype(np.int64)

    def get_config(self) -> dict[str, Any]:
        return {"rule": self.rule, "value": self.value, "target_fpr": self.target_fpr}


def normal_quantile_threshold(
    calibration_vehicle_scores: np.ndarray,
    calibration_vehicle_labels: np.ndarray,
    target_fpr: float = 0.10,
) -> Threshold:
    """Threshold at the ``1 - target_fpr`` quantile of normal calibration vehicles."""
    scores = np.asarray(calibration_vehicle_scores, dtype=np.float64)
    labels = np.asarray(calibration_vehicle_labels)
    normal = scores[labels == 0]
    if len(normal) == 0:
        raise ValueError("no normal calibration vehicles to fit a threshold on")
    return Threshold(
        value=float(np.quantile(normal, 1.0 - target_fpr)),
        rule="normal_quantile",
        target_fpr=target_fpr,
    )


def snippet_tail_threshold(
    calibration_window_scores: np.ndarray,
    calibration_window_labels: np.ndarray,
    tail_grid: range = range(1, 100),
) -> Threshold:
    """Legacy rule: scan the top ``n/1000`` of calibration windows.

    Kept so the reported degeneracy can be reproduced, not because it is
    recommended.
    """
    scores = np.asarray(calibration_window_scores, dtype=np.float64)
    labels = np.asarray(calibration_window_labels)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]

    best = None
    for n in tail_grid:
        count = top_count(n / 1000.0, len(scores))
        ratio = float(np.mean(sorted_labels[:count]))
        # Ties keep the smallest n, i.e. the most selective candidate.
        if best is None or ratio > best[0]:
            best = (ratio, float(sorted_scores[count - 1]))
    return Threshold(value=best[1], rule="snippet_tail")


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Vehicle-level confusion matrix and the rates derived from it."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(recall) and precision + recall > 0
        else 0.0
    )
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (tn + fp) if tn + fp else float("nan"),
    }
