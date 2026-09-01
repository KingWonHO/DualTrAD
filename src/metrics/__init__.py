"""Scoring, decision rules and ranking metrics."""

from .decision import (
    Threshold,
    confusion,
    normal_quantile_threshold,
    snippet_tail_threshold,
    top_count,
)
from .ranking import average_precision, roc_auc
from .scoring import EvidenceNormalizer, aggregate_vehicles, fuse, window_residuals

__all__ = [
    "window_residuals", "EvidenceNormalizer", "fuse", "aggregate_vehicles",
    "Threshold", "normal_quantile_threshold", "snippet_tail_threshold",
    "confusion", "top_count", "roc_auc", "average_precision",
]
