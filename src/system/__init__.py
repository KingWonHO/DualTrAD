"""Training and evaluation loops."""

from .evaluator import (
    WindowEvidence,
    collect_evidence,
    evaluate,
    fit_threshold,
    vehicle_scores,
)
from .runner import build_model, build_objective, run_from_config
from .trainer import Trainer, TrainerConfig

__all__ = [
    "Trainer", "TrainerConfig",
    "collect_evidence", "vehicle_scores", "fit_threshold", "evaluate", "WindowEvidence",
    "build_model", "build_objective", "run_from_config",
]
