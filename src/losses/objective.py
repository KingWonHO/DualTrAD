"""Training objective.

The total loss is

    L = L_pred + λ_r · L_rec + λ_t · L_traj + λ_c · L_cons

Every term after the first is optional: setting its weight to zero removes it
without changing the rest, which is how the ablation variants are built.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
from torch.nn import functional


@dataclass
class LossBreakdown:
    """Each term is reported separately so a run's log stays diagnosable."""

    total: torch.Tensor
    prediction: torch.Tensor
    reconstruction: torch.Tensor
    trajectory: torch.Tensor
    consistency: torch.Tensor

    def detached(self) -> dict[str, float]:
        # Read the fields directly: dataclasses.asdict would deep-copy tensors,
        # which fails for anything still attached to the autograd graph.
        return {
            field.name: float(getattr(self, field.name).detach())
            for field in fields(self)
        }


class DualTrADObjective(nn.Module):
    """Weighted multi-task objective for the dual-evidence detector."""

    def __init__(
        self,
        prediction: str = "huber",
        huber_delta: float = 1.0,
        horizon_weights: list | None = None,
        reconstruction: str = "l1",
        derivative_weight: float = 0.0,
        lambda_reconstruction: float = 0.0,
        lambda_trajectory: float = 0.0,
        lambda_consistency: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super(DualTrADObjective, self).__init__()

        if prediction not in {"huber", "mse"}:
            raise ValueError(f"unknown prediction loss {prediction!r}")
        if reconstruction not in {"l1", "mse"}:
            raise ValueError(f"unknown reconstruction loss {reconstruction!r}")

        self.prediction = prediction
        self.huber_delta = huber_delta
        self.horizon_weights = horizon_weights
        self.reconstruction = reconstruction
        self.derivative_weight = derivative_weight
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_trajectory = lambda_trajectory
        self.lambda_consistency = lambda_consistency

    def _point_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-horizon loss, optionally weighted across horizons."""
        if self.prediction == "huber":
            per_element = functional.huber_loss(
                prediction, target, delta=self.huber_delta, reduction="none"
            )
        else:
            per_element = functional.mse_loss(prediction, target, reduction="none")
        per_horizon = per_element.mean(dim=(0, 2))
        if self.horizon_weights is None:
            return per_horizon.mean()
        weights = torch.as_tensor(
            self.horizon_weights, dtype=per_horizon.dtype, device=per_horizon.device
        )
        return (per_horizon * weights).sum() / weights.sum()

    def _reconstruction_loss(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.reconstruction == "l1":
            return functional.l1_loss(reconstruction, target)
        return functional.mse_loss(reconstruction, target)

    def forward(self, output, target: torch.Tensor, clean_context: torch.Tensor) -> LossBreakdown:
        zero = torch.zeros((), device=target.device, dtype=target.dtype)

        prediction_loss = self._point_loss(output.prediction, target)

        # Reconstruction is scored against the clean window even though the
        # branch was fed a corrupted one.
        reconstruction_loss = zero
        if output.reconstruction is not None and self.lambda_reconstruction > 0:
            value = self._reconstruction_loss(output.reconstruction, clean_context)
            derivative = zero
            if self.derivative_weight > 0 and clean_context.size(1) > 1:
                derivative = self._reconstruction_loss(
                    output.reconstruction[:, 1:] - output.reconstruction[:, :-1],
                    clean_context[:, 1:] - clean_context[:, :-1],
                )
            reconstruction_loss = value + self.derivative_weight * derivative

        # Penalise trajectories that disagree between neighbouring horizons.
        trajectory_loss = zero
        if self.lambda_trajectory > 0 and target.size(1) > 1:
            trajectory_loss = functional.huber_loss(
                output.prediction[:, 1:] - output.prediction[:, :-1],
                target[:, 1:] - target[:, :-1],
                delta=self.huber_delta,
            )

        # Pull the two branches together in representation space only.
        consistency_loss = zero
        if self.lambda_consistency > 0 and output.aligned_reconstruction is not None:
            prediction_state = output.prediction_features.mean(dim=1)
            consistency_loss = 1.0 - functional.cosine_similarity(
                output.aligned_reconstruction, prediction_state, dim=-1
            ).mean()

        total = (
            prediction_loss
            + self.lambda_reconstruction * reconstruction_loss
            + self.lambda_trajectory * trajectory_loss
            + self.lambda_consistency * consistency_loss
        )
        return LossBreakdown(
            total=total,
            prediction=prediction_loss,
            reconstruction=reconstruction_loss,
            trajectory=trajectory_loss,
            consistency=consistency_loss,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "DualTrADObjective",
            "prediction": self.prediction,
            "huber_delta": self.huber_delta,
            "horizon_weights": self.horizon_weights,
            "reconstruction": self.reconstruction,
            "derivative_weight": self.derivative_weight,
            "lambda_reconstruction": self.lambda_reconstruction,
            "lambda_trajectory": self.lambda_trajectory,
            "lambda_consistency": self.lambda_consistency,
        }
