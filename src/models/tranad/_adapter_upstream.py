from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from ..third_party.dlutils import PositionalEncoding
from ..third_party.models import TranAD_without_sigmoid
from ..shared.forecast_interface import ForecastOutput


class TranADQASForecastAdapter(nn.Module):
    """Causal multi-horizon adapter around the repository's TranAD model.

    The original TranAD implementation reconstructs the last value already
    present in a ten-step input window.  The QAS comparison instead predicts
    three genuinely future values from a twenty-step context.  One independent
    repository-native ``TranAD_without_sigmoid`` branch is therefore assigned
    to each horizon.  Every decoder query is the last *observed* context value;
    a future target is deliberately not part of this module's forward API.

    ``TranAD_without_sigmoid`` is the repository-provided variant appropriate
    for standardized data.  Its identity output head avoids constraining QAS
    standardized/offset values to the interval [0, 1].
    """

    DEFAULT_CONTEXT_LENGTH = 20
    DEFAULT_HORIZON_STEPS = (1, 3, 5)

    def __init__(
        self,
        input_dim: int = 1,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        horizon_steps: Sequence[int] = DEFAULT_HORIZON_STEPS,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if context_length < 1:
            raise ValueError("context_length must be positive")

        horizons = tuple(int(step) for step in horizon_steps)
        if not horizons or any(step < 1 for step in horizons):
            raise ValueError("horizon_steps must contain positive integers")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizon_steps must not contain duplicates")

        self.input_dim = int(input_dim)
        self.context_length = int(context_length)
        self.horizon_steps = horizons
        self.branches = nn.ModuleList(
            [self._make_context_compatible_branch() for _ in self.horizon_steps]
        )

        branch_parameters = self._count_trainable_parameters(self.branches[0])
        self.adaptation_metadata: dict[str, Any] = {
            "schema_version": "qas-tranad-forecast-adapter-v1",
            "source_model": "src.models.TranAD_without_sigmoid",
            "source_model_reused": True,
            "input_dim": self.input_dim,
            "original_context_length": 10,
            "comparison_context_length": self.context_length,
            "horizon_steps": list(self.horizon_steps),
            "independent_horizon_branches": len(self.horizon_steps),
            "decoder_query": "last_observed_context_value",
            "future_target_used_as_model_input": False,
            "output_activation": "identity",
            "phase1_shape": "[B,K,F]",
            "prediction_shape": "[B,K,F]",
            "trainable_parameters_per_branch": branch_parameters,
            "trainable_parameters": self.trainable_parameter_count,
        }

    @staticmethod
    def _count_trainable_parameters(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    @property
    def trainable_parameter_count(self) -> int:
        """Total trainable parameters, including every horizon branch."""

        return self._count_trainable_parameters(self)

    def _make_context_compatible_branch(self) -> TranAD_without_sigmoid:
        branch = TranAD_without_sigmoid(self.input_dim)

        # The source model creates a non-trainable positional buffer of length
        # ten.  Rebuild only that buffer for the shared QAS context length.  No
        # learned source-model layer or parameter is replaced.
        branch.n_window = self.context_length
        branch.n = branch.n_feats * branch.n_window
        branch.pos_encoder = PositionalEncoding(
            2 * self.input_dim,
            dropout=0.1,
            max_len=self.context_length,
        )
        return branch

    def forward(self, context: torch.Tensor) -> ForecastOutput:
        """Predict all configured horizons using historical context only.

        Args:
            context: Tensor shaped ``[batch, context_length, input_dim]``.

        Returns:
            ``ForecastOutput`` whose ``phase1`` and ``prediction`` tensors are
            both shaped ``[batch, horizons, input_dim]``.
        """

        if context.ndim != 3:
            raise ValueError(
                "context must have shape [B,L,F], "
                f"received {tuple(context.shape)}"
            )
        if context.size(1) != self.context_length:
            raise ValueError(
                f"context length must be {self.context_length}, "
                f"received {context.size(1)}"
            )
        if context.size(2) != self.input_dim:
            raise ValueError(
                f"input feature dimension must be {self.input_dim}, "
                f"received {context.size(2)}"
            )
        if not context.is_floating_point():
            raise TypeError("context must be a floating-point tensor")

        # Repository TranAD uses sequence-first tensors [L,B,F].  The query is
        # selected exclusively from src, so future labels cannot enter either
        # phase even accidentally through this public API.
        src = context.transpose(0, 1)
        last_observed_query = src[-1:].contiguous()

        phase1_by_horizon: list[torch.Tensor] = []
        phase2_by_horizon: list[torch.Tensor] = []
        for branch in self.branches:
            phase1, phase2 = branch(src, last_observed_query)
            phase1_by_horizon.append(phase1.squeeze(0))
            phase2_by_horizon.append(phase2.squeeze(0))

        phase1_prediction = torch.stack(phase1_by_horizon, dim=1)
        phase2_prediction = torch.stack(phase2_by_horizon, dim=1)
        return ForecastOutput(
            prediction=phase2_prediction,
            phase1=phase1_prediction,
        )

