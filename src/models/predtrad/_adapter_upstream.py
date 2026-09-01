"""Leakage-safe QAS adapter for the checked-in PredTrAD_v1 model.

The original implementation remains in :mod:`src.PredTrAD_v1`.  This module
only adapts its one-step sequence-to-sequence core to the common QAS protocol:
20 observed context points in and forecasts for steps 1, 3, and 5 out.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from ..third_party.PredTrAD_v1 import PredTrAD_v1

from ..shared.forecast_interface import ForecastOutput


ORIGINAL_SOURCE_PATH = Path(__file__).resolve().parents[1] / "third_party" / "PredTrAD_v1.py"
ORIGINAL_SOURCE_SHA256 = (
    "e02a1483b267546bab1b7763434d15126258d9599a4b5334ba2c4109fea14134"
)


def original_source_sha256() -> str:
    """Return the hash of the exact checked-in core used by this adapter."""

    return hashlib.sha256(ORIGINAL_SOURCE_PATH.read_bytes()).hexdigest()


class DTypeSafePredTrADV1(PredTrAD_v1):
    """PredTrAD_v1 with a causal mask that is safe for FP32 execution.

    The upstream implementation always creates a float64 attention mask.  Its
    loader also converts the whole model to float64, so that happened to match
    there.  The QAS comparison runs models in FP32.  A boolean causal mask has
    identical masking semantics and is accepted independently of the query
    floating-point dtype, without adding trainable parameters.
    """

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        if sz < 1:
            raise ValueError("causal mask size must be positive")
        device = self.fc.weight.device
        return torch.triu(
            torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1
        )


class PredTrADV1QASAdapter(nn.Module):
    """Autoregressively adapt PredTrAD_v1 to QAS horizons 1, 3, and 5.

    Every rollout call receives exactly the current 20-point history.  The
    encoder consumes the first 11 points and the decoder the last 10 points;
    point 10 is deliberately shared at their boundary.  Consequently all 20
    observed points are represented while retaining PredTrAD's one-point
    encoder/decoder overlap.  Only the model's own last prediction is appended
    before the next rollout, so no future target can enter the input.
    """

    context_length = 20
    encoder_length = 11
    decoder_length = 10
    encoder_decoder_overlap = 1

    def __init__(
        self,
        *,
        core: nn.Module | None = None,
        input_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        ff_hidden_dim: int = 1024,
        dropout: float = 0.1,
        horizons: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        selected_horizons = tuple(int(step) for step in horizons)
        if not selected_horizons:
            raise ValueError("at least one forecast horizon is required")
        if any(step < 1 for step in selected_horizons):
            raise ValueError("forecast horizons must be positive")
        if tuple(sorted(set(selected_horizons))) != selected_horizons:
            raise ValueError("forecast horizons must be unique and increasing")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")

        if core is None:
            core = DTypeSafePredTrADV1(
                input_dim=input_dim,
                d_model=d_model,
                n_heads=n_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
            )
        if not isinstance(core, nn.Module):
            raise TypeError("core must be a torch.nn.Module")

        self.core = core
        self.input_dim = int(input_dim)
        self.horizons = selected_horizons
        self.rollout_steps = max(selected_horizons)
        actual_hash = original_source_sha256()
        self.adaptation_metadata: dict[str, Any] = {
            "baseline": "PredTrAD_v1",
            "original_source_path": str(ORIGINAL_SOURCE_PATH),
            "original_source_sha256": actual_hash,
            "expected_original_source_sha256": ORIGINAL_SOURCE_SHA256,
            "original_source_hash_matches": actual_hash
            == ORIGINAL_SOURCE_SHA256,
            "context_length": self.context_length,
            "encoder_length": self.encoder_length,
            "decoder_length": self.decoder_length,
            "encoder_decoder_overlap": self.encoder_decoder_overlap,
            "forecast_horizons": list(self.horizons),
            "rollout_steps": self.rollout_steps,
            "rollout_mode": "autoregressive_model_predictions_only",
            "future_target_used_as_input": False,
            "causal_mask_adaptation": "boolean_upper_triangular",
            "default_core_dtype": "float32",
        }

    def forward(self, context: torch.Tensor) -> ForecastOutput:
        if context.ndim != 3:
            raise ValueError(
                "context must have shape [batch, 20, features], "
                f"got {tuple(context.shape)}"
            )
        if context.shape[1] != self.context_length:
            raise ValueError(
                f"context length must be {self.context_length}, got {context.shape[1]}"
            )
        if context.shape[2] != self.input_dim:
            raise ValueError(
                f"context feature dimension must be {self.input_dim}, "
                f"got {context.shape[2]}"
            )
        if not context.is_floating_point():
            raise TypeError("context must be a floating-point tensor")

        history = context
        selected_predictions: list[torch.Tensor] = []
        horizon_set = set(self.horizons)
        for step in range(1, self.rollout_steps + 1):
            encoder_input = history[:, : self.encoder_length, :]
            decoder_input = history[:, -self.decoder_length :, :]
            sequence_prediction = self.core(encoder_input, decoder_input)
            expected_shape = (
                context.shape[0],
                self.decoder_length,
                self.input_dim,
            )
            if tuple(sequence_prediction.shape) != expected_shape:
                raise RuntimeError(
                    "PredTrAD_v1 core returned an unexpected shape: "
                    f"expected {expected_shape}, got {tuple(sequence_prediction.shape)}"
                )
            next_value = sequence_prediction[:, -1:, :]
            if step in horizon_set:
                selected_predictions.append(next_value)
            history = torch.cat((history[:, 1:, :], next_value), dim=1)

        prediction = torch.cat(selected_predictions, dim=1)
        return ForecastOutput(prediction=prediction, phase1=None)


__all__ = [
    "DTypeSafePredTrADV1",
    "ORIGINAL_SOURCE_PATH",
    "ORIGINAL_SOURCE_SHA256",
    "PredTrADV1QASAdapter",
    "original_source_sha256",
]
