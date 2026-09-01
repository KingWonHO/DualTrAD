"""Leakage-safe, multi-horizon QAS adapter for repository DTAAD.

The checked-in :class:`src.models.DTAAD_without_sigmoid` predicts one value
from a ten-point window.  This module retains its local/global TCN, positional
encoding, custom Transformer, residual FCN, and two-phase callback design,
while adapting the input to the comparison's twenty observed points and the
output to future horizons 1, 3, and 5.

Only the horizon-specific linear decoders are replicated.  The local and
global feature extractors are shared, so this is one model with six small
heads rather than three independent DTAAD copies.  The public ``forward`` API
accepts context only; future targets can therefore be used by a loss function
but cannot enter either prediction phase.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from ..third_party.dlutils import (
    PositionalEncoding,
    Tcn_Global,
    Tcn_Local,
    TransformerEncoderLayer,
)

from ..shared.forecast_interface import ForecastOutput


class DTAADQASForecastAdapter(nn.Module):
    """Adapt ``DTAAD_without_sigmoid`` to causal QAS forecasting.

    Input uses the common comparison layout ``[batch, context, features]``.
    Both returned phases use ``[batch, horizons, features]``.  With the locked
    QAS settings this is ``[B, 20, 1]`` in and two ``[B, 3, 1]`` tensors out.

    Phase 1 is evaluated once with the shared local backbone.  Each phase-1
    horizon prediction is then explicitly expanded across all context points,
    added to the observed context, and passed through the same global
    backbone.  The corresponding phase-2 horizon head turns that callback
    representation into the final prediction.
    """

    DEFAULT_CONTEXT_LENGTH = 20
    DEFAULT_HORIZON_STEPS = (1, 3, 5)

    def __init__(
        self,
        input_dim: int = 1,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        horizon_steps: Sequence[int] = DEFAULT_HORIZON_STEPS,
        *,
        local_kernel_size: int = 4,
        global_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        transformer_dropout: float = 0.1,
        feedforward_dim: int = 16,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if context_length < 2:
            raise ValueError("context_length must be at least two")
        if local_kernel_size < 2 or global_kernel_size < 2:
            raise ValueError("TCN kernel sizes must be at least two")
        if feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive")
        if not 0.0 <= tcn_dropout < 1.0:
            raise ValueError("tcn_dropout must be in [0, 1)")
        if not 0.0 <= transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")

        horizons = tuple(int(step) for step in horizon_steps)
        if not horizons or any(step < 1 for step in horizons):
            raise ValueError("horizon_steps must contain positive integers")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizon_steps must not contain duplicates")

        self.input_dim = int(input_dim)
        self.context_length = int(context_length)
        self.horizon_steps = horizons

        # Repository-native DTAAD_without_sigmoid components.  These modules
        # are deliberately shared by all forecast horizons.
        self.l_tcn = Tcn_Local(
            num_outputs=self.input_dim,
            kernel_size=int(local_kernel_size),
            dropout=float(tcn_dropout),
        )
        self.g_tcn = Tcn_Global(
            num_inputs=self.context_length,
            num_outputs=self.input_dim,
            kernel_size=int(global_kernel_size),
            dropout=float(tcn_dropout),
        )
        self.pos_encoder = PositionalEncoding(
            self.input_dim,
            dropout=float(transformer_dropout),
            max_len=self.context_length,
        )

        encoder_layer1 = TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.input_dim,
            dim_feedforward=int(feedforward_dim),
            dropout=float(transformer_dropout),
        )
        encoder_layer2 = TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.input_dim,
            dim_feedforward=int(feedforward_dim),
            dropout=float(transformer_dropout),
        )
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layer1,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.transformer_encoder2 = nn.TransformerEncoder(
            encoder_layer2,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.fcn = nn.Linear(self.input_dim, self.input_dim)

        # DTAAD_without_sigmoid uses linear (identity-output) decoders.  One
        # pair per horizon avoids forcing all horizons to share the same map.
        self.decoder1_heads = nn.ModuleList(
            nn.Linear(self.context_length, 1) for _ in self.horizon_steps
        )
        self.decoder2_heads = nn.ModuleList(
            nn.Linear(self.context_length, 1) for _ in self.horizon_steps
        )

        self.adaptation_metadata: dict[str, Any] = {
            "schema_version": "qas-dtaad-forecast-adapter-v1",
            "source_model": "src.models.DTAAD_without_sigmoid",
            "source_components": [
                "src.dlutils.Tcn_Local",
                "src.dlutils.Tcn_Global",
                "src.dlutils.PositionalEncoding",
                "src.dlutils.TransformerEncoderLayer",
            ],
            "source_components_reused": True,
            "input_dim": self.input_dim,
            "original_context_length": 10,
            "comparison_context_length": self.context_length,
            "horizon_steps": list(self.horizon_steps),
            "shared_local_global_backbone": True,
            "horizon_specific_decoder1_heads": len(self.decoder1_heads),
            "horizon_specific_decoder2_heads": len(self.decoder2_heads),
            "callback_condition": "phase1_prediction_broadcast_over_context",
            "future_target_used_as_model_input": False,
            "output_activation": "identity",
            "phase1_shape": "[B,K,F]",
            "prediction_shape": "[B,K,F]",
            "trainable_parameters": self.trainable_parameter_count,
            "total_parameters": self.total_parameter_count,
        }

    @staticmethod
    def _count_parameters(module: nn.Module, *, trainable_only: bool) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )

    @property
    def trainable_parameter_count(self) -> int:
        """Number of parameters updated by the optimizer."""

        return self._count_parameters(self, trainable_only=True)

    @property
    def total_parameter_count(self) -> int:
        """Number of all model parameters (trainable and frozen)."""

        return self._count_parameters(self, trainable_only=False)

    def _phase1_representation(self, source: torch.Tensor) -> torch.Tensor:
        """Return native DTAAD phase-1 representation ``[L,B,F]``."""

        local_attention = self.l_tcn(source)
        encoded_source = local_attention.permute(2, 0, 1)
        encoded_source = encoded_source * math.sqrt(self.input_dim)
        encoded_source = self.pos_encoder(encoded_source)
        latent = self.transformer_encoder1(encoded_source)
        return latent + self.fcn(latent)

    def _phase2_representation(
        self,
        source: torch.Tensor,
        phase1_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Run DTAAD callback with phase-1 condition broadcast over context."""

        broadcast_condition = phase1_prediction.expand(
            -1,
            -1,
            self.context_length,
        )
        callback_source = source + broadcast_condition
        global_attention = self.g_tcn(callback_source)
        encoded_source = global_attention.permute(2, 0, 1)
        encoded_source = encoded_source * math.sqrt(self.input_dim)
        encoded_source = self.pos_encoder(encoded_source)
        latent = self.transformer_encoder2(encoded_source)
        return latent + self.fcn(latent)

    def forward(self, context: torch.Tensor) -> ForecastOutput:
        """Forecast configured horizons using observed context only."""

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

        # Repository DTAAD uses [B,F,L].  No value outside this observed
        # context is accepted or constructed before phase 1.
        source = context.transpose(1, 2).contiguous()
        phase1_representation = self._phase1_representation(source)
        decoder1_input = phase1_representation.permute(1, 2, 0)
        phase1_native = [
            decoder(decoder1_input) for decoder in self.decoder1_heads
        ]

        phase2_native: list[torch.Tensor] = []
        for phase1_prediction, decoder2 in zip(
            phase1_native,
            self.decoder2_heads,
            strict=True,
        ):
            phase2_representation = self._phase2_representation(
                source,
                phase1_prediction,
            )
            decoder2_input = phase2_representation.permute(1, 2, 0)
            phase2_native.append(decoder2(decoder2_input))

        phase1 = torch.stack(
            [prediction.squeeze(-1) for prediction in phase1_native],
            dim=1,
        )
        phase2 = torch.stack(
            [prediction.squeeze(-1) for prediction in phase2_native],
            dim=1,
        )
        return ForecastOutput(prediction=phase2, phase1=phase1)


__all__ = ["DTAADQASForecastAdapter"]
