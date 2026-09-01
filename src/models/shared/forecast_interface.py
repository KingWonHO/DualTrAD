from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class ForecastOutput:
    """Common output used by the protocol-locked baseline trainer.

    ``prediction`` is always the final prediction used at validation and test
    time.  Two-phase models may additionally expose ``phase1`` so their native
    training objective can be retained without changing the evaluation rule.
    """

    prediction: torch.Tensor
    phase1: torch.Tensor | None = None


class ForecastAdapter(Protocol):
    adaptation_metadata: dict[str, Any]

    def __call__(self, context: torch.Tensor) -> ForecastOutput: ...
