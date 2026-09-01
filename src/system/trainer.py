"""Training loop.

Kept deliberately small: it owns the optimiser, the epoch loop and early
stopping, and nothing else. Corruption belongs to the data module, the loss to
``src.losses``, and every scoring decision to ``src.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    amp: bool = False
    log_interval: int = 50


class Trainer:
    """Fit one detector and keep the best checkpoint by validation loss.

    ``amp`` defaults to off. Gradient clipping runs with
    ``error_if_nonfinite=True`` so a numerical blow-up stops the run instead of
    being silently skipped; with channels whose scales differ by orders of
    magnitude, fp16 overflow is a real failure mode and should be visible.
    """

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        corruption,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.objective = objective
        self.corruption = corruption
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        self.amp_enabled = bool(config.amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        examples = 0
        for batch in loader:
            clean_context = batch["context"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            model_input = self.corruption(clean_context) if training else clean_context

            with torch.set_grad_enabled(training), torch.amp.autocast(
                "cuda", enabled=self.amp_enabled
            ):
                output = self.model(model_input)
                breakdown = self.objective(output, target, clean_context)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(breakdown.total).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            size = clean_context.size(0)
            examples += size
            for name, value in breakdown.detached().items():
                totals[name] = totals.get(name, 0.0) + value * size
        if not examples:
            raise RuntimeError("the loader produced no batches")
        return {name: value / examples for name, value in totals.items()}

    def fit(
        self, train_loader: DataLoader, validation_loader: DataLoader, checkpoint_path: str | Path
    ) -> list[dict[str, Any]]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, Any]] = []
        best_loss = float("inf")
        stale_epochs = 0

        for epoch in range(self.config.epochs):
            train_metrics = self.run_epoch(train_loader, training=True)
            with torch.no_grad():
                validation_metrics = self.run_epoch(validation_loader, training=False)
            self.scheduler.step(validation_metrics["total"])

            improved = validation_metrics["total"] < best_loss
            if improved:
                best_loss = validation_metrics["total"]
                stale_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "model_config": self.model.get_config(),
                        "objective_config": self.objective.get_config(),
                        "validation_loss": best_loss,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1

            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "best_validation_loss": best_loss,
                    "checkpoint_written": improved,
                }
            )
            if stale_epochs >= self.config.patience:
                break
        return history
