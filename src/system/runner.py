"""One configuration in, one finished run out.

``run_from_config`` is the only entry point a user needs: it builds the splits,
the model and the objective from a YAML file, trains, fixes the operating point
on calibration data, and evaluates once on test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..datas import ContextCorruption, StandardScaler, WindowDataset, read_aggregate_split
from ..losses import DualTrADObjective
from ..metrics import EvidenceNormalizer
from ..models import get as get_model
from ..utils import seed_everything, select_device
from .evaluator import collect_evidence, evaluate, fit_threshold, vehicle_scores
from .trainer import Trainer, TrainerConfig


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the detector named by ``model.name``."""
    model_config = dict(config["model"])
    name = model_config.pop("name")
    return get_model(name)(**model_config)


def build_objective(config: dict[str, Any]) -> DualTrADObjective:
    return DualTrADObjective(**config.get("loss", {}))


def _loader(dataset: WindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_from_config(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Train and evaluate one model. Returns the metrics it wrote to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = config["experiment"]
    data = config["data"]
    seed_everything(experiment["seed"], experiment.get("deterministic", True))
    device = select_device(experiment.get("device", "auto"))

    root = Path(data["aggregate_dir"])
    features = data["features"]
    read = lambda split, labels=None: read_aggregate_split(root / f"{split}.npz", features, labels)

    train_vehicles = read(data["train_split"], {0})
    scaler = StandardScaler.fit(train_vehicles)

    def windows(vehicles, cap):
        return WindowDataset(
            [v.scaled(scaler) for v in vehicles],
            context_length=data["context_length"],
            horizon_steps=data["horizon_steps"],
            stride=data.get("stride", 1),
            offset_mode=data.get("offset_mode", "series_first"),
            max_windows_per_vehicle=cap,
        )

    train_cap = data.get("max_train_windows_per_vehicle")
    eval_cap = data.get("max_eval_windows_per_vehicle")
    train = windows(train_vehicles, train_cap)
    validation = windows(read(data["validation_split"], {0}), train_cap)
    calibration = windows(read(data["calibration_split"], {0}), eval_cap)
    test_normal = windows(read(data["test_normal_split"], {0}), eval_cap)
    test_abnormal = windows(read(data["test_abnormal_split"], {1}), eval_cap)

    batch_size = data.get("batch_size", 512)
    model = build_model(config)
    objective = build_objective(config)
    corruption = ContextCorruption(**config.get("corruption", {}))

    trainer = Trainer(
        model, objective, corruption, TrainerConfig(**config.get("training", {})), device
    )
    history = trainer.fit(
        _loader(train, batch_size, True),
        _loader(validation, batch_size, False),
        output_dir / "best.pt",
    )
    model.load_state_dict(torch.load(output_dir / "best.pt", weights_only=True)["model_state"])

    scoring = config.get("scoring", {})
    gather = lambda ds: collect_evidence(
        model,
        _loader(ds, batch_size, False),
        device,
        scoring.get("prediction_error", "squared"),
        config.get("loss", {}).get("derivative_weight", 0.0),
    )

    # Scales come from normal validation windows; the threshold from normal
    # calibration vehicles. Neither touches test data.
    reference = gather(validation)
    normalizer = EvidenceNormalizer.fit(
        reference.forecast_residual,
        reference.reconstruction_residual,
        scoring.get("normalizer_quantile", 0.99),
    )
    fusion = scoring.get("fusion", "forecast_only")
    quantile = scoring.get("vehicle_quantile", 0.99)

    calibration_scores, calibration_order = vehicle_scores(
        gather(calibration), normalizer, fusion, quantile
    )
    threshold = fit_threshold(
        calibration_scores,
        calibration.labels[calibration_order],
        scoring.get("target_fpr", 0.10),
    )

    test_scores, test_order, test_labels = [], [], []
    for dataset in (test_normal, test_abnormal):
        scores, order = vehicle_scores(gather(dataset), normalizer, fusion, quantile)
        test_scores.append(scores)
        test_labels.append(dataset.labels[order])
    import numpy as np

    metrics = evaluate(
        np.concatenate(test_scores), np.concatenate(test_labels), threshold
    )

    payload = {
        "experiment": experiment,
        "model": model.get_config(),
        "objective": objective.get_config(),
        "corruption": corruption.get_config(),
        "scaler": scaler.get_config(),
        "normalizer": normalizer.get_config(),
        "datasets": {
            name: ds.get_config()
            for name, ds in (
                ("train", train), ("validation", validation), ("calibration", calibration),
                ("test_normal", test_normal), ("test_abnormal", test_abnormal),
            )
        },
        "metrics": metrics,
        "history": history,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    return payload
