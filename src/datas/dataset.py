"""Vehicle series, per-channel standardisation, and sliding windows.

Two invariants hold everywhere in this module:

* A window never spans two vehicles, and never bridges a gap in the source row
  index. If the rows are not contiguous the loader fails loudly instead of
  inventing a time link.
* The scaler is fitted on training-normal vehicles only. Calibration and test
  vehicles are transformed with it, never used to fit it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class StandardScaler:
    """Per-channel mean/std. A single channel keeps scalar statistics."""

    means: tuple[float, ...]
    stds: tuple[float, ...]

    @classmethod
    def fit(cls, vehicles: Iterable["VehicleSeries"]) -> "StandardScaler":
        count = 0
        total = square_total = None
        for vehicle in vehicles:
            values = np.asarray(vehicle.values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"VehicleSeries.values must be [N, F], got {values.shape}")
            if total is None:
                total = np.zeros(values.shape[1])
                square_total = np.zeros(values.shape[1])
            elif values.shape[1] != total.size:
                raise ValueError("vehicles disagree on the channel count")
            count += values.shape[0]
            total += values.sum(axis=0)
            square_total += np.square(values).sum(axis=0)
        if not count:
            raise ValueError("no values to fit the scaler on")
        mean = total / count
        std = np.sqrt(np.maximum(square_total / count - mean * mean, 0.0))
        if not np.all(np.isfinite(mean)) or not np.all(std > 0):
            raise ValueError(f"degenerate scaler statistics: mean={mean}, std={std}")
        return cls(tuple(float(v) for v in mean), tuple(float(v) for v in std))

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.means, dtype=np.float32)
        std = np.asarray(self.stds, dtype=np.float32)
        if values.shape[-1] != mean.size:
            raise ValueError(
                f"scaler has {mean.size} channels but values have {values.shape[-1]}"
            )
        return ((values - mean) / std).astype(np.float32, copy=False)

    def get_config(self) -> dict[str, Any]:
        return {"type": "StandardScaler", "means": list(self.means), "stds": list(self.stds)}


@dataclass(frozen=True)
class VehicleSeries:
    """One vehicle's contiguous, valid rows."""

    vehicle_id: str
    values: np.ndarray            # [N, F]
    binary_label: int
    source_row_index: np.ndarray  # [N]

    def scaled(self, scaler: StandardScaler) -> "VehicleSeries":
        return replace(self, values=scaler.transform(self.values))


def read_aggregate_split(
    path: str | Path, features: str | Sequence[str], allowed_labels: set[int] | None = None
) -> list[VehicleSeries]:
    """Read one split NPZ and select the requested channels, in order."""
    archive = np.load(Path(path), allow_pickle=False)
    columns = [str(name) for name in archive["columns"].tolist()]
    names = [features] if isinstance(features, str) else list(features)
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"features {missing!r} are not in columns {columns}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate features requested: {names!r}")
    indices = [columns.index(name) for name in names]

    values, mask = archive["X"], archive["valid_mask"]
    offsets, ids = archive["vehicle_offsets"], archive["vehicle_ids"]
    labels, rows = archive["vehicle_binary_labels"], archive["source_row_index"]

    series: list[VehicleSeries] = []
    for index, vehicle_id in enumerate(ids):
        label = int(labels[index])
        if allowed_labels is not None and label not in allowed_labels:
            continue
        start, stop = int(offsets[index]), int(offsets[index + 1])
        valid = mask[start:stop]
        selected = np.asarray(values[start:stop][:, indices], dtype=np.float32)[valid]
        if not np.isfinite(selected).all():
            raise ValueError(f"vehicle {vehicle_id}: non-finite values in {names}")
        series.append(
            VehicleSeries(
                vehicle_id=str(vehicle_id),
                values=selected,
                binary_label=label,
                source_row_index=np.asarray(rows[start:stop], dtype=np.int64)[valid],
            )
        )
    return series


class WindowDataset(Dataset):
    """Sliding context/target windows over a set of vehicles.

    Each item carries the corrupted context the model sees, the clean context
    the reconstruction loss is scored against, the future target, and the
    vehicle it came from.
    """

    def __init__(
        self,
        vehicles: Sequence[VehicleSeries],
        context_length: int,
        horizon_steps: Sequence[int],
        stride: int = 1,
        offset_mode: str = "series_first",
        max_windows_per_vehicle: int | None = None,
        response_indices: Sequence[int] | None = None,
    ) -> None:
        if offset_mode not in {"series_first", "window_first", "none"}:
            raise ValueError(f"unknown offset_mode {offset_mode!r}")
        self.vehicles = list(vehicles)
        self.context_length = context_length
        self.horizon_steps = tuple(int(step) for step in horizon_steps)
        self.offset_mode = offset_mode
        self.response_indices = (
            None if response_indices is None else list(response_indices)
        )
        self.index: list[tuple[int, int]] = []
        span = context_length + max(self.horizon_steps)
        for vehicle_index, vehicle in enumerate(self.vehicles):
            rows = vehicle.source_row_index
            starts = []
            for start in range(0, len(vehicle.values) - span + 1, stride):
                block = rows[start : start + span]
                # Contiguity check: a gap means the window would bridge unseen time.
                if block[-1] - block[0] != span - 1:
                    continue
                starts.append(start)
            if max_windows_per_vehicle is not None and len(starts) > max_windows_per_vehicle:
                keep = np.linspace(0, len(starts) - 1, max_windows_per_vehicle).astype(int)
                starts = [starts[i] for i in keep]
            self.index.extend((vehicle_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        vehicle_index, start = self.index[item]
        vehicle = self.vehicles[vehicle_index]
        values = vehicle.values
        context = values[start : start + self.context_length]
        targets = np.stack(
            [values[start + self.context_length - 1 + step] for step in self.horizon_steps]
        )

        if self.offset_mode == "series_first":
            offset = values[0]
        elif self.offset_mode == "window_first":
            offset = context[0]
        else:
            offset = np.zeros(values.shape[1], dtype=np.float32)
        context = context - offset
        targets = targets - offset

        if self.response_indices is not None:
            targets = targets[:, self.response_indices]

        return {
            "context": torch.from_numpy(np.ascontiguousarray(context)),
            "target": torch.from_numpy(np.ascontiguousarray(targets)),
            "offset": torch.from_numpy(np.ascontiguousarray(offset)),
            "vehicle_index": torch.tensor(vehicle_index, dtype=torch.long),
            "start_index": torch.tensor(start, dtype=torch.long),
        }

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([vehicle.binary_label for vehicle in self.vehicles], dtype=np.int64)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "WindowDataset",
            "vehicles": len(self.vehicles),
            "windows": len(self.index),
            "context_length": self.context_length,
            "horizon_steps": list(self.horizon_steps),
            "offset_mode": self.offset_mode,
        }
