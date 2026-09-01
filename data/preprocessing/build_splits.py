from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "hybridtrad-qas-aggregate-v1"


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vehicles = payload.get("vehicles")
    if not isinstance(vehicles, list):
        raise ValueError(f"vehicles 배열이 없는 manifest: {path}")
    return [entry for entry in vehicles if str(entry["manufacturer"]) == "QAS"]


def _split_train(
    entries: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(entries, key=lambda entry: str(entry["vehicle_id"]))
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, int(round(len(ordered) * validation_fraction)))
    validation_count = min(validation_count, len(ordered) - 1)
    return ordered[validation_count:], ordered[:validation_count]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _aggregate_split(
    role: str,
    entries: list[dict[str, Any]],
    source_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    chunks: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    vehicle_ids: list[str] = []
    labels: list[int] = []
    statuses: list[str] = []
    source_files: list[str] = []
    source_hashes: list[str] = []
    manufacturers: list[str] = []
    offsets = [0]
    columns: np.ndarray | None = None
    units: np.ndarray | None = None

    for entry in sorted(entries, key=lambda item: str(item["vehicle_id"])):
        relative = Path(str(entry["npz_file"]))
        source_path = (source_root / relative).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"source root 밖의 NPZ: {relative}") from error
        with np.load(source_path, allow_pickle=False) as archive:
            manufacturer = str(np.asarray(archive["manufacturer"]).reshape(-1)[0])
            vehicle_id = str(np.asarray(archive["vehicle_id"]).reshape(-1)[0])
            if manufacturer != "QAS" or vehicle_id != str(entry["vehicle_id"]):
                raise ValueError(f"manifest/NPZ ID 불일치: QAS/{entry['vehicle_id']}")
            mask = np.asarray(archive["valid_mask"], dtype=bool)
            x = np.asarray(archive["X"])[mask].astype(np.float32, copy=False)
            rows = np.asarray(archive["source_row_index"], dtype=np.int64)[mask]
            if not np.isfinite(x).all():
                raise ValueError(f"{source_path}: valid X에 NaN/Inf가 있습니다.")
            if len(rows) > 1 and not np.all(np.diff(rows) == 1):
                raise ValueError(f"{source_path}: valid source_row_index가 연속적이지 않습니다.")
            current_columns = np.asarray(archive["columns"])
            current_units = np.asarray(archive["units"])
            if columns is None:
                columns = current_columns
                units = current_units
            elif not np.array_equal(columns, current_columns):
                raise ValueError(f"{source_path}: columns schema 불일치")
            chunks.append(np.ascontiguousarray(x))
            source_rows.append(np.ascontiguousarray(rows))
            vehicle_ids.append(vehicle_id)
            labels.append(int(np.asarray(archive["binary_label"]).reshape(-1)[0]))
            statuses.append(str(np.asarray(archive["label_status"]).reshape(-1)[0]))
            source_files.append(relative.as_posix())
            source_hashes.append(str(np.asarray(archive["source_sha256"]).reshape(-1)[0]))
            manufacturers.append(manufacturer)
            offsets.append(offsets[-1] + len(x))

    if columns is None or units is None or not chunks:
        raise ValueError(f"비어 있는 split: {role}")
    x_all = np.concatenate(chunks, axis=0)
    row_all = np.concatenate(source_rows, axis=0)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            X=x_all,
            valid_mask=np.ones(len(x_all), dtype=bool),
            source_row_index=row_all,
            vehicle_offsets=np.asarray(offsets, dtype=np.int64),
            vehicle_ids=np.asarray(vehicle_ids),
            vehicle_binary_labels=np.asarray(labels, dtype=np.int64),
            vehicle_label_status=np.asarray(statuses),
            source_npz_files=np.asarray(source_files),
            source_sha256=np.asarray(source_hashes),
            vehicle_manufacturers=np.asarray(manufacturers),
            columns=columns,
            units=units,
            manufacturer=np.asarray(["QAS"]),
            sample_period_seconds=np.asarray([30], dtype=np.int64),
            split_role=np.asarray([role]),
            schema_version=np.asarray([SCHEMA_VERSION]),
        )
    os.replace(temporary, output_path)
    return {
        "vehicles": len(vehicle_ids),
        "rows": len(x_all),
        "normal_vehicles": int(np.count_nonzero(np.asarray(labels) == 0)),
        "abnormal_vehicles": int(np.count_nonzero(np.asarray(labels) == 1)),
        "file": output_path.name,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def _manifest_payload(role: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split": role,
        "manufacturer_policy": "QAS_only_DTI_excluded",
        "sample_period_seconds": 30,
        "vehicles": [
            {
                "manufacturer": "QAS",
                "vehicle_id": str(entry["vehicle_id"]),
                "npz_file": str(entry["npz_file"]),
                "label_status": str(entry["label_status"]),
            }
            for entry in sorted(entries, key=lambda item: str(item["vehicle_id"]))
        ],
    }


def build(
    source_root: Path,
    source_split_dir: Path,
    output_dir: Path,
    validation_fraction: float,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_split_dir = source_split_dir.resolve()
    output_dir = output_dir.resolve()
    manifests_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    source_train = _read_manifest(source_split_dir / "train.json")
    train, validation = _split_train(source_train, validation_fraction, seed)
    roles = {
        "train": train,
        "validation": validation,
        "calibration_normal": _read_manifest(
            source_split_dir / "calibration_normal.json"
        ),
        "test_normal": _read_manifest(source_split_dir / "test_normal.json"),
        "test_abnormal": _read_manifest(source_split_dir / "test_abnormal.json"),
    }
    roles["test"] = roles["test_normal"] + roles["test_abnormal"]
    all_keys: set[tuple[str, str]] = set()
    for role, entries in roles.items():
        if role == "test":
            continue
        for entry in entries:
            key = (str(entry["manufacturer"]), str(entry["vehicle_id"]))
            if key in all_keys:
                raise ValueError(f"split leakage: {key}")
            all_keys.add(key)

    expected = [output_dir / f"{role}.npz" for role in roles]
    expected += [manifests_dir / f"{role}.json" for role in roles]
    expected += [output_dir / "summary.json", output_dir / "scaler_temperature.json"]
    conflicts = [path for path in expected if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "기존 QAS split 산출물이 있습니다. --overwrite가 필요합니다: "
            + ", ".join(str(path) for path in conflicts)
        )

    split_summaries: dict[str, Any] = {}
    for role, entries in roles.items():
        _atomic_json(manifests_dir / f"{role}.json", _manifest_payload(role, entries))
        split_summaries[role] = _aggregate_split(
            role, entries, source_root, output_dir / f"{role}.npz"
        )

    with np.load(output_dir / "train.npz", allow_pickle=False) as archive:
        t_index = [str(value) for value in archive["columns"].tolist()].index("T")
        temperatures = np.asarray(archive["X"][:, t_index], dtype=np.float64)
    scaler = {
        "schema_version": "hybridtrad-qas-inner-train-standard-scaler-v1",
        "fit_split": "train",
        "manufacturer": "QAS",
        "feature": "T",
        "count": int(len(temperatures)),
        "mean": float(temperatures.mean()),
        "std": float(temperatures.std(ddof=0)),
        "ddof": 0,
        "target_data_used": False,
    }
    _atomic_json(output_dir / "scaler_temperature.json", scaler)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manufacturer_policy": "QAS_only_DTI_excluded",
        "source_root": str(source_root),
        "source_split_dir": str(source_split_dir),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "splits": split_summaries,
        "temperature_scaler": scaler,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed QAS-only aggregate NPZ splits.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(
        args.source_root,
        args.source_split_dir,
        args.output_dir,
        args.validation_fraction,
        args.seed,
        args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
