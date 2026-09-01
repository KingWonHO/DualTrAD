"""YAML configuration with single-parent inheritance.

A config may name a parent with ``extends:``. Parents are merged depth-first
and the child always wins, so a variant only states what it changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_extends(path: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Read one YAML file and merge its ancestors into it."""
    config_path = Path(path).resolve()
    seen = seen or set()
    if config_path in seen:
        raise ValueError(f"circular extends chain at {config_path}")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    parent_name = payload.pop("extends", None)
    if parent_name is None:
        return payload
    parent = resolve_extends(config_path.parent / parent_name, seen)
    return _deep_merge(parent, payload)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a run configuration and record where it came from."""
    config = resolve_extends(path)
    config.setdefault("meta", {})["config_path"] = str(Path(path).resolve())
    return config
