"""Config loading and override utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Top level of config must be a mapping.")

    return data


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply dot-separated CLI overrides like `train.lr=0.1`."""

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override, expected k=v format: {override}")
        key, raw_value = override.split("=", maxsplit=1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ValueError(f"Failed parsing override '{override}': {exc}") from exc
        target[parts[-1]] = value
    return config


def ensure_keys(config: Dict[str, Any], required_keys: Iterable[str]) -> None:
    """Validate that all required dotted keys are present."""

    missing = []
    for dotted in required_keys:
        target = config
        for part in dotted.split("."):
            if not isinstance(target, dict) or part not in target:
                missing.append(dotted)
                break
            target = target[part]
    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"Missing required config keys: {joined}")
