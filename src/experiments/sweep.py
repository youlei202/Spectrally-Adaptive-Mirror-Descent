"""Utility to run a deterministic sweep over YAML configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import List

from src.experiments.run_experiment import run_training
from src.utils import config as config_utils
from src.utils import logging as logging_utils


LOGGER = logging_utils.get_logger("sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--filter", default=None, help="Substring to filter config names.")
    return parser.parse_args()


def discover_configs(config_dir: Path, pattern: str | None) -> List[Path]:
    configs = sorted(config_dir.glob("*.yaml"))
    if pattern:
        configs = [cfg for cfg in configs if pattern in cfg.stem]
    return configs


def _set_nested_value(cfg: dict, dotted_key: str, value) -> None:
    target = cfg
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        target = target.setdefault(key, {})
    target[parts[-1]] = value


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    artifacts_dir = Path(args.artifacts_dir)
    logs_dir = artifacts_dir / "logs"
    tables_dir = artifacts_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for cfg_path in discover_configs(config_dir, args.filter):
        base_cfg = config_utils.load_config(cfg_path)
        base_cfg.setdefault("name", cfg_path.stem)
        sweep_cfg = base_cfg.get("sweep")
        if sweep_cfg:
            param = sweep_cfg.get("param")
            values = sweep_cfg.get("values", [])
            if not param or not values:
                raise ValueError(f"Invalid sweep section in {cfg_path}")
            for value in values:
                cfg = copy.deepcopy(base_cfg)
                _set_nested_value(cfg, param, value)
                cfg["name"] = f"{base_cfg['name']}_{sweep_cfg.get('label', 'val')}{value}"
                cfg.pop("sweep", None)
                LOGGER.info("Running config %s (sweep=%s)", cfg_path.name, value)
                summary = run_training(cfg, logs_dir)
                summaries.append(summary)
        else:
            LOGGER.info("Running config %s", cfg_path.name)
            summary = run_training(base_cfg, logs_dir)
            summaries.append(summary)

    summary_path = tables_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    LOGGER.info("Wrote sweep summary to %s", summary_path)


if __name__ == "__main__":
    main()
