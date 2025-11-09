#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "[run_all] Ensuring artifacts directories exist"
mkdir -p "${ROOT_DIR}/artifacts/logs" "${ROOT_DIR}/artifacts/figures" "${ROOT_DIR}/artifacts/tables"

echo "[run_all] Running unit tests"
pytest -q

echo "[run_all] Launching experiment sweep"
python -m src.experiments.sweep \
  --config-dir "${ROOT_DIR}/src/experiments/configs" \
  --artifacts-dir "${ROOT_DIR}/artifacts"

echo "[run_all] Executing notebook deterministically"
python -m papermill \
  "${ROOT_DIR}/notebooks/main_experiments.ipynb" \
  "${ROOT_DIR}/notebooks/main_experiments.ipynb" \
  -k python3

echo "[run_all] Done"
