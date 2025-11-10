#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

WIN_USER="${USERNAME:-${USER:-}}"
WIN_USER_HOME=""
if [ -n "${WIN_USER}" ]; then
  if [ -d "/mnt/c/Users/${WIN_USER}" ]; then
    WIN_USER_HOME="/mnt/c/Users/${WIN_USER}"
  elif [ -d "/c/Users/${WIN_USER}" ]; then
    WIN_USER_HOME="/c/Users/${WIN_USER}"
  fi
fi

if [ -z "${PYTHON_BIN+x}" ] && [ -n "${WIN_USER_HOME}" ] && [ -x "${WIN_USER_HOME}/Anaconda3/python.exe" ]; then
  PYTHON_BIN=("${WIN_USER_HOME}/Anaconda3/python.exe")
fi

if [ -z "${PYTHON_BIN+x}" ]; then
  PYTHON_CANDIDATES=("py -3" "py" "python" "python3")
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    IFS=' ' read -r -a parts <<< "$candidate"
    if command -v "${parts[0]}" >/dev/null 2>&1; then
      PYTHON_BIN=("${parts[@]}")
      break
    fi
  done
else
  IFS=' ' read -r -a parts <<< "$PYTHON_BIN"
  PYTHON_BIN=("${parts[@]}")
fi

if [ -z "${PYTHON_BIN[0]+x}" ]; then
    echo "[run_all] Python executable not found." >&2
    exit 1
fi

echo "[run_all] Ensuring artifacts directories exist"
mkdir -p "${ROOT_DIR}/artifacts/logs" "${ROOT_DIR}/artifacts/figures" "${ROOT_DIR}/artifacts/tables"

echo "[run_all] Running unit tests"
"${PYTHON_BIN[@]}" -m pytest -q

echo "[run_all] Launching experiment sweep"
"${PYTHON_BIN[@]}" -m src.experiments.sweep \
  --config-dir "${ROOT_DIR}/src/experiments/configs" \
  --artifacts-dir "${ROOT_DIR}/artifacts"

echo "[run_all] Executing notebook deterministically"
"${PYTHON_BIN[@]}" -m papermill \
  "${ROOT_DIR}/notebooks/main_experiments.ipynb" \
  "${ROOT_DIR}/notebooks/main_experiments.ipynb" \
  -k python3

echo "[run_all] Done"
