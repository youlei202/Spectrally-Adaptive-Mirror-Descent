"""Stability and generalization proxies."""

from __future__ import annotations

import numpy as np


def path_stability(weights_history: np.ndarray) -> float:
    """Compute cumulative movement of parameters."""

    diffs = np.linalg.norm(np.diff(weights_history, axis=0), axis=1)
    return float(np.sum(diffs))


def generalization_gap(train_losses: np.ndarray, test_losses: np.ndarray) -> float:
    """Return mean test - train losses."""

    return float(np.mean(test_losses) - np.mean(train_losses))
