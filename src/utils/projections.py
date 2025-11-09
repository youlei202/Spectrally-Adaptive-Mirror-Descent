"""Projection operators used by the optimizers."""

from __future__ import annotations

import numpy as np


def project_l2_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    """Project onto an L2 ball of the given radius."""

    norm = np.linalg.norm(vector)
    if norm <= radius:
        return vector
    if radius <= 0:
        return np.zeros_like(vector)
    return vector * (radius / norm)


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex."""

    if vector.ndim != 1:
        raise ValueError("vector must be 1-D for simplex projection")
    u = np.sort(vector)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, len(u) + 1)
    cond = u - cssv / ind > 0
    if not cond.any():
        theta = 0.0
    else:
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
    return np.maximum(vector - theta, 0.0)
