"""Log-determinant based metrics."""

from __future__ import annotations

import numpy as np

from src.utils.linalg import stable_logdet


def logdet_ratio(gram: np.ndarray, lambda_reg: float) -> float:
    """Compute log det((lambda I + G) / (lambda I))."""

    gram = np.asarray(gram, dtype=np.float64)
    lambda_reg = float(lambda_reg)
    dimension = gram.shape[0]
    shifted = gram + lambda_reg * np.eye(dimension)
    return stable_logdet(shifted) - dimension * np.log(lambda_reg)


def elliptical_potential_check(sum_potential: float, logdet_value: float, epsilon: float) -> bool:
    """Return True if S_T <= 2/(1-eps) logdet."""

    if epsilon >= 1.0:
        raise ValueError("epsilon must be < 1")
    upper_bound = 2 * logdet_value / (1 - epsilon)
    return sum_potential <= upper_bound * (1 + 1e-6)
