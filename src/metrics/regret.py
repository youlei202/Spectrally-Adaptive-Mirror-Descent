"""Instance-dependent regret utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def cumulative_regret(losses: Iterable[float], optimum: float) -> np.ndarray:
    """Compute cumulative regret relative to a reference loss."""

    losses = np.asarray(list(losses), dtype=np.float64)
    return np.cumsum(losses - optimum)


@dataclass
class RegretBoundResult:
    bound: float
    satisfied: bool


def instance_regret_bound(
    domain_diameter: float,
    lambda_reg: float,
    epsilon: float,
    logdet_ratio: float,
    observed_regret: float,
) -> RegretBoundResult:
    """Evaluate the theoretical regret bound."""

    if epsilon >= 1.0:
        raise ValueError("epsilon must be smaller than 1.")
    coeff = (2 * lambda_reg) / (1 - epsilon)
    bound = domain_diameter * np.sqrt(coeff * logdet_ratio)
    satisfied = observed_regret <= bound + 1e-6
    return RegretBoundResult(bound=float(bound), satisfied=bool(satisfied))
