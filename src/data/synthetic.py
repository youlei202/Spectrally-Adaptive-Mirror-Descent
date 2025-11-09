"""Deterministic synthetic dataset generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.utils.reproducibility import set_global_seeds


@dataclass
class Dataset:
    features: np.ndarray
    targets: np.ndarray
    name: str


def make_regression(
    n_samples: int,
    n_features: int,
    noise: float,
    condition_number: float,
    seed: int = 0,
) -> Tuple[Dataset, np.ndarray]:
    """Generate a well-controlled regression problem."""

    rng = set_global_seeds(seed)
    covariance = _spiked_covariance(n_features, condition_number, rng)
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)
    w_star = rng.normal(scale=1.0, size=n_features)
    y = X @ w_star + noise * rng.normal(size=n_samples)
    dataset = Dataset(features=X, targets=y, name="synthetic_regression")
    return dataset, w_star


def make_logistic(
    n_samples: int,
    n_features: int,
    seed: int = 0,
) -> Tuple[Dataset, np.ndarray]:
    """Generate a logistic regression task with balanced labels."""

    rng = set_global_seeds(seed)
    w_star = rng.normal(scale=1.0, size=n_features)
    X = rng.normal(scale=1.0, size=(n_samples, n_features))
    logits = X @ w_star
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)
    dataset = Dataset(features=X, targets=y, name="synthetic_logistic")
    return dataset, w_star


def _spiked_covariance(
    dimension: int, condition_number: float, rng: np.random.Generator
) -> np.ndarray:
    diag = np.linspace(condition_number, 1.0, num=dimension)
    Q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return Q @ np.diag(diag) @ Q.T
