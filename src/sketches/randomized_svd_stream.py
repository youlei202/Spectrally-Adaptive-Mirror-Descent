"""Randomized streaming SVD sketch used by SAMD-SS."""

from __future__ import annotations

import numpy as np

from src.utils.reproducibility import set_global_seeds


class RandomizedSVDStream:
    """Low-rank approximation via random projections."""

    def __init__(
        self,
        dimension: int,
        rank: int,
        ridge: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if rank <= 0 or rank > dimension:
            raise ValueError("rank must be between 1 and dimension")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.seed = seed
        self._init_storage()

    def _init_storage(self) -> None:
        rng = set_global_seeds(self.seed)
        self.test_matrix = rng.normal(size=(self.dimension, self.rank))
        self.sketch = np.zeros((self.rank, self.dimension))

    def reset(self) -> None:
        self._init_storage()

    def update(self, gradient: np.ndarray) -> None:
        if gradient.shape != (self.dimension,):
            raise ValueError("gradient shape mismatch")
        compressed = self.test_matrix.T @ gradient
        self.sketch += np.outer(compressed, gradient)

    def covariance(self) -> np.ndarray:
        approx = self.test_matrix @ self.sketch
        approx = 0.5 * (approx + approx.T)
        eigvals, eigvecs = np.linalg.eigh(approx)
        eigvals = np.clip(eigvals, 0.0, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def metric(self) -> np.ndarray:
        return self.covariance() + self.ridge * np.eye(self.dimension)

    def factor(self) -> np.ndarray:
        return self.sketch.T
