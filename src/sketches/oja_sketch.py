"""Oja-style streaming sketch for principal subspace tracking."""

from __future__ import annotations

import numpy as np

from src.utils.reproducibility import set_global_seeds


class OjaSketch:
    """Maintains an orthonormal basis tracking the covariance subspace."""

    def __init__(
        self,
        dimension: int,
        rank: int,
        step_size: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.step_size = float(step_size)
        self.seed = seed
        self._init_basis()

    def _init_basis(self) -> None:
        rng = set_global_seeds(self.seed)
        Q, _ = np.linalg.qr(rng.normal(size=(self.dimension, self.rank)))
        self.basis = Q
        self.eigenvalues = np.zeros(self.rank)

    def reset(self) -> None:
        self._init_basis()

    def update(self, gradient: np.ndarray) -> None:
        if gradient.shape != (self.dimension,):
            raise ValueError("gradient shape mismatch")
        projected = self.basis.T @ gradient
        self.eigenvalues = (1 - self.step_size) * self.eigenvalues + self.step_size * (
            projected**2
        )
        delta = np.outer(gradient, projected)
        self.basis += self.step_size * delta
        self._reorthogonalize()

    def _reorthogonalize(self) -> None:
        Q, _ = np.linalg.qr(self.basis)
        self.basis = Q

    def covariance(self) -> np.ndarray:
        Lambda = np.diag(np.maximum(self.eigenvalues, 1e-8))
        return self.basis @ Lambda @ self.basis.T

    def metric(self, ridge: float = 1e-6) -> np.ndarray:
        return self.covariance() + ridge * np.eye(self.dimension)

    def factor(self) -> np.ndarray:
        scales = np.sqrt(np.maximum(self.eigenvalues, 0.0))
        return self.basis * scales
