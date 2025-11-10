"""Implementation of the Frequent Directions sketch."""

from __future__ import annotations

import numpy as np


class FrequentDirections:
    """Streaming sketch approximating the covariance matrix."""

    def __init__(self, dimension: int, rank: int, ridge: float = 1e-6) -> None:
        if rank <= 0 or rank > dimension:
            raise ValueError("rank must be between 1 and dimension")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.buffer = np.zeros((self.rank, self.dimension))
        self.next_row = 0

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.next_row = 0

    def update(self, gradient: np.ndarray) -> None:
        if gradient.shape != (self.dimension,):
            raise ValueError("gradient shape mismatch")
        self.buffer[self.next_row] = gradient
        self.next_row += 1
        if self.next_row == self.rank:
            self._shrink()

    def _shrink(self) -> None:
        U, s, Vt = np.linalg.svd(self.buffer, full_matrices=False)
        delta = s[-1] ** 2
        shrunk = np.sqrt(np.maximum(s**2 - delta, 0.0))
        self.buffer = np.diag(shrunk) @ Vt
        self.buffer[-1] = 0.0
        self.next_row = self.rank - 1

    def covariance(self) -> np.ndarray:
        return self.buffer.T @ self.buffer

    def metric(self) -> np.ndarray:
        return self.covariance() + self.ridge * np.eye(self.dimension)

    def factor(self) -> np.ndarray:
        if self.rank == 0:
            return np.zeros((self.dimension, 0))
        return self.buffer.T
