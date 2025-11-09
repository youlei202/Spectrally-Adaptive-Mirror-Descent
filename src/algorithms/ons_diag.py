"""Diagonal Online Newton Step baseline."""

from __future__ import annotations

import numpy as np


class ONSDiag:
    """Approximate ONS using a diagonal Hessian surrogate."""

    def __init__(
        self,
        dimension: int,
        eta: float = 1.0,
        alpha: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.dimension = dimension
        self.eta = eta
        self.alpha = alpha
        self.epsilon = epsilon
        self.h_diag = alpha * np.ones(dimension)

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, dict]:
        if gradient.shape != (self.dimension,):
            raise ValueError("Gradient shape mismatch for ONSDiag.")
        self.h_diag += gradient**2
        precond_grad = gradient / (self.h_diag + self.epsilon)
        new_weights = weights - self.eta * precond_grad
        stats = {"max_diag": float(self.h_diag.max())}
        return new_weights, stats
