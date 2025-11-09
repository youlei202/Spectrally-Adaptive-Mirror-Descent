"""Diagonal AdaGrad implementation."""

from __future__ import annotations

import numpy as np


class AdaGradDiag:
    """AdaGrad with diagonal preconditioner."""

    def __init__(self, dimension: int, lr: float = 1.0, epsilon: float = 1e-8) -> None:
        self.dimension = dimension
        self.lr = lr
        self.epsilon = epsilon
        self.accumulator = np.zeros(dimension)

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, dict]:
        if gradient.shape != (self.dimension,):
            raise ValueError("Gradient shape mismatch for AdaGradDiag.")
        self.accumulator += gradient**2
        denom = np.sqrt(self.accumulator) + self.epsilon
        step = gradient / denom
        new_weights = weights - self.lr * step
        stats = {"conditioning": float(denom.max() / denom.min())}
        return new_weights, stats
