"""Squared loss with ridge regularization."""

from __future__ import annotations

import numpy as np


class SquaredLoss:
    """Standard squared loss with optional ridge penalty."""

    def __init__(self, l2_reg: float = 0.0) -> None:
        self.l2_reg = float(l2_reg)

    def loss(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        residual = x @ weights - y
        data_loss = 0.5 * np.mean(residual**2)
        reg_loss = 0.5 * self.l2_reg * float(weights @ weights)
        return data_loss + reg_loss

    def grad(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        residual = x @ weights - y
        grad = x.T @ residual / x.shape[0]
        return grad + self.l2_reg * weights
