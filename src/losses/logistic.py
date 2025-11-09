"""Binary logistic loss with ridge penalty."""

from __future__ import annotations

import numpy as np


class LogisticLoss:
    """Smooth logistic loss for binary classification."""

    def __init__(self, l2_reg: float = 0.0) -> None:
        self.l2_reg = float(l2_reg)

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits))

    def loss(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        logits = x @ weights
        probs = self._sigmoid(logits)
        eps = 1e-12
        data_loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        reg_loss = 0.5 * self.l2_reg * float(weights @ weights)
        return data_loss + reg_loss

    def grad(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        logits = x @ weights
        probs = self._sigmoid(logits)
        grad = x.T @ (probs - y) / x.shape[0]
        return grad + self.l2_reg * weights
