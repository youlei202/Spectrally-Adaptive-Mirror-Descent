"""Full-matrix AdaGrad baseline."""

from __future__ import annotations

import numpy as np

from src.utils.linalg import solve_psd


class AdaGradFull:
    """Maintains the full Gram matrix for exact preconditioning."""

    def __init__(
        self,
        dimension: int,
        lr: float = 1.0,
        damping: float = 1e-3,
    ) -> None:
        self.dimension = dimension
        self.lr = lr
        self.damping = damping
        self.gram = damping * np.eye(dimension)

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, dict]:
        if gradient.shape != (self.dimension,):
            raise ValueError("Gradient shape mismatch for AdaGradFull.")
        self.gram += np.outer(gradient, gradient)
        precond_grad = solve_psd(self.gram, gradient)
        new_weights = weights - self.lr * precond_grad
        stats = {"logdet": float(np.linalg.slogdet(self.gram)[1])}
        return new_weights, stats
