"""Classical stochastic gradient descent baseline."""

from __future__ import annotations

import numpy as np


class SGD:
    """Momentum SGD with constant step size."""

    def __init__(self, dimension: int, lr: float = 0.1, momentum: float = 0.0) -> None:
        self.dimension = dimension
        self.lr = lr
        self.momentum = momentum
        self.velocity = np.zeros(dimension)

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, dict]:
        if gradient.shape != (self.dimension,):
            raise ValueError("Gradient shape mismatch for SGD.")
        if self.momentum:
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradient
            step = self.velocity
        else:
            step = gradient
        new_weights = weights - self.lr * step
        return new_weights, {"lr": self.lr}
