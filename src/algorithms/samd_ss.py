"""Spectrally Adaptive Mirror Descent with Streaming Sketches (SAMD-SS)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from src.utils import projections


Schedule = Callable[[int], float]


def _identity_schedule(_: int) -> float:
    return 1.0


@dataclass
class ConstraintSpec:
    """Supported constraint description."""

    type: str
    radius: float


@dataclass
class SAMDSSState:
    """Internal book-keeping exposed via `get_state`."""

    step: int = 0
    sum_potential: float = 0.0
    q_history: list[float] = field(default_factory=list)
    last_lr: float = 0.0


class SAMDSS:
    """Implements SAMD-SS with a streaming sketch preconditioner."""

    def __init__(
        self,
        dim: int,
        lambda_ridge: float,
        step_schedule: Schedule | None,
        sketch_backend,
        sketch_kwargs: Optional[Dict] = None,
        constraint: Optional[Dict] = None,
        alpha_strong_convexity: Optional[float] = None,
    ) -> None:
        self.dim = int(dim)
        self.lambda_ridge = float(lambda_ridge)
        if self.lambda_ridge <= 0:
            raise ValueError("lambda_ridge must be positive.")
        self.step_schedule = step_schedule or _identity_schedule
        self.sketch_backend = sketch_backend
        self.sketch_kwargs = sketch_kwargs or {}
        self.alpha_strong_convexity = alpha_strong_convexity
        self.constraint = None
        if constraint:
            if constraint.get("type") != "l2_ball":
                raise ValueError(f"Unsupported constraint: {constraint}")
            self.constraint = ConstraintSpec(
                type="l2_ball", radius=float(constraint.get("radius", 1.0))
            )

        self.state = SAMDSSState()
        self._rng = np.random.default_rng()
        self.sketch = self._instantiate_sketch()
        self.x = np.zeros(self.dim, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Initialization utilities
    # ------------------------------------------------------------------ #
    def _instantiate_sketch(self):
        sketch = self.sketch_backend(dimension=self.dim, **self.sketch_kwargs)
        if hasattr(sketch, "reset"):
            sketch.reset()
        return sketch

    def reset(
        self,
        x0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Reset iterate, RNG, and sketch state."""

        self._rng = rng or np.random.default_rng()
        self.x = np.zeros(self.dim, dtype=np.float64) if x0 is None else x0.astype(
            np.float64
        )
        if self.x.shape != (self.dim,):
            raise ValueError("x0 has incompatible shape.")
        if hasattr(self.sketch, "reset"):
            self.sketch.reset()
        else:
            self.sketch = self._instantiate_sketch()
        self.state = SAMDSSState()

    # ------------------------------------------------------------------ #
    # Core step
    # ------------------------------------------------------------------ #
    def step(self, grad: np.ndarray, t: Optional[int] = None) -> tuple[np.ndarray, Dict]:
        """Perform one SAMD-SS update with gradient `grad`."""

        grad = grad.astype(np.float64)
        if grad.shape != (self.dim,):
            raise ValueError("Gradient shape mismatch for SAMDSS.")

        if t is None:
            self.state.step += 1
        else:
            self.state.step = int(t)

        # Update sketch with current gradient.
        self.sketch.update(grad)

        eta_t = float(self.step_schedule(self.state.step))
        if eta_t <= 0:
            raise ValueError("Step size schedule must produce positive values.")

        direction = self._preconditioned_direction(grad)
        q_t = float(grad @ direction)
        self.state.sum_potential += q_t
        self.state.q_history.append(q_t)
        self.state.last_lr = eta_t

        new_x = self.x - eta_t * direction
        if self.constraint is not None and self.constraint.radius > 0:
            new_x = projections.project_l2_ball(new_x, self.constraint.radius)

        self.x = new_x
        info = {
            "lr": eta_t,
            "elliptical_potential": q_t,
            "preconditioned_norm": float(np.linalg.norm(direction)),
        }
        return self.x.copy(), info

    def _preconditioned_direction(self, grad: np.ndarray) -> np.ndarray:
        """Solve H^{-1} grad via Woodbury using the sketch factor."""

        B = self._sketch_factor()  # shape (dim, r)
        lambda_inv = 1.0 / self.lambda_ridge
        if B.size == 0:
            return lambda_inv * grad

        Bt_grad = B.T @ grad  # shape (r,)
        gram = B.T @ B
        system = np.eye(gram.shape[0], dtype=np.float64) + lambda_inv * gram
        rhs = Bt_grad
        middle = np.linalg.solve(system, rhs)
        return lambda_inv * grad - (lambda_inv**2) * (B @ middle)

    def _sketch_factor(self) -> np.ndarray:
        """Return a factor B whose Gram defines the sketch matrix."""

        if hasattr(self.sketch, "factor"):
            factor = self.sketch.factor()
            if factor is None:
                return np.zeros((self.dim, 0))
            if factor.shape[0] != self.dim:
                raise ValueError("Sketch factor has mismatched dimension.")
            return factor
        # Fallback: build dense metric and compute its Cholesky (costly but correct).
        metric = self.sketch.metric()
        chol = np.linalg.cholesky(
            metric - self.lambda_ridge * np.eye(self.dim, dtype=np.float64)
        )
        return chol.T

    # ------------------------------------------------------------------ #
    # Compatibility helpers
    # ------------------------------------------------------------------ #
    def get_state(self) -> SAMDSSState:
        return self.state

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, Dict]:
        """Compatibility wrapper used by the experiment runner."""

        self.x = weights.astype(np.float64)
        new_w, info = self.step(gradient)
        return new_w, info

    @property
    def supports_potential(self) -> bool:
        return True
