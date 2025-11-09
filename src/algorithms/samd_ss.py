"""Spectrally Adaptive Mirror Descent with Streaming Sketches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from src.sketches.frequent_directions import FrequentDirections
from src.sketches.oja_sketch import OjaSketch
from src.sketches.randomized_svd_stream import RandomizedSVDStream
from src.utils.linalg import elliptical_potential, solve_psd

SketchName = Literal["frequent_directions", "oja", "randomized_svd"]


@dataclass
class SAMDSSConfig:
    dimension: int
    sketch_type: SketchName
    sketch_rank: int
    eta: float = 1.0
    lambda_reg: float = 1e-3
    epsilon: float = 0.1
    min_lr: float = 0.0


class SAMDSS:
    """Implements SAMD-SS using a user-selected sketch."""

    def __init__(self, config: SAMDSSConfig) -> None:
        config.sketch_rank = int(config.sketch_rank)
        config.lambda_reg = float(config.lambda_reg)
        config.eta = float(config.eta)
        config.epsilon = float(config.epsilon)
        config.min_lr = float(config.min_lr)
        self.config = config
        self.step_count = 0
        self.sketch = self._build_sketch()

    def _build_sketch(self):
        cfg = self.config
        if cfg.sketch_type == "frequent_directions":
            return FrequentDirections(cfg.dimension, cfg.sketch_rank, ridge=cfg.lambda_reg)
        if cfg.sketch_type == "oja":
            return OjaSketch(cfg.dimension, cfg.sketch_rank, seed=0)
        if cfg.sketch_type == "randomized_svd":
            return RandomizedSVDStream(cfg.dimension, cfg.sketch_rank, ridge=cfg.lambda_reg)
        raise ValueError(f"Unknown sketch type: {cfg.sketch_type}")

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, Dict]:
        if gradient.shape != (self.config.dimension,):
            raise ValueError("Gradient shape mismatch for SAMD-SS.")
        self.step_count += 1
        self.sketch.update(gradient)
        metric = self.sketch.metric()
        base_lr = (
            self.config.eta * self.config.lambda_reg / max(np.sqrt(self.step_count), 1.0)
        )
        lr = max(base_lr, self.config.min_lr)
        direction = solve_psd(metric, gradient)
        new_weights = weights - lr * direction
        stats = {
            "lr": lr,
            "elliptical_potential": elliptical_potential(metric, gradient),
        }
        return new_weights, stats
