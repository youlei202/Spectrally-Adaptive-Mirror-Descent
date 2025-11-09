"""Utilities to guarantee deterministic behavior across the codebase."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


def set_global_seeds(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and hash randomization."""

    if seed is None:
        raise ValueError("seed must be an integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def deterministic_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic permutation using the provided RNG."""

    perm = np.arange(n)
    rng.shuffle(perm)
    return perm


def shard_indices(
    n: int, num_shards: int, rng: np.random.Generator | None = None
) -> List[np.ndarray]:
    """Split indices into shards while keeping determinism."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    rng = rng or np.random.default_rng()
    perm = deterministic_permutation(n, rng)
    return np.array_split(perm, num_shards)


@dataclass
class SeededConfig:
    """Helper to propagate RNG instances through configs."""

    seed: int

    def spawn(self, namespace: str) -> Dict[str, int]:
        child_seed = abs(hash((self.seed, namespace))) % (2**32)
        return {"seed": child_seed}
