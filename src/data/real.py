"""Wrappers around scikit-learn datasets used in experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.reproducibility import set_global_seeds


@dataclass
class RealDataset:
    train_features: np.ndarray
    train_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    name: str


def load_breast_cancer_dataset(seed: int = 0, test_size: float = 0.2) -> RealDataset:
    """Load and standardize the Breast Cancer dataset."""

    rng = set_global_seeds(seed)
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return RealDataset(
        train_features=X_train,
        train_targets=y_train.astype(np.float64),
        test_features=X_test,
        test_targets=y_test.astype(np.float64),
        name="breast_cancer",
    )
