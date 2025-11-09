"""Single-experiment runner configured by YAML files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from src.algorithms.adagrad_diag import AdaGradDiag
from src.algorithms.adagrad_full import AdaGradFull
from src.algorithms.ons_diag import ONSDiag
from src.algorithms.samd_ss import SAMDSS, SAMDSSConfig
from src.algorithms.sgd import SGD
from src.data import real, synthetic
from src.losses.logistic import LogisticLoss
from src.losses.squared import SquaredLoss
from src.metrics import logdet, regret, stability
from src.utils import config as config_utils
from src.utils import logging as logging_utils
from src.utils import projections
from src.utils.reproducibility import set_global_seeds


LOGGER = logging_utils.get_logger("run_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory storing outputs.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dot-separated overrides (repeatable).",
    )
    return parser.parse_args()


def build_dataset(cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data_cfg = cfg["data"]
    seed = cfg.get("seed", 0)
    if data_cfg["name"] == "synthetic_regression":
        dataset, _ = synthetic.make_regression(
            n_samples=data_cfg["n_samples"],
            n_features=data_cfg["n_features"],
            noise=data_cfg.get("noise", 0.1),
            condition_number=data_cfg.get("condition_number", 10.0),
            seed=seed,
        )
        extras = {}
        return dataset.features, dataset.targets, extras
    if data_cfg["name"] == "synthetic_logistic":
        dataset, _ = synthetic.make_logistic(
            n_samples=data_cfg["n_samples"],
            n_features=data_cfg["n_features"],
            seed=seed,
        )
        extras = {}
        return dataset.features, dataset.targets, extras
    if data_cfg["name"] == "breast_cancer":
        dataset = real.load_breast_cancer_dataset(seed=seed, test_size=data_cfg.get("test_size", 0.2))
        extras = {
            "test_features": dataset.test_features,
            "test_targets": dataset.test_targets,
        }
        return dataset.train_features, dataset.train_targets, extras
    raise ValueError(f"Unknown dataset {data_cfg['name']}")


def build_loss(cfg: Dict):
    loss_cfg = cfg["loss"]
    if loss_cfg["name"] == "squared":
        return SquaredLoss(l2_reg=loss_cfg.get("l2_reg", 0.0))
    if loss_cfg["name"] == "logistic":
        return LogisticLoss(l2_reg=loss_cfg.get("l2_reg", 0.0))
    raise ValueError(f"Unknown loss {loss_cfg['name']}")


def build_optimizer(cfg: Dict, dimension: int):
    algo_cfg = cfg["optimizer"]
    name = algo_cfg["name"]
    if name == "sgd":
        return SGD(dimension=dimension, lr=algo_cfg.get("lr", 0.1), momentum=algo_cfg.get("momentum", 0.0))
    if name == "adagrad_diag":
        return AdaGradDiag(dimension=dimension, lr=algo_cfg.get("lr", 1.0))
    if name == "adagrad_full":
        return AdaGradFull(dimension=dimension, lr=algo_cfg.get("lr", 1.0))
    if name == "ons_diag":
        return ONSDiag(
            dimension=dimension,
            eta=algo_cfg.get("eta", 0.5),
            alpha=algo_cfg.get("alpha", 1.0),
        )
    if name == "samd_ss":
        samd_cfg = SAMDSSConfig(
            dimension=dimension,
            sketch_type=algo_cfg.get("sketch_type", "frequent_directions"),
            sketch_rank=algo_cfg.get("sketch_rank", min(64, dimension)),
            eta=algo_cfg.get("eta", 1.0),
            lambda_reg=algo_cfg.get("lambda_reg", 1e-3),
            epsilon=algo_cfg.get("epsilon", 0.1),
        )
        return SAMDSS(samd_cfg)
    raise ValueError(f"Unknown optimizer {name}")


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    perm = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx], y[idx]


def run_training(cfg: Dict, artifacts_dir: Path) -> Dict:
    seed = cfg.get("seed", 0)
    rng = set_global_seeds(seed)
    X, y, extras = build_dataset(cfg)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, dimension=X.shape[1])
    weights = np.zeros(X.shape[1])
    history = [weights.copy()]
    gram = np.zeros((X.shape[1], X.shape[1]))
    potentials = []
    train_losses = []

    train_cfg = cfg["train"]
    num_epochs = train_cfg.get("epochs", 1)
    batch_size = train_cfg.get("batch_size", len(X))
    domain_radius = train_cfg.get(
        "domain_radius", train_cfg.get("domain_diameter", 1.0) / 2.0
    )
    enforce_projection = train_cfg.get("enforce_projection", True)

    for epoch in range(num_epochs):
        for batch_x, batch_y in iterate_minibatches(X, y, batch_size, rng):
            grad = loss_fn.grad(weights, batch_x, batch_y)
            gram += np.outer(grad, grad)
            weights, info = optimizer.update(weights, grad)
            if enforce_projection and domain_radius > 0:
                weights = projections.project_l2_ball(weights, domain_radius)
            potentials.append(info.get("elliptical_potential", float(grad @ grad)))
            batch_loss = loss_fn.loss(weights, batch_x, batch_y)
            train_losses.append(batch_loss)
            history.append(weights.copy())
        LOGGER.info("Epoch %d done. Latest loss=%.4f", epoch + 1, train_losses[-1])

    metrics = summarize_metrics(
        cfg=cfg,
        weights_history=np.stack(history),
        train_losses=np.array(train_losses),
        gram=gram,
        potentials=np.array(potentials),
        extras=extras,
        loss_fn=loss_fn,
        final_weights=weights,
    )
    save_summary(metrics, artifacts_dir)
    return metrics


def summarize_metrics(
    cfg: Dict,
    weights_history: np.ndarray,
    train_losses: np.ndarray,
    gram: np.ndarray,
    potentials: np.ndarray,
    extras: Dict[str, np.ndarray],
    loss_fn,
    final_weights: np.ndarray,
) -> Dict:
    epsilon = float(cfg["optimizer"].get("epsilon", 0.1))
    lambda_reg = float(cfg["optimizer"].get("lambda_reg", 1e-3))
    logdet_value = logdet.logdet_ratio(gram, lambda_reg)
    sum_potential = float(np.sum(potentials))
    ellip_ok = logdet.elliptical_potential_check(sum_potential, logdet_value, epsilon)

    if extras:
        test_X = extras["test_features"]
        test_y = extras["test_targets"]
        test_loss = loss_fn.loss(final_weights, test_X, test_y)
    else:
        test_loss = train_losses[-1]

    regret_result = regret.instance_regret_bound(
        domain_diameter=float(cfg["train"].get("domain_diameter", 1.0)),
        lambda_reg=lambda_reg,
        epsilon=epsilon,
        logdet_ratio=logdet_value,
        observed_regret=float(np.sum(train_losses) - len(train_losses) * train_losses.min()),
    )

    metrics = {
        "experiment": cfg["name"],
        "elliptical_potential_ok": bool(ellip_ok),
        "sum_potential": sum_potential,
        "logdet_value": logdet_value,
        "regret_bound": regret_result.bound,
        "regret_satisfied": bool(regret_result.satisfied),
        "stability": stability.path_stability(weights_history),
        "generalization_gap": stability.generalization_gap(
            train_losses, np.array([test_loss])
        ),
        "train_loss_final": float(train_losses[-1]),
        "test_loss": float(test_loss),
    }
    return metrics


def save_summary(summary: Dict, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / f"{summary['experiment']}_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Saved summary to %s", output_path)


def main() -> None:
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    config_utils.apply_overrides(cfg, args.override)
    artifacts_dir = Path(args.artifacts_dir)
    run_training(cfg, artifacts_dir / "logs")


if __name__ == "__main__":
    main()
