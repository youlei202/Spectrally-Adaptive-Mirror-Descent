from pathlib import Path

from src.experiments.run_experiment import run_training


def test_end_to_end_training(tmp_path: Path):
    cfg = {
        "name": "unit_test_run",
        "seed": 0,
        "data": {"name": "synthetic_regression", "n_samples": 32, "n_features": 8, "noise": 0.1, "condition_number": 5.0},
        "loss": {"name": "squared", "l2_reg": 1e-3},
        "optimizer": {
            "name": "samd_ss",
            "lambda_reg": 1.0,
            "epsilon": 0.2,
            "step_schedule": {"type": "inverse_sqrt", "eta0": 0.5},
            "sketch": {"name": "frequent_directions", "rank": 4, "ridge": 1e-3},
            "constraint": {"type": "l2_ball", "radius": 1.0},
        },
        "train": {"epochs": 1, "batch_size": 16, "domain_diameter": 2.0},
    }

    summary = run_training(cfg, tmp_path)
    assert "elliptical_potential_ok" in summary
    assert summary["experiment"] == "unit_test_run"
