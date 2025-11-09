import numpy as np

from src.metrics import logdet, regret, stability


def test_logdet_ratio_positive():
    gram = np.eye(3)
    value = logdet.logdet_ratio(gram, lambda_reg=1e-2)
    assert value > 0


def test_regret_bound_satisfied():
    result = regret.instance_regret_bound(
        domain_diameter=2.0,
        lambda_reg=1e-2,
        epsilon=0.1,
        logdet_ratio=1.5,
        observed_regret=0.1,
    )
    assert result.satisfied


def test_stability_path_is_non_negative():
    weights = np.zeros((4, 2))
    weights[1] = 1.0
    value = stability.path_stability(weights)
    assert value >= 0
