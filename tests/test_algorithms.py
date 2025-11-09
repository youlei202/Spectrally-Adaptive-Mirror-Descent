import numpy as np

from src.algorithms.adagrad_diag import AdaGradDiag
from src.algorithms.samd_ss import SAMDSS, SAMDSSConfig
from src.algorithms.sgd import SGD


def test_samd_ss_produces_finite_update():
    cfg = SAMDSSConfig(
        dimension=4,
        sketch_type="frequent_directions",
        sketch_rank=2,
        eta=0.5,
        lambda_reg=1e-2,
    )
    opt = SAMDSS(cfg)
    w = np.zeros(4)
    grad = np.array([1.0, -0.5, 0.25, 0.75])

    new_w, info = opt.update(w, grad)
    assert np.isfinite(new_w).all()
    assert info["elliptical_potential"] > 0


def test_adagrad_diag_improves_conditioning():
    opt = AdaGradDiag(dimension=3, lr=0.1)
    w = np.zeros(3)
    grad = np.ones(3)

    new_w, stats = opt.update(w, grad)
    assert not np.allclose(new_w, w)
    assert stats["conditioning"] >= 1.0


def test_sgd_momentum_update():
    opt = SGD(dimension=2, lr=0.1, momentum=0.9)
    w = np.zeros(2)
    grad = np.array([1.0, 2.0])
    new_w, _ = opt.update(w, grad)
    assert np.linalg.norm(new_w) > 0
