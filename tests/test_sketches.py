import numpy as np

from src.sketches.frequent_directions import FrequentDirections
from src.sketches.oja_sketch import OjaSketch
from src.sketches.randomized_svd_stream import RandomizedSVDStream


def _random_grad(dimension: int) -> np.ndarray:
    return np.linspace(1.0, 2.0, num=dimension)


def test_frequent_directions_metric_is_psd():
    sketch = FrequentDirections(dimension=6, rank=3, ridge=1e-3)
    grad = _random_grad(6)
    for _ in range(5):
        sketch.update(grad)
    metric = sketch.metric()
    eigvals = np.linalg.eigvalsh(metric)
    assert np.all(eigvals > 0)


def test_oja_sketch_covariance_shape():
    sketch = OjaSketch(dimension=5, rank=2, seed=123)
    grad = _random_grad(5)
    sketch.update(grad)
    cov = sketch.covariance()
    assert cov.shape == (5, 5)


def test_randomized_svd_stream_covariance_psd():
    sketch = RandomizedSVDStream(dimension=4, rank=2, ridge=1e-4, seed=0)
    grad = _random_grad(4)
    sketch.update(grad)
    eigvals = np.linalg.eigvalsh(sketch.metric())
    assert np.all(eigvals > 0)
