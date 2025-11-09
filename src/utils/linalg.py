"""Numerically robust linear algebra helpers."""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError


def add_jitter(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Add jitter to the diagonal of a square matrix."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix + jitter * np.eye(matrix.shape[0])


def safe_cholesky(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Compute a Cholesky factor with automatic jitter."""

    attempt = matrix
    for _ in range(5):
        try:
            return np.linalg.cholesky(attempt)
        except LinAlgError:
            attempt = add_jitter(attempt, jitter)
            jitter *= 10
    raise LinAlgError("Cholesky failed even after jitter growth.")


def solve_psd(matrix: np.ndarray, rhs: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Solve (matrix + ridge I) x = rhs for PSD matrices."""

    work = matrix + ridge * np.eye(matrix.shape[0])
    try:
        chol = safe_cholesky(work)
        y = np.linalg.solve(chol, rhs)
        return np.linalg.solve(chol.T, y)
    except LinAlgError:
        return pseudo_inverse_solve(work, rhs)


def pseudo_inverse_solve(matrix: np.ndarray, rhs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Fallback solver using SVD pseudo-inverse."""

    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    s_inv = np.where(s > eps, 1.0 / s, 0.0)
    rhs_reshaped = rhs[:, None] if rhs.ndim == 1 else rhs
    proj = u.T @ rhs_reshaped
    proj = s_inv[:, None] * proj
    solution = vh.T @ proj
    if rhs.ndim == 1:
        return solution[:, 0]
    return solution


def stable_logdet(matrix: np.ndarray, jitter: float = 1e-8) -> float:
    """Stable log determinant via slogdet."""

    attempt = matrix
    for _ in range(5):
        sign, logdet = np.linalg.slogdet(attempt)
        if sign > 0:
            return float(logdet)
        attempt = add_jitter(attempt, jitter)
        jitter *= 10
    raise LinAlgError("Matrix is not positive definite.")


def spectral_norm(matrix: np.ndarray) -> float:
    """Return the spectral norm (largest singular value)."""

    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def quadratic_form(matrix: np.ndarray, vector: np.ndarray) -> float:
    """Compute v^T M v."""

    return float(vector.T @ matrix @ vector)


def elliptical_potential(precision: np.ndarray, gradient: np.ndarray) -> float:
    """g^T H^{-1} g, with H = precision."""

    inv_grad = solve_psd(precision, gradient)
    return float(gradient.T @ inv_grad)
