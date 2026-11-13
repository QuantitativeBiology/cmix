"""Numerical core of cmix.

The model is a single-trait linear mixed model

    y = M b + g + e,    g ~ N(0, sigma_g^2 K),    e ~ N(0, sigma_e^2 I)

Writing K = U S U^T (eigendecomposition) and rotating everything by U^T turns
the covariance into a diagonal matrix, so every operation below is a weighted
least squares problem with weights 1 / (S_i + delta), delta = sigma_e^2 /
sigma_g^2.  The rotation is done once per kinship matrix and the per-variant
test is a handful of matrix multiplications.  This is the FaST-LMM / limix
strategy.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, special

__all__ = ["eigh_kinship", "fit_null", "scan_variants", "LOG_DELTA_BOUNDS"]

LOG_DELTA_BOUNDS = (-10.0, 10.0)
_GRID_SIZE = 64
_REFINE_SIZE = 16
_REFINE_STEPS = 3


def eigh_kinship(K):
    """Eigendecomposition of a kinship matrix.

    Negative eigenvalues produced by round-off are clipped to zero.

    Returns
    -------
    S : ndarray, shape (n,)
        Eigenvalues.
    U : ndarray, shape (n, n)
        Eigenvectors, ``K = U @ diag(S) @ U.T``.
    """
    S, U = linalg.eigh(np.asarray(K, dtype=float), driver="evd", check_finite=False)
    return np.clip(S, 0.0, None), U


def _weighted_residual(y, M, w):
    """Residual of the weighted regression of y on M and its RSS.

    y may have several columns.  Returns (residuals, rss, beta).
    """
    MtW = (M * w[:, None]).T
    beta = np.linalg.pinv(MtW @ M) @ (MtW @ y)
    r = y - M @ beta
    rss = np.einsum("i,i...->...", w, r * r)
    return r, rss, beta


def _lml(log_delta, yt, Mt, S):
    """Profile log marginal likelihood (ML) of the null model.

    ``log_delta`` may be a vector: all values are evaluated in one batch of
    small dense operations, which keeps the Python overhead of the grid search
    negligible.
    """
    n = yt.shape[0]
    log_delta = np.atleast_1d(np.asarray(log_delta, dtype=float))
    w = 1.0 / (S[None, :] + np.exp(log_delta)[:, None])  # (k, n)
    WM = w[:, :, None] * Mt[None, :, :]  # (k, n, c)
    A = np.matmul(WM.transpose(0, 2, 1), Mt)  # (k, c, c)
    b = (w * yt) @ Mt  # (k, c)
    beta = np.einsum("kcd,kd->kc", np.linalg.pinv(A), b)  # (k, c)
    r = yt[None, :] - beta @ Mt.T  # (k, n)
    rss = (w * r * r).sum(axis=1)
    return -0.5 * (n * np.log(2 * np.pi * rss / n) + n - np.log(w).sum(axis=1))


def fit_null(yt, Mt, S):
    """Fit the null model (covariates only) by maximum likelihood.

    Parameters
    ----------
    yt : ndarray, shape (n,)
        Rotated phenotype, ``U.T @ y``.
    Mt : ndarray, shape (n, c)
        Rotated covariates, ``U.T @ M``.
    S : ndarray, shape (n,)
        Eigenvalues of the kinship matrix.

    Returns
    -------
    dict with ``delta``, ``sigma_g``, ``sigma_e`` (variances), ``h2``,
    ``lml`` and ``beta`` (covariate effects).
    """
    # Coarse grid over log(delta), then successive batched refinements of the
    # bracket around the maximum.  Each stage is one vectorised call.
    grid = np.linspace(*LOG_DELTA_BOUNDS, _GRID_SIZE)
    for _ in range(_REFINE_STEPS + 1):
        values = _lml(grid, yt, Mt, S)
        i = int(np.argmax(values))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
        grid = np.linspace(lo, hi, _REFINE_SIZE)
    log_delta = grid[np.argmax(_lml(grid, yt, Mt, S))]

    n = yt.shape[0]
    delta = float(np.exp(log_delta))
    w = 1.0 / (S + delta)
    _, rss, beta = _weighted_residual(yt, Mt, w)
    sigma_g = float(rss / n)
    sigma_e = sigma_g * delta
    scale = float(S.mean())  # mean diagonal of K, 1 for a normalised kinship
    return {
        "delta": delta,
        "sigma_g": sigma_g,
        "sigma_e": sigma_e,
        "h2": sigma_g * scale / (sigma_g * scale + sigma_e),
        "lml": float(_lml(log_delta, yt, Mt, S)[0]),
        "beta": beta,
    }


def scan_variants(Yt, Mt, Gt, w):
    """Test every column of Gt against every column of Yt with fixed weights.

    All inputs live in the rotated space.  For each variant the model
    ``y ~ [M, g]`` is compared with ``y ~ M`` by a likelihood ratio test
    (chi-square, one degree of freedom) while keeping ``delta`` fixed at its
    null estimate; the residual variance is re-estimated per variant.

    Redundant covariate columns (for example an all-zero dummy variable in a
    sample subset) are tolerated through a pseudo-inverse; the residual
    degrees of freedom use the rank of the covariate matrix.

    Parameters
    ----------
    Yt : ndarray, shape (n, q)
    Mt : ndarray, shape (n, c)
    Gt : ndarray, shape (n, p)
    w : ndarray, shape (n,)
        Weights ``1 / (S + delta)`` (all ones for a plain linear model).

    Returns
    -------
    beta, beta_se, lrt, pvalue : ndarray, shape (p, q)
    """
    n = Mt.shape[0]
    MtW = (Mt * w[:, None]).T  # (c, n)
    A = MtW @ Mt  # (c, c)
    proj = np.linalg.pinv(A) @ MtW  # (c, n); pinv tolerates redundant covariates
    c = np.linalg.matrix_rank(A)

    # Residualise phenotypes and variants on the covariates (Frisch-Waugh).
    Ry = Yt - Mt @ (proj @ Yt)  # (n, q)
    Rg = Gt - Mt @ (proj @ Gt)  # (n, p)

    rss0 = np.einsum("i,iq,iq->q", w, Ry, Ry)  # (q,)
    den = np.einsum("i,ip,ip->p", w, Rg, Rg)  # (p,)
    num = Rg.T @ (Ry * w[:, None])  # (p, q)  the single BLAS call that matters

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = num / den[:, None]
        rss1 = np.maximum(rss0[None, :] - beta * num, np.finfo(float).tiny)
        lrt = n * (np.log(rss0)[None, :] - np.log(rss1))
        beta_se = np.sqrt(rss1 / max(n - c - 1, 1) / den[:, None])

    bad = ~(den > n * np.finfo(float).eps * np.max(den, initial=0.0))
    beta[bad] = beta_se[bad] = lrt[bad] = np.nan
    lrt = np.clip(lrt, 0.0, None)
    pvalue = special.erfc(np.sqrt(lrt / 2))  # chi-square survival function, 1 d.o.f.
    return beta, beta_se, lrt, pvalue
