"""High-level association scan."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from ._lmm import eigh_kinship, fit_null, scan_variants

__all__ = ["scan", "ScanResult"]

_VARIANCE_COLUMNS = ["n", "h2", "sigma_g", "sigma_e", "delta", "lml"]


@dataclass(repr=False)
class ScanResult:
    """Output of :func:`scan`.

    The wide tables have one row per variant (column of ``G``) and one column
    per phenotype (column of ``Y``).
    """

    beta: pd.DataFrame
    beta_se: pd.DataFrame
    lrt: pd.DataFrame
    pvalue: pd.DataFrame
    variances: pd.DataFrame
    elapsed: float = field(default=np.nan)

    @cached_property
    def stats(self):
        """Long table with one row per (phenotype, variant) test."""
        p, q = self.pvalue.shape
        n = self.variances["n"].to_numpy()
        return pd.DataFrame(
            {
                "phenotype": pd.Categorical.from_codes(
                    np.repeat(np.arange(q), p), categories=self.pvalue.columns
                ),
                "variant": pd.Categorical.from_codes(
                    np.tile(np.arange(p), q), categories=self.pvalue.index
                ),
                "beta": self.beta.to_numpy().ravel(order="F"),
                "beta_se": self.beta_se.to_numpy().ravel(order="F"),
                "lrt": self.lrt.to_numpy().ravel(order="F"),
                "pvalue": self.pvalue.to_numpy().ravel(order="F"),
                "n": np.repeat(n, p),
            }
        )

    def __repr__(self):
        p, q = self.pvalue.shape
        h2 = self.variances["h2"]
        lines = [
            f"ScanResult: {q} phenotype(s) x {p} variant(s) in {self.elapsed:.2f} s",
            f"  h2 (median): {np.nanmedian(h2):.3f}"
            if h2.notna().any()
            else "  linear model (no kinship)",
            f"  smallest p-value: {np.nanmin(self.pvalue.to_numpy()):.3g}",
        ]
        return "\n".join(lines)


def _as_2d(x, name, allow_nan=False):
    """Return (float ndarray with 2 dims, column names or None, index or None)."""
    index = columns = None
    if isinstance(x, pd.Series):
        index = x.index
        columns = pd.Index([x.name if x.name is not None else name])
        x = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        index, columns = x.index, x.columns
        x = x.to_numpy()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{name} must be 1- or 2-dimensional, got shape {x.shape}")
    if not allow_nan and not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")
    if columns is None:
        columns = pd.Index([f"{name}{i}" for i in range(x.shape[1])])
    return x, columns, index


def _check_index(reference, index, name):
    if reference is not None and index is not None and not reference.equals(index):
        raise ValueError(f"the sample index of {name} does not match the one of Y")


def scan(Y, G, K=None, M=None, verbose=False):
    """Test the association of every column of ``G`` with every column of ``Y``.

    For each phenotype a null model ``y ~ M`` (plus the random effect when a
    kinship matrix is given) is fitted by maximum likelihood.  Every variant is
    then added one at a time to the fixed effects and compared with the null
    model by a likelihood ratio test, keeping the ratio of the variance
    components fixed at its null estimate.

    Parameters
    ----------
    Y : array-like, shape (n,) or (n, q)
        Phenotypes (dependent variables), one column each.  ``NaN`` is allowed:
        a phenotype is tested on its observed samples only.
    G : array-like, shape (n, p)
        Variants (independent variables), tested one column at a time.
    K : array-like, shape (n, n), optional
        Kinship (sample covariance) matrix for the random effect.  When omitted
        a plain linear model is fitted.
    M : array-like, shape (n, c), optional
        Covariates.  They must include an intercept if one is wanted; the
        default is a single column of ones.
    verbose : bool
        Print progress and timing.

    Returns
    -------
    ScanResult
        Effect sizes, standard errors, likelihood ratio statistics and
        p-values (variants x phenotypes) plus the variance components of
        each null model.
    """
    start = time.perf_counter()

    Y, pheno_names, index = _as_2d(Y, "y", allow_nan=True)
    n = Y.shape[0]
    G, variant_names, g_index = _as_2d(G, "g")
    _check_index(index, g_index, "G")
    if G.shape[0] != n:
        raise ValueError(f"G has {G.shape[0]} rows but Y has {n}")

    if M is None:
        M = np.ones((n, 1))
    M, _, m_index = _as_2d(M, "m")
    _check_index(index, m_index, "M")
    if M.shape[0] != n:
        raise ValueError(f"M has {M.shape[0]} rows but Y has {n}")

    if K is not None:
        if isinstance(K, pd.DataFrame):
            _check_index(index, K.index, "K")
            _check_index(index, K.columns, "K")
        K = np.asarray(K, dtype=float)
        if K.shape != (n, n):
            raise ValueError(f"K must be ({n}, {n}), got {K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("K contains non-finite values")

    p, q = G.shape[1], Y.shape[1]
    c = M.shape[1]
    beta = np.full((p, q), np.nan)
    beta_se = np.full((p, q), np.nan)
    lrt = np.full((p, q), np.nan)
    pvalue = np.full((p, q), np.nan)
    variances = np.full((q, len(_VARIANCE_COLUMNS)), np.nan)

    # Phenotypes sharing a missingness pattern share the sample subset, hence
    # the rotation of G and M (and the kinship eigendecomposition).
    observed = ~np.isnan(Y)
    _, first, inverse = np.unique(observed, axis=1, return_index=True, return_inverse=True)
    for k, j0 in enumerate(first):
        cols = np.flatnonzero(inverse.ravel() == k)
        rows = observed[:, j0]
        n_k = int(rows.sum())
        if n_k < c + 2:
            warnings.warn(
                f"{cols.size} phenotype(s) skipped: {n_k} samples for {c} covariates",
                stacklevel=2,
            )
            continue
        Yk, Gk, Mk = Y[rows][:, cols], G[rows], M[rows]
        variances[cols, 0] = n_k

        if K is None:
            w = np.ones(n_k)
            out = scan_variants(Yk, Mk, Gk, w)
            rss0 = _null_rss(Yk, Mk)
            variances[cols, 3] = rss0 / n_k  # sigma_e
            variances[cols, 5] = -0.5 * n_k * (np.log(2 * np.pi * rss0 / n_k) + 1)
        else:
            S, U = eigh_kinship(K[np.ix_(rows, rows)])
            Yt, Gt, Mt = U.T @ Yk, U.T @ Gk, U.T @ Mk
            out = [np.empty((p, cols.size)) for _ in range(4)]
            for i, j in enumerate(cols):
                null = fit_null(Yt[:, i], Mt, S)
                w = 1.0 / (S + null["delta"])
                for arr, val in zip(out, scan_variants(Yt[:, [i]], Mt, Gt, w), strict=True):
                    arr[:, i] = val[:, 0]
                variances[j, 1:] = [null[key] for key in _VARIANCE_COLUMNS[1:]]
                if verbose and (i + 1) % 100 == 0:
                    elapsed = time.perf_counter() - start
                    print(f"  {i + 1}/{cols.size} phenotypes ({elapsed:.1f} s)")

        beta[:, cols], beta_se[:, cols], lrt[:, cols], pvalue[:, cols] = out

    elapsed = time.perf_counter() - start
    if verbose:
        print(f"cmix.scan: {q} phenotype(s) x {p} variant(s) in {elapsed:.2f} s")

    frame = lambda a: pd.DataFrame(a, index=variant_names, columns=pheno_names)  # noqa: E731
    return ScanResult(
        beta=frame(beta),
        beta_se=frame(beta_se),
        lrt=frame(lrt),
        pvalue=frame(pvalue),
        variances=pd.DataFrame(variances, index=pheno_names, columns=_VARIANCE_COLUMNS),
        elapsed=elapsed,
    )


def _null_rss(Y, M):
    """Residual sum of squares of the ordinary least squares fit of Y on M."""
    coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
    r = Y - M @ coef
    return (r * r).sum(axis=0)
