"""Kinship (sample relatedness) matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["kinship"]


def kinship(X, standardize=True):
    """Realised relationship matrix from a samples x features matrix.

    Features are centred (and scaled to unit variance when ``standardize`` is
    true), constant features are dropped, and ``K = X X^T`` is normalised so
    that its diagonal has mean one.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Samples by features (for example binary mutation calls or copy-number
        states).  Must not contain missing values.
    standardize : bool
        Scale features to unit variance in addition to centring them.

    Returns
    -------
    K : ndarray or DataFrame, shape (n, n)
        A ``DataFrame`` indexed by the samples when ``X`` is one.
    """
    index = X.index if isinstance(X, pd.DataFrame) else None
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional samples x features matrix")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")

    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    X = X[:, sd > 0]
    if standardize:
        X = X / sd[sd > 0]
    if X.shape[1] == 0:
        raise ValueError("X has no variable features")

    K = X @ X.T
    K /= np.diag(K).mean()
    if index is not None:
        K = pd.DataFrame(K, index=index, columns=index)
    return K
