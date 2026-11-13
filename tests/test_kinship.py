import numpy as np
import pandas as pd
import pytest

import cmix


def test_kinship_is_symmetric_psd_and_normalised(simulated):
    K = cmix.kinship(simulated["X"])
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    np.testing.assert_allclose(np.diag(K).mean(), 1.0)


def test_kinship_keeps_sample_index():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.binomial(1, 0.3, (20, 50)), index=[f"c{i}" for i in range(20)])
    K = cmix.kinship(X)
    assert isinstance(K, pd.DataFrame)
    assert list(K.index) == list(X.index) and list(K.columns) == list(X.index)


def test_kinship_drops_constant_features_and_rejects_bad_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 10))
    X[:, 0] = 1.0
    np.testing.assert_allclose(cmix.kinship(X), cmix.kinship(X[:, 1:]))
    with pytest.raises(ValueError, match="non-finite"):
        cmix.kinship(np.full((5, 3), np.nan))
    with pytest.raises(ValueError, match="variable"):
        cmix.kinship(np.ones((5, 3)))


def test_kinship_without_standardisation_only_centres():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 8)) * np.arange(1, 9)
    K = cmix.kinship(X, standardize=False)
    Xc = X - X.mean(0)
    expected = Xc @ Xc.T
    expected /= np.diag(expected).mean()
    np.testing.assert_allclose(K, expected)
