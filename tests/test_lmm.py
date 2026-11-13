"""Numerical core against dense reference implementations."""

import numpy as np
from scipy import linalg, optimize

from cmix._lmm import eigh_kinship, fit_null, scan_variants


def _whiten(L, A):
    return linalg.solve_triangular(L, A, lower=True)


def _dense_gls(y, X, V):
    L = linalg.cholesky(V, lower=True)
    Xw, yw = _whiten(L, X), _whiten(L, y)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    r = yw - Xw @ beta
    return beta, r @ r


def _dense_negative_lml(log_delta, y, M, K):
    n = y.size
    V = K + np.exp(log_delta) * np.eye(n)
    L = linalg.cholesky(V, lower=True)
    _, rss = _dense_gls(y, M, V)
    return 0.5 * (n * np.log(2 * np.pi * rss / n) + n + 2 * np.log(np.diag(L)).sum())


def test_eigh_kinship_reconstructs_matrix(simulated):
    K = simulated["K"].to_numpy()
    S, U = eigh_kinship(K)
    assert np.all(S >= 0)
    np.testing.assert_allclose((U * S) @ U.T, K, atol=1e-10)


def test_fit_null_matches_dense_maximum_likelihood(simulated):
    y, M, K = (simulated[k].to_numpy() for k in ("y", "M", "K"))
    M = np.c_[np.ones(y.size), M]
    S, U = eigh_kinship(K)
    null = fit_null(U.T @ y, U.T @ M, S)

    ref = optimize.minimize_scalar(
        _dense_negative_lml, args=(y, M, K), bounds=(-10, 10), method="bounded"
    )
    np.testing.assert_allclose(null["lml"], -ref.fun, rtol=1e-8)
    np.testing.assert_allclose(np.log(null["delta"]), ref.x, atol=1e-3)
    assert 0 < null["h2"] < 1
    assert null["sigma_g"] > 0 and null["sigma_e"] > 0


def test_fit_null_recovers_simulated_heritability():
    # 60 families of 10 samples, sigma_g = 0.6 and sigma_e = 0.4 (h2 = 0.6)
    rng = np.random.default_rng(11)
    n = 600
    family = np.repeat(np.arange(60), 10)
    kinship = np.equal.outer(family, family) * 0.8 + 0.2 * np.eye(n)
    kinship /= np.diag(kinship).mean()
    S, U = eigh_kinship(kinship)
    M = np.ones((n, 1))
    estimates = []
    for _ in range(5):
        y = rng.multivariate_normal(np.zeros(n), 0.6 * kinship) + rng.normal(
            scale=np.sqrt(0.4), size=n
        )
        estimates.append(fit_null(U.T @ y, U.T @ M, S)["h2"])
    assert abs(np.mean(estimates) - 0.6) < 0.15


def test_scan_variants_matches_dense_gls(simulated):
    y, M, K, G = (simulated[k].to_numpy() for k in ("y", "M", "K", "G"))
    n = y.size
    M = np.c_[np.ones(n), M]
    S, U = eigh_kinship(K)
    null = fit_null(U.T @ y, U.T @ M, S)
    w = 1.0 / (S + null["delta"])
    beta, beta_se, lrt, pvalue = scan_variants(U.T @ y[:, None], U.T @ M, U.T @ G, w)

    V = K + null["delta"] * np.eye(n)
    _, rss0 = _dense_gls(y, M, V)
    for j in [0, 10, 123, 299]:
        b, rss1 = _dense_gls(y, np.c_[M, G[:, j]], V)
        np.testing.assert_allclose(beta[j, 0], b[-1], rtol=1e-8)
        np.testing.assert_allclose(lrt[j, 0], n * np.log(rss0 / rss1), rtol=1e-6, atol=1e-8)
        # standard error from the dense weighted regression
        Xw = _whiten(linalg.cholesky(V, lower=True), np.c_[M, G[:, j]])
        cov = np.linalg.inv(Xw.T @ Xw) * rss1 / (n - Xw.shape[1])
        np.testing.assert_allclose(beta_se[j, 0], np.sqrt(cov[-1, -1]), rtol=1e-8)
    assert pvalue.shape == (G.shape[1], 1)
    assert np.all((pvalue >= 0) & (pvalue <= 1))


def test_scan_variants_handles_constant_and_collinear_columns():
    rng = np.random.default_rng(3)
    n = 50
    M = np.c_[np.ones(n), rng.normal(size=n)]
    y = rng.normal(size=(n, 1))
    G = np.c_[np.zeros(n), M[:, 1], rng.normal(size=n)]
    beta, beta_se, lrt, pvalue = scan_variants(y, M, G, np.ones(n))
    assert np.isnan(beta[0, 0]) and np.isnan(pvalue[0, 0])
    assert np.isnan(beta[1, 0]) and np.isnan(pvalue[1, 0])
    assert np.isfinite(beta[2, 0]) and np.isfinite(pvalue[2, 0])


def test_scan_variants_many_phenotypes_equals_one_at_a_time():
    rng = np.random.default_rng(4)
    n, p, q = 60, 20, 5
    M, G, Y = np.ones((n, 1)), rng.normal(size=(n, p)), rng.normal(size=(n, q))
    w = rng.uniform(0.5, 2, size=n)
    out = scan_variants(Y, M, G, w)
    for j in range(q):
        single = scan_variants(Y[:, [j]], M, G, w)
        for a, b in zip(out, single, strict=True):
            np.testing.assert_allclose(a[:, j], b[:, 0])
