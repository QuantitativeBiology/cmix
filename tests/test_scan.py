"""High-level scan interface."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import cmix


def test_scan_finds_causal_variant_with_kinship(simulated):
    res = cmix.scan(simulated["y"], simulated["G"], K=simulated["K"], M=simulated["M"])
    assert res.pvalue.shape == (300, 1)
    assert res.pvalue["trait"].idxmin() == simulated["causal"]
    assert res.pvalue.loc[simulated["causal"], "trait"] < 0.05 / 300
    assert res.beta.loc[simulated["causal"], "trait"] > 0.5
    assert res.variances.loc["trait", "n"] == 240
    assert 0 < res.variances.loc["trait", "h2"] < 1
    assert "ScanResult" in repr(res)


def test_scan_without_kinship_matches_statsmodels(simulated):
    y, G, M = simulated["y"], simulated["G"], simulated["M"]
    res = cmix.scan(y, G, M=M)
    assert res.variances["h2"].isna().all()
    M0 = np.c_[np.ones(len(y)), M.to_numpy()]
    for v in ["v0", "v10", "v42"]:
        X = np.c_[M0, G[v].to_numpy()]  # covariates first, tested variant last
        fit = sm.OLS(y.to_numpy(), X).fit()
        null = sm.OLS(y.to_numpy(), M0).fit()
        np.testing.assert_allclose(res.beta.loc[v, "trait"], fit.params[-1], rtol=1e-10)
        np.testing.assert_allclose(res.beta_se.loc[v, "trait"], fit.bse[-1], rtol=1e-10)
        np.testing.assert_allclose(res.lrt.loc[v, "trait"], 2 * (fit.llf - null.llf), rtol=1e-8)
    np.testing.assert_allclose(res.variances.loc["trait", "lml"], null.llf, rtol=1e-10)


def test_identity_kinship_reduces_to_linear_model(simulated):
    y, G, M = simulated["y"], simulated["G"], simulated["M"]
    lm = cmix.scan(y, G, M=M)
    lmm = cmix.scan(y, G, K=np.eye(len(y)), M=M)
    pd.testing.assert_frame_equal(lm.pvalue, lmm.pvalue, rtol=1e-6)
    pd.testing.assert_frame_equal(lm.beta, lmm.beta, rtol=1e-6)


def test_null_pvalues_are_calibrated():
    rng = np.random.default_rng(7)
    n, p = 200, 2000
    G = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    res = cmix.scan(y, G)
    pv = res.pvalue.to_numpy().ravel()
    assert abs((pv < 0.05).mean() - 0.05) < 0.02
    assert abs((pv < 0.5).mean() - 0.5) < 0.05


def test_multiple_phenotypes_and_long_table(simulated):
    y, G, K, M = simulated["y"], simulated["G"], simulated["K"], simulated["M"]
    Y = pd.DataFrame({"a": y, "b": y.to_numpy()[::-1]}, index=y.index)
    res = cmix.scan(Y, G, K=K, M=M)
    single = cmix.scan(Y["a"], G, K=K, M=M)
    np.testing.assert_allclose(res.pvalue["a"], single.pvalue["a"])
    stats = res.stats
    assert len(stats) == 2 * 300
    assert list(stats.columns) == ["phenotype", "variant", "beta", "beta_se", "lrt", "pvalue", "n"]
    row = stats[(stats.phenotype == "b") & (stats.variant == "v3")].iloc[0]
    assert row.pvalue == res.pvalue.loc["v3", "b"]
    assert row.beta == res.beta.loc["v3", "b"]


def test_missing_phenotype_values_use_observed_samples(simulated):
    y, G, K, M = simulated["y"], simulated["G"], simulated["K"], simulated["M"]
    y_nan = y.copy()
    y_nan.iloc[:40] = np.nan
    res = cmix.scan(pd.DataFrame({"full": y, "partial": y_nan}), G, K=K, M=M)
    keep = y_nan.notna()
    sub = cmix.scan(y[keep], G[keep], K=K.loc[keep, keep], M=M[keep])
    np.testing.assert_allclose(res.pvalue["partial"], sub.pvalue.iloc[:, 0])
    assert res.variances.loc["partial", "n"] == 200
    assert res.variances.loc["full", "n"] == 240


def test_numpy_inputs_get_default_names():
    rng = np.random.default_rng(0)
    res = cmix.scan(rng.normal(size=30), rng.normal(size=(30, 4)))
    assert list(res.pvalue.index) == ["g0", "g1", "g2", "g3"]
    assert list(res.pvalue.columns) == ["y0"]


def test_input_validation(simulated):
    y, G, K, M = simulated["y"], simulated["G"], simulated["K"], simulated["M"]
    with pytest.raises(ValueError, match="non-finite"):
        cmix.scan(y, G.mask(G > 2, np.nan).assign(v0=np.nan), K=K)
    with pytest.raises(ValueError, match="index"):
        cmix.scan(y, G.iloc[::-1], K=K)
    with pytest.raises(ValueError, match="rows"):
        cmix.scan(y.to_numpy(), G.to_numpy()[:-1])
    with pytest.raises(ValueError, match="K must be"):
        cmix.scan(y.to_numpy(), G.to_numpy(), K=K.to_numpy()[:-1, :-1])
    with pytest.warns(UserWarning, match="skipped"):
        too_few = y.copy()
        too_few.iloc[3:] = np.nan
        cmix.scan(too_few, G, M=M)


def test_redundant_covariate_column_is_tolerated(simulated):
    y, G, K, M = simulated["y"], simulated["G"], simulated["K"], simulated["M"]
    M_zero = M.assign(unused=0.0)
    M_dup = M.assign(twice=M["age"])
    ref = cmix.scan(y, G, K=K, M=M)
    for M_bad in (M_zero, M_dup):
        res = cmix.scan(y, G, K=K, M=M_bad)
        pd.testing.assert_frame_equal(res.pvalue, ref.pvalue, rtol=1e-6)
        pd.testing.assert_frame_equal(res.beta_se, ref.beta_se, rtol=1e-6)
    ref_lm = cmix.scan(y, G, M=M)
    res_lm = cmix.scan(y, G, M=M_zero)
    pd.testing.assert_frame_equal(res_lm.pvalue, ref_lm.pvalue, rtol=1e-6)


def test_threads_argument_does_not_change_results(simulated):
    y, G, K, M = simulated["y"], simulated["G"], simulated["K"], simulated["M"]
    ref = cmix.scan(y, G, K=K, M=M)
    for threads in (1, None):
        res = cmix.scan(y, G, K=K, M=M, threads=threads)
        pd.testing.assert_frame_equal(res.pvalue, ref.pvalue, rtol=1e-8)


def test_intercept_argument(simulated):
    y, G, K, M = simulated["y"], simulated["G"], simulated["K"], simulated["M"]
    auto = cmix.scan(y, G, K=K, M=M)  # intercept added by default
    manual = cmix.scan(y, G, K=K, M=M.assign(intercept=1.0), intercept=False)
    pd.testing.assert_frame_equal(auto.pvalue, manual.pvalue, rtol=1e-8)
    pd.testing.assert_frame_equal(auto.beta, manual.beta, rtol=1e-8)
    none = cmix.scan(y, G, K=K, M=M, intercept=False)
    assert not np.allclose(none.pvalue, auto.pvalue)
    with pytest.raises(ValueError, match="no covariates"):
        cmix.scan(y, G, intercept=False)
