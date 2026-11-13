import numpy as np
import pandas as pd
import pytest

import cmix


@pytest.fixture(scope="session")
def simulated():
    """Structured population (24 families), one causal variant, known covariates."""
    rng = np.random.default_rng(1)
    n, p = 240, 300
    group = np.repeat(np.arange(24), n // 24)
    # allele frequencies drift between families so the kinship carries structure
    freq = np.clip(rng.beta(0.5, 0.5, size=(24, 1500)), 0.02, 0.98)
    X = rng.binomial(1, freq[group]).astype(float)
    K = cmix.kinship(X)
    M = np.c_[np.ones(n), rng.normal(size=n)]
    G = rng.binomial(1, 0.25, size=(n, p)).astype(float)
    g = rng.multivariate_normal(np.zeros(n), 0.7 * K)
    y = M @ [0.5, 0.3] + 1.0 * G[:, 10] + g + rng.normal(scale=np.sqrt(0.3), size=n)
    samples = pd.Index([f"s{i}" for i in range(n)])
    return {
        "y": pd.Series(y, index=samples, name="trait"),
        "G": pd.DataFrame(G, index=samples, columns=[f"v{j}" for j in range(p)]),
        "M": pd.DataFrame(M, index=samples, columns=["intercept", "age"]),
        "K": pd.DataFrame(K, index=samples, columns=samples),
        "X": X,
        "causal": "v10",
    }
