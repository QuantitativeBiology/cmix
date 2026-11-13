# API

## `cmix.scan(Y, G, K=None, M=None, intercept=True, verbose=False, threads=8)`

Test the association of every column of `G` with every column of `Y`.

| argument  | shape          | description                                                                 |
|-----------|----------------|-----------------------------------------------------------------------------|
| `Y`       | `(n,)`/`(n, q)`| Phenotypes. `NaN` allowed (each phenotype uses its observed samples).       |
| `G`       | `(n, p)`       | Variants (genes, mutations, ...) tested one at a time.                      |
| `K`       | `(n, n)`       | Kinship matrix for the random effect. `None` gives a plain linear model.    |
| `M`       | `(n, c)`       | Covariates (fixed effects); they enter before the tested variant.           |
| `intercept` | bool         | Prepend a column of ones to `M` (default). Use `False` if `M` has one.      |
| `verbose` | bool           | Print progress and total time.                                              |
| `threads` | int or None    | Cap on BLAS threads during the scan (`None`: leave BLAS untouched). The matrices are small; on a 48-core server the default cap of 8 made a scan with many missingness patterns 10x faster. |

`numpy` arrays and `pandas` objects are accepted. With `pandas` input the
sample index of `G`, `M` and `K` must equal the index of `Y` (same order); the
column names of `Y` and `G` become the phenotype and variant names.

Returns a `ScanResult`:

| attribute   | type                       | content                                               |
|-------------|----------------------------|-------------------------------------------------------|
| `beta`      | DataFrame `(p, q)`         | effect size of each variant on each phenotype         |
| `beta_se`   | DataFrame `(p, q)`         | standard error of the effect size                     |
| `lrt`       | DataFrame `(p, q)`         | likelihood ratio statistic (1 degree of freedom)      |
| `pvalue`    | DataFrame `(p, q)`         | p-value of the likelihood ratio test                  |
| `variances` | DataFrame `(q, 6)`         | `n, h2, sigma_g, sigma_e, delta, lml` per phenotype   |
| `stats`     | DataFrame `(p * q, 7)`     | long table: `phenotype, variant, beta, beta_se, lrt, pvalue, n` |
| `elapsed`   | float                      | wall-clock seconds                                    |

`stats` is built on first access, which is what you want for a single
phenotype; for thousands of phenotypes work with the wide tables.

```python
import cmix

result = cmix.scan(y, G, K=K, M=M)
result.stats.sort_values("pvalue").head()
result.variances
```

## `cmix.kinship(X, standardize=True)`

Realised relationship matrix from a samples x features matrix `X`:
features are centred (and scaled to unit variance when `standardize=True`),
constant features are dropped, and `K = X X^T` is normalised so that its
diagonal has mean one. A `DataFrame` input returns a `DataFrame` indexed by
the samples.

## `cmix.plot`

See [plotting.md](plotting.md).

## Lower level

`cmix._lmm` contains the numerical core used by `scan`:

- `eigh_kinship(K) -> (S, U)`
- `fit_null(yt, Mt, S) -> dict(delta, sigma_g, sigma_e, h2, lml, beta)`
- `scan_variants(Yt, Mt, Gt, w) -> (beta, beta_se, lrt, pvalue)`

All of them operate on rotated matrices (`U.T @ y` and so on). They are kept
free of pandas so they can be reused in other pipelines.
