# cmix

Lean linear mixed models for gene-by-gene association scans, with a kinship
matrix for the random effect and a covariate matrix for the fixed effects.
`cmix` is a small, dependency-light replacement for the parts of
[limix](https://github.com/limix/limix) used in systematic association studies
such as drug response or CRISPR-Cas9 gene essentiality versus genomic features
across cell lines.

- One function, `cmix.scan(Y, G, K, M)`, tests every column of `G` against
  every column of `Y`.
- Pure Python on top of NumPy/SciPy. The kinship matrix is eigendecomposed
  once and every test becomes a weighted least squares problem; all variants
  (and all phenotypes sharing a sample set) are handled with a handful of BLAS
  matrix multiplications.
- Missing phenotype values are handled per phenotype.
- Manhattan, quantile-quantile and kinship plots with a compact default style.

## Installation

```bash
pip install git+https://github.com/QuantitativeBiology/cmix.git
# or, for development
git clone https://github.com/QuantitativeBiology/cmix.git && cd cmix
pip install -e ".[dev]"
pytest
```

Requires Python 3.10 or later, NumPy, SciPy, pandas and matplotlib. Use a NumPy
build linked to OpenBLAS or MKL (the PyPI wheels are); the scan is a sequence
of BLAS calls.

## Quick start

```python
import numpy as np
import pandas as pd
import cmix

# samples x features (for example binary mutation calls)
X = pd.read_csv("genomics.csv", index_col=0)
K = cmix.kinship(X)                      # samples x samples, diagonal mean 1

y = pd.read_csv("phenotype.csv", index_col=0)["drug"]   # samples, NaN allowed
M = pd.DataFrame({"intercept": 1.0, "suspension": X_suspension}, index=X.index)

result = cmix.scan(y, X, K=K, M=M)       # one LMM null fit, 1 LRT per feature
result.stats.sort_values("pvalue").head()
#   phenotype   variant      beta   beta_se        lrt        pvalue      n
#   drug        TP53_mut  -1.52     0.11        168.3      1.9e-38      889

result.variances                          # n, h2, sigma_g, sigma_e, delta, lml

import cmix.plot
cmix.plot.manhattan(result.pvalue["drug"])
cmix.plot.qqplot(result.pvalue["drug"])
cmix.plot.kinship(K, cluster=True)
```

`Y` may have many columns (phenotypes); `result.beta`, `result.beta_se`,
`result.lrt` and `result.pvalue` are then `variants x phenotypes` tables and
`result.stats` the long version. Without `K` the same call fits ordinary
linear regressions.

## Documentation

- [docs/model.md](docs/model.md): the statistical model and how it is computed.
- [docs/api.md](docs/api.md): `scan`, `ScanResult`, `kinship` and the low-level functions.
- [docs/plotting.md](docs/plotting.md): the plotting functions and default style.
- [examples/cell_lines_association.ipynb](examples/cell_lines_association.ipynb):
  proof of concept on ~1,000 cancer cell lines (drug response and CRISPR-Cas9
  gene essentiality versus 678 genomic features), linear model versus mixed
  model, with timings.

## Speed

Timings from the example notebook (shared 48-core server, OpenBLAS capped at
8 threads by `scan`; `cmix.scan` alone, data loading excluded):

| scan                                            | tests  | linear model | mixed model |
|-------------------------------------------------|--------|--------------|-------------|
| 819 drugs x 529 genomic features (up to 991 lines, 515 missingness patterns) | 430k | 1.6 s | 31 s |
| 17,486 CRISPR genes x 395 features (535 lines)  | 6.9M   | 2-13 s*      | 77 s        |

\* the linear model scan of the CRISPR data is a single matrix product; it ran
in about 2 s in isolation and 13 s while the server was busy.

The mixed model cost is one variance component fit per phenotype (a batched
1-D likelihood search) plus, when phenotypes have different missing samples,
one eigendecomposition per missingness pattern. `scan` caps the BLAS thread
pool (default 8): on many-core machines letting BLAS use every core for these
small matrices made the drug scan more than ten times slower.

## Layout

```
cmix/_lmm.py      numerical core: eigendecomposition, null fit, batched tests
cmix/_scan.py     scan() and ScanResult
cmix/kinship.py   kinship()
cmix/plot.py      manhattan(), qqplot(), kinship(), set_style()
tests/            pytest suite (dense reference implementations, statsmodels)
docs/             model, API and plotting notes
examples/         proof-of-concept notebook
```

## License

MIT
