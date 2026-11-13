# The model

`cmix` fits one single-trait linear mixed model per phenotype and tests the
candidate variants one at a time.

## Null model

For a phenotype vector `y` of `n` samples, a covariate matrix `M` (`n x c`) and
a kinship matrix `K` (`n x n`):

```
y = M b + g + e        g ~ N(0, sigma_g^2 K)        e ~ N(0, sigma_e^2 I)
```

The covariance is `sigma_g^2 (K + delta I)` with `delta = sigma_e^2 / sigma_g^2`.
With the eigendecomposition `K = U S U^T` every quantity is rotated by `U^T`:

```
y~ = U^T y       M~ = U^T M       G~ = U^T G
```

so that `y~ ~ N(M~ b, sigma_g^2 diag(S + delta))`. Everything is then a
weighted least squares problem with weights `w_i = 1 / (S_i + delta)`.

`delta` is estimated by maximum likelihood over `log(delta) in [-10, 10]`: a
grid of 64 values is evaluated in one vectorised call and the bracket around
the maximum is refined three times with 16-point grids (final resolution
about 0.001 in `log(delta)`). `sigma_g^2` is profiled out analytically. Reported
variance components:

| column    | meaning                                                            |
|-----------|--------------------------------------------------------------------|
| `n`       | samples used (phenotype values that are not missing)               |
| `h2`      | `sigma_g^2 mean(diag K) / (sigma_g^2 mean(diag K) + sigma_e^2)`    |
| `sigma_g` | genetic (kinship) variance                                         |
| `sigma_e` | residual variance                                                  |
| `delta`   | `sigma_e^2 / sigma_g^2`                                            |
| `lml`     | log marginal likelihood of the null model                          |

Without a kinship matrix `h2`, `sigma_g` and `delta` are undefined and the
model is an ordinary linear regression (`sigma_e` is the maximum likelihood
residual variance).

## Testing each variant

For every column `g` of `G` the alternative model `y ~ [M, g]` (intercept
and covariates first, the tested variant as the last column) is compared with
the null `y ~ M` by a likelihood ratio test with one degree of freedom.
`delta` is kept at its null estimate (the usual FaST-LMM / limix shortcut) and
the residual variance is re-estimated for every variant.

With `delta` fixed, both models are weighted regressions and all variants can
be processed together. After residualising `y~` and all columns of `G~` on
`M~` (Frisch-Waugh), the effect size, residual sum of squares and test
statistic of every variant are:

```
beta_g  = (r_g^T W r_y) / (r_g^T W r_g)
RSS_1   = RSS_0 - beta_g * (r_g^T W r_y)
LRT     = n * log(RSS_0 / RSS_1)             ~  chi-square(1)
se(beta_g) = sqrt( RSS_1 / (n - c - 1) / (r_g^T W r_g) )
```

The `r_g^T W r_y` terms for all variants and all phenotypes sharing the same
weights are a single matrix product (`G~^T W Y~`), which is where the speed
comes from: the whole scan is a few BLAS calls on `n x p` and `n x q`
matrices rather than `p x q` separate regressions.

Variants that are constant or collinear with the covariates get `NaN`
effect sizes and p-values.

## Missing phenotype values

`NaN` values in `Y` are allowed. Each phenotype is analysed on its observed
samples: phenotypes sharing the same missingness pattern share the sample
subset, the kinship eigendecomposition and the rotation of `G` and `M`.
Missing values in `G`, `M` or `K` raise an error.

## Complexity

| step                                | cost                               |
|-------------------------------------|------------------------------------|
| eigendecomposition of `K`           | `O(n^3)`, once per sample subset   |
| rotation of `G` and `M`             | `O(n^2 (p + c))`, once per subset  |
| null model per phenotype            | `O(n c^2)` per likelihood value    |
| all variants for one phenotype      | `O(n p c)`                         |
| all variants, all phenotypes (LM)   | one `O(n p q)` matrix product      |

In the example notebook (shared 48-core server, BLAS capped at 8 threads)
17,486 CRISPR phenotypes x 395 variants on 535 cell lines took about 2 s as a
linear model and 77 s as a mixed model with five covariates; 819 drugs x 529
variants with 515 different missingness patterns took 1.6 s and 31 s.
