# Plotting

`import cmix.plot` applies a compact publication style to matplotlib
(Arial/Helvetica/DejaVu Sans, 8 pt text, no top/right spines, 0.6 pt lines,
editable text in PDF/SVG, 300 dpi when saving, 150 dpi in notebooks). Call
`cmix.plot.set_style()` to re-apply it after changing the rcParams. The two
default colours are `cmix.plot.GREY` (`#8a949e`) and `cmix.plot.RED`
(`#c0392b`).

Every function draws on the `ax` given or creates a new figure, and returns
the `Axes`.

## `manhattan(pvalue, chrom=None, pos=None, alpha=None, label=True, max_labels=15, ax=None)`

Manhattan plot of one scan.

- `pvalue`: p-values in drawing order. Pass a `Series` to label the hits with
  its index (for example `result.pvalue["drug"]`).
- `chrom`, `pos`: optional grouping (chromosome, feature type, ...) and
  position inside the group. Groups alternate between two greys and are named
  on the x axis. Without `chrom` the variants are drawn in the given order.
- `alpha`: significance threshold, drawn as a dashed line; variants below it
  are red. Default: Bonferroni `0.05 / p`.
- `label`, `max_labels`: annotate up to `max_labels` of the most significant
  hits.

```python
ax = cmix.plot.manhattan(result.pvalue["Nutlin-3a"])
```

## `qqplot(pvalue, band=True, color=GREY, label=None, ax=None)`

Observed against expected `-log10` p-values with the identity line in red and
the 95% confidence band of the order statistics in light grey. Call it
several times on the same `ax` with different `color`/`label` to compare
scans (for example linear model versus mixed model).

```python
fig, ax = plt.subplots()
cmix.plot.qqplot(lm.pvalue.to_numpy().ravel(), ax=ax, label="LM")
cmix.plot.qqplot(lmm.pvalue.to_numpy().ravel(), ax=ax, label="LMM", color=cmix.plot.RED)
```

## `kinship(K, cluster=False, cmap="Reds", vmin=None, vmax=None, ax=None)`

Heatmap of a kinship matrix with a colour bar. `cluster=True` reorders the
samples by average-linkage hierarchical clustering so related groups appear
as blocks. The colour range defaults to the 1st-99th percentiles of the
off-diagonal values so the diagonal does not hide the structure.

```python
cmix.plot.kinship(K, cluster=True)
```
