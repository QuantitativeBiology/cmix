"""Plotting helpers: Manhattan plot, quantile-quantile plot and kinship heatmap.

Importing this module applies a compact publication style to matplotlib
(see :func:`set_style`).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["GREY", "RED", "set_style", "manhattan", "qqplot", "kinship"]

GREY, RED = "#8a949e", "#c0392b"
LIGHT_GREY = "#c5ccd3"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
}


def set_style():
    """Apply the default cmix figure style to matplotlib."""
    mpl.rcParams.update(STYLE)


set_style()


def _neglog10(pvalue):
    pvalue = np.asarray(pvalue, dtype=float)
    return -np.log10(np.clip(pvalue, np.finfo(float).tiny, 1.0))


def manhattan(pvalue, chrom=None, pos=None, alpha=None, label=True, max_labels=15, ax=None):
    """Manhattan plot of association p-values.

    Parameters
    ----------
    pvalue : array-like or Series, shape (p,)
        P-values in the order the variants should be drawn.  The index of a
        ``Series`` is used to label significant hits.
    chrom : array-like, optional
        Grouping of the variants (chromosomes, or any category).  Groups are
        drawn in alternating shades and named on the x axis.
    pos : array-like, optional
        Position of each variant inside its group.  Variants are drawn in the
        given order when omitted.
    alpha : float, optional
        Significance threshold; defaults to Bonferroni, ``0.05 / p``.
    label : bool
        Annotate significant variants with their names (``Series`` input only).
    max_labels : int
        Maximum number of annotated variants (the most significant ones).
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    names = pvalue.index if isinstance(pvalue, pd.Series) else None
    y = _neglog10(pvalue)
    p = y.size
    if alpha is None:
        alpha = 0.05 / max(p, 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.2))

    if chrom is None:
        order = np.arange(p)
        x = np.arange(p, dtype=float)
        colors = np.full(p, GREY, dtype=object)
    else:
        chrom = np.asarray(chrom)
        pos = np.arange(p) if pos is None else np.asarray(pos, dtype=float)
        groups = pd.unique(chrom)
        order = np.lexsort((pos, pd.Categorical(chrom, categories=groups).codes))
        x = np.empty(p)
        colors = np.empty(p, dtype=object)
        offset, centres = 0.0, []
        for i, g in enumerate(groups):
            idx = order[chrom[order] == g]
            x[idx] = offset + pos[idx] - pos[idx].min()
            colors[idx] = GREY if i % 2 == 0 else LIGHT_GREY
            width = pos[idx].max() - pos[idx].min()
            centres.append(offset + width / 2)
            offset += width + max(width * 0.03, 1.0)
        ax.set_xticks(centres)
        ax.set_xticklabels([str(g) for g in groups], rotation=90 if len(groups) > 12 else 0)

    hit = np.asarray(pvalue, dtype=float) < alpha
    colors[hit] = RED
    ax.scatter(x, y, c=list(colors), s=6, lw=0, zorder=2)
    ax.axhline(-np.log10(alpha), color=GREY, lw=0.6, ls="--", zorder=1)

    if label and names is not None and hit.any():
        top = np.flatnonzero(hit)
        top = top[np.argsort(y[top])[::-1]][:max_labels]
        for i in top:
            ax.annotate(
                str(names[i]),
                (x[i], y[i]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
                color=RED,
            )

    ax.set_ylabel(r"$-\log_{10}$(p-value)")
    ax.set_xlabel("variants" if chrom is None else "")
    ax.set_ylim(0, None)
    if chrom is None:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    return ax


def qqplot(pvalue, band=True, color=GREY, label=None, ax=None):
    """Quantile-quantile plot of p-values against the uniform distribution.

    Parameters
    ----------
    pvalue : array-like, shape (p,)
    band : bool
        Draw the 95% confidence band of the order statistics.
    color : str
        Colour of the points.
    label : str, optional
        Legend entry, useful when several scans share one axis.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    pvalue = np.asarray(pvalue, dtype=float)
    pvalue = pvalue[np.isfinite(pvalue)]
    n = pvalue.size
    observed = np.sort(_neglog10(pvalue))[::-1]
    ranks = np.arange(1, n + 1)
    expected = -np.log10(ranks / (n + 1))

    if ax is None:
        _, ax = plt.subplots(figsize=(2.4, 2.4))
    if band:
        lo = -np.log10(stats.beta.ppf(0.975, ranks, n + 1 - ranks))
        hi = -np.log10(stats.beta.ppf(0.025, ranks, n + 1 - ranks))
        ax.fill_between(expected, lo, hi, color=LIGHT_GREY, alpha=0.4, lw=0, zorder=0)
    top = max(expected.max(), observed.max()) * 1.05
    ax.plot([0, top], [0, top], color=RED, lw=0.8, zorder=1)
    ax.scatter(expected, observed, s=6, lw=0, color=color, label=label, zorder=2)
    ax.set_xlabel(r"expected $-\log_{10}$(p-value)")
    ax.set_ylabel(r"observed $-\log_{10}$(p-value)")
    ax.set_xlim(0, top)
    ax.set_ylim(0, None)
    if label is not None:
        ax.legend(frameon=False)
    return ax


def kinship(K, cluster=False, cmap="Reds", ax=None):
    """Heatmap of a kinship matrix.

    Parameters
    ----------
    K : array-like or DataFrame, shape (n, n)
    cluster : bool
        Reorder the samples by hierarchical clustering (average linkage).
    cmap : str
        Sequential matplotlib colormap.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    K = np.asarray(K, dtype=float)
    if cluster:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        d = np.max(K) - K
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0.0)
        order = hierarchy.leaves_list(hierarchy.linkage(squareform(d, checks=False), "average"))
        K = K[np.ix_(order, order)]
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6, 2.4))
    img = ax.imshow(K, cmap=cmap, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("samples")
    ax.set_ylabel("samples")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="kinship")
    return ax
