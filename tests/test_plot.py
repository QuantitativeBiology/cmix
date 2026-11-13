import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cmix  # noqa: E402
from cmix import plot  # noqa: E402


def test_style_is_applied_on_import():
    assert matplotlib.rcParams["font.size"] == 8
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert plot.GREY == "#8a949e" and plot.RED == "#c0392b"


def test_manhattan_plain_and_grouped(simulated):
    res = cmix.scan(simulated["y"], simulated["G"], K=simulated["K"], M=simulated["M"])
    pv = res.pvalue["trait"]
    ax = plot.manhattan(pv)
    assert ax.get_ylabel().startswith("$-\\log")
    texts = [t.get_text() for t in ax.texts]
    assert simulated["causal"] in texts
    plt.close("all")

    chrom = np.repeat(np.arange(1, 7), 50)
    pos = np.tile(np.arange(50) * 1000, 6)
    ax = plot.manhattan(pv.to_numpy(), chrom=chrom, pos=pos, alpha=1e-3)
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(i) for i in range(1, 7)]
    plt.close("all")


def test_qqplot_runs_and_handles_nan():
    pv = np.r_[np.random.default_rng(0).uniform(size=500), np.nan]
    fig, ax = plt.subplots()
    out = plot.qqplot(pv, ax=ax, label="null")
    assert out is ax
    assert ax.get_legend() is not None
    plt.close("all")


def test_kinship_plot(simulated):
    ax = plot.kinship(simulated["K"])
    assert len(ax.images) == 1
    plt.close("all")
    ax = plot.kinship(pd.DataFrame(simulated["K"]), cluster=True)
    assert ax.images[0].get_array().shape == simulated["K"].shape
    plt.close("all")
