"""cmix: lean linear mixed models for association scans.

>>> import cmix
>>> result = cmix.scan(y, G, K=cmix.kinship(X), M=covariates)
>>> result.stats.sort_values("pvalue").head()
"""

from ._scan import ScanResult, scan
from .kinship import kinship

__all__ = ["scan", "ScanResult", "kinship", "plot"]
__version__ = "0.1.0"


def __getattr__(name):
    # Import matplotlib lazily so that the numerical core stays light.
    if name == "plot":
        import importlib

        return importlib.import_module("cmix.plot")
    raise AttributeError(f"module 'cmix' has no attribute '{name}'")
