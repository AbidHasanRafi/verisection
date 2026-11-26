"""Small statistics helpers for multi-replicate experiments.

Units of analysis are replicates (specimens/phantoms), never voxels. Paired comparisons use
the Wilcoxon signed-rank test; intervals are percentile bootstraps over replicates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Summary:
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def bootstrap_ci(x, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> Summary:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return Summary(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return Summary(float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), len(x))


def paired_comparison(a, b, seed: int = 0) -> dict:
    """Paired difference ``a - b``: mean, bootstrap CI, Wilcoxon p, and a rank-biserial effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    out = {"n": int(ok.sum()), "mean_diff": float(d.mean()) if len(d) else float("nan")}
    if len(d) >= 2:
        s = bootstrap_ci(d, seed=seed)
        out["ci"] = [s.lo, s.hi]
        nz = d[d != 0]
        if len(nz) >= 2:
            w = stats.wilcoxon(nz)
            out["wilcoxon_p"] = float(w.pvalue)
            ranks = stats.rankdata(np.abs(nz))
            out["rank_biserial"] = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        else:
            out["wilcoxon_p"] = float("nan")
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for a family of named comparisons."""
    items = sorted(((k, p) for k, p in pvalues.items() if np.isfinite(p)), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
