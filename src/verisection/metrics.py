"""Calibration and accuracy metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error. ``p`` is (C, N); ``y`` is (N,)."""
    conf = p.max(axis=0)
    correct = (p.argmax(axis=0) == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(p)
    onehot[y, np.arange(p.shape[1])] = 1.0
    return float(((p - onehot) ** 2).sum(axis=0).mean())


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[y, np.arange(p.shape[1])], 1e-12, 1.0)).mean())


def dice(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        a, b = pred == c, truth == c
        den = a.sum() + b.sum()
        if den > 0:
            out[c] = 2.0 * (a & b).sum() / den
    return out


def sparsification_curve(order_key: np.ndarray, err: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Mean error of the retained set after removing the top-``f`` fraction by ``order_key``."""
    order = np.argsort(-order_key, kind="stable")
    e = err[order].astype(float)
    n = len(e)
    # suffix means: mean of e[i:] for i = removed count
    suffix = np.cumsum(e[::-1])[::-1] / np.arange(n, 0, -1)
    fracs = np.linspace(0.0, 0.95, n_points)
    return suffix[np.minimum((fracs * n).astype(int), n - 1)]


def ause(unc: np.ndarray, err: np.ndarray, n_points: int = 20) -> float:
    """Area under the sparsification error curve vs the oracle (0 = perfect ranking)."""
    curve = sparsification_curve(unc, err, n_points)
    oracle = sparsification_curve(err, err, n_points)
    return float(np.mean(curve - oracle))


def spearman(unc: np.ndarray, err: np.ndarray) -> float:
    if np.ptp(unc) == 0 or np.ptp(err) == 0:
        return float("nan")
    return float(spearmanr(unc, err).statistic)
