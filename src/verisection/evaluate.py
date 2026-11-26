"""Evaluation against simulation truth and leave-one-section-out."""
from __future__ import annotations

import numpy as np

from . import metrics
from .interpolate import InterpConfig
from .provenance import Provenance
from .register import RegistrationEnsemble
from .segment import SegmentationEnsemble
from .uncertainty import UncertaintyMaps, propagate
from .volume import SectionStack


def truth_on_grid(stack: SectionStack, z_planes_um: np.ndarray) -> np.ndarray:
    """True labels (Z, Y, X) at the given plane centres (simulation only)."""
    vol = stack.meta["truth_volume"]
    dz = stack.meta["dz_fine_um"]
    idx = np.clip(np.floor(np.asarray(z_planes_um) / dz).astype(int), 0, vol.shape[0] - 1)
    return vol[idx]


def score_maps(
    maps: UncertaintyMaps,
    truth: np.ndarray,
    rng: np.random.Generator,
    max_vox: int = 150_000,
) -> dict:
    """Accuracy, calibration and uncertainty-vs-error metrics, overall and by provenance."""
    out: dict = {}
    valid = maps.provenance > 0
    pred = maps.map_labels
    for name, mask in (
        ("all", valid),
        ("observed", maps.provenance == Provenance.OBSERVED),
        ("interpolated", maps.provenance == Provenance.INTERPOLATED),
    ):
        if not mask.any():
            continue
        flat = np.flatnonzero(mask.ravel())
        if flat.size > max_vox:
            flat = rng.choice(flat, size=max_vox, replace=False)
        p = maps.p_hat.reshape(maps.p_hat.shape[0], -1)[:, flat]
        y = truth.ravel()[flat].astype(int)
        err = (pred.ravel()[flat] != y).astype(float)
        u = maps.u_total.ravel()[flat]
        out[name] = {
            "n": int(flat.size),
            "accuracy": float(1.0 - err.mean()),
            "ece": metrics.ece(p, y),
            "brier": metrics.brier(p, y),
            "nll": metrics.nll(p, y),
            "ause": metrics.ause(u, err),
            "spearman_u_err": metrics.spearman(u, err),
            "dice": np.round(metrics.dice(pred.ravel()[flat], y, p.shape[0]), 4).tolist(),
        }
    return out


def error_vs_gap(maps: UncertaintyMaps, truth: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Error and mean uncertainty by distance to the nearest slab."""
    rows = []
    gap = np.broadcast_to(maps.gap_distance_um[:, None, None], truth.shape)
    valid = maps.provenance > 0
    err = maps.map_labels != truth
    levels = np.unique(np.round(gap[valid], 6))
    if len(levels) <= n_bins:  # discrete gap distances (typical with a coarse z-grid)
        groups = [(g, g, valid & np.isclose(gap, g)) for g in levels]
    else:
        edges = np.unique(np.quantile(gap[valid], np.linspace(0, 1, n_bins + 1)))
        groups = [
            (lo, hi, valid & (gap >= lo) & (gap <= hi if hi == edges[-1] else gap < hi))
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    for lo, hi, m in groups:
        if m.any():
            rows.append(
                {"gap_lo_um": float(lo), "gap_hi_um": float(hi), "error": float(err[m].mean()),
                 "mean_u_bits": float(maps.u_total[m].mean()), "n": int(m.sum())}
            )
    return rows


def loso_scores(
    stack: SectionStack,
    reg: RegistrationEnsemble,
    seg: SegmentationEnsemble,
    interp: InterpConfig,
    class_names: list[str],
    held_out: list[int] | None = None,
) -> list[dict]:
    """Leave-one-section-out: hide section k's labels, predict its plane, compare with truth.

    The held-out section's *transform* is taken from the full-stack registration (it is a
    registration of the image, not of the labels); end sections are skipped (no extrapolation).
    Returns per-section records with ``p`` (C, N) and ``y`` (N,) for calibration/conformal use.
    """
    K = stack.n_sections
    ks = held_out if held_out is not None else list(range(1, K - 1))
    records = []
    for k in ks:
        keep = [i for i in range(K) if i != k]
        sub = stack.subset(keep)
        sub_reg = reg.subset(keep)
        maps = propagate(sub, sub_reg, seg, interp, class_names)
        j = int(np.argmin(np.abs(maps.z_um - stack.z_um[k])))
        truth = truth_on_grid(stack, maps.z_um[[j]])[0]
        valid = maps.provenance[j] > 0
        p = maps.p_hat[:, j][:, valid]
        y = truth[valid].astype(int)
        records.append({"section": k, "p": p, "y": y, "u": maps.u_total[j][valid]})
    return records
