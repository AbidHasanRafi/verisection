"""Image-processing diagnostics and rule-based section QC.

The v0 decision rule is robust-z thresholding on a few diagnostics. A calibrated risk
model that predicts downstream failure is future work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance


def focus_laplacian_var(img: np.ndarray) -> float:
    """Variance of the Laplacian: higher = sharper."""
    return float(ndi.laplace(np.asarray(img, dtype=np.float64)).var())


def noise_sigma_haar(img: np.ndarray) -> float:
    """Noise sd from the finest diagonal Haar detail band: median(|d|)/0.6745."""
    x = np.asarray(img, dtype=np.float64)
    x = x[: x.shape[0] // 2 * 2, : x.shape[1] // 2 * 2]
    a, b = x[0::2, 0::2], x[0::2, 1::2]
    c, d = x[1::2, 0::2], x[1::2, 1::2]
    hh = (a - b - c + d) / 2.0
    return float(np.median(np.abs(hh)) / 0.6745)


def background_flatness(img: np.ndarray, tissue_mask: np.ndarray) -> float:
    """RMS residual of a plane fitted to background pixels (illumination non-uniformity)."""
    bg = ~tissue_mask
    if bg.sum() < 50:
        return float("nan")
    yy, xx = np.nonzero(bg)
    A = np.c_[np.ones(len(yy)), yy, xx]
    coef, *_ = np.linalg.lstsq(A, img[bg].astype(np.float64), rcond=None)
    return float(np.sqrt(np.mean((img[bg] - A @ coef) ** 2)))


def continuity_wasserstein(a: np.ndarray, b: np.ndarray, max_pix: int = 20000) -> float:
    """Distance between intensity histograms of adjacent sections, in units of pooled sd."""
    a, b = np.ravel(a), np.ravel(b)
    step_a, step_b = max(1, a.size // max_pix), max(1, b.size // max_pix)
    sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
    return float(wasserstein_distance(a[::step_a], b[::step_b]) / sd)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return (x - med) / max(mad, 1e-12)


def passport(img: np.ndarray, prev: np.ndarray | None = None) -> dict:
    """Quality passport for one section."""
    p = {
        "focus_lapvar": focus_laplacian_var(img),
        "noise_sigma": noise_sigma_haar(img),
        "mean": float(np.mean(img)),
        "sd": float(np.std(img)),
    }
    if prev is not None:
        p["continuity_wasserstein"] = continuity_wasserstein(prev, img)
    return p


@dataclass
class QCDecision:
    section: int
    verdict: str  # accept | flag | exclude
    reasons: list[str]


def qc_decisions(
    passports: list[dict], z_flag: float = 3.5, z_exclude: float = 6.0
) -> list[QCDecision]:
    """Rule-based (v0): accept / flag / exclude by robust-z outliers in the *bad* direction."""
    K = len(passports)
    bad = {
        "focus_lapvar": -1.0,  # low focus is bad
        "noise_sigma": +1.0,  # high noise is bad
        "continuity_wasserstein": +1.0,  # large jump vs neighbour is bad
    }
    z = {}
    for key, sign in bad.items():
        vals = np.array([p.get(key, np.nan) for p in passports], dtype=float)
        z[key] = sign * robust_z(vals) if np.isfinite(vals).sum() >= 3 else np.zeros(K)
    out = []
    for k in range(K):
        reasons = [f"{key} z={z[key][k]:.1f}" for key in bad if z[key][k] > z_flag]
        worst = max(z[key][k] for key in bad)
        verdict = "exclude" if worst > z_exclude else ("flag" if worst > z_flag else "accept")
        out.append(QCDecision(k, verdict, reasons))
    return out


def stain_passport(rgb: np.ndarray, od_threshold: float = 0.15, i0: float | np.ndarray | None = None, rule: str = "all") -> dict:
    """Stain diagnostics for one RGB section: estimated vectors vs Ruifrok defaults,
    mean H/E concentration over tissue, and the OD model residual. Returns ``{'ok': False, ...}`` if the
    section has too little tissue for a stable estimate.

    ``i0`` is the white point; the default (per-channel 99th percentile) suits slides with a bright background. Full-field
    patches without background need a fixed white point (``i0=1.0``), otherwise most tissue is read as background."""
    from .stain import RUIFROK_HE, deconvolve, estimate_stain_vectors, reconstruction_residual, rgb_to_od, tissue_mask

    od = rgb_to_od(rgb, i0=i0)
    tissue = tissue_mask(od, od_threshold, rule)
    try:
        V, diag = estimate_stain_vectors(od, od_threshold, rule=rule)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc), "tissue_fraction": float(tissue.mean())}
    conc = deconvolve(od, V)
    return {
        "ok": True, "tissue_fraction": float(tissue.mean()), **diag,
        "mean_h": float(conc[tissue][:, 0].mean()), "mean_e": float(conc[tissue][:, 1].mean()),
        "residual_rms": reconstruction_residual(od[tissue][None], V, conc[tissue][None]),
        "angle_default_h_deg": float(np.degrees(np.arccos(np.clip(V[0] @ RUIFROK_HE[0], -1, 1)))),
    }
