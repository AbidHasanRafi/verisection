"""Gap interpolators: I0 nearest, I1 linear, I3 correlation-length-aware.

I3 is simple indicator kriging with an exponential covariance along z. Far from
the observed slabs the posterior returns to the class prior, so the renderer can
show those voxels as "unknown" rather than as a confident smoothed guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["nearest", "linear", "kriging"]


@dataclass
class InterpConfig:
    mode: Mode = "kriging"
    xi_um: np.ndarray | None = None  # (C,) class correlation lengths
    nugget: float = 0.05  # observation-noise fraction of the sill
    dz_um: float | None = None  # grid step; defaults to section thickness

    def xi_for(self, n_classes: int) -> np.ndarray:
        if self.xi_um is None:
            raise ValueError("kriging requires class correlation lengths (xi_um)")
        xi = np.asarray(self.xi_um, dtype=float)
        if xi.shape != (n_classes,):
            raise ValueError(f"xi_um must have shape ({n_classes},)")
        return xi


def kriging_weights_two(
    d_lo: float, d_hi: float, spacing: float, xi: float, sill: float, nugget: float
) -> tuple[np.ndarray, float]:
    """Simple-kriging weights for two flanking observations and the relative error variance."""
    r1, r2, r12 = np.exp(-d_lo / xi), np.exp(-d_hi / xi), np.exp(-spacing / xi)
    K = sill * np.array([[1.0 + nugget, r12], [r12, 1.0 + nugget]])
    c0 = sill * np.array([r1, r2])
    w = np.linalg.solve(K, c0)
    return w, float(sill - w @ c0)


def interpolate_plane(
    mode: Mode,
    p_lo: np.ndarray,
    p_hi: np.ndarray,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    d_lo: float,
    d_hi: float,
    priors: np.ndarray,
    xi_um: np.ndarray | None,
    nugget: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one gap plane from its two flanking section posteriors.

    ``p_*`` have shape (C, H, W); ``v_*`` are boolean validity masks (H, W).
    Returns ``(p, valid)``.
    """
    if mode == "nearest":
        near_lo = d_lo <= d_hi
        base, other = (p_lo, p_hi) if near_lo else (p_hi, p_lo)
        vb, vo = (v_lo, v_hi) if near_lo else (v_hi, v_lo)
        out = np.where(vb[None], base, other)
    elif mode == "linear":
        w_hi = d_lo / (d_lo + d_hi)
        both = (1.0 - w_hi) * p_lo + w_hi * p_hi
        out = np.where((v_lo & v_hi)[None], both, np.where(v_lo[None], p_lo, p_hi))
    elif mode == "kriging":
        if xi_um is None:
            raise ValueError("kriging requires xi_um")
        C = p_lo.shape[0]
        spacing = d_lo + d_hi
        both = np.empty_like(p_lo)
        only_lo = np.empty_like(p_lo)
        only_hi = np.empty_like(p_lo)
        for c in range(C):
            pi = float(priors[c])
            sill = max(pi * (1.0 - pi), 1e-6)
            w, _ = kriging_weights_two(d_lo, d_hi, spacing, float(xi_um[c]), sill, nugget)
            wl = np.exp(-d_lo / xi_um[c]) / (1.0 + nugget)
            wh = np.exp(-d_hi / xi_um[c]) / (1.0 + nugget)
            both[c] = pi + w[0] * (p_lo[c] - pi) + w[1] * (p_hi[c] - pi)
            only_lo[c] = pi + wl * (p_lo[c] - pi)
            only_hi[c] = pi + wh * (p_hi[c] - pi)
        out = np.where(
            (v_lo & v_hi)[None], both, np.where(v_lo[None], only_lo, only_hi)
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown interpolation mode: {mode}")
    out = np.clip(out, 1e-6, None)
    out = out / out.sum(axis=0, keepdims=True)
    return out.astype(np.float32), (v_lo | v_hi)
