"""Texture on anisotropic, gapped volumes.

Design rules implemented here:

1. physical scale: offsets are given in µm and converted with the pixel size;
2. fixed quantisation: gray levels use a fixed bin width (OD units), never per-image min-max;
3. observed-support restriction: only Observed voxels enter (callers pass masks);
4. registration-limited scale and z-coherence decide 3D vs 2.5D vs masked;
5. features are *certified* by ICC under a perturbation suite before use.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------- basics
def optical_density(img: np.ndarray, i0: float | None = None, eps: float = 1e-6) -> np.ndarray:
    """OD = -log10(I / I0); ``I0`` defaults to the 99th percentile (bright background)."""
    img = np.asarray(img, dtype=np.float64)
    i0 = float(np.percentile(img, 99)) if i0 is None else i0
    return -np.log10(np.clip(img, eps, None) / max(i0, eps))


def quantize_fixed(x: np.ndarray, bin_width: float, n_levels: int, vmin: float = 0.0) -> np.ndarray:
    """Fixed-bin-width quantisation (not per-image rescaling)."""
    q = np.floor((np.asarray(x, dtype=np.float64) - vmin) / bin_width + 1e-9).astype(np.int64)
    return np.clip(q, 0, n_levels - 1)


def physical_offset_px(offset_um: float, pixel_um: float, angle_deg: float) -> tuple[int, int]:
    """(dy, dx) pixel offset for a physical distance and direction."""
    if offset_um < pixel_um / 2:
        raise ValueError("offset is below half a pixel; cannot be represented at this resolution")
    a = np.deg2rad(angle_deg)
    r = offset_um / pixel_um
    dy, dx = int(round(-r * np.sin(a))), int(round(r * np.cos(a)))
    if dy == 0 and dx == 0:
        dx = 1
    return dy, dx


def glcm_offset(
    levels: np.ndarray, mask: np.ndarray, dy: int, dx: int, n_levels: int, symmetric: bool = True
) -> tuple[np.ndarray, int]:
    """Normalised gray-level co-occurrence matrix and the number of pairs used."""
    H, W = levels.shape
    y0, y1 = max(0, -dy), H - max(0, dy)
    x0, x1 = max(0, -dx), W - max(0, dx)
    a = levels[y0:y1, x0:x1]
    b = levels[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    v = mask[y0:y1, x0:x1] & mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return _counts_to_glcm(a[v], b[v], n_levels, symmetric)


def glcm_z(
    levels_a: np.ndarray, levels_b: np.ndarray, mask: np.ndarray, n_levels: int, symmetric: bool = True
) -> tuple[np.ndarray, int]:
    """Co-occurrence between corresponding pixels of two registered, *observed* sections."""
    return _counts_to_glcm(levels_a[mask], levels_b[mask], n_levels, symmetric)


def _counts_to_glcm(a: np.ndarray, b: np.ndarray, n_levels: int, symmetric: bool):
    n = int(a.size)
    if n == 0:
        return np.zeros((n_levels, n_levels)), 0
    P = np.bincount(a * n_levels + b, minlength=n_levels * n_levels).reshape(n_levels, n_levels).astype(float)
    if symmetric:
        P = P + P.T
    return P / P.sum(), n


def glcm_features(P: np.ndarray) -> dict[str, float]:
    """Haralick-style features of a normalised GLCM."""
    n = P.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mu_i, mu_j = (P * i).sum(), (P * j).sum()
    sd_i = np.sqrt((P * (i - mu_i) ** 2).sum())
    sd_j = np.sqrt((P * (j - mu_j) ** 2).sum())
    nz = P > 0
    corr = 0.0 if sd_i * sd_j < 1e-12 else float((P * (i - mu_i) * (j - mu_j)).sum() / (sd_i * sd_j))
    return {
        "contrast": float((P * (i - j) ** 2).sum()),
        "dissimilarity": float((P * np.abs(i - j)).sum()),
        "homogeneity": float((P / (1.0 + (i - j) ** 2)).sum()),
        "asm": float((P**2).sum()),
        "entropy": float(-(P[nz] * np.log2(P[nz])).sum()),
        "correlation": corr,
    }


def window_features_2d(
    img_od: np.ndarray,
    mask: np.ndarray,
    pixel_um: float,
    offset_um: float = 2.0,
    bin_width: float = 0.05,
    n_levels: int = 32,
    angles_deg: tuple[float, ...] = (0, 45, 90, 135),
) -> dict[str, float]:
    """In-plane GLCM features averaged over directions (physical offset, fixed bins)."""
    lv = quantize_fixed(img_od, bin_width, n_levels)
    feats, pairs = [], 0
    for ang in angles_deg:
        dy, dx = physical_offset_px(offset_um, pixel_um, ang)
        P, n = glcm_offset(lv, mask, dy, dx, n_levels)
        if n:
            feats.append(glcm_features(P))
            pairs += n
    if not feats:
        return {k: float("nan") for k in glcm_features(np.eye(2) / 2)}
    return {k: float(np.mean([f[k] for f in feats])) for k in feats[0]} | {"n_pairs": float(pairs)}


# ----------------------------------------------------------------- validity
class Validity(str, Enum):
    FULL_3D = "3D"
    TWO_POINT_FIVE_D = "2.5D"
    MASKED = "masked"


@dataclass
class ValidityDecision:
    mode: Validity
    reasons: list[str] = field(default_factory=list)
    observed_fraction: float = 1.0

    @property
    def ici(self) -> float:
        """Interpolation-contamination index = 1 - observed fraction."""
        return 1.0 - self.observed_fraction


def min_scale_um(pixel_um: float, sigma_reg_um: float) -> float:
    """Registration-limited minimum feature scale: max(pixel, 2*sigma_reg)."""
    return max(pixel_um, 2.0 * sigma_reg_um)


def texture_validity(
    observed_fraction: float,
    *,
    ell_xy_um: float,
    pixel_um: float,
    sigma_reg_um: float,
    dz_um: float,
    xi_um: float,
    needs_z: bool,
    rho_min: float = 0.8,
    kappa: float = 1.0,
) -> ValidityDecision:
    """Decide whether a feature may be computed in 3D, must fall back to 2.5D, or is masked."""
    d = ValidityDecision(Validity.FULL_3D, [], observed_fraction)
    if observed_fraction < rho_min:
        d.mode = Validity.MASKED
        d.reasons.append(f"observed fraction {observed_fraction:.2f} < {rho_min}")
        return d
    if not needs_z:
        d.mode = Validity.TWO_POINT_FIVE_D
        d.reasons.append("in-plane feature aggregated across sections")
        return d
    if ell_xy_um < min_scale_um(pixel_um, sigma_reg_um):
        d.mode = Validity.TWO_POINT_FIVE_D
        d.reasons.append("feature scale below registration-limited minimum scale")
    if dz_um > kappa * xi_um:
        d.mode = Validity.TWO_POINT_FIVE_D
        d.reasons.append("section spacing exceeds correlation length (z-decorrelated)")
    return d


# --------------------------------------------------------- reliability and certification
def icc_2_1(data: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    ``data`` has shape (n_subjects, n_conditions).
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = data.mean()
    ms_r = k * ((data.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((data.mean(0) - grand) ** 2).sum() / (k - 1)
    ss_e = ((data - grand) ** 2).sum() - (n - 1) * ms_r - (k - 1) * ms_c
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 0 else float("nan")


def icc_bootstrap_lower(
    data: np.ndarray, rng: np.random.Generator, n_boot: int = 300, q: float = 0.025
) -> float:
    """Lower CI bound of ICC(2,1) by bootstrap over subjects (windows/specimens)."""
    n = data.shape[0]
    vals = []
    for _ in range(n_boot):
        v = icc_2_1(data[rng.integers(0, n, n)])
        if np.isfinite(v):
            vals.append(v)
    return float(np.quantile(vals, q)) if vals else float("nan")


def perturb(img: np.ndarray, kind: str, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Acquisition/processing perturbations of the stability suite."""
    x = np.asarray(img, dtype=np.float64)
    if kind == "noise":
        return x + rng.normal(0, strength * (x.std() + 1e-12), x.shape)
    if kind == "blur":
        return ndi.gaussian_filter(x, strength)
    if kind == "od_scale":  # thickness-like OD scaling
        return x * (1.0 + strength)
    if kind == "offset":
        return x + strength * (x.std() + 1e-12)
    raise ValueError(f"unknown perturbation: {kind}")


@dataclass
class StabilityResult:
    feature: str
    icc: float
    icc_lower: float
    certified: bool
    n_windows: int
    conditions: list[str]


def stability_suite(
    feature_fn: Callable[[np.ndarray], float],
    windows: list[np.ndarray],
    conditions: list[tuple[str, float]],
    rng: np.random.Generator,
    feature_name: str = "feature",
    threshold: float = 0.75,
) -> StabilityResult:
    """Certify a feature if the lower CI bound of its ICC across perturbations is >= ``threshold``.

    Column 0 of the data matrix is the unperturbed condition; windows are the subjects.
    """
    cols = [np.array([feature_fn(w) for w in windows])]
    for kind, s in conditions:
        cols.append(np.array([feature_fn(perturb(w, kind, s, rng)) for w in windows]))
    data = np.stack(cols, axis=1)
    icc = icc_2_1(data)
    lo = icc_bootstrap_lower(data, rng)
    return StabilityResult(
        feature_name, icc, lo, bool(np.isfinite(lo) and lo >= threshold), len(windows),
        ["baseline"] + [f"{k}:{s}" for k, s in conditions],
    )
