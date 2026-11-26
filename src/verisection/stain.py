"""Optical density and stain separation.

OD = -log10(I / I0) per RGB channel, linear in absorbance under Beer-Lambert. Stain separation
solves OD = C @ V for non-negative concentrations ``C`` given stain vectors ``V`` (rows):

* ``RUIFROK_HE`` fixed H&E vectors (Ruifrok & Johnston, 2001);
* ``estimate_stain_vectors`` Macenko-style estimation (SVD plane of the OD cloud, extreme angles).

Caveats: the linear model breaks down at saturation and for strongly overlapping stains, and
estimated vectors are unstable on slides with little of one stain. Residual and vector-angle
diagnostics are returned so the QC layer can flag such sections.
"""
from __future__ import annotations

import numpy as np

# Rows: hematoxylin, eosin, (residual/third) unit OD vectors in RGB (Ruifrok & Johnston 2001 defaults).
RUIFROK_HE = np.array(
    [[0.644, 0.717, 0.267],
     [0.093, 0.954, 0.283],
     [0.636, 0.001, 0.771]]
)
RUIFROK_HE = RUIFROK_HE / np.linalg.norm(RUIFROK_HE, axis=1, keepdims=True)


def rgb_to_od(rgb: np.ndarray, i0: float | np.ndarray | None = None, eps: float = 1.0 / 255.0) -> np.ndarray:
    """OD of an (H, W, 3) image with values in [0, 1]; ``i0`` defaults to the 99th percentile per channel."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected an (H, W, 3) RGB image")
    if i0 is None:
        i0 = np.percentile(rgb.reshape(-1, 3), 99, axis=0)
    return -np.log10(np.clip(rgb, eps, None) / np.maximum(i0, eps))


def od_to_rgb(od: np.ndarray, i0: float | np.ndarray = 1.0) -> np.ndarray:
    return np.asarray(i0) * 10.0 ** (-np.asarray(od))


def deconvolve(od: np.ndarray, vectors: np.ndarray = RUIFROK_HE) -> np.ndarray:
    """Concentrations (H, W, n_stains) from OD (H, W, 3); negative values are clipped to zero."""
    V = np.asarray(vectors, dtype=np.float64)
    C = od.reshape(-1, 3) @ np.linalg.pinv(V)
    return np.clip(C, 0.0, None).reshape(od.shape[:2] + (V.shape[0],))


def reconstruction_residual(od: np.ndarray, vectors: np.ndarray, conc: np.ndarray) -> float:
    """RMS OD residual of the stain model over tissue pixels (large = model does not fit)."""
    rec = conc.reshape(-1, conc.shape[-1]) @ vectors
    return float(np.sqrt(np.mean((od.reshape(-1, 3) - rec) ** 2)))


def tissue_mask(od: np.ndarray, od_threshold: float = 0.15, rule: str = "all") -> np.ndarray:
    """Pixels carrying stain. ``all``: OD above the threshold in every channel (Macenko); ``sum``: summed OD above
    3x the threshold (keeps pale, eosin-dominant tissue whose red-channel OD is small)."""
    if rule == "all":
        return (od > od_threshold).all(axis=-1)
    if rule == "sum":
        return od.sum(axis=-1) > 3.0 * od_threshold
    raise ValueError(f"unknown tissue rule: {rule}")


def estimate_stain_vectors(
    od: np.ndarray, od_threshold: float = 0.15, percentile: float = 1.0, rule: str = "all"
) -> tuple[np.ndarray, dict]:
    """Macenko-style 2-stain estimate; returns (3x3 vectors with a complementary third row, diagnostics).

    Pixels below ``od_threshold`` (background) are ignored. Raises ``ValueError`` if too little tissue.
    """
    flat = od.reshape(-1, 3)
    keep = flat[tissue_mask(flat, od_threshold, rule)]
    if len(keep) < 500:
        raise ValueError("too few tissue pixels to estimate stain vectors")
    # SVD plane of the OD cloud
    _, _, vt = np.linalg.svd(np.cov(keep.T))
    plane = vt[:2].T  # (3, 2)
    if plane[:, 0].sum() < 0:
        plane[:, 0] *= -1
    if plane[:, 1].sum() < 0:
        plane[:, 1] *= -1
    proj = keep @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [percentile, 100 - percentile])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    # hematoxylin has the larger red-channel OD
    h, e = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    H = np.vstack([h / np.linalg.norm(h), e / np.linalg.norm(e)])
    third = np.cross(H[0], H[1])
    V = np.vstack([H, third / np.linalg.norm(third)])
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))  # noqa: E731
    diag = {
        "n_tissue_pixels": int(len(keep)),
        "angle_to_ruifrok_h_deg": ang(V[0], RUIFROK_HE[0]),
        "angle_to_ruifrok_e_deg": ang(V[1], RUIFROK_HE[1]),
        "angle_between_stains_deg": ang(V[0], V[1]),
    }
    return V, diag
