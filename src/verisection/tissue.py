"""A tissue-like 3D phantom: branching glands (lumen + epithelium), immune-cell clusters, stroma, nuclei.

Unlike the Gaussian-random-field phantom, structures here have recognisable morphology that looks
different along each axis: tubes meander through the volume, so an x–y slice shows round or oval
gland cross-sections while an x–z slice shows long sheared profiles. Intensities are OD-like (higher =
darker): lumen ≈ white, stroma pink, epithelium purple, immune cells dark blue; nuclei are small dark
spots, stroma carries oriented fibre texture.

This is still a synthetic phantom for exercising the pipeline. It is not a model of any real tissue,
and no claim about real histology should be drawn from results on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Classes are indexed by increasing optical density, the order an intensity-based segmenter recovers.
CLASS_NAMES = ["lumen", "stroma", "epithelium", "immune"]
CLASS_MEANS = (0.10, 0.42, 0.68, 0.90)
_OLD_MEANS = (0.42, 0.68, 0.10, 0.90)  # generation order: stroma, epithelium, lumen, immune
_REMAP = np.array([1, 2, 0, 3], dtype=np.int8)  # generation index -> OD-ordered index


@dataclass
class TissuePhantom:
    labels: np.ndarray  # (Z, Y, X) int8
    intensity: np.ndarray  # (Z, Y, X) float32, noise-free texture + nuclei
    nuclei_um: np.ndarray  # (N, 3) z, y, x nucleus centres in µm
    centerlines_um: list[np.ndarray]


def _tube_centerline(rng, start, direction, n_steps, step_um, steer, extent=None):
    """Persistent random walk with smoothly varying direction; bounces off the volume faces."""
    pts = [np.asarray(start, float)]
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    for _ in range(n_steps):
        d = d + rng.normal(0, steer, 3)
        d /= np.linalg.norm(d)
        nxt = pts[-1] + d * step_um
        if extent is not None:
            for a in range(3):
                if nxt[a] < 0.05 * extent[a] or nxt[a] > 0.95 * extent[a]:
                    d[a] = -d[a]
            nxt = pts[-1] + d * step_um
        pts.append(nxt)
    return np.array(pts)


def make_tissue_phantom(
    shape_zyx: tuple[int, int, int] = (120, 96, 96),
    dz_um: float = 2.0,
    pixel_um: float = 2.0,
    n_glands: int = 7,
    gland_radius_um: float = 20.0,
    lumen_fraction: float = 0.42,
    n_immune_clusters: int = 10,
    nuclei_density_per_um3: float = 6e-4,
    seed: int = 0,
) -> TissuePhantom:
    """Branching glands in stroma with immune clusters and nuclei; returns labels, intensity and nuclei."""
    rng = np.random.default_rng(seed)
    Z, Y, X = shape_zyx
    ext = np.array([Z * dz_um, Y * pixel_um, X * pixel_um])
    zz, yy, xx = np.meshgrid(np.arange(Z) * dz_um, np.arange(Y) * pixel_um, np.arange(X) * pixel_um, indexing="ij")
    grid = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

    # --- gland centrelines (with side branches) ---
    lines: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    for _ in range(n_glands):
        start = rng.uniform(0.15, 0.85, 3) * ext
        direction = rng.normal(size=3)
        direction[0] *= 0.3  # glands run mostly in-plane but wander through z
        n = int(rng.integers(130, 190))
        main = _tube_centerline(rng, start, direction, n, 3.0, 0.09, ext)
        main_r = gland_radius_um * (1.0 + 0.18 * np.sin(np.linspace(0, rng.uniform(2, 5) * np.pi, len(main)) + rng.uniform(0, 6)))
        lines.append(main)
        radii.append(main_r * rng.uniform(0.85, 1.15))
        for _ in range(int(rng.integers(2, 4))):  # branches
            k = int(rng.integers(len(main) // 4, 3 * len(main) // 4))
            bdir = rng.normal(size=3)
            br = _tube_centerline(rng, main[k], bdir, int(rng.integers(45, 80)), 3.0, 0.11, ext)
            lines.append(br)
            radii.append(main_r[k] * 0.7 * np.linspace(1.0, 0.75, len(br)))
    pts = np.vstack(lines)
    rad = np.concatenate(radii)
    d, idx = cKDTree(pts).query(grid, k=1)
    r_out = rad[idx]
    d = d.reshape(Z, Y, X)
    r_out = r_out.reshape(Z, Y, X)
    labels = np.zeros((Z, Y, X), dtype=np.int8)  # stroma
    labels[d < r_out] = 1  # epithelium (wall + lumen first)
    labels[d < lumen_fraction * r_out] = 2  # lumen
    # smooth boundaries slightly so tubes are not voxel-perfect
    smooth = ndi.gaussian_filter((labels == 2).astype(np.float32), 0.8) > 0.5
    labels[(labels == 2) & ~smooth] = 1

    # --- immune clusters in stroma ---
    for _ in range(n_immune_clusters):
        c = rng.uniform(0.1, 0.9, 3) * ext
        rr = rng.uniform(9.0, 16.0)
        n_cells = int(rng.integers(14, 26))
        for _ in range(n_cells):
            p = c + rng.normal(0, rr * 0.55, 3)
            m = (np.linalg.norm(grid - p, axis=1) < rng.uniform(4.6, 6.6)).reshape(Z, Y, X)
            labels[m & (labels == 0)] = 3

    # --- nuclei: dark spots in epithelium and immune cells, sparse elongated ones in stroma ---
    n_expected = nuclei_density_per_um3 * ext.prod()
    cand = rng.uniform(0, 1, (int(n_expected * 5), 3)) * ext
    zi = np.clip((cand[:, 0] / dz_um).astype(int), 0, Z - 1)
    yi = np.clip((cand[:, 1] / pixel_um).astype(int), 0, Y - 1)
    xi = np.clip((cand[:, 2] / pixel_um).astype(int), 0, X - 1)
    cls = labels[zi, yi, xi]
    keep_p = np.select([cls == 1, cls == 3, cls == 0], [0.9, 1.0, 0.12], default=0.0)
    nuclei = cand[rng.random(len(cand)) < keep_p / 5.0 * 1.0]
    nuc_field = np.zeros((Z, Y, X), dtype=np.float32)
    zi = np.clip((nuclei[:, 0] / dz_um).astype(int), 0, Z - 1)
    yi = np.clip((nuclei[:, 1] / pixel_um).astype(int), 0, Y - 1)
    xi = np.clip((nuclei[:, 2] / pixel_um).astype(int), 0, X - 1)
    nuc_field[zi, yi, xi] = 1.0
    nuc_field = ndi.gaussian_filter(nuc_field, (1.2 / (dz_um / 2.0), 1.3, 1.3))
    nuc_field = nuc_field / max(nuc_field.max(), 1e-9)
    nuc_field = np.clip(nuc_field * 3.0, 0, 1)

    # --- intensity: class means + collagen-like oriented texture in stroma + nuclei ---
    means = np.asarray(_OLD_MEANS, dtype=np.float32)
    inten = means[labels].astype(np.float32)
    fibre = ndi.gaussian_filter(rng.normal(size=(Z, Y, X)), (1.0, 0.8, 5.0))
    fibre = fibre / fibre.std() * 0.035
    inten += np.where(labels == 0, fibre, 0.0).astype(np.float32)
    epi_tex = ndi.gaussian_filter(rng.normal(size=(Z, Y, X)), (0.8, 1.0, 1.0))
    inten += np.where(labels == 1, epi_tex / epi_tex.std() * 0.035, 0.0).astype(np.float32)
    inten += (0.20 * nuc_field * np.isin(labels, (1, 3, 0))).astype(np.float32)
    return TissuePhantom(_REMAP[labels], inten.astype(np.float32), nuclei, lines)
