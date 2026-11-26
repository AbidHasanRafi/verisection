"""Virtual sectioning of a ground-truth volume.

A correlated random phantom stands in for real 3D tissue so that the true 3D label field is
known. A fine-grid *intensity* volume (class mean + spatially correlated texture) is cut into
virtual sections that carry the error sources the pipeline must survive: per-section rigid
drift and rotation, smooth non-rigid deformation (registration), per-class stain shifts and
class-dependent noise (segmentation), slab averaging / partial volume, and a gap between
sections (interpolation).

Presets: ``SectioningConfig()`` is the easy default; ``SectioningConfig.hard()`` makes
segmentation and registration genuinely difficult.

Simulation results are *sanity checks of the machinery*, not evidence about real tissue
.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi

from .transform import smooth_random_field
from .volume import SectionStack


def make_phantom(
    shape_zyx: tuple[int, int, int] = (120, 64, 64),
    dz_um: float = 2.0,
    pixel_um: float = 2.0,
    xi_z_um: float = 36.0,
    xi_xy_um: float = 14.0,
    fractions: tuple[float, ...] = (0.5, 0.3, 0.2),
    seed: int = 0,
) -> np.ndarray:
    """Correlated multi-class label volume (Z, Y, X) with nominal correlation lengths."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape_zyx)
    sigma = (xi_z_um / dz_um / 2.0, xi_xy_um / pixel_um / 2.0, xi_xy_um / pixel_um / 2.0)
    field = ndi.gaussian_filter(noise, sigma, mode="wrap")
    field = (field - field.mean()) / field.std()
    cuts = np.quantile(field, np.cumsum(fractions)[:-1])
    return np.digitize(field, cuts).astype(np.int8)


@dataclass
class SectioningConfig:
    n_sections: int = 18
    first_plane: int = 2
    spacing_planes: int = 6  # fine-grid planes between consecutive sections
    thickness_um: float = 4.0
    class_means: tuple[float, ...] = (0.2, 0.55, 0.9)
    noise_sigma: float = 0.12
    noise_by_class: tuple[float, ...] = (1.0, 1.0, 1.0)  # multiplicative, heteroscedastic noise
    blur_sigma_px: float = 0.8
    stain_scale_sd: float = 0.03
    stain_offset_sd: float = 0.03
    class_offset_sd: float = 0.0  # per-section, per-class stain shift of the class mean
    drift_step_px: float = 0.8  # random-walk step per section
    jitter_px: float = 0.4  # independent per-section jitter
    rotation_step_deg: float = 0.0  # random-walk rotation step per section
    nonrigid_amp_px: float = 0.0  # RMS amplitude of the smooth per-section deformation
    nonrigid_sigma_px: float = 10.0  # smoothness of that deformation
    texture_sd: float = 0.0  # sd of spatially correlated intensity texture (within class)
    texture_sigma_xy_px: float = 1.5
    texture_sigma_z_planes: float = 1.0

    @classmethod
    def tissue(cls, **overrides) -> "SectioningConfig":
        """Preset for the gland/immune/stroma phantom (4 classes; intensity carries texture and nuclei)."""
        base = cls(
            class_means=(0.10, 0.42, 0.68, 0.90), noise_by_class=(0.6, 1.0, 1.0, 1.0), noise_sigma=0.07,
            blur_sigma_px=0.7, stain_scale_sd=0.04, stain_offset_sd=0.03, class_offset_sd=0.02,
            drift_step_px=0.5, jitter_px=0.3, rotation_step_deg=0.3, nonrigid_amp_px=1.5, texture_sd=0.0,
        )
        return replace(base, **overrides)

    @classmethod
    def hard(cls, **overrides) -> "SectioningConfig":
        """Segmentation- and registration-challenging preset."""
        base = cls(
            class_means=(0.20, 0.50, 0.80), noise_sigma=0.15, noise_by_class=(0.8, 1.0, 1.3),
            stain_scale_sd=0.05, stain_offset_sd=0.04, class_offset_sd=0.04,
            drift_step_px=0.9, jitter_px=0.5, rotation_step_deg=0.7,
            nonrigid_amp_px=3.0, texture_sd=0.06,
        )
        return replace(base, **overrides)


def make_intensity_volume(
    labels: np.ndarray, cfg: SectioningConfig, rng: np.random.Generator
) -> np.ndarray:
    """Fine-grid noise-free-ish intensity truth: class mean + correlated texture (float32)."""
    vol = np.asarray(cfg.class_means, dtype=np.float32)[labels]
    if cfg.texture_sd > 0:
        tex = ndi.gaussian_filter(
            rng.normal(size=labels.shape),
            (cfg.texture_sigma_z_planes, cfg.texture_sigma_xy_px, cfg.texture_sigma_xy_px),
            mode="wrap",
        )
        tex = tex / tex.std() * cfg.texture_sd
        vol = vol + tex.astype(np.float32)
    return vol


def virtual_sections(
    truth: np.ndarray,
    cfg: SectioningConfig,
    dz_um: float = 2.0,
    pixel_um: float = 2.0,
    seed: int = 0,
    intensity: np.ndarray | None = None,
) -> SectionStack:
    """Cut ``truth`` into noisy, deformed, stain-varying sections.

    Section ``k`` averages the planes of its slab (partial volume at the slab faces). Truth stays in
    ``meta`` and is read only by evaluation code.
    """
    rng = np.random.default_rng(seed + 10_000)
    Z, H, W = truth.shape
    idx = cfg.first_plane + cfg.spacing_planes * np.arange(cfg.n_sections)
    if idx[-1] >= Z:
        raise ValueError("sections do not fit in the phantom; reduce n_sections or spacing")
    z_um = (idx + 0.5) * dz_um
    means = np.asarray(cfg.class_means, dtype=float)
    n_cls = len(means)
    intens = make_intensity_volume(truth, cfg, rng) if intensity is None else np.asarray(intensity, dtype=np.float32)

    # --- per-section geometry: rigid random walk (+ rotation) and smooth deformation ---
    shifts = np.zeros((cfg.n_sections, 2))
    thetas = np.zeros(cfg.n_sections)
    walk, th = np.zeros(2), 0.0
    for k in range(1, cfg.n_sections):
        walk = walk + rng.normal(0, cfg.drift_step_px, 2)
        th = th + np.deg2rad(rng.normal(0, cfg.rotation_step_deg))
        shifts[k] = walk + rng.normal(0, cfg.jitter_px, 2)
        thetas[k] = th
    grid = np.stack(np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij"))
    o = np.array([(H - 1) / 2.0, (W - 1) / 2.0])[:, None, None]

    images, true_h = [], []
    for k, zi in enumerate(idx):
        # slab average of planes (weights 1/2, 1, 1/2) and class-dependent noise
        lo, hi = max(zi - 1, 0), min(zi + 1, Z - 1)
        slab = 0.25 * intens[lo] + 0.5 * intens[zi] + 0.25 * intens[hi]
        lab = truth[zi]
        off = rng.normal(0, cfg.class_offset_sd, n_cls) if cfg.class_offset_sd > 0 else np.zeros(n_cls)
        sd = cfg.noise_sigma * np.asarray(cfg.noise_by_class)[lab]
        img = slab + off[lab] + rng.normal(0, 1, lab.shape) * sd
        img = ndi.gaussian_filter(img, cfg.blur_sigma_px)
        img = (1.0 + rng.normal(0, cfg.stain_scale_sd)) * img + rng.normal(0, cfg.stain_offset_sd)

        # h_k: raw pixel coordinates -> common-frame coordinates
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        R = np.array([[c, -s], [s, c]])
        h = (R @ (grid - o).reshape(2, -1)).reshape(grid.shape) + o - shifts[k][:, None, None]
        if cfg.nonrigid_amp_px > 0 and k > 0:
            h = h + smooth_random_field((H, W), cfg.nonrigid_amp_px, cfg.nonrigid_sigma_px, rng)
        raw = ndi.map_coordinates(img, h, order=1, mode="nearest")
        images.append(raw.astype(np.float32))
        true_h.append(h)

    return SectionStack(
        images=images,
        z_um=z_um,
        thickness_um=cfg.thickness_um,
        pixel_um=pixel_um,
        meta={
            "truth_volume": truth,
            "truth_intensity": intens,
            "dz_fine_um": dz_um,
            "true_shifts_px": shifts,
            "true_theta_rad": thetas,
            "true_h": true_h,
            "section_planes": idx,
            "sectioning": cfg,
        },
    )


def make_pilot_block(
    truth: np.ndarray, cfg: SectioningConfig, start_plane: int, n_sections: int,
    dz_um: float = 2.0, pixel_um: float = 2.0, seed: int = 0, intensity: np.ndarray | None = None,
) -> SectionStack:
    """A dense run of consecutive thin sections (spacing = thickness): the 'pilot sub-block'
    from which correlation lengths are estimated without using the truth labels."""
    spacing_planes = max(int(round(cfg.thickness_um / dz_um)), 1)
    dense = replace(cfg, n_sections=n_sections, first_plane=start_plane, spacing_planes=spacing_planes)
    return virtual_sections(truth, dense, dz_um, pixel_um, seed + 777, intensity=intensity)
