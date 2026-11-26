"""Volume construction from registered section posteriors.

Observed slabs copy the (warped) section posterior; gap planes are interpolated;
voxels outside any warped section are marked UNSAMPLED. Provenance is assigned
here and nowhere else.
"""
from __future__ import annotations

import numpy as np

from .interpolate import InterpConfig, interpolate_plane
from .provenance import Provenance
from .transform import Transform
from .volume import ProbVolume, SectionStack, make_grid


def warp_posterior(
    p: np.ndarray, transform: Transform, prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (C, H, W) section posterior into the common frame; returns (posterior, valid).

    Pixels whose source location falls outside the section are marked invalid and carry the prior.
    """
    out, valid = transform.apply(p, order=1, cval=0.0)
    out = np.where(valid[None], out, prior[:, None, None])
    return out.astype(np.float32), valid


def section_priors(section_probs: list[np.ndarray]) -> np.ndarray:
    pri = np.mean([p.mean(axis=(1, 2)) for p in section_probs], axis=0)
    return pri / pri.sum()


def build_volume(
    stack: SectionStack,
    transforms: list[Transform],
    section_probs: list[np.ndarray],
    interp: InterpConfig,
    class_names: list[str],
    priors: np.ndarray | None = None,
) -> ProbVolume:
    """Assemble a ``ProbVolume`` from per-section posteriors and one registration sample
    (one ``Transform`` per section)."""
    C = section_probs[0].shape[0]
    H, W = stack.shape
    dz = interp.dz_um or stack.thickness_um
    centers, plane_section, gap = make_grid(stack.z_um, stack.thickness_um, dz)
    pri = section_priors(section_probs) if priors is None else priors
    xi = interp.xi_for(C) if interp.mode == "kriging" else None

    warped, valid = [], []
    for k, p in enumerate(section_probs):
        w, v = warp_posterior(p, transforms[k], pri)
        warped.append(w)
        valid.append(v)

    Z = len(centers)
    prob = np.empty((C, Z, H, W), dtype=np.float32)
    prov = np.zeros((Z, H, W), dtype=np.uint8)
    for j in range(Z):
        k = plane_section[j]
        if k >= 0:
            prob[:, j] = warped[k]
            prov[j] = np.where(valid[k], Provenance.OBSERVED, Provenance.UNSAMPLED)
        else:
            hi = int(np.searchsorted(stack.z_um, centers[j]))
            lo = hi - 1
            d_lo = float(centers[j] - stack.z_um[lo])
            d_hi = float(stack.z_um[hi] - centers[j])
            p, v = interpolate_plane(
                interp.mode, warped[lo], warped[hi], valid[lo], valid[hi],
                d_lo, d_hi, pri, xi, interp.nugget,
            )
            prob[:, j] = p
            prov[j] = np.where(v, Provenance.INTERPOLATED, Provenance.UNSAMPLED)
        # unsampled voxels carry the prior so that sums stay 1
        prob[:, j] = np.where((prov[j] > 0)[None], prob[:, j], pri[:, None, None])
    return ProbVolume(
        label_prob=prob,
        provenance=prov,
        z_um=centers,
        pixel_um=stack.pixel_um,
        dz_um=dz,
        gap_distance_um=gap,
        class_names=class_names,
    )
