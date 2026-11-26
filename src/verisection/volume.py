"""Core data model: section stacks and probabilistic volumes.

The volume grid is anisotropic: in-plane pixels are ``pixel_um`` and the z-step is
``dz_um``. Sections are slabs of physical thickness; the space between slabs is
unobserved and must be filled by an interpolator that reports uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .provenance import Manifest, Provenance, ProvenanceError, validate_provenance


@dataclass
class SectionStack:
    """Serial sections with physical positions.

    ``images`` are 2-D arrays (one per section) in a common pixel size.
    ``meta`` carries optional simulation ground truth (never read by the pipeline).
    """

    images: list[np.ndarray]
    z_um: np.ndarray
    thickness_um: float
    pixel_um: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z_um = np.asarray(self.z_um, dtype=float)
        if len(self.images) != len(self.z_um):
            raise ValueError("images and z_um must have the same length")
        if len(self.images) < 2:
            raise ValueError("need at least 2 sections")
        if len({im.shape for im in self.images}) != 1:
            raise ValueError("all sections must share one shape")
        spacing = np.diff(self.z_um)
        if np.any(spacing <= 0):
            raise ValueError("z positions must be strictly increasing")
        if np.any(spacing < self.thickness_um - 1e-9):
            raise ValueError("section spacing is smaller than thickness (overlapping slabs)")

    @property
    def n_sections(self) -> int:
        return len(self.images)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.images[0].shape)  # type: ignore[return-value]

    @property
    def spacing_um(self) -> np.ndarray:
        return np.diff(self.z_um)

    def subset(self, idx: list[int] | np.ndarray) -> "SectionStack":
        idx = list(idx)
        return SectionStack(
            images=[self.images[i] for i in idx],
            z_um=self.z_um[idx],
            thickness_um=self.thickness_um,
            pixel_um=self.pixel_um,
            meta=self.meta,
        )


def make_grid(
    z_um: np.ndarray, thickness_um: float, dz_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-grid of plane centres covering all slabs.

    Returns ``(centers, plane_section, gap_distance_um)`` where ``plane_section[j]``
    is the index of the section whose slab contains plane ``j`` (or -1 for a gap
    plane) and ``gap_distance_um[j]`` is the distance from plane ``j`` to the
    nearest slab surface (0 inside slabs).
    """
    z_um = np.asarray(z_um, dtype=float)
    z_lo = z_um[0] - thickness_um / 2
    z_hi = z_um[-1] + thickness_um / 2
    n = int(round((z_hi - z_lo) / dz_um))
    centers = z_lo + dz_um * (np.arange(n) + 0.5)
    d = np.abs(centers[:, None] - z_um[None, :])
    k = d.argmin(axis=1)
    dmin = d[np.arange(n), k]
    in_slab = dmin <= thickness_um / 2 + 1e-9
    plane_section = np.where(in_slab, k, -1)
    gap = np.maximum(dmin - thickness_um / 2, 0.0)
    return centers, plane_section, gap


@dataclass
class ProbVolume:
    """Per-voxel label posterior with mandatory provenance."""

    label_prob: np.ndarray  # (C, Z, Y, X) float32
    provenance: np.ndarray  # (Z, Y, X) uint8
    z_um: np.ndarray  # (Z,) plane centres
    pixel_um: float
    dz_um: float
    gap_distance_um: np.ndarray  # (Z,)
    class_names: list[str]

    def __post_init__(self) -> None:
        c, z, y, x = self.label_prob.shape
        validate_provenance(self.provenance, (z, y, x))
        if len(self.z_um) != z or len(self.gap_distance_um) != z:
            raise ValueError("z_um / gap_distance_um length must equal Z")
        if len(self.class_names) != c:
            raise ValueError("class_names length must equal C")
        valid = self.provenance > 0
        if valid.any():
            sums = self.label_prob.sum(axis=0)[valid]
            if not np.allclose(sums, 1.0, atol=1e-3):
                raise ValueError("label probabilities must sum to 1 on sampled voxels")

    @property
    def map_labels(self) -> np.ndarray:
        return self.label_prob.argmax(axis=0).astype(np.int8)

    def observed_fraction(self, region: np.ndarray | None = None) -> float:
        """Fraction of sampled voxels (optionally within ``region``) that are Observed."""
        prov = self.provenance if region is None else self.provenance[region]
        sampled = prov > 0
        if not sampled.any():
            return 0.0
        return float((prov == Provenance.OBSERVED).sum() / sampled.sum())


def export_volume(vol: ProbVolume, path: str | Path, manifest: Manifest) -> Path:
    """Write a volume + manifest. Refuses to export without provenance or a manifest."""
    if vol.provenance is None:
        raise ProvenanceError("refusing to export a volume without provenance")
    if not isinstance(manifest, Manifest):
        raise ProvenanceError("refusing to export a volume without a manifest")
    validate_provenance(vol.provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        label_prob=vol.label_prob.astype(np.float16),
        provenance=vol.provenance,
        z_um=vol.z_um,
        gap_distance_um=vol.gap_distance_um,
        pixel_um=vol.pixel_um,
        dz_um=vol.dz_um,
        class_names=np.array(vol.class_names),
    )
    manifest.write(path.with_suffix(".manifest.json"))
    return path
