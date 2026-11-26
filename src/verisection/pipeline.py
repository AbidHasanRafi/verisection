"""End-to-end Mode A pipeline on a ``SectionStack``."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .interpolate import InterpConfig
from .provenance import Manifest
from .register import RegistrationEnsemble, register_series, register_series_chain
from .segment import SegmentationEnsemble
from .uncertainty import UncertaintyMaps, propagate
from .validate import DataWarning, validate_stack
from .volume import ProbVolume, SectionStack, export_volume


@dataclass
class PipelineResult:
    stack: SectionStack
    reg: RegistrationEnsemble
    seg: SegmentationEnsemble
    interp: InterpConfig
    maps: UncertaintyMaps
    manifest: Manifest
    class_names: list[str]

    def to_probvolume(self) -> ProbVolume:
        """Aggregated posterior with provenance, ready for export."""
        m = self.maps
        return ProbVolume(
            label_prob=m.p_hat, provenance=m.provenance, z_um=m.z_um, pixel_um=m.pixel_um,
            dz_um=m.dz_um, gap_distance_um=m.gap_distance_um, class_names=self.class_names,
        )

    def export(self, path) -> "Path":
        """Write volume + manifest (refuses without provenance)."""
        return export_volume(self.to_probvolume(), path, self.manifest)


def run_pipeline(
    stack: SectionStack,
    n_classes: int = 3,
    n_reg: int = 6,
    n_seg: int = 6,
    interp_mode: str = "kriging",
    xi_um: np.ndarray | None = None,
    nugget: float = 0.05,
    seed: int = 0,
    measure_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    class_names: list[str] | None = None,
    reg: RegistrationEnsemble | None = None,
    seg: SegmentationEnsemble | None = None,
    registration_method: str = "translation",
    sigma_scale: float | None = None,
    validate: bool = True,
    grid_dz_um: float | None = None,
) -> PipelineResult:
    """Validate -> register -> segment -> reconstruct -> propagate uncertainty.

    ``registration_method``: 'translation' | 'rigid' | 'nonrigid'. ``sigma_scale`` recalibrates the
    registration-ensemble spread (see ``register.fit_sigma_scale``); ``None`` leaves it raw.
    Registration failures (sections that fell back to the previous transform) are recorded in the
    manifest so downstream decisions can see them. Returns everything needed to inspect,
    evaluate and export (with provenance and a manifest).
    """
    if validate:
        validate_stack(stack)
    if interp_mode == "kriging" and xi_um is None:
        raise ValueError(
            "kriging interpolation needs class correlation lengths (xi_um): estimate them from a dense "
            "pilot block (variogram.estimate_class_params_from_pilot) or choose interp_mode='linear'"
        )
    rng = np.random.default_rng(seed)
    names = class_names or [f"class_{c}" for c in range(n_classes)]
    cfg = {
        "n_classes": n_classes, "n_reg": n_reg, "n_seg": n_seg, "interp_mode": interp_mode,
        "xi_um": None if xi_um is None else [float(x) for x in xi_um], "nugget": nugget,
        "n_sections": stack.n_sections, "thickness_um": stack.thickness_um,
        "pixel_um": stack.pixel_um, "registration_method": registration_method, "grid_dz_um": grid_dz_um,
        "sigma_scale": sigma_scale,
    }
    manifest = Manifest(seed=seed, config=cfg)
    manifest.inputs = [{"id": f"sec_{k:03d}", "z_um": float(z)} for k, z in enumerate(stack.z_um)]

    if reg is None:
        if registration_method == "translation":
            reg = register_series(stack.images, stack.pixel_um, n_reg, rng)
        else:
            reg = register_series_chain(stack.images, stack.pixel_um, n_reg, rng, method=registration_method)
    stage = {"backend": reg.method, "n_samples": reg.n_samples}
    if reg.failed is not None and reg.failed.any():
        stage["failed_pairs"] = {
            int(k): int(reg.failed[:, k].sum()) for k in range(reg.n_sections) if reg.failed[:, k].any()
        }
        warnings.warn(
            f"registration fell back to the previous transform for sections {sorted(stage['failed_pairs'])}",
            DataWarning, stacklevel=2,
        )
    if reg.ncc is not None:
        stage["median_ncc"] = float(np.median(reg.ncc[:, 1:]))
    # record what actually ran (ensembles may have been supplied by the caller)
    manifest.config.update({"n_reg": reg.n_samples, "registration_method": reg.method})
    manifest.add_stage("register", **stage)
    if sigma_scale is not None and sigma_scale != 1.0:
        reg = reg.calibrated(sigma_scale)
        manifest.add_stage("recalibrate_registration_spread", scale=float(sigma_scale))
    if seg is None:
        seg = SegmentationEnsemble.fit(stack.images, n_classes, n_seg, rng)
    manifest.config.update({"n_seg": seg.n_members})
    manifest.add_stage("segment", backend="intensity_mixture_bootstrap", n_members=seg.n_members)

    interp = InterpConfig(mode=interp_mode, xi_um=xi_um, nugget=nugget, dz_um=grid_dz_um)  # type: ignore[arg-type]
    maps = propagate(stack, reg, seg, interp, names, measure_fns)
    manifest.add_stage("reconstruct", interpolation=interp_mode, dz_um=maps.dz_um)
    manifest.uncertainty = {"M": reg.n_samples, "N": seg.n_members, "conformal": None}
    return PipelineResult(stack, reg, seg, interp, maps, manifest, names)
