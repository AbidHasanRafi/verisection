"""One-call simulation for the UI and the CLI: phantom -> virtual sections -> pipeline -> metrics + 3D view.

Everything is synthetic with known truth; results check the machinery and are not evidence about tissue.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from . import evaluate, planner
from .decision import EvidenceVector, VerdictPolicy
from .measure import class_volume_fraction, interval_from_samples, observed_support_fraction
from .pipeline import PipelineResult, run_pipeline
from .register import register_series, register_series_chain, registration_tre_px
from .segment import SegmentationEnsemble, SupervisedSegmentationEnsemble
from .simulate import SectioningConfig, make_phantom, make_pilot_block, virtual_sections
from .tissue import CLASS_NAMES as TISSUE_CLASSES
from .tissue import make_tissue_phantom
from .variogram import conservative_xi, estimate_class_params_from_pilot
from .viewer3d import ViewData, view_data_from_maps

PX = 2.0


@dataclass
class SimParams:
    preset: str = "tissue"  # 'tissue' (glands, stroma, immune cells) | 'hard' | 'easy' (random-field blobs)
    seed: int = 0
    n_sections: int = 20
    spacing_planes: int = 3  # fine planes (2 µm each) between sections: 3 -> 6 µm
    thickness_um: float = 4.0
    grid_dz_um: float = 2.0  # z-step of the reconstruction grid
    xi_z_um: float = 36.0  # tissue z-correlation length of the phantom
    method: str = "translation"  # 'translation' | 'rigid' | 'nonrigid'
    n_reg: int = 3
    n_seg: int = 3
    interp_mode: str = "kriging"
    sigma_scale: float = 2.2  # registration-spread multiplier
    noise_sigma: float | None = None  # None = preset default
    nonrigid_amp_px: float | None = None
    texture_sd: float | None = None

    def sectioning(self) -> SectioningConfig:
        cfg = {"hard": SectioningConfig.hard, "tissue": SectioningConfig.tissue}.get(self.preset, SectioningConfig)()
        over = {"n_sections": self.n_sections, "spacing_planes": self.spacing_planes, "thickness_um": self.thickness_um}
        for name, val in (("noise_sigma", self.noise_sigma), ("nonrigid_amp_px", self.nonrigid_amp_px),
                          ("texture_sd", self.texture_sd)):
            if val is not None:
                over[name] = val
        return replace(cfg, **over)


@dataclass
class SimResult:
    view: ViewData
    metrics: dict
    result: PipelineResult
    truth: np.ndarray
    log: list[str] = field(default_factory=list)
    params: SimParams | None = None


def _phantom(params: SimParams, seed: int):
    """Truth labels, optional precomputed intensity, class names."""
    if params.preset == "tissue":
        ph = make_tissue_phantom(seed=seed)
        return ph.labels, ph.intensity, list(TISSUE_CLASSES)
    truth = make_phantom(seed=seed, xi_z_um=params.xi_z_um)
    n = len(SectioningConfig().class_means)
    return truth, None, [f"class_{c}" for c in range(n)]


def _max_sections(params: SimParams, planes: int = 120) -> int:
    return max((planes - 1 - 2) // params.spacing_planes + 1, 3)


def run_simulation(params: SimParams, progress: Callable[[str, float], None] | None = None) -> SimResult:
    """Run the full pipeline on a simulated stack and package metrics and the 3D view data."""
    t0 = time.time()
    log: list[str] = []

    def step(msg: str, frac: float) -> None:
        log.append(f"{time.time() - t0:5.1f}s  {msg}")
        if progress:
            progress(msg, frac)

    if params.interp_mode not in ("kriging", "linear", "nearest"):
        raise ValueError("interp_mode must be kriging, linear or nearest")
    n_max = _max_sections(params)
    if params.n_sections > n_max:
        params = replace(params, n_sections=n_max)
        log.append(f"n_sections reduced to {n_max} to fit the 240 µm phantom")
    cfg = params.sectioning()
    step("building phantom and virtual sections", 0.05)
    truth, intensity, class_names = _phantom(params, params.seed)
    n_cls = len(class_names)
    stack = virtual_sections(truth, cfg, seed=params.seed, intensity=intensity)
    rng = np.random.default_rng(params.seed)

    step(f"registering ({params.method}, {params.n_reg} samples)", 0.2)
    if params.method == "translation":
        reg = register_series(stack.images, PX, params.n_reg, rng)
    else:
        reg = register_series_chain(stack.images, PX, params.n_reg, rng, method=params.method)
    step("fitting segmentation ensemble", 0.5)
    if params.preset == "tissue":
        # a supervised pixel classifier trained on a *different* phantom (a stand-in for a pretrained model)
        tt, ti, _ = _phantom(params, params.seed + 9000)
        train_cfg = replace(cfg, drift_step_px=0.0, jitter_px=0.0, rotation_step_deg=0.0, nonrigid_amp_px=0.0,
                            n_sections=min(cfg.n_sections, 10), spacing_planes=max(cfg.spacing_planes, 3))
        tstack = virtual_sections(tt, train_cfg, seed=params.seed + 9000, intensity=ti)
        tlabels = [tt[i] for i in tstack.meta["section_planes"]]
        seg = SupervisedSegmentationEnsemble.fit(tstack.images, tlabels, n_cls, params.n_seg, rng)
    else:
        seg = SegmentationEnsemble.fit(stack.images, n_cls, params.n_seg, rng)

    xi = None
    if params.interp_mode == "kriging":
        step("estimating correlation lengths from a dense pilot block", 0.55)
        ptruth, pint, _ = _phantom(params, params.seed + 500)
        pilot = make_pilot_block(ptruth, cfg, start_plane=10, n_sections=14, seed=params.seed + 500, intensity=pint)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xi = conservative_xi(estimate_class_params_from_pilot(pilot, seg, n_cls, rng=rng))
    step("propagating uncertainty and reconstructing", 0.65)
    tgt = 2 if params.preset == "tissue" else 1  # epithelium for the tissue phantom
    fn = {f"frac{c}": (lambda p, prov, c=c: class_volume_fraction(p, prov, c)) for c in range(n_cls)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_pipeline(stack, n_classes=n_cls, class_names=class_names, xi_um=xi, seed=params.seed, reg=reg, seg=seg,
                           interp_mode=params.interp_mode, grid_dz_um=params.grid_dz_um,
                           sigma_scale=params.sigma_scale if params.sigma_scale != 1.0 else None,
                           measure_fns=fn, validate=False)
    step("scoring against the known truth", 0.9)
    maps = res.maps
    tg = evaluate.truth_on_grid(stack, maps.z_um)
    score = evaluate.score_maps(maps, tg, np.random.default_rng(params.seed), max_vox=80_000)
    tre = registration_tre_px(res.reg.point().transforms[0], stack.meta["true_h"], stack.images[0].shape) * PX

    # measurement of the class-1 volume fraction with a verdict
    domain = (stack.z_um[0] - stack.thickness_um / 2, stack.z_um[-1] + stack.thickness_um / 2)
    prior = float((truth == tgt).mean())
    xi1 = float(xi[tgt]) if xi is not None else 36.0
    fm = planner.FieldModel(prior, xi1, n_eff_xy=max((truth.shape[1] * PX / 14.0) ** 2, 1.0))
    kvar = planner.mean_estimation_variance(stack.z_um, domain, fm)
    o = observed_support_fraction(maps.provenance)
    iv = interval_from_samples(maps.measurements[f"frac{tgt}"], kvar, o, level=0.9)
    true_frac = float((tg[maps.provenance > 0] == tgt).mean())
    verdict, reasons = VerdictPolicy().decide(EvidenceVector(o=o, w=iv.relative_halfwidth, c="untested"), 0.3)
    sp = float(np.median(stack.spacing_um))
    recs = planner.recommend(stack.z_um, np.arange(domain[0] + 1, domain[1], 1.0), domain, fm, top_k=5,
                             min_separation_um=max(1.0, 0.4 * sp))

    metrics = {
        "all": score["all"], "observed": score.get("observed"), "interpolated": score.get("interpolated"),
        "registration": {"mean_tre_um": float(tre[1:].mean()), "max_tre_um": float(tre[1:].max()),
                         "sigma_um_mean": float(res.reg.sigma_um()[1:].mean()),
                         "failed_pairs": int(res.reg.failed.sum()) if res.reg.failed is not None else 0},
        "measurement": {"target": f"volume fraction of {class_names[tgt]}", "estimate": iv.estimate, "lo90": iv.lo, "hi90": iv.hi,
                        "truth": true_frac, "covered": bool(iv.contains(true_frac)), "observed_fraction": o,
                        "verdict": verdict.value, "reasons": reasons + ["calibration untested in this run"]},
        "planner_next_sections_um": [{"z_um": r["z_um"], "relative_sd_reduction": r["relative_sd_reduction"]} for r in recs],
        "xi_um_used": None if xi is None else [float(x) for x in xi],
        "n_sections": stack.n_sections, "spacing_um": float(np.median(stack.spacing_um)),
        "runtime_s": time.time() - t0,
    }
    notes = [f"preset={params.preset}, seed={params.seed}, registration={params.method}, interpolation={params.interp_mode}",
             f"accuracy vs truth {score['all']['accuracy']:.2f}; ECE {score['all']['ece']:.3f}; mean TRE {metrics['registration']['mean_tre_um']:.1f} µm",
             "Synthetic phantom with known truth: machinery check, not evidence about real tissue."]
    view = view_data_from_maps(maps, truth=tg, title=f"VeriSection simulation (seed {params.seed})", notes=notes)
    step("done", 1.0)
    return SimResult(view, metrics, res, truth, log, params)
