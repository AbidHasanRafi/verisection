"""Multi-replicate simulation experiments.

Differences from ``demo`` (a single-seed smoke run):

* many independent phantoms (replicates are the unit of analysis; bootstrap CIs, paired Wilcoxon tests,
  Holm correction);
* a *hard* preset (rotation, non-rigid deformation, per-class stain shift, class-dependent noise);
* correlation lengths estimated from an independent dense **pilot block** with the pipeline's own
  segmentation (never from the truth labels of the stack under test);
* registration-ensemble spread recalibrated on separate calibration replicates, then applied to the
  test replicates;
* a check that texture computed on observed planes is less biased than texture computed
  after interpolation.

Everything is synthetic: results check the machinery, they are not evidence about real tissue.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import evaluate, texture
from .pipeline import run_pipeline
from .register import fit_sigma_scale, register_series, register_series_chain, registration_tre_px
from .segment import SegmentationEnsemble
from .simulate import SectioningConfig, make_phantom, make_pilot_block, virtual_sections
from .stats import bootstrap_ci, holm, paired_comparison
from .variogram import estimate_class_params, estimate_class_params_from_pilot

METRICS = ("accuracy", "ece", "brier", "nll", "ause", "spearman_u_err")
PX = 2.0  # µm per pixel in the simulator


def _cfg(preset: str) -> SectioningConfig:
    return SectioningConfig.hard() if preset == "hard" else SectioningConfig()


def _register(images, method: str, M: int, rng):
    if method == "translation":
        return register_series(images, PX, M, rng)
    return register_series_chain(images, PX, M, rng, method=method)


def _tre_sigma(reg, stack):
    """Per-section TRE (µm) of the mean transform and the raw ensemble spread (µm)."""
    mean_t = reg.point().transforms[0]
    tre = registration_tre_px(mean_t, stack.meta["true_h"], stack.images[0].shape) * PX
    return tre, reg.sigma_um()


def run_replicate(
    seed: int,
    preset: str = "hard",
    method: str = "nonrigid",
    M: int = 4,
    N: int = 4,
    sigma_scale: float | None = None,
    full: bool = True,
    grid_scales: tuple[float, ...] | None = None,
    extra_scales: dict[str, float] | None = None,
) -> dict:
    """One independent phantom.

    ``full=False`` measures registration and xi (calibration phase); with ``grid_scales`` it also
    records the downstream ECE/AUSE of the joint pipeline at each candidate spread multiplier, so the
    multiplier can be chosen on what matters (calibrated posteriors), not only on spread magnitude.
    ``extra_scales`` adds named joint variants (e.g. ``{"ece": 4.0}`` -> ``joint_ece``).
    """
    t0 = time.time()
    cfg = _cfg(preset)
    truth = make_phantom(seed=seed)
    stack = virtual_sections(truth, cfg, seed=seed)
    rng = np.random.default_rng(seed)
    reg = _register(stack.images, method, M, rng)
    tre, sig = _tre_sigma(reg, stack)
    out: dict = {
        "seed": seed,
        "tre_um": tre.tolist(), "sigma_um": sig.tolist(),
        "reg_failed_pairs": int(reg.failed.sum()) if reg.failed is not None else 0,
    }
    seg = SegmentationEnsemble.fit(stack.images, 3, N, rng)
    ptruth = make_phantom(seed=seed + 500)  # independent block of the same tissue type
    pilot = make_pilot_block(ptruth, cfg, start_plane=10, n_sections=14, seed=seed + 500)
    par_true = estimate_class_params(truth, 2.0, 3)
    par = estimate_class_params_from_pilot(pilot, seg, 3, rng=rng)
    xi = np.where(np.isfinite(par["xi_um"]), par["xi_um"], np.nanmedian(par["xi_um"]))
    if not np.isfinite(xi).all():
        xi = np.full(3, 30.0)
    out["xi_est"], out["xi_true"] = xi.tolist(), par_true["xi_um"].tolist()
    out["xi_rel_error"] = float(np.mean(np.abs(xi - par_true["xi_um"]) / par_true["xi_um"]))
    if not full:
        if grid_scales:
            out["grid"] = {}
            sr = np.random.default_rng(seed)
            for g in grid_scales:
                res = run_pipeline(stack, xi_um=xi, seed=seed, validate=False, reg=reg, seg=seg,
                                   interp_mode="kriging", sigma_scale=g)
                tg = evaluate.truth_on_grid(stack, res.maps.z_um)
                sc_ = evaluate.score_maps(res.maps, tg, sr, max_vox=40_000)["all"]
                out["grid"][str(g)] = {"ece": sc_["ece"], "ause": sc_["ause"]}
        out["runtime_s"] = time.time() - t0
        return out

    sc = sigma_scale
    variants = {
        "joint_raw": dict(reg=reg, seg=seg, interp_mode="kriging"),
        "joint_cal": dict(reg=reg, seg=seg, interp_mode="kriging", sigma_scale=sc),
        "seg_only": dict(reg=reg.point(), seg=seg, interp_mode="kriging"),
        "reg_only_cal": dict(reg=reg, seg=seg.point(), interp_mode="kriging", sigma_scale=sc),
        "point_linear": dict(reg=reg.point(), seg=seg.point(), interp_mode="linear"),
        "point_kriging": dict(reg=reg.point(), seg=seg.point(), interp_mode="kriging"),
    }
    if sc is None:
        variants.pop("joint_cal"), variants.pop("reg_only_cal")
    for nm, g in (extra_scales or {}).items():
        variants[f"joint_{nm}"] = dict(reg=reg, seg=seg, interp_mode="kriging", sigma_scale=g)
    score_rng = np.random.default_rng(seed)
    out["variants"] = {}
    for name, kw in variants.items():
        res = run_pipeline(stack, xi_um=xi, seed=seed, validate=False, **kw)
        tg = evaluate.truth_on_grid(stack, res.maps.z_um)
        s = evaluate.score_maps(res.maps, tg, score_rng, max_vox=60_000)
        out["variants"][name] = {
            "all": {m: s["all"][m] for m in METRICS},
            "interpolated": {m: s["interpolated"][m] for m in ("accuracy", "ece", "ause")},
            "observed": {m: s["observed"][m] for m in ("accuracy", "ece", "ause")},
        }
        if name == "joint_cal" or (sc is None and name == "joint_raw"):
            m_ = res.maps
            v = m_.provenance > 0
            out["share_var_reg"] = float(m_.var_reg[v].mean() / max(m_.var_reg[v].mean() + m_.var_seg[v].mean(), 1e-12))
    out["runtime_s"] = time.time() - t0
    return out


def _run(args):
    return run_replicate(*args)


def _pool(fn, items, workers):
    if workers <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


GRID = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def run_multiseed(
    n_cal: int = 8, n_test: int = 16, preset: str = "hard", method: str = "nonrigid",
    M: int = 4, N: int = 4, workers: int = 4, seed0: int = 100,
) -> dict:
    """Calibration phase (registration spread) then test phase (all variants).

    Two ways to choose the spread multiplier on the calibration replicates, compared on the test set:
    ``mag`` matches the average registration error (ratio of means); ``ece`` picks the grid value with
    the lowest downstream ECE of the full pipeline.
    """
    t0 = time.time()
    cal = _pool(_run, [(seed0 + i, preset, method, M, N, None, False, GRID) for i in range(n_cal)], workers)
    tre = np.concatenate([np.array(c["tre_um"])[1:] for c in cal])
    sig = np.concatenate([np.array(c["sigma_um"])[1:] for c in cal])
    scale_mag = fit_sigma_scale(sig, tre)
    grid_ece = {g: float(np.mean([c["grid"][str(g)]["ece"] for c in cal])) for g in GRID}
    grid_ause = {g: float(np.mean([c["grid"][str(g)]["ause"] for c in cal])) for g in GRID}
    scale_ece = min(grid_ece, key=lambda g: (grid_ece[g], g))
    test = _pool(
        _run,
        [(seed0 + 1000 + i, preset, method, M, N, scale_mag, True, None, {"ece": scale_ece}) for i in range(n_test)],
        workers,
    )

    out: dict = {
        "preset": preset, "method": method, "M": M, "N": N, "n_cal": n_cal, "n_test": n_test,
        "sigma_scale": scale_mag, "sigma_scale_ece_selected": scale_ece,
        "calibration_grid": {"ece": grid_ece, "ause": grid_ause},
        "calibration_registration": {
            "mean_tre_um": float(tre.mean()), "mean_sigma_um": float(sig.mean()),
            "spearman_sigma_tre": float(np.corrcoef(np.argsort(np.argsort(sig)), np.argsort(np.argsort(tre)))[0, 1]),
        },
        "xi_rel_error_mean": float(np.mean([t["xi_rel_error"] for t in test])),
        "reg_failed_pairs_mean": float(np.mean([t["reg_failed_pairs"] for t in test])),
        "variants": {}, "comparisons": {},
    }
    names = list(test[0]["variants"].keys())
    for v in names:
        out["variants"][v] = {
            m: str(bootstrap_ci([t["variants"][v]["all"][m] for t in test])) for m in METRICS
        } | {
            "interpolated_ause": str(bootstrap_ci([t["variants"][v]["interpolated"]["ause"] for t in test])),
            "observed_ece": str(bootstrap_ci([t["variants"][v]["observed"]["ece"] for t in test])),
        }
    pairs = [("joint_cal", "point_linear"), ("joint_cal", "joint_raw"), ("joint_cal", "seg_only"),
             ("joint_cal", "point_kriging"), ("joint_ece", "point_linear"), ("joint_ece", "joint_cal")]
    pvals = {}
    for a, b in pairs:
        for m in ("ause", "ece"):
            key = f"{a} vs {b} [{m}]"
            c = paired_comparison(
                [t["variants"][a]["all"][m] for t in test], [t["variants"][b]["all"][m] for t in test]
            )
            out["comparisons"][key] = c
            pvals[key] = c.get("wilcoxon_p", float("nan"))
    for k, p in holm(pvals).items():
        out["comparisons"][k]["holm_p"] = p
    # post-calibration check of the registration spread on the test replicates
    tre_t = np.concatenate([np.array(t["tre_um"])[1:] for t in test])
    sig_t = np.concatenate([np.array(t["sigma_um"])[1:] for t in test])
    out["test_registration"] = {
        "mean_tre_um": float(tre_t.mean()),
        "raw_sigma_to_error": float(sig_t.mean() / tre_t.mean()),
        "calibrated_sigma_to_error": float(scale_mag * sig_t.mean() / tre_t.mean()),
        "ece_selected_sigma_to_error": float(scale_ece * sig_t.mean() / tre_t.mean()),
    }
    out["share_var_reg_mean"] = float(np.mean([t["share_var_reg"] for t in test]))
    out["runtime_s"] = round(time.time() - t0, 1)
    out["replicates"] = test
    return out


# ------------------------------------------------------------------ texture on observed vs interpolated planes
def texture_h8_replicate(seed: int, spacing_planes: int, cls: int = 1) -> dict:
    """Bias of texture features estimated from observed planes vs from interpolated planes.

    Truth: mean feature over *all* fine planes of the true intensity volume inside the class mask.
    A (observed-support): mean over the section planes only (never interpolated).
    B (naive): mean over all planes after linearly interpolating intensity between sections
    (class mask from the truth labels, i.e. an oracle mask, to isolate the interpolation effect).
    """
    truth = make_phantom(seed=seed)
    first = 2
    n_sec = min(18, (truth.shape[0] - 1 - first) // spacing_planes + 1)
    cfg = SectioningConfig(texture_sd=0.08, class_means=(0.25, 0.5, 0.75), spacing_planes=spacing_planes,
                           n_sections=n_sec, first_plane=first)
    stack = virtual_sections(truth, cfg, seed=seed)
    inten = stack.meta["truth_intensity"]
    sec = np.asarray(stack.meta["section_planes"])
    lo, hi = sec[0], sec[-1]
    planes = np.arange(lo, hi + 1)

    def feats(img, mask):
        f = texture.window_features_2d(img, mask, PX, offset_um=PX, bin_width=0.04, n_levels=64)
        return np.array([f["contrast"], f["entropy"], f["homogeneity"]])

    def mean_over(planes_, getter):
        vals = []
        for z in planes_:
            m = truth[z] == cls
            if m.sum() > 200:
                vals.append(feats(getter(z), m))
        return np.mean(vals, axis=0) if vals else np.full(3, np.nan)

    truth_f = mean_over(planes, lambda z: inten[z])
    obs_f = mean_over(sec, lambda z: inten[z])

    def interp(z):
        j = int(np.searchsorted(sec, z, side="right") - 1)
        j = min(j, len(sec) - 2)
        w = (z - sec[j]) / (sec[j + 1] - sec[j])
        return (1 - w) * inten[sec[j]] + w * inten[sec[j + 1]]

    naive_f = mean_over(planes, interp)
    return {"seed": seed, "spacing_planes": spacing_planes, "truth": truth_f.tolist(),
            "observed": obs_f.tolist(), "naive_interp": naive_f.tolist()}


def run_texture_experiment(n_seeds: int = 24, seed0: int = 300, workers: int = 4) -> dict:
    names = ["contrast", "entropy", "homogeneity"]
    out: dict = {"n_seeds": n_seeds, "by_spacing_planes": {}}
    for sp in (2, 4, 6, 10):
        reps = _pool(_tex_run, [(seed0 + i, sp) for i in range(n_seeds)], workers)
        T = np.array([r["truth"] for r in reps])
        A = np.array([r["observed"] for r in reps])
        B = np.array([r["naive_interp"] for r in reps])
        row = {}
        for j, nm in enumerate(names):
            ea, eb = (A[:, j] - T[:, j]) / T[:, j], (B[:, j] - T[:, j]) / T[:, j]
            row[nm] = {
                "observed_rel_bias": str(bootstrap_ci(ea)), "naive_interp_rel_bias": str(bootstrap_ci(eb)),
                "observed_rel_rmse": float(np.sqrt(np.mean(ea**2))), "naive_rel_rmse": float(np.sqrt(np.mean(eb**2))),
                "abs_error_paired": paired_comparison(np.abs(ea), np.abs(eb)),
            }
        out["by_spacing_planes"][str(sp)] = row
    return out


def _tex_run(args):
    return texture_h8_replicate(*args)


# --------------------------------------------------------------- nuclei linking and counts
def nuclei_replicate(seed: int, spacing_um: float, reg_sigma_um: float, thickness_um: float = 4.0) -> dict:
    """One simulated nuclear field: counts and linking quality under a given misregistration."""
    from . import cellular as C

    rng = np.random.default_rng(seed)
    box = (200.0, 200.0, 200.0)
    margin = 15.0
    nuc = C.simulate_nuclei(box, seed=seed)
    n_sec = int((box[0] - 40) // spacing_um)
    z0 = 20.0 + np.arange(n_sec) * spacing_um
    profs = []
    for k, z in enumerate(z0):
        p = C.cut_section(nuc, z, thickness_um, k, rng, centroid_jitter_um=0.3, miss_rate=0.03)
        off = rng.normal(0.0, reg_sigma_um, 2)  # section-level misregistration
        p.xy = p.xy + off
        sel = ((p.xy[:, 0] > margin) & (p.xy[:, 0] < box[1] - margin)
               & (p.xy[:, 1] > margin) & (p.xy[:, 1] < box[2] - margin))
        profs.append(C.Profiles(p.section, p.xy[sel], p.radius[sel], p.ids[sel]))
    c = nuc.centers
    lo, hi = z0[0] - spacing_um / 2, z0[-1] + spacing_um / 2
    truth = int(((c[:, 0] >= lo) & (c[:, 0] < hi) & (c[:, 1] > margin) & (c[:, 1] < box[1] - margin)
                 & (c[:, 2] > margin) & (c[:, 2] < box[2] - margin)).sum())
    tol = float(np.sqrt(0.3**2 + reg_sigma_um**2))
    area = (box[1] - 2 * margin) * (box[2] - 2 * margin)
    d_true = 2 * nuc.semi_axes[:, 0].mean()
    known = C.estimate_counts(profs, thickness_um, spacing_um, area, tol, caliper_um=d_true)
    est_d = C.estimate_counts(profs, thickness_um, spacing_um, area, tol, caliper_um=None)
    sc = C.linking_scores(C.link_series(profs, tol), profs)
    return {
        "truth": truth, "d_true": float(d_true), "d_est": est_d["caliper_um"],
        "naive": known["naive"] / truth - 1, "aber_known_D": known["abercrombie"] / truth - 1,
        "aber_est_D": est_d["abercrombie"] / truth - 1, "tracks_ht": known["tracks_ht"] / truth - 1,
        "link_f1": sc["pair_f1"],
    }


def run_nuclei_experiment(n_rep: int = 20, seed0: int = 5000) -> dict:
    out: dict = {"n_rep": n_rep, "cells": {}}
    for spacing in (4.0, 6.0, 8.0, 12.0):
        for reg_sigma in (0.3, 1.5, 3.0):
            reps = [nuclei_replicate(seed0 + i, spacing, reg_sigma) for i in range(n_rep)]
            out["cells"][f"spacing={spacing:g}um,reg_sigma={reg_sigma:g}um"] = {
                k: str(bootstrap_ci([r[k] for r in reps]))
                for k in ("naive", "aber_known_D", "aber_est_D", "tracks_ht", "link_f1")
            } | {"d_true": reps[0]["d_true"], "d_est_mean": float(np.mean([r["d_est"] for r in reps]))}
    return out


# ------------------------------------------- conformal calibration and verdicts (hard simulator)
def _prepare(seed: int, preset: str, method: str, M: int, N: int, spacing_planes: int | None = None,
             xi_z_um: float = 36.0):
    """Phantom, stack, registration and segmentation ensembles, and pilot-estimated correlation lengths."""
    from dataclasses import replace

    from .variogram import conservative_xi

    cfg = _cfg(preset)
    truth = make_phantom(seed=seed, xi_z_um=xi_z_um)
    if spacing_planes is not None:
        cfg = replace(cfg, spacing_planes=spacing_planes,
                      n_sections=min(18, (truth.shape[0] - 1 - cfg.first_plane) // spacing_planes + 1))
    stack = virtual_sections(truth, cfg, seed=seed)
    rng = np.random.default_rng(seed)
    reg = _register(stack.images, method, M, rng)
    seg = SegmentationEnsemble.fit(stack.images, 3, N, rng)
    pilot = make_pilot_block(make_phantom(seed=seed + 500, xi_z_um=xi_z_um), cfg, start_plane=10,
                             n_sections=14, seed=seed + 500)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi = conservative_xi(estimate_class_params_from_pilot(pilot, seg, 3, rng=rng))
    return truth, stack, reg, seg, xi


def conformal_replicate(seed: int, preset: str, method: str, M: int, N: int, sigma_scale: float,
                        stride: int = 2) -> list[dict]:
    """LOSO records (p, y per held-out section) of one hard-simulator replicate."""
    from .interpolate import InterpConfig

    _, stack, reg, seg, xi = _prepare(seed, preset, method, M, N)
    ks = list(range(2, stack.n_sections - 2, stride))
    recs = evaluate.loso_scores(stack, reg.calibrated(sigma_scale), seg, InterpConfig("kriging", xi_um=xi),
                                ["c0", "c1", "c2"], held_out=ks)
    return [{"p": r["p"].astype(np.float32), "y": r["y"]} for r in recs]


def _conf_run(args):
    return conformal_replicate(*args)


def run_conformal_hard(n_cal: int = 6, n_test: int = 8, preset: str = "hard", method: str = "nonrigid",
                       sigma_scale: float = 2.16, alpha: float = 0.1, workers: int = 4, seed0: int = 7000,
                       M: int = 3, N: int = 3) -> dict:
    """Split-conformal sets calibrated on held-out sections of calibration phantoms, evaluated on others."""
    from .conformal import SplitConformal

    cal_sets = _pool(_conf_run, [(seed0 + i, preset, method, M, N, sigma_scale) for i in range(n_cal)], workers)
    test_sets = _pool(_conf_run, [(seed0 + 500 + i, preset, method, M, N, sigma_scale) for i in range(n_test)], workers)
    cal = [r for rs in cal_sets for r in rs]
    cp = SplitConformal(alpha=alpha).calibrate(cal, rng=np.random.default_rng(0))
    per_rep = [cp.evaluate(rs) for rs in test_sets]
    cov = [e["coverage_mean"] for e in per_rep]
    return {
        "alpha": alpha, "nominal": 1 - alpha, "qhat": cp.qhat, "n_cal_phantoms": n_cal, "n_test_phantoms": n_test,
        "coverage_per_test_phantom": cov, "coverage": str(bootstrap_ci(cov)),
        "coverage_min": float(min(cov)), "frac_within_5pts": float(np.mean([abs(c - (1 - alpha)) <= 0.05 for c in cov])),
        "mean_set_size": str(bootstrap_ci([e["mean_set_size"] for e in per_rep])),
        "sigma_scale": sigma_scale,
    }


def verdict_unit_hard(seed: int, spacing_planes: int, xi_z_um: float, preset: str, method: str,
                      sigma_scale: float, M: int = 3, N: int = 3) -> dict:
    """One specimen of the hard simulator: measurement interval for a compartment fraction and its truth."""
    from . import planner
    from .measure import class_volume_fraction, interval_from_samples, observed_support_fraction

    truth, stack, reg, seg, xi = _prepare(seed, preset, method, M, N, spacing_planes, xi_z_um)
    fn = {"frac": lambda p, prov: class_volume_fraction(p, prov, 1)}
    res = run_pipeline(stack, xi_um=xi, seed=seed, reg=reg, seg=seg, sigma_scale=sigma_scale,
                       measure_fns=fn, validate=False)
    maps = res.maps
    domain = (stack.z_um[0] - stack.thickness_um / 2, stack.z_um[-1] + stack.thickness_um / 2)
    pri = float((truth == 1).mean())
    fm = planner.FieldModel(pri, float(xi[1]), n_eff_xy=max((truth.shape[1] * PX / 14.0) ** 2, 1.0))
    kvar = planner.mean_estimation_variance(stack.z_um, domain, fm)
    o = observed_support_fraction(maps.provenance)
    iv = interval_from_samples(maps.measurements["frac"], kvar, o, level=0.9)
    tg = evaluate.truth_on_grid(stack, maps.z_um)
    tf = float((tg[maps.provenance > 0] == 1).mean())
    return {"spacing_planes": spacing_planes, "xi_z_um": xi_z_um, "o": o, "rel_halfwidth": iv.relative_halfwidth,
            "estimate": iv.estimate, "truth": tf, "covered": bool(iv.contains(tf))}


def _unit_run(args):
    return verdict_unit_hard(*args)


def run_verdicts_hard(n_units: int = 48, conformal_coverage: float | None = None, preset: str = "hard",
                      method: str = "nonrigid", sigma_scale: float = 2.16, workers: int = 4,
                      seed0: int = 9000, alpha: float = 0.1) -> dict:
    """Risk-controlled verdicts on the hard simulator, averaged over random calibration/test splits."""
    from .demo import evaluate_verdicts

    spacings, xis = [4, 6, 8, 10], [20.0, 36.0, 60.0]
    items = [(seed0 + i, spacings[i % 4], xis[(i // 4) % 3], preset, method, sigma_scale) for i in range(n_units)]
    raw = _pool(_unit_run, items, workers)
    status = "pass" if conformal_coverage is None or abs(conformal_coverage - (1 - alpha)) <= 0.05 else "fail"
    return {"assumed_pass": evaluate_verdicts(raw, "pass", 0, alpha), "measured_status": status,
            "measured": evaluate_verdicts(raw, status, 0, alpha), "units": raw}


def run_all_experiments(out_dir: str | Path, n_cal: int = 8, n_test: int = 16, workers: int = 4,
                        preset: str = "hard", method: str = "nonrigid") -> dict:
    """Run both experiment families; each stage is written to disk as soon as it finishes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"disclaimer": "Synthetic phantoms with known truth; machinery checks, not evidence about real tissue."}
    res["multiseed"] = run_multiseed(n_cal, n_test, preset, method, workers=workers)
    (out_dir / "experiments.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    res["texture_h8"] = run_texture_experiment(max(n_test, 12), workers=workers)
    (out_dir / "experiments.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    res["nuclei"] = run_nuclei_experiment(20)
    (out_dir / "experiments.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    res["conformal_hard"] = run_conformal_hard(workers=workers)
    (out_dir / "experiments.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    cov = float(np.mean(res["conformal_hard"]["coverage_per_test_phantom"]))
    res["verdicts_hard"] = run_verdicts_hard(48, cov, workers=workers)
    (out_dir / "experiments.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res
