"""Reproducible smoke experiments on simulated stacks (``verisection demo``).

IMPORTANT: everything here uses a synthetic phantom with known truth. The numbers
show whether the machinery behaves as designed; they are *not* evidence about real
tissue, and the simulator is deliberately simple. Correlation
lengths are estimated from the phantom's dense truth, standing in for the dense pilot
sub-block a real study would cut.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import evaluate, metrics, planner
from .conformal import SplitConformal
from .decision import (
    EvidenceVector,
    VerdictPolicy,
    conformal_risk_control,
)
from .measure import class_volume_fraction, interval_from_samples, observed_support_fraction
from .pipeline import run_pipeline
from .register import register_series, registration_error_px
from .segment import SegmentationEnsemble
from .simulate import SectioningConfig, make_phantom, virtual_sections
from .variogram import estimate_class_params

XI_XY_UM = 14.0  # nominal in-plane correlation length of the phantom
TARGET_CLASS = 1  # compartment whose volume fraction is the target measurement


def _setup(seed: int, cfg: SectioningConfig | None = None, xi_z_um: float = 36.0):
    truth = make_phantom(seed=seed, xi_z_um=xi_z_um, xi_xy_um=XI_XY_UM)
    cfg = cfg or SectioningConfig()
    stack = virtual_sections(truth, cfg, seed=seed)
    params = estimate_class_params(truth, 2.0, 3)
    return truth, stack, params


def _fit_ensembles(stack, M: int, N: int, seed: int):
    rng = np.random.default_rng(seed)
    reg = register_series(stack.images, stack.pixel_um, M, rng)
    seg = SegmentationEnsemble.fit(stack.images, 3, N, rng)
    return reg, seg


# ----------------------------------------------------------------------------------
def experiment_uncertainty(seed: int = 0, fast: bool = False) -> dict:
    """H1-style check: joint vs single-source uncertainty, decomposition, registration error."""
    M = N = 3 if fast else 6
    truth, stack, par = _setup(seed)
    reg, seg = _fit_ensembles(stack, M, N, seed)
    true_shifts = stack.meta["true_shifts_px"]
    err = registration_error_px(reg.mean_shifts(), true_shifts)
    sig = reg.sigma_um() / stack.pixel_um
    rng = np.random.default_rng(seed)
    variants = {
        "joint": dict(reg=reg, seg=seg, interp_mode="kriging"),
        "seg_only": dict(reg=reg.point(), seg=seg, interp_mode="kriging"),
        "reg_only": dict(reg=reg, seg=seg.point(), interp_mode="kriging"),
        "point_estimate_linear": dict(reg=reg.point(), seg=seg.point(), interp_mode="linear"),
    }
    out: dict = {"seed": seed, "M": M, "N": N, "variants": {}}
    maps_joint = None
    for name, kw in variants.items():
        res = run_pipeline(stack, xi_um=par["xi_um"], seed=seed, **kw)
        truth_grid = evaluate.truth_on_grid(stack, res.maps.z_um)
        sc = evaluate.score_maps(res.maps, truth_grid, rng)
        out["variants"][name] = sc
        if name == "joint":
            maps_joint = res.maps
            out["error_vs_gap"] = evaluate.error_vs_gap(res.maps, truth_grid)
            m = res.maps
            valid = m.provenance > 0
            out["decomposition"] = {
                "mean_var_total": float(m.var_total[valid].mean()),
                "mean_var_reg_total_effect": float(m.var_reg[valid].mean()),
                "mean_var_seg_total_effect": float(m.var_seg[valid].mean()),
                "mean_var_interaction": float(m.var_interaction[valid].mean()),
                "mean_expected_entropy_bits": float(m.expected_entropy[valid].mean()),
                "mean_mutual_information_bits": float(m.mutual_information[valid].mean()),
                "mean_u_interp_bits_in_gaps": float(
                    m.expected_entropy[m.provenance == 2].mean()
                ),
            }
    out["registration"] = {
        "mean_error_px": float(err.mean()),
        "max_error_px": float(err.max()),
        "mean_sigma_px": float(sig.mean()),
        "sigma_to_error_ratio": float(sig[1:].mean() / max(err[1:].mean(), 1e-9)),
        "corr_sigma_error": float(np.corrcoef(sig[1:], err[1:])[0, 1]) if sig[1:].std() > 0 else None,
    }
    out["_maps"] = maps_joint  # stripped before JSON
    out["_stack"] = stack
    out["_truth"] = truth
    return out


def experiment_conformal(seed: int = 0, fast: bool = False, alpha: float = 0.1) -> dict:
    """LOSO conformal calibration on replicate A phantoms, coverage on a different phantom."""
    M = N = 2 if fast else 3
    n_cal_reps = 1 if fast else 2

    def loso(rep_seed: int):
        truth, stack, par = _setup(rep_seed)
        reg, seg = _fit_ensembles(stack, M, N, rep_seed)
        from .interpolate import InterpConfig

        interp = InterpConfig("kriging", xi_um=par["xi_um"])
        ks = list(range(2, stack.n_sections - 2, 3 if fast else 2))
        return evaluate.loso_scores(stack, reg, seg, interp, ["c0", "c1", "c2"], held_out=ks)

    cal = [r for s in range(n_cal_reps) for r in loso(seed + 1000 + s)]
    test = loso(seed)
    cp = SplitConformal(alpha=alpha).calibrate(cal, rng=np.random.default_rng(seed))
    res = cp.evaluate(test)
    res["qhat"] = cp.qhat
    res["n_calibration_scores"] = cp.n_cal
    # reference: ECE of the held-out predictions (not conformal)
    p = np.concatenate([r["p"] for r in test], axis=1)
    y = np.concatenate([r["y"] for r in test])
    res["loso_accuracy"] = float((p.argmax(0) == y).mean())
    res["loso_ece"] = metrics.ece(p, y)
    return res


def _unit(seed: int, spacing_planes: int, xi_z_um: float, M: int, N: int):
    """One specimen: run the pipeline and return (evidence-ready numbers, truth)."""
    cfg = SectioningConfig(spacing_planes=spacing_planes)
    Z = 120
    cfg.n_sections = min(18, (Z - 1 - cfg.first_plane) // spacing_planes + 1)
    truth, stack, par = _setup(seed, cfg, xi_z_um)
    reg, seg = _fit_ensembles(stack, M, N, seed)
    fn = {"frac": lambda p, prov: class_volume_fraction(p, prov, TARGET_CLASS)}
    res = run_pipeline(stack, xi_um=par["xi_um"], seed=seed, reg=reg, seg=seg, measure_fns=fn)
    maps = res.maps
    domain = (stack.z_um[0] - stack.thickness_um / 2, stack.z_um[-1] + stack.thickness_um / 2)
    n_eff = max((truth.shape[1] * stack.pixel_um / XI_XY_UM) ** 2, 1.0)
    fm = planner.FieldModel(float(par["priors"][TARGET_CLASS]), float(par["xi_um"][TARGET_CLASS]), n_eff_xy=n_eff)
    kvar = planner.mean_estimation_variance(stack.z_um, domain, fm)
    o = observed_support_fraction(maps.provenance)
    iv = interval_from_samples(maps.measurements["frac"], kvar, o, level=0.9)
    truth_grid = evaluate.truth_on_grid(stack, maps.z_um)
    true_frac = float((truth_grid[maps.provenance > 0] == TARGET_CLASS).mean())
    return iv, true_frac


def collect_units(n_units: int = 48, seed: int = 0, fast: bool = False) -> list[dict]:
    """Run ``n_units`` simulated specimens (varying spacing and z-correlation) once."""
    M = N = 2 if fast else 3
    spacings = [4, 6, 8, 10]
    xis = [20.0, 36.0, 60.0]
    raw = []
    for i in range(n_units):
        sp, xi = spacings[i % len(spacings)], xis[(i // len(spacings)) % len(xis)]
        iv, tf = _unit(seed + 5000 + i, sp, xi, M, N)
        raw.append({"spacing_planes": sp, "xi_z_um": xi, "o": iv.observed_fraction,
                    "rel_halfwidth": iv.relative_halfwidth, "estimate": iv.estimate,
                    "truth": tf, "covered": iv.contains(tf)})
    return raw


def evaluate_verdicts(
    raw: list[dict], calibration_status: str, seed: int = 0, alpha: float = 0.1, n_splits: int = 200
) -> dict:
    """H9-style check: risk-controlled verdicts vs naive rules, averaged over random splits.

    Each split calibrates the strictness on half of the specimens (conformal risk control) and
    evaluates false-claim and abstention rates on the other half. A single split of ~24 units is
    far too noisy to judge a risk guarantee, so results are averaged over ``n_splits`` splits.
    """
    from .decision import LAMBDAS

    rng = np.random.default_rng(seed)
    units = [(EvidenceVector(o=r["o"], w=r["rel_halfwidth"], s=1.0, c=calibration_status, u=False),
              bool(r["covered"])) for r in raw]
    n = len(units)
    covered = np.array([c for _, c in units])
    policies = {
        "full (o, w)": VerdictPolicy(),
        "B1 observed-fraction only": VerdictPolicy(use_w=False),
        "B2 interval-width only": VerdictPolicy(use_o=False),
    }
    out: dict = {
        "alpha": alpha, "calibration_status": calibration_status, "n_units": n, "n_splits": n_splits,
        "interval_coverage_all_units": float(covered.mean()), "policies": {},
    }
    splits = [rng.permutation(n) for _ in range(n_splits)]
    for name, pol in policies.items():
        R = np.array([[pol.reportable(e, lam) for lam in LAMBDAS] for e, _ in units])  # (n, n_lam)
        L = (R & ~covered[:, None]).astype(float)
        fc, ab, lam_hats, feas = [], [], [], []
        for idx in splits:
            cal, test = idx[: n // 2], idx[n // 2 :]
            crc = conformal_risk_control(L[cal], alpha)
            j = int(np.argmin(np.abs(LAMBDAS - crc["lam_hat"])))
            fc.append(float(L[test][:, j].mean()))
            ab.append(float(1.0 - R[test][:, j].mean()))
            lam_hats.append(crc["lam_hat"])
            feas.append(crc["feasible"])
        out["policies"][name] = {
            "false_claim_rate": float(np.mean(fc)), "false_claim_rate_sd_over_splits": float(np.std(fc)),
            "abstention_rate": float(np.mean(ab)), "lam_hat_mean": float(np.mean(lam_hats)),
            "feasible_fraction": float(np.mean(feas)),
        }
    out["report_everything"] = {
        "false_claim_rate": float((~covered).mean()), "abstention_rate": 0.0,
    }
    return out


def experiment_verdicts(
    n_units: int = 48, seed: int = 0, fast: bool = False, alpha: float = 0.1,
    calibration_status: str = "pass",
) -> dict:
    raw = collect_units(n_units, seed, fast)
    out = evaluate_verdicts(raw, calibration_status, seed, alpha)
    out["units"] = raw
    return out


def experiment_planner(seed: int = 0, n_phantoms: int = 40) -> dict:
    """H10-style check on the geostatistical model: planner vs random vs bisect, clustered start."""
    domain = (0.0, 240.0)
    cand = np.arange(2.0, 240.0, 4.0)  # candidate section centres (4 µm grid)
    rng = np.random.default_rng(seed)
    results = {k: {n: [] for n in (2, 4, 6, 8)} for k in ("planner", "random", "bisect")}
    for i in range(n_phantoms):
        truth = make_phantom(seed=seed + 7000 + i)
        f = (truth == TARGET_CLASS).mean(axis=(1, 2))  # per fine plane fraction
        z_plane = (np.arange(len(f)) + 0.5) * 2.0
        true_mean = float(f.mean())
        par = estimate_class_params(truth, 2.0, 3)
        m = planner.FieldModel(float(par["priors"][TARGET_CLASS]), float(par["xi_um"][TARGET_CLASS]),
                               nugget=0.02, n_eff_xy=1.0)
        start = np.array([6.0, 14.0, 22.0])  # clustered at one end
        for n_add in (2, 4, 6, 8):
            designs = {}
            designs["planner"] = planner.greedy_design(start, cand[~np.isin(cand, start)], domain, m, n_add)
            designs["random"] = np.sort(np.append(start, rng.choice(cand[~np.isin(cand, start)], n_add, replace=False)))
            z = list(start)
            for _ in range(n_add):  # bisect the largest gap (incl. the domain ends)
                zs = np.sort(np.array(z))
                edges = np.concatenate([[domain[0]], zs, [domain[1]]])
                j = int(np.argmax(np.diff(edges)))
                mid = 0.5 * (edges[j] + edges[j + 1])
                z.append(float(cand[np.argmin(np.abs(cand - mid))]))
            designs["bisect"] = np.sort(np.array(z))
            for name, zd in designs.items():
                obs = np.array([f[int(np.argmin(np.abs(z_plane - zz)))] for zz in zd])
                est = planner.kriged_domain_mean(zd, obs, domain, m)
                results[name][n_add].append(abs(est - true_mean))
    return {
        "n_phantoms": n_phantoms,
        "mean_abs_error": {k: {str(n): float(np.mean(v)) for n, v in d.items()} for k, d in results.items()},
        "note": "Clustered start (3 sections at one end). Stationary phantom: 'bisect' is expected to "
                "match the planner; benefits require non-stationary or multi-target settings (R20).",
    }


# ----------------------------------------------------------------------------------
def run_all(out_dir: str | Path = "runs/demo", seed: int = 0, fast: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: dict = {"seed": seed, "fast": fast, "disclaimer": (
        "Synthetic phantom with known truth: machinery sanity checks only, not evidence about real tissue."
    )}
    u = experiment_uncertainty(seed, fast)
    maps, stack, truth = u.pop("_maps"), u.pop("_stack"), u.pop("_truth")
    results["uncertainty"] = u
    c = experiment_conformal(seed, fast)
    results["conformal"] = c
    status = "pass" if abs(c["coverage_mean"] - c["nominal"]) <= 0.05 else "fail"
    raw = collect_units(24 if fast else 48, seed, fast)
    results["calibration_status_measured"] = status
    results["verdict_units"] = raw
    # 'assumed_pass' isolates the evidence rule (o, w); 'measured' applies the LOSO conformal check:
    # a failed calibration check makes every verdict "insufficient evidence", by design.
    results["verdicts_assumed_pass"] = evaluate_verdicts(raw, "pass", seed)
    results["verdicts"] = evaluate_verdicts(raw, status, seed)
    results["planner"] = experiment_planner(seed, 15 if fast else 40)
    results["runtime_s"] = round(time.time() - t0, 1)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    try:
        _figure(out_dir / "figure.png", maps, stack, truth)
    except Exception as exc:  # matplotlib is optional
        results["figure"] = f"skipped: {exc}"
    return results


def _figure(path: Path, maps, stack, truth) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = maps.p_hat.shape[2] // 2
    tg = evaluate.truth_on_grid(stack, maps.z_um)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ext = [0, maps.p_hat.shape[3] * maps.pixel_um, maps.z_um[-1], maps.z_um[0]]
    ax[0].imshow(tg[:, y], cmap="viridis", aspect="auto", extent=ext, interpolation="nearest")
    ax[0].set_title("truth (x–z slice)")
    ax[1].imshow(maps.map_labels[:, y], cmap="viridis", aspect="auto", extent=ext, interpolation="nearest")
    ax[1].set_title("reconstruction (MAP)")
    im = ax[2].imshow(maps.u_total[:, y], cmap="magma", aspect="auto", extent=ext, interpolation="nearest")
    ax[2].set_title("uncertainty (bits)")
    fig.colorbar(im, ax=ax[2])
    prov = maps.provenance[:, y]
    ax[3].imshow(prov, cmap="Greys", vmin=0, vmax=3, aspect="auto", extent=ext, interpolation="nearest")
    ax[3].set_title("provenance (dark = observed)")
    for a in ax:
        a.set_xlabel("x (µm)")
    ax[0].set_ylabel("z (µm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
