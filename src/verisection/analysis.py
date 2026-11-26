"""Numerical analysis for the workstation's analysis dock.

``build_analysis`` turns a simulation result into plain JSON-serialisable tables and curves:
segmentation and calibration by provenance, uncertainty against depth/gap/provenance, variance
attribution, registration and section QC, texture per class (with validity rules and the
observed-vs-interpolated bias check when truth intensity exists), per-class measurements with
intervals and evidence-gated verdicts, and the section planner.

Everything is computed against *simulation truth*, so it checks the machinery; none of it is
evidence about real tissue.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import metrics, planner, texture
from .decision import EvidenceVector, VerdictPolicy
from .measure import interval_from_samples
from .provenance import Provenance
from .qc import passport, qc_decisions
from .register import registration_tre_px


def _f(x: Any, nd: int = 4):
    """JSON-safe float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def _sample(rng: np.random.Generator, mask: np.ndarray, cap: int = 200_000) -> np.ndarray:
    idx = np.flatnonzero(mask.ravel())
    return idx if idx.size <= cap else rng.choice(idx, cap, replace=False)


def _reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 12) -> dict:
    conf = p.max(axis=0)
    correct = (p.argmax(axis=0) == y).astype(float)
    edges = np.linspace(1.0 / p.shape[0], 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"conf": _f(conf[m].mean()), "acc": _f(correct[m].mean()), "n": int(m.sum()),
                         "lo": _f(edges[b]), "hi": _f(edges[b + 1])})
    return {"bins": rows, "ece": _f(metrics.ece(p, y, n_bins)), "brier": _f(metrics.brier(p, y)),
            "nll": _f(metrics.nll(p, y)), "accuracy": _f(correct.mean()), "n": int(len(y)),
            "mean_confidence": _f(conf.mean())}


def _confusion(pred: np.ndarray, y: np.ndarray, C: int) -> list[list[int]]:
    return np.bincount(y * C + pred, minlength=C * C).reshape(C, C).tolist()


def _class_table(cm: np.ndarray) -> list[dict]:
    out = []
    for c in range(cm.shape[0]):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        out.append({"class": c, "support": int(cm[c].sum()), "predicted": int(cm[:, c].sum()),
                    "precision": _f(tp / max(tp + fp, 1)), "recall": _f(tp / max(tp + fn, 1)),
                    "dice": _f(2 * tp / max(2 * tp + fp + fn, 1)), "iou": _f(tp / max(tp + fp + fn, 1))})
    return out


def _sparsification(unc: np.ndarray, err: np.ndarray, n: int = 21) -> dict:
    cu = metrics.sparsification_curve(unc, err, n)
    co = metrics.sparsification_curve(err, err, n)
    fr = np.linspace(0, 0.95, n)
    return {"fraction_removed": fr.round(3).tolist(), "by_uncertainty": cu.round(4).tolist(),
            "oracle": co.round(4).tolist(), "random": [round(float(err.mean()), 4)] * n,
            "ause": _f(metrics.ause(unc, err, n)), "spearman": _f(metrics.spearman(unc, err))}


def _glcm_features(img: np.ndarray, mask: np.ndarray, pixel_um: float) -> dict | None:
    if mask.sum() < 300:
        return None
    od = np.clip(img, 0, None)
    f = texture.window_features_2d(od, mask, pixel_um, offset_um=max(pixel_um, 2.0), bin_width=0.04, n_levels=48)
    return {k: float(v) for k, v in f.items() if k in ("contrast", "entropy", "homogeneity", "correlation")}


def build_analysis(sim, params, extended: bool = False, progress=None) -> dict:
    """Assemble every table and curve shown in the analysis dock."""
    rng = np.random.default_rng(0)
    res, view = sim.result, sim.view
    maps, stack = res.maps, res.stack
    C = maps.p_hat.shape[0]
    names = list(view.class_names)
    tg = view.truth
    prov = maps.provenance
    Z = prov.shape[0]
    valid = prov > 0
    out: dict[str, Any] = {"class_names": names, "disclaimer": "Synthetic phantom with known truth: machinery check, not evidence about real tissue."}

    # ---------------- summary
    pred = maps.map_labels
    prov_frac = {Provenance(i).name.lower(): _f((prov == i).sum() / max(valid.sum(), 1)) for i in (1, 2, 3)}
    out["summary"] = {
        "dims": list(prov.shape), "n_sections": int(stack.n_sections), "section_spacing_um": _f(np.median(stack.spacing_um)),
        "section_thickness_um": _f(stack.thickness_um), "pixel_um": _f(stack.pixel_um), "grid_dz_um": _f(maps.dz_um),
        "provenance_fraction": prov_frac, "n_reg_samples": int(res.reg.n_samples), "n_seg_members": int(res.seg.n_members),
        "registration_method": res.reg.method, "interpolation": res.interp.mode,
        "true_class_fraction": [_f((tg[valid] == c).mean()) for c in range(C)] if tg is not None else None,
        "predicted_class_fraction": [_f((pred[valid] == c).mean()) for c in range(C)],
        "mean_entropy_bits": _f(maps.u_total[valid].mean()),
        "params": params if isinstance(params, dict) else {},
    }
    if tg is None:
        return out

    # ---------------- segmentation (vs truth)
    seg: dict[str, Any] = {}
    cal: dict[str, Any] = {}
    unc_out: dict[str, Any] = {}
    flat_p = maps.p_hat.reshape(C, -1)
    for key, mask in (("all", valid), ("observed", prov == Provenance.OBSERVED), ("interpolated", prov == Provenance.INTERPOLATED)):
        if not mask.any():
            continue
        idx = _sample(rng, mask)
        p = flat_p[:, idx]
        y = tg.ravel()[idx].astype(int)
        pr = p.argmax(axis=0)
        cm = np.array(_confusion(pr, y, C))
        seg[key] = {"confusion": cm.tolist(), "per_class": _class_table(cm), "accuracy": _f((pr == y).mean()),
                    "balanced_accuracy": _f(np.mean([cm[c, c] / max(cm[c].sum(), 1) for c in range(C)]))}
        cal[key] = _reliability(p, y)
    out["segmentation"] = seg
    out["calibration"] = cal

    # ---------------- uncertainty
    idx = _sample(rng, valid)
    err = (pred.ravel()[idx] != tg.ravel()[idx]).astype(float)
    u = maps.u_total.ravel()[idx]
    unc_out["sparsification"] = _sparsification(u, err)
    per_z = []
    wrong = pred != tg
    for k in range(Z):
        vm = prov[k] > 0
        if not vm.any():
            continue
        per_z.append({"z_um": _f(maps.z_um[k], 2), "provenance": int(np.bincount(prov[k][vm]).argmax()),
                      "entropy": _f(maps.u_total[k][vm].mean()), "error": _f(wrong[k][vm].mean()),
                      "expected_entropy": _f(maps.expected_entropy[k][vm].mean()),
                      "mutual_information": _f(maps.mutual_information[k][vm].mean()),
                      "var_reg": _f(maps.var_reg[k][vm].mean(), 6), "var_seg": _f(maps.var_seg[k][vm].mean(), 6),
                      "gap_um": _f(maps.gap_distance_um[k], 2)})
    unc_out["per_plane"] = per_z
    hist = {}
    for key, mask in (("observed", prov == Provenance.OBSERVED), ("interpolated", prov == Provenance.INTERPOLATED)):
        if mask.any():
            h, e = np.histogram(maps.u_total[mask], bins=24, range=(0, float(np.log2(C))))
            hist[key] = {"edges": e.round(3).tolist(), "counts": h.tolist()}
    unc_out["entropy_histogram"] = hist
    gap_rows = []
    gap = np.broadcast_to(maps.gap_distance_um[:, None, None], prov.shape)
    for g in np.unique(np.round(gap[valid], 3)):
        m = valid & np.isclose(gap, g)
        gap_rows.append({"gap_um": _f(g, 2), "n": int(m.sum()), "error": _f(wrong[m].mean()), "entropy": _f(maps.u_total[m].mean())})
    unc_out["by_gap"] = gap_rows
    v = valid
    unc_out["variance_attribution"] = {
        "total": _f(maps.var_total[v].mean(), 6), "registration_total_effect": _f(maps.var_reg[v].mean(), 6),
        "segmentation_total_effect": _f(maps.var_seg[v].mean(), 6),
        "registration_first_order": _f(maps.var_reg_first[v].mean(), 6),
        "segmentation_first_order": _f(maps.var_seg_first[v].mean(), 6), "interaction": _f(maps.var_interaction[v].mean(), 6),
        "mean_expected_entropy_bits": _f(maps.expected_entropy[v].mean()), "mean_mutual_information_bits": _f(maps.mutual_information[v].mean()),
    }
    out["uncertainty"] = unc_out
    if progress:
        progress("analysis: registration and QC", 0.93)

    # ---------------- registration
    reg = res.reg
    tre = registration_tre_px(reg.point().transforms[0], stack.meta["true_h"], stack.images[0].shape) * stack.pixel_um
    sig = reg.sigma_um()
    out["registration"] = {
        "sections": [{"index": k, "z_um": _f(stack.z_um[k], 1), "tre_um": _f(tre[k]), "sigma_um": _f(sig[k]),
                      "ncc": _f(np.median(reg.ncc[:, k])) if reg.ncc is not None else None,
                      "failed_samples": int(reg.failed[:, k].sum()) if reg.failed is not None else 0} for k in range(stack.n_sections)],
        "mean_tre_um": _f(tre[1:].mean()), "max_tre_um": _f(tre[1:].max()), "mean_sigma_um": _f(sig[1:].mean()),
        "sigma_to_error_ratio": _f(sig[1:].mean() / max(tre[1:].mean(), 1e-9)), "method": reg.method,
        "spearman_sigma_tre": _f(metrics.spearman(sig[1:], tre[1:])), "n_samples": int(reg.n_samples),
    }

    # ---------------- section QC
    pps = [passport(im, stack.images[k - 1] if k else None) for k, im in enumerate(stack.images)]
    dec = qc_decisions(pps)
    out["qc"] = [{"index": k, "z_um": _f(stack.z_um[k], 1), "focus": _f(pps[k]["focus_lapvar"], 5), "noise": _f(pps[k]["noise_sigma"], 5),
                  "mean": _f(pps[k]["mean"]), "sd": _f(pps[k]["sd"]), "continuity": _f(pps[k].get("continuity_wasserstein")),
                  "verdict": dec[k].verdict, "reasons": dec[k].reasons} for k in range(stack.n_sections)]

    # ---------------- texture per class on registered observed sections
    tex_rows: list[dict] = []
    mean_t = reg.point().transforms[0]
    obs_planes = [k for k in range(Z) if (prov[k] == Provenance.OBSERVED).mean() > 0.5]
    # map observed planes to section index by z
    sec_of_plane = {k: int(np.argmin(np.abs(stack.z_um - maps.z_um[k]))) for k in obs_planes}
    per_cls: dict[int, list[dict]] = {c: [] for c in range(C)}
    for k, s_idx in sec_of_plane.items():
        reg_img, ok = mean_t[s_idx].apply(stack.images[s_idx].astype(np.float64), order=1, cval=0.0)
        for c in range(C):
            feats = _glcm_features(reg_img, (pred[k] == c) & ok & (prov[k] == Provenance.OBSERVED), stack.pixel_um)
            if feats:
                per_cls[c].append(feats)
    sigma_med = float(np.median(sig[1:])) if len(sig) > 1 else 0.0
    xi = res.interp.xi_um
    for c in range(C):
        rows = per_cls[c]
        stats = {f: {"mean": _f(np.mean([r[f] for r in rows])), "sd": _f(np.std([r[f] for r in rows])), "n": len(rows)} for f in ("contrast", "entropy", "homogeneity", "correlation")} if rows else {}
        xi_c = float(xi[c]) if xi is not None else 36.0
        d = texture.texture_validity(float(prov_frac["observed"] or 0), ell_xy_um=8.0, pixel_um=float(stack.pixel_um),
                                     sigma_reg_um=sigma_med, dz_um=float(np.median(stack.spacing_um)), xi_um=xi_c, needs_z=True)
        tex_rows.append({"class": names[c], "n_sections": len(rows), "features": stats, "z_feature_mode": d.mode.value,
                         "z_feature_reasons": d.reasons, "min_scale_um": _f(texture.min_scale_um(float(stack.pixel_um), sigma_med), 2), "xi_um": _f(xi_c, 1)})
    tex_out: dict[str, Any] = {"per_class": tex_rows, "note": "GLCM features at 2 µm offset, fixed 0.04 OD bins, computed only on observed registered sections inside the reconstructed class mask."}
    ti = stack.meta.get("truth_intensity")
    if ti is not None:
        sec_planes = np.asarray(stack.meta["section_planes"])
        tv = stack.meta["truth_volume"]
        rows = []
        planes_all = np.arange(sec_planes[0], sec_planes[-1] + 1)

        def feat_at(z, getter, c):
            m = tv[z] == c
            if m.sum() < 300:
                return None
            f = _glcm_features(getter(z), m, stack.pixel_um)
            return None if f is None else np.array([f["contrast"], f["entropy"], f["homogeneity"]])

        def interp_plane(z):
            j = int(np.clip(np.searchsorted(sec_planes, z, side="right") - 1, 0, len(sec_planes) - 2))
            w = (z - sec_planes[j]) / max(sec_planes[j + 1] - sec_planes[j], 1)
            return (1 - w) * ti[sec_planes[j]] + w * ti[sec_planes[j + 1]]
        for c in range(C):
            truth_f = [feat_at(z, lambda q: ti[q], c) for z in planes_all]
            obs_f = [feat_at(z, lambda q: ti[q], c) for z in sec_planes]
            nav_f = [feat_at(z, interp_plane, c) for z in planes_all]
            tr_ = [x for x in truth_f if x is not None]
            if not tr_:
                continue
            tm = np.mean(tr_, axis=0)
            om = np.mean([x for x in obs_f if x is not None], axis=0) if any(x is not None for x in obs_f) else np.full(3, np.nan)
            nm = np.mean([x for x in nav_f if x is not None], axis=0)
            rows.append({"class": names[c], "contrast": {"truth": _f(tm[0]), "observed_only": _f(om[0]), "naive_interpolated": _f(nm[0])},
                         "entropy": {"truth": _f(tm[1]), "observed_only": _f(om[1]), "naive_interpolated": _f(nm[1])},
                         "homogeneity": {"truth": _f(tm[2]), "observed_only": _f(om[2]), "naive_interpolated": _f(nm[2])}})
        tex_out["observed_vs_interpolated_bias"] = rows
    out["texture"] = tex_out

    # ---------------- measurements and verdicts per class
    meas = []
    domain = (stack.z_um[0] - stack.thickness_um / 2, stack.z_um[-1] + stack.thickness_um / 2)
    pol = VerdictPolicy()
    o_frac = float(prov_frac["observed"] or 0)
    fm_cache = {}
    for c in range(C):
        samples = maps.measurements.get(f"frac{c}")
        prior = float((tg[valid] == c).mean())
        xi_c = float(xi[c]) if xi is not None else 36.0
        fm = planner.FieldModel(max(prior, 1e-3), xi_c, n_eff_xy=max((prov.shape[1] * stack.pixel_um / 14.0) ** 2, 1.0))
        fm_cache[c] = fm
        kv = planner.mean_estimation_variance(stack.z_um, domain, fm)
        est = float(maps.p_hat[c][valid].mean())
        if samples is None:
            samples = np.full((1, 1), est)
        iv = interval_from_samples(samples, kv, o_frac, 0.9)
        verdict, reasons = pol.decide(EvidenceVector(o=o_frac, w=iv.relative_halfwidth, c="untested"), 0.3)
        meas.append({"class": names[c], "estimate": _f(iv.estimate), "lo90": _f(iv.lo), "hi90": _f(iv.hi), "truth": _f(prior),
                     "covered": bool(iv.contains(prior)), "abs_error": _f(abs(iv.estimate - prior)), "rel_halfwidth": _f(iv.relative_halfwidth),
                     "sd_mc": _f(iv.sd_mc, 5), "sd_sectioning": _f(iv.sd_krig, 5), "observed_fraction": _f(o_frac),
                     "verdict": verdict.value, "reasons": reasons + ["calibration untested in this run"]})
    out["measurements"] = {"rows": meas, "level": 0.9,
                           "note": "Interval = Monte-Carlo spread ⊕ kriging sectioning variance (Gaussian). On the hard simulator this under-covered (54% at nominal 90%); read verdicts accordingly."}

    # ---------------- planner
    tgt = int(np.argmax([m["truth"] for m in meas]))
    fm = fm_cache[tgt]
    cand = np.arange(domain[0] + 1.0, domain[1], 2.0)
    recs = planner.recommend(stack.z_um, cand, domain, fm, top_k=8, min_separation_um=3.0)
    cur = float(np.sqrt(planner.mean_estimation_variance(stack.z_um, domain, fm)))
    unif = []
    for n in range(2, 41, 2):
        zz = domain[0] + (np.arange(n) + 0.5) * (domain[1] - domain[0]) / n
        unif.append({"n": n, "sd": _f(np.sqrt(planner.mean_estimation_variance(zz, domain, fm)), 5)})
    greedy = planner.greedy_design(stack.z_um, cand[::2], domain, fm, 6)
    added = [z for z in greedy if not np.any(np.isclose(z, stack.z_um))]
    seq = [{"added": 0, "sd": _f(cur, 5)}]
    zc = stack.z_um.copy()
    for k, z in enumerate(added):
        zc = np.append(zc, z)
        seq.append({"added": k + 1, "sd": _f(np.sqrt(planner.mean_estimation_variance(zc, domain, fm)), 5), "z_um": _f(z, 1)})
    out["planner"] = {"target_class": names[tgt], "current_sd": _f(cur, 5), "recommendations": [{**r, "z_um": _f(r["z_um"], 1)} for r in recs],
                      "uniform_designs": unif, "greedy_sequence": seq, "current_sections": int(stack.n_sections),
                      "note": "Kriging expected-value-of-information under a stationary exponential covariance."}

    # ---------------- provenance / manifest
    out["provenance"] = {"legend": {"0": "unsampled", "1": "observed", "2": "interpolated", "3": "synthetic"},
                         "fraction": prov_frac, "planes": [{"z_um": _f(maps.z_um[k], 1), "provenance": int(np.bincount(prov[k][prov[k] > 0]).argmax()) if (prov[k] > 0).any() else 0,
                                                            "gap_um": _f(maps.gap_distance_um[k], 2)} for k in range(Z)]}
    out["manifest"] = res.manifest.finalize().model_dump()
    out["log"] = sim.log

    if extended:
        out["conformal"] = _conformal(sim, progress)
    return out


def _conformal(sim, progress=None) -> dict:
    """LOSO conformal on this stack: calibrate on half of the held-out sections, evaluate on the other half."""
    from .conformal import SplitConformal
    from .evaluate import loso_scores

    res = sim.result
    K = res.stack.n_sections
    ks = list(range(1, K - 1, 2))
    if progress:
        progress("extended analysis: leave-one-section-out", 0.95)
    recs = loso_scores(res.stack, res.reg, res.seg, res.interp, res.class_names, held_out=ks)
    half = max(len(recs) // 2, 1)
    out = {"n_sections": len(recs), "alpha": 0.1, "directions": {}}
    for name, cal, test in (("first→second half", recs[:half], recs[half:]), ("second→first half", recs[half:], recs[:half])):
        if not cal or not test:
            continue
        cp = SplitConformal(alpha=0.1).calibrate(cal, rng=np.random.default_rng(0))
        ev = cp.evaluate(test)
        out["directions"][name] = {k: _f(v) for k, v in ev.items() if isinstance(v, float)} | {"qhat": _f(cp.qhat)}
    out["note"] = "Single specimen, blocked split along z; coverage is relative to simulation truth."
    return out
