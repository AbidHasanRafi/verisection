"""Evaluation on a *real* dense serial-section stack without ground truth.

A dense stack (spacing close to thickness) is thinned to every ``step``-th section; the omitted
sections are held out. The pipeline reconstructs from the kept sections and its posterior at each
held-out position is compared with that section's **own segmentation** (registered with the full
stack's transform). This is a *proxy* truth: it contains segmentation error, so absolute accuracy
is a lower bound and only comparisons between interpolators, calibration (ECE) and the dependence
on gap size are interpretable.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .interpolate import InterpConfig
from .provenance import Provenance
from .register import register_series, register_series_chain
from .segment import SegmentationEnsemble
from .uncertainty import propagate
from .validate import validate_stack
from .variogram import conservative_xi, estimate_class_params_from_pilot
from .volume import SectionStack


def real_subsample_experiment(
    stack: SectionStack,
    step: int = 4,
    n_classes: int = 3,
    method: str = "nonrigid",
    M: int = 3,
    N: int = 3,
    seed: int = 0,
    sigma_scale: float | None = None,
    return_records: bool = False,
) -> dict:
    """Thin ``stack`` by ``step`` and score interpolation at the omitted sections."""
    validate_stack(stack)
    rng = np.random.default_rng(seed)
    K = stack.n_sections
    if step < 2 or K < 2 * step + 1:
        raise ValueError("need step >= 2 and at least 2*step+1 sections")
    reg_full = (
        register_series(stack.images, stack.pixel_um, M, rng)
        if method == "translation"
        else register_series_chain(stack.images, stack.pixel_um, M, rng, method=method)
    )
    seg = SegmentationEnsemble.fit(stack.images, n_classes, N, rng)
    par = estimate_class_params_from_pilot(stack, seg, n_classes, registration=reg_full.point(), rng=rng)
    xi = conservative_xi(par)  # unreliable (extrapolated) lengths are clamped to the sampled extent
    keep = list(range(0, K, step))
    held = [j for j in range(keep[0] + 1, keep[-1]) if j not in keep]
    sparse = stack.subset(keep)
    reg_sparse = reg_full.subset(keep)
    if sigma_scale:
        reg_sparse = reg_sparse.calibrated(sigma_scale)
    mean_t = reg_full.point().transforms[0]

    # proxy truth: each held-out section's own MAP segmentation in the common frame
    ref = {}
    for j in held:
        p = np.mean([seg.probs(stack.images[j], n) for n in range(seg.n_members)], axis=0)
        w, valid = mean_t[j].apply(p, order=1, cval=0.0)
        ref[j] = (w.argmax(axis=0), valid)

    names = [f"c{c}" for c in range(n_classes)]
    out: dict = {"n_sections": K, "step": step, "n_kept": len(keep), "n_heldout": len(held),
                 "xi_um_estimated": par["xi_um"].tolist(), "xi_um_used": xi.tolist(), "xi_reliable": par["reliable"].tolist(), "method": method, "variants": {}}
    for name, interp in {
        "kriging": InterpConfig("kriging", xi_um=xi),
        "linear": InterpConfig("linear"),
        "nearest": InterpConfig("nearest"),
    }.items():
        maps = propagate(sparse, reg_sparse if name == "kriging" else reg_sparse.point(),
                         seg if name == "kriging" else seg.point(), interp, names)
        per_gap: dict[int, list] = {}
        P, Y, U = [], [], []
        recs = []
        for j in held:
            plane = int(np.argmin(np.abs(maps.z_um - stack.z_um[j])))
            lab, valid = ref[j]
            sampled = (maps.provenance[plane] != Provenance.UNSAMPLED) & valid
            if sampled.sum() < 100:
                continue
            p = maps.p_hat[:, plane][:, sampled]
            y = lab[sampled]
            gap_sections = min(abs(j - k) for k in keep)
            per_gap.setdefault(gap_sections, []).append((p, y))
            recs.append({"section": j, "p": p.astype(np.float32), "y": y})
            P.append(p)
            Y.append(y)
            U.append(maps.u_total[plane][sampled])
        P, Y, U = np.concatenate(P, axis=1), np.concatenate(Y), np.concatenate(U)
        err = (P.argmax(0) != Y).astype(float)
        if return_records:
            out.setdefault("_records", {})[name] = recs
        out["variants"][name] = {
            "agreement_with_heldout_segmentation": float(1 - err.mean()),
            "ece": metrics.ece(P, Y), "brier": metrics.brier(P, Y), "nll": metrics.nll(P, Y),
            "ause": metrics.ause(U, err), "spearman_u_err": metrics.spearman(U, err),
            "by_distance_to_nearest_kept_section": {
                str(g): float(np.mean([(p.argmax(0) == y).mean() for p, y in v])) for g, v in sorted(per_gap.items())
            },
        }
    out["note"] = ("Proxy truth = the held-out section's own segmentation; agreement is a lower bound on accuracy. "
                   "Interpret comparisons between interpolators and calibration, not absolute values.")
    return out


def real_conformal_experiment(
    stack: SectionStack, step: int = 4, alpha: float = 0.1, n_classes: int = 3, method: str = "nonrigid",
    M: int = 3, N: int = 3, seed: int = 0, sigma_scale: float | None = None,
) -> dict:
    """Blocked split-conformal check on a real stack (proxy truth).

    Held-out sections are split into a first and a second half along z. Sets are calibrated on one
    half and evaluated on the other (both directions). Neighbouring sections are strongly dependent,
    so this is a *stringent*, small-sample check of the exchangeability assumption, on one specimen.
    """
    from .conformal import SplitConformal

    r = real_subsample_experiment(stack, step, n_classes, method, M, N, seed, sigma_scale, return_records=True)
    recs = sorted(r.pop("_records")["kriging"], key=lambda d: d["section"])
    half = len(recs) // 2
    out = {"alpha": alpha, "nominal": 1 - alpha, "n_heldout": len(recs), "step": step, "directions": {}}
    for name, cal, test in (("first->second", recs[:half], recs[half:]), ("second->first", recs[half:], recs[:half])):
        cp = SplitConformal(alpha=alpha).calibrate(cal, rng=np.random.default_rng(seed))
        out["directions"][name] = cp.evaluate(test) | {"qhat": cp.qhat}
    out["coverage_mean_over_directions"] = float(np.mean([d["coverage_mean"] for d in out["directions"].values()]))
    out["note"] = ("Proxy truth (held-out section's own segmentation); one specimen; blocked split along z. "
                   "Coverage is relative to that proxy label.")
    return out
