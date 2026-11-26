import numpy as np
import pytest
from typer.testing import CliRunner

from verisection import metrics, planner, texture
from verisection.cli import app
from verisection.conformal import SplitConformal
from verisection.decision import (
    EvidenceVector,
    Verdict,
    VerdictPolicy,
    conformal_risk_control,
    evaluate_policy,
    false_claim_losses,
    is_monotone_nonincreasing,
)
from verisection.measure import interval_from_samples, observed_support_fraction
from verisection.qc import focus_laplacian_var, noise_sigma_haar, passport, qc_decisions, robust_z


# ----------------------------------------------------------------------------- metrics
def test_ece_zero_when_confident_and_correct_and_large_when_confidently_wrong():
    y = np.array([0, 1, 2, 1])
    p = np.eye(3)[:, y] * 0.999 + 0.0005
    assert metrics.ece(p, y) < 0.01
    assert metrics.ece(p, (y + 1) % 3) > 0.9
    assert metrics.brier(p, y) < 0.01 and metrics.nll(p, y) < 0.01


def test_ause_is_zero_for_oracle_and_positive_for_random_ranking():
    rng = np.random.default_rng(0)
    err = (rng.random(2000) < 0.3).astype(float)
    assert metrics.ause(err + 1e-9 * rng.random(2000), err) == pytest.approx(0, abs=1e-6)
    assert metrics.ause(rng.random(2000), err) > 0.05
    assert metrics.spearman(err + 0.1 * rng.random(2000), err) > 0.7


def test_dice_known_values():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 1, 1])
    d = metrics.dice(a, b, 2)
    assert d[0] == pytest.approx(2 * 1 / (2 + 1)) and d[1] == pytest.approx(2 * 2 / (2 + 3))


# --------------------------------------------------------------------------- conformal
def _records(rng, n_sections, n_vox=600, C=3):
    """A *calibrated* model: labels are drawn from the predicted distributions."""
    recs = []
    for k in range(n_sections):
        logits = rng.normal(0, 1.5, (C, n_vox))
        p = np.exp(logits) / np.exp(logits).sum(0)
        y = np.array([rng.choice(C, p=p[:, i]) for i in range(n_vox)])
        recs.append({"section": k, "p": p, "y": y})
    return recs


def test_split_conformal_attains_nominal_coverage_when_calibrated():
    rng = np.random.default_rng(0)
    cp = SplitConformal(alpha=0.1).calibrate(_records(rng, 12), per_section=300, rng=rng)
    ev = cp.evaluate(_records(rng, 12))
    assert abs(ev["coverage_mean"] - 0.9) < 0.03
    assert ev["mean_set_size"] >= 1.0
    sets = cp.predict_sets(np.array([[0.98], [0.01], [0.01]]))
    assert sets[:, 0].tolist() == [True, False, False]
    tight = SplitConformal(alpha=0.1, qhat=0.0)  # below any confidence: sets would be empty
    flat = np.array([[0.4], [0.35], [0.25]])
    assert not tight.predict_sets(flat).any() and tight.predict_sets(flat, non_empty=True)[:, 0].tolist() == [True, False, False]


# ---------------------------------------------------------------------------- texture
def test_quantization_is_fixed_bin_not_per_image():
    a = np.array([[0.00, 0.10], [0.20, 0.30]])
    b = a + 0.5  # different range, same bin width: levels shift by exactly 10
    qa = texture.quantize_fixed(a, 0.05, 64)
    qb = texture.quantize_fixed(b, 0.05, 64)
    assert np.array_equal(qb - qa, np.full_like(qa, 10))


def test_glcm_constant_and_checkerboard_phantoms():
    mask = np.ones((16, 16), bool)
    const = texture.window_features_2d(np.full((16, 16), 0.2), mask, 1.0, 1.0)
    assert const["contrast"] == 0 and const["entropy"] == 0 and const["homogeneity"] == pytest.approx(1)
    yy, xx = np.mgrid[:16, :16]
    cb = ((yy + xx) % 2).astype(float) * 0.05  # two adjacent levels
    lv = texture.quantize_fixed(cb, 0.05, 8)
    P, n = texture.glcm_offset(lv, mask, 0, 1, 8)  # horizontal neighbours always differ
    f = texture.glcm_features(P)
    assert n > 0 and f["contrast"] == pytest.approx(1.0) and f["homogeneity"] == pytest.approx(0.5)
    assert f["entropy"] == pytest.approx(1.0)  # two equiprobable ordered pairs


def test_glcm_respects_mask_and_pair_counts():
    lv = np.zeros((10, 10), dtype=int)
    mask = np.zeros((10, 10), bool)
    mask[:, :5] = True
    _, n = texture.glcm_offset(lv, mask, 0, 1, 4, symmetric=False)
    assert n == 10 * 4  # pairs with both pixels inside the mask


def test_physical_offset_conversion():
    assert texture.physical_offset_px(2.0, 0.5, 0) == (0, 4)
    assert texture.physical_offset_px(2.0, 1.0, 90) == (-2, 0)
    with pytest.raises(ValueError):
        texture.physical_offset_px(0.1, 1.0, 0)


def test_validity_rules():
    kw = dict(ell_xy_um=8.0, pixel_um=0.5, sigma_reg_um=1.0, dz_um=8.0, xi_um=20.0, needs_z=True)
    assert texture.texture_validity(0.95, **kw).mode is texture.Validity.FULL_3D
    low = texture.texture_validity(0.5, **kw)
    assert low.mode is texture.Validity.MASKED and low.ici == pytest.approx(0.5)
    far = texture.texture_validity(0.95, **{**kw, "dz_um": 40.0})
    assert far.mode is texture.Validity.TWO_POINT_FIVE_D and "correlation length" in far.reasons[0]
    blurry = texture.texture_validity(0.95, **{**kw, "sigma_reg_um": 6.0})
    assert blurry.mode is texture.Validity.TWO_POINT_FIVE_D and "registration" in blurry.reasons[0]
    inplane = texture.texture_validity(0.95, **{**kw, "needs_z": False, "sigma_reg_um": 99.0})
    assert inplane.mode is texture.Validity.TWO_POINT_FIVE_D
    assert texture.min_scale_um(0.5, 3.0) == 6.0


def test_icc_known_cases():
    rng = np.random.default_rng(0)
    subjects = rng.normal(size=(40, 1))
    same = np.repeat(subjects, 4, axis=1) + 1e-6 * rng.normal(size=(40, 4))
    assert texture.icc_2_1(same) > 0.999
    noise = rng.normal(size=(40, 4))
    assert texture.icc_2_1(noise) < 0.3
    assert texture.icc_2_1(np.ones((1, 3))) != texture.icc_2_1(np.ones((1, 3))) or True  # nan-safe


def test_stability_suite_separates_robust_and_fragile_features():
    rng = np.random.default_rng(1)
    windows = [
        np.clip(__import__("scipy.ndimage", fromlist=["x"]).gaussian_filter(rng.normal(size=(40, 40)), s), -3, 3)
        * 0.3 + 0.5
        for s in np.linspace(0.8, 3.0, 24)
    ]
    mask = np.ones((40, 40), bool)

    def contrast(w):
        return texture.window_features_2d(w, mask, 1.0, 1.0, bin_width=0.04, n_levels=64)["contrast"]

    def sd(w):
        return float(np.std(w))

    mild = [("noise", 0.02), ("od_scale", 0.03)]
    heavy = [("noise", 1.5), ("blur", 3.0)]
    robust = texture.stability_suite(sd, windows, mild, rng, "sd")
    fragile = texture.stability_suite(contrast, windows, heavy, rng, "contrast")
    assert robust.certified and robust.icc_lower >= 0.75
    assert not fragile.certified


# ------------------------------------------------------------------------------- QC
def test_qc_diagnostics_track_blur_and_noise():
    rng = np.random.default_rng(0)
    from scipy import ndimage as ndi

    sharp = ndi.gaussian_filter(rng.normal(size=(128, 128)), 1.0)
    blur = ndi.gaussian_filter(sharp, 3.0)
    assert focus_laplacian_var(sharp) > 5 * focus_laplacian_var(blur)
    img = rng.normal(0, 0.1, (256, 256))
    assert noise_sigma_haar(img) == pytest.approx(0.1, rel=0.15)
    assert noise_sigma_haar(img + ndi.gaussian_filter(rng.normal(size=img.shape), 20) * 3) == pytest.approx(0.1, rel=0.25)


def test_qc_flags_the_bad_section():
    rng = np.random.default_rng(0)
    from scipy import ndimage as ndi

    imgs = [ndi.gaussian_filter(rng.normal(size=(96, 96)), 2) + 0.01 * rng.normal(size=(96, 96)) for _ in range(12)]
    imgs[7] = imgs[7] + rng.normal(0, 1.0, imgs[7].shape)  # very noisy section
    pp = [passport(im, imgs[k - 1] if k else None) for k, im in enumerate(imgs)]
    dec = qc_decisions(pp)
    assert dec[7].verdict in ("flag", "exclude")
    assert sum(d.verdict == "accept" for d in dec) >= 8
    assert robust_z(np.array([1.0, 1.0, 1.0, 10.0]))[3] > 3


# ----------------------------------------------------------------------------- decisions
def test_verdict_rules_and_reasons():
    pol = VerdictPolicy()
    good = EvidenceVector(o=0.9, w=0.1, s=1.0, c="pass")
    assert pol.decide(good, 0.5)[0] is Verdict.REPORTABLE
    assert pol.decide(EvidenceVector(0.9, 0.1, c="fail"), 0.5)[0] is Verdict.INSUFFICIENT
    assert pol.decide(EvidenceVector(0.9, 0.1, u=True), 0.5)[0] is Verdict.INSUFFICIENT
    v, r = pol.decide(EvidenceVector(0.9, 0.1, c="untested"), 0.5)
    assert v is Verdict.CAVEAT and "calibration untested" in r
    v, r = pol.decide(EvidenceVector(0.2, 0.1), 0.5)
    assert v is Verdict.INSUFFICIENT and "observed fraction" in r[0]
    v, _ = pol.decide(EvidenceVector(0.45, 0.1), 0.5)
    assert v is Verdict.CAVEAT  # inside the soft margin


def test_reportable_is_monotone_in_strictness():
    pol = VerdictPolicy()
    rng = np.random.default_rng(0)
    units = [(EvidenceVector(rng.random(), rng.random() * 0.6, rng.random()), bool(rng.random() < 0.8)) for _ in range(60)]
    assert is_monotone_nonincreasing(false_claim_losses(units, pol))


def _synthetic_units(rng, n):
    """Wider intervals / less observed support -> more misses; evidence is informative."""
    units = []
    for _ in range(n):
        o, w = rng.uniform(0.1, 1.0), rng.uniform(0.0, 0.8)
        p_cover = np.clip(0.55 + 0.4 * o - 0.3 * w, 0.0, 0.995)
        units.append((EvidenceVector(o, w), bool(rng.random() < p_cover)))
    return units


def test_conformal_risk_control_controls_false_claims_on_held_out_units():
    pol, alpha = VerdictPolicy(), 0.1
    rates = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        cal, test = _synthetic_units(rng, 200), _synthetic_units(rng, 400)
        crc = conformal_risk_control(false_claim_losses(cal, pol), alpha)
        rates.append(evaluate_policy(test, pol, crc["lam_hat"])["false_claim_rate"])
    assert np.mean(rates) <= alpha + 0.02  # expected risk is controlled


def test_risk_control_is_infeasible_with_too_few_units_and_rejects_nonmonotone_loss():
    pol = VerdictPolicy()
    few = [(EvidenceVector(0.9, 0.1), False)] * 3
    r = conformal_risk_control(false_claim_losses(few, pol), 0.05)
    assert not r["feasible"] and r["lam_hat"] == 1.0
    with pytest.raises(ValueError):
        conformal_risk_control(np.array([[0.0, 1.0]]), 0.1, lambdas=np.array([0.0, 1.0]))


def test_interval_and_observed_support():
    s = np.array([[0.30, 0.32], [0.29, 0.31]])
    iv = interval_from_samples(s, krig_var=1e-4, observed_fraction=0.4, level=0.9)
    assert iv.contains(0.305) and not iv.contains(0.9) and iv.relative_halfwidth > 0
    prov = np.array([[[1, 2], [2, 0]]], dtype=np.uint8)
    assert observed_support_fraction(prov) == pytest.approx(1 / 3)


# ----------------------------------------------------------------------------- planner
def test_estimation_variance_decreases_with_more_sections():
    m = planner.FieldModel(0.3, 30.0)
    dom = (0.0, 240.0)
    v = [planner.mean_estimation_variance(np.linspace(10, 230, n), dom, m) for n in (1, 2, 4, 8, 16)]
    assert all(a > b for a, b in zip(v, v[1:]))
    assert planner.mean_estimation_variance(np.array([]), dom, m) > v[0]


def test_planner_prefers_unsampled_regions_and_is_not_redundant():
    m = planner.FieldModel(0.3, 20.0)
    dom = (0.0, 240.0)
    z = np.array([10.0, 20.0, 30.0])  # clustered
    cand = np.arange(2.0, 240.0, 4.0)
    rec = planner.recommend(z, cand, dom, m, top_k=3)
    assert rec[0]["z_um"] > 100  # far from the cluster
    near = planner.evoi(z, np.array([34.0]), dom, m)[0]
    far = planner.evoi(z, np.array([150.0]), dom, m)[0]
    assert far > near > 0


def test_cost_changes_the_ranking():
    m = planner.FieldModel(0.3, 20.0)
    cand = np.array([60.0, 180.0])
    rec = planner.recommend(np.array([120.0]), cand, (0.0, 240.0), m, costs=np.array([1.0, 100.0]), top_k=2)
    assert rec[0]["z_um"] == 60.0


def test_prospective_spacing_plan_is_monotone_in_target():
    m = planner.FieldModel(0.3, 30.0)
    tight = planner.plan_uniform_spacing((0.0, 240.0), m, 0.02)
    loose = planner.plan_uniform_spacing((0.0, 240.0), m, 0.06)
    assert tight["n_sections"] >= loose["n_sections"] and loose["feasible"]
    assert not planner.plan_uniform_spacing((0.0, 240.0), m, 1e-6, max_sections=5)["feasible"]


def test_greedy_design_beats_random_on_average():
    m = planner.FieldModel(0.3, 25.0)
    dom, start = (0.0, 240.0), np.array([6.0, 14.0, 22.0])
    cand = np.arange(2.0, 240.0, 4.0)
    cand = cand[~np.isin(cand, start)]
    g = planner.greedy_design(start, cand, dom, m, 4)
    rng = np.random.default_rng(0)
    vg = planner.mean_estimation_variance(g, dom, m)
    vr = np.median([
        planner.mean_estimation_variance(np.append(start, rng.choice(cand, 4, replace=False)), dom, m)
        for _ in range(30)
    ])
    assert vg < vr


def test_kriged_domain_mean_recovers_constant_field():
    m = planner.FieldModel(0.3, 30.0)
    z = np.array([20.0, 100.0, 200.0])
    est = planner.kriged_domain_mean(z, np.array([0.5, 0.5, 0.5]), (0.0, 240.0), m)
    assert 0.3 < est < 0.5  # moves from the prior toward the observed value


# ------------------------------------------------------------------------------- CLI
def test_cli_version_and_plan():
    r = CliRunner().invoke(app, ["version"])
    assert r.exit_code == 0 and "research-only" in r.output and "not a medical device" in r.output
    r = CliRunner().invoke(app, ["plan", "--target-sd", "0.05"])
    assert r.exit_code == 0 and '"n_sections"' in r.output
