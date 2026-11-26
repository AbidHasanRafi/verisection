import numpy as np
import pytest
from scipy import ndimage as ndi

from verisection.interpolate import interpolate_plane, kriging_weights_two
from verisection.pipeline import run_pipeline
from verisection.provenance import Provenance
from verisection.register import (
    pairwise_shift,
    register_series,
    registration_error_px,
    solve_pose_graph,
)
from verisection.segment import SegmentationEnsemble
from verisection.simulate import SectioningConfig, make_phantom, virtual_sections
from verisection.variogram import estimate_class_params, indicator_variogram_z


# ----------------------------------------------------------------------- simulation
def test_phantom_is_deterministic_and_has_requested_fractions():
    a = make_phantom((40, 32, 32), seed=3)
    b = make_phantom((40, 32, 32), seed=3)
    assert np.array_equal(a, b)
    assert np.allclose(np.bincount(a.ravel()) / a.size, [0.5, 0.3, 0.2], atol=0.01)


def test_longer_correlation_length_gives_longer_fitted_range():
    short = make_phantom((120, 32, 32), xi_z_um=12.0, seed=1)
    long = make_phantom((120, 32, 32), xi_z_um=60.0, seed=1)
    xs = estimate_class_params(short, 2.0, 3)["xi_um"]
    xl = estimate_class_params(long, 2.0, 3)["xi_um"]
    assert np.nanmean(xl) > 1.5 * np.nanmean(xs)


def test_variogram_starts_near_zero_and_rises():
    lab = make_phantom((100, 32, 32), seed=2)
    lags, g = indicator_variogram_z(lab, 2.0, 0, 30)
    assert g[0] < g[-1] and g[0] < 0.1


# ----------------------------------------------------------------------- interpolation
def test_kriging_weights_sum_and_limits():
    w, var = kriging_weights_two(1.0, 11.0, 12.0, xi=30.0, sill=0.2, nugget=0.01)
    assert w[0] > w[1] > 0  # closer observation weighs more
    assert 0 <= var <= 0.2
    w_far, var_far = kriging_weights_two(500.0, 500.0, 1000.0, xi=10.0, sill=0.2, nugget=0.01)
    assert np.all(np.abs(w_far) < 1e-6) and var_far == pytest.approx(0.2, abs=1e-6)


def _planes(C=3, H=4, W=4):
    p_lo = np.zeros((C, H, W), np.float32)
    p_lo[0] = 1.0
    p_hi = np.zeros((C, H, W), np.float32)
    p_hi[1] = 1.0
    v = np.ones((H, W), bool)
    return p_lo, p_hi, v


def test_interpolators_return_normalised_posteriors_and_use_nearest():
    p_lo, p_hi, v = _planes()
    pri = np.array([0.5, 0.3, 0.2])
    xi = np.array([30.0, 30.0, 30.0])
    for mode in ("nearest", "linear", "kriging"):
        out, valid = interpolate_plane(mode, p_lo, p_hi, v, v, 2.0, 10.0, pri, xi)
        assert np.allclose(out.sum(0), 1.0, atol=1e-5) and valid.all()
    near, _ = interpolate_plane("nearest", p_lo, p_hi, v, v, 2.0, 10.0, pri, xi)
    assert near[0].mean() > 0.99
    lin, _ = interpolate_plane("linear", p_lo, p_hi, v, v, 6.0, 6.0, pri, xi)
    assert lin[0].mean() == pytest.approx(lin[1].mean(), abs=1e-3)


def test_kriging_reverts_to_prior_far_from_observations():
    p_lo, p_hi, v = _planes()
    pri = np.array([0.5, 0.3, 0.2])
    out, _ = interpolate_plane("kriging", p_lo, p_hi, v, v, 400.0, 400.0, pri, np.array([10.0] * 3))
    assert np.allclose(out.mean(axis=(1, 2)), pri, atol=1e-3)
    near, _ = interpolate_plane("kriging", p_lo, p_hi, v, v, 1.0, 20.0, pri, np.array([30.0] * 3))
    assert near[0].mean() > 0.8  # close to a class-0 slab: confident


def test_interpolation_uses_one_sided_validity():
    p_lo, p_hi, v = _planes()
    inv = np.zeros_like(v)
    out, valid = interpolate_plane("linear", p_lo, p_hi, v, inv, 5.0, 5.0, np.ones(3) / 3, None)
    assert valid.all() and out[0].mean() > 0.99
    _, valid2 = interpolate_plane("linear", p_lo, p_hi, inv, inv, 5.0, 5.0, np.ones(3) / 3, None)
    assert not valid2.any()


# ----------------------------------------------------------------------- segmentation
def test_segmentation_ensemble_separates_intensity_classes():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 3, size=(40, 40))
    img = np.array([0.2, 0.55, 0.9])[lab] + rng.normal(0, 0.05, lab.shape)
    ens = SegmentationEnsemble.fit([img, img + 0.01], 3, 4, rng)
    p = ens.probs(img, 1)
    assert np.allclose(p.sum(0), 1.0, atol=1e-5)
    assert (p.argmax(0) == lab).mean() > 0.97
    assert ens.point().n_members == 1


# ----------------------------------------------------------------------- registration
def test_pairwise_shift_recovers_known_translation():
    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.normal(size=(96, 96)), 3)
    mov = ndi.shift(base, (3.0, -2.0), order=1, mode="nearest")
    # ref=base, mov=base shifted by a; the shift that registers mov to ref is -a
    est = pairwise_shift(base, mov)
    assert np.allclose(est, (-3.0, 2.0), atol=0.35)


def test_pose_graph_recovers_positions_from_exact_pairs():
    rng = np.random.default_rng(1)
    K = 8
    a = np.vstack([[0, 0], np.cumsum(rng.normal(0, 1, (K - 1, 2)), 0)])
    pairs = {(k, k + 1): a[k] - a[k + 1] for k in range(K - 1)}
    pairs |= {(k, k + 2): a[k] - a[k + 2] for k in range(K - 2)}
    w = {k: 1.0 for k in pairs}
    assert np.abs(solve_pose_graph(pairs, w, K) - a).max() < 1e-6


def test_pose_graph_downweights_a_gross_outlier():
    K = 8
    a = np.vstack([[0, 0], np.cumsum(np.random.default_rng(2).normal(0, 1, (K - 1, 2)), 0)])
    pairs = {(k, k + 1): a[k] - a[k + 1] for k in range(K - 1)}
    pairs |= {(k, k + 2): a[k] - a[k + 2] for k in range(K - 2)}
    pairs[(2, 4)] = pairs[(2, 4)] + np.array([40.0, -40.0])  # failed correlation
    w = {k: 1.0 for k in pairs}
    assert np.abs(solve_pose_graph(pairs, w, K) - a).max() < 0.5


def test_register_series_on_simulated_stack():
    truth = make_phantom((60, 40, 40), seed=4)
    st = virtual_sections(truth, SectioningConfig(n_sections=8), seed=4)
    ens = register_series(st.images, st.pixel_um, 3, np.random.default_rng(0))
    assert ens.shifts_px.shape == (3, 8, 2) and np.allclose(ens.shifts_px[:, 0], 0)
    err = registration_error_px(ens.mean_shifts(), st.meta["true_shifts_px"])
    assert err.mean() < 3.0
    assert ens.sigma_um().shape == (8,) and ens.point().n_samples == 1


# ----------------------------------------------------------------------- pipeline / uncertainty
@pytest.fixture(scope="module")
def small_result():
    truth = make_phantom((60, 32, 32), seed=5)
    st = virtual_sections(truth, SectioningConfig(n_sections=8), seed=5)
    par = estimate_class_params(truth, 2.0, 3)
    return run_pipeline(st, xi_um=par["xi_um"], n_reg=2, n_seg=2, seed=0)


def test_pipeline_posteriors_and_provenance(small_result):
    m = small_result.maps
    valid = m.provenance > 0
    assert np.allclose(m.p_hat.sum(0)[valid], 1.0, atol=1e-3)
    assert set(np.unique(m.provenance)) <= {0, 1, 2, 3}
    assert (m.provenance == Provenance.OBSERVED).any() and (m.provenance == Provenance.INTERPOLATED).any()
    assert not (m.provenance == Provenance.SYNTHETIC).any()  # Mode A never produces Synthetic
    assert (m.u_total >= -1e-6).all() and (m.mutual_information >= 0).all()
    assert (m.expected_entropy >= -1e-6).all() and (m.var_total >= -1e-9).all()


def test_gap_planes_are_more_uncertain_than_observed_planes(small_result):
    m = small_result.maps
    obs = m.provenance == Provenance.OBSERVED
    gap = m.provenance == Provenance.INTERPOLATED
    assert m.u_total[gap].mean() > m.u_total[obs].mean()


def test_single_sample_has_no_sample_variance():
    truth = make_phantom((60, 32, 32), seed=6)
    st = virtual_sections(truth, SectioningConfig(n_sections=8), seed=6)
    par = estimate_class_params(truth, 2.0, 3)
    res = run_pipeline(st, xi_um=par["xi_um"], n_reg=1, n_seg=1, seed=0)
    assert np.allclose(res.maps.var_total, 0) and np.allclose(res.maps.mutual_information, 0, atol=1e-6)


def test_measurement_functions_are_evaluated_on_every_sample():
    from verisection.measure import class_volume_fraction

    truth = make_phantom((60, 32, 32), seed=7)
    st = virtual_sections(truth, SectioningConfig(n_sections=8), seed=7)
    par = estimate_class_params(truth, 2.0, 3)
    res = run_pipeline(
        st, xi_um=par["xi_um"], n_reg=2, n_seg=3, seed=0,
        measure_fns={"f": lambda p, prov: class_volume_fraction(p, prov, 1)},
    )
    f = res.maps.measurements["f"]
    assert f.shape == (2, 3) and np.all((f > 0) & (f < 1))


def test_kriging_requires_correlation_lengths():
    truth = make_phantom((60, 32, 32), seed=8)
    st = virtual_sections(truth, SectioningConfig(n_sections=8), seed=8)
    with pytest.raises(ValueError):
        run_pipeline(st, xi_um=None, n_reg=1, n_seg=1)


def test_export_roundtrip_has_provenance_and_manifest(small_result, tmp_path):
    out = small_result.export(tmp_path / "vol.npz")
    assert out.exists() and out.with_suffix(".manifest.json").exists()
    with np.load(out) as z:
        assert z["provenance"].shape == small_result.maps.provenance.shape
