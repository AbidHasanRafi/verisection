import numpy as np
import pytest

from verisection.experiments import run_replicate, texture_h8_replicate
from verisection.pipeline import run_pipeline
from verisection.real_eval import real_subsample_experiment
from verisection.segment import SegmentationEnsemble
from verisection.simulate import SectioningConfig, make_phantom, make_pilot_block, virtual_sections
from verisection.validate import DataWarning
from verisection.variogram import estimate_class_params, estimate_class_params_from_pilot


def test_manifest_records_what_actually_ran():
    truth = make_phantom((60, 32, 32), seed=2)
    st = virtual_sections(truth, SectioningConfig(n_sections=7), seed=2)
    par = estimate_class_params(truth, 2.0, 3)
    from verisection.register import register_series

    reg = register_series(st.images, 2.0, 2, np.random.default_rng(0))
    seg = SegmentationEnsemble.fit(st.images, 3, 3, np.random.default_rng(0))
    res = run_pipeline(st, xi_um=par["xi_um"], reg=reg, seg=seg)  # caller-supplied ensembles
    cfg = res.manifest.config
    assert cfg["n_reg"] == 2 and cfg["n_seg"] == 3 and cfg["registration_method"] == "translation"


def test_pilot_flags_correlation_lengths_longer_than_the_sampled_extent():
    truth = make_phantom((120, 40, 40), xi_z_um=200.0, seed=4)  # far longer than a short pilot
    cfg = SectioningConfig()
    pilot = make_pilot_block(truth, cfg, start_plane=10, n_sections=8, seed=4)
    seg = SegmentationEnsemble.fit(pilot.images, 3, 2, np.random.default_rng(0))
    with pytest.warns(DataWarning, match="exceed the pilot extent|could not be fitted"):
        out = estimate_class_params_from_pilot(pilot, seg, 3)
    assert not out["reliable"].all()


def test_experiment_replicate_smoke_with_calibrated_spread():
    r = run_replicate(3, preset="easy", method="translation", M=2, N=2, sigma_scale=2.0, full=True)
    assert set(r["variants"]) == {"joint_raw", "joint_cal", "seg_only", "reg_only_cal", "point_linear", "point_kriging"}
    for v in r["variants"].values():
        assert 0 <= v["all"]["accuracy"] <= 1 and np.isfinite(v["all"]["ece"])
    assert len(r["tre_um"]) == len(r["sigma_um"]) and np.isfinite(r["xi_rel_error"])
    cal_only = run_replicate(3, preset="easy", method="translation", M=2, N=2, full=False)
    assert "variants" not in cal_only


def test_texture_h8_observed_planes_are_unbiased_and_naive_interpolation_is_not():
    rows = [texture_h8_replicate(s, 6) for s in range(10, 16)]
    truth = np.array([r["truth"] for r in rows])
    obs = np.array([r["observed"] for r in rows])
    naive = np.array([r["naive_interp"] for r in rows])
    obs_err = np.abs((obs - truth) / truth).mean(axis=0)
    naive_err = np.abs((naive - truth) / truth).mean(axis=0)
    assert np.all(obs_err < 0.05)
    assert naive_err[0] > 5 * obs_err[0]  # contrast is strongly biased by interpolation


def test_real_subsample_experiment_on_a_simulated_dense_stack():
    cfg = SectioningConfig(n_sections=17, spacing_planes=2, first_plane=2, thickness_um=4.0, drift_step_px=0.3,
                           jitter_px=0.1)
    truth = make_phantom((60, 40, 40), seed=8)
    st = virtual_sections(truth, cfg, seed=8)
    r = real_subsample_experiment(st, step=4, n_classes=3, method="translation", M=1, N=2)
    assert set(r["variants"]) == {"kriging", "linear", "nearest"}
    assert r["n_kept"] == 5 and r["n_heldout"] == 12
    for v in r["variants"].values():
        assert 0.4 < v["agreement_with_heldout_segmentation"] <= 1.0
        assert 0 <= v["ece"] < 0.5
        assert set(v["by_distance_to_nearest_kept_section"]) <= {"1", "2"}
    with pytest.raises(ValueError):
        real_subsample_experiment(st.subset(range(5)), step=4)


def test_conservative_xi_clamps_unreliable_lengths_to_the_sampled_extent():
    from verisection.variogram import conservative_xi

    par = {"xi_um": np.array([20.0, 500.0, np.nan]), "reliable": np.array([True, False, False]),
           "max_lag_um": np.array(60.0)}
    assert conservative_xi(par).tolist() == [20.0, 60.0, 60.0]
    with pytest.raises(ValueError):
        conservative_xi({"xi_um": np.array([1.0]), "reliable": np.array([False]), "max_lag_um": np.array(0.0)})


def test_real_conformal_blocked_split_on_a_simulated_dense_stack():
    from verisection.real_eval import real_conformal_experiment

    cfg = SectioningConfig(n_sections=25, spacing_planes=2, first_plane=2, thickness_um=4.0, drift_step_px=0.3,
                           jitter_px=0.1)
    truth = make_phantom((60, 40, 40), seed=9)
    st = virtual_sections(truth, cfg, seed=9)
    r = real_conformal_experiment(st, step=4, alpha=0.2, method="translation", M=1, N=2)
    assert set(r["directions"]) == {"first->second", "second->first"}
    for d in r["directions"].values():
        assert 0.0 <= d["coverage_mean"] <= 1.0 and d["mean_set_size"] >= 0.0  # empty sets are allowed
    assert 0.0 <= r["coverage_mean_over_directions"] <= 1.0
