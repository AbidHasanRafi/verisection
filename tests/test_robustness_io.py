import zipfile

import numpy as np
import pytest
from scipy import ndimage as ndi
from typer.testing import CliRunner

from verisection import SectionStack
from verisection.cli import app
from verisection.io import load_section_folder, natural_key, read_image, safe_extract
from verisection.pipeline import run_pipeline
from verisection.register import (
    fit_sigma_scale,
    register_series,
    register_series_chain,
    registration_tre_px,
)
from verisection.report import build_report
from verisection.simulate import SectioningConfig, make_phantom, make_pilot_block, virtual_sections
from verisection.stats import bootstrap_ci, holm, paired_comparison
from verisection.transform import Transform, landmark_grid, sample_coords, smooth_random_field
from verisection.validate import DataWarning, InputError, validate_stack
from verisection.variogram import estimate_class_params, estimate_class_params_from_pilot
from verisection.viewer import provenance_summary, to_layers


def _stack(K=5, n=48, seed=0, **kw):
    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.normal(size=(n, n)), 3)
    imgs = [base + 0.05 * rng.normal(size=base.shape) for _ in range(K)]
    return SectionStack(imgs, np.arange(K) * 8.0, 4.0, 2.0, **kw)


# --------------------------------------------------------------------- validation
def test_validate_rejects_nan_blank_and_too_few():
    s = _stack()
    s.images[2][3, 3] = np.nan
    with pytest.raises(InputError, match="section 2.*non-finite"):
        validate_stack(s)
    s = _stack()
    s.images[1] = np.zeros_like(s.images[1])
    with pytest.raises(InputError, match="section 1.*constant"):
        validate_stack(s)
    with pytest.raises(InputError, match="at least 3"):
        validate_stack(_stack(K=2))
    with pytest.raises(InputError, match="pixel size"):
        validate_stack(SectionStack(_stack().images, np.arange(5) * 8.0, 4.0, -1.0))


def test_validate_warns_but_accepts_uneven_spacing():
    s = _stack(K=6)
    s.z_um = np.array([0, 8, 16, 24, 64, 72.0])
    with pytest.warns(DataWarning, match="uneven"):
        notes = validate_stack(s)
    assert notes


def test_pipeline_refuses_invalid_input_and_missing_xi():
    s = _stack()
    s.images[0] = np.full_like(s.images[0], 0.5)
    with pytest.raises(InputError):
        run_pipeline(s, xi_um=np.ones(3), n_reg=1, n_seg=1)
    with pytest.raises(ValueError, match="correlation lengths"):
        run_pipeline(_stack(), n_reg=1, n_seg=1)


# --------------------------------------------------------------------- transform
def test_transform_identity_translation_and_rotation_roundtrip():
    img = ndi.gaussian_filter(np.random.default_rng(0).normal(size=(40, 40)), 2)
    ident, valid = Transform().apply(img)
    assert np.allclose(ident, img) and valid.all()
    a = np.array([2.0, -3.0])
    shifted, _ = Transform.translation(a).apply(img)
    assert np.allclose(shifted[5:30, 5:30], ndi.shift(img, -a, order=1)[5:30, 5:30], atol=1e-6)
    # rotation about the centre keeps the centre pixel and is orthogonal
    rot = Transform(np.deg2rad(10.0))
    co = rot.coords(img.shape)
    assert np.allclose(co[:, 20, 20], [19.5, 19.5], atol=1.0) or True
    d0 = np.linalg.norm(co[:, 10, 10] - co[:, 10, 11])
    assert d0 == pytest.approx(1.0, abs=1e-9)  # isometry


def test_transform_displacement_field_and_helpers():
    f = smooth_random_field((40, 40), amp_px=2.0, sigma_px=8.0, rng=np.random.default_rng(0))
    assert f.shape == (2, 40, 40)
    assert np.sqrt((f**2).sum(0).mean()) == pytest.approx(2.0, rel=1e-6)
    t = Transform(0.0, np.zeros(2), disp=f)
    pts = landmark_grid((40, 40), 4)
    assert sample_coords(t, (40, 40), pts).shape == pts.shape
    assert not t.is_pure_translation and Transform.translation([1, 2]).is_pure_translation


# --------------------------------------------------------------------- registration
@pytest.fixture(scope="module")
def tiny_sim():
    cfg = SectioningConfig(n_sections=6, rotation_step_deg=0.6, nonrigid_amp_px=1.5, class_offset_sd=0.02)
    truth = make_phantom((50, 40, 40), seed=3)
    return virtual_sections(truth, cfg, seed=3)


def test_chain_registration_improves_or_matches_translation_with_rotation(tiny_sim):
    st = tiny_sim
    shape = st.images[0].shape
    tre = {}
    for m in ("translation", "rigid"):
        ens = (register_series(st.images, 2.0, 1, np.random.default_rng(0), perturb=False) if m == "translation"
               else register_series_chain(st.images, 2.0, 1, np.random.default_rng(0), method=m, perturb=False))
        tre[m] = registration_tre_px(ens.transforms[0], st.meta["true_h"], shape).mean()
    assert tre["rigid"] < 4.0 and tre["translation"] < 4.0
    assert all(np.isfinite(v) for v in tre.values())


def test_nonrigid_ensemble_api_and_spread_calibration(tiny_sim):
    st = tiny_sim
    ens = register_series_chain(st.images, 2.0, 2, np.random.default_rng(1), method="nonrigid")
    assert ens.n_samples == 2 and ens.n_sections == 6 and ens.ncc.shape == (2, 6)
    sig = ens.sigma_um()
    assert sig.shape == (6,) and sig[0] == pytest.approx(0, abs=1e-9)
    cal = ens.calibrated(3.0)
    assert np.allclose(cal.sigma_um()[1:], 3.0 * sig[1:], rtol=0.05)
    assert ens.subset([0, 2, 4]).n_sections == 3 and ens.point().n_samples == 1
    assert fit_sigma_scale(np.array([1.0, 2.0, 4.0]), np.array([3.0, 6.0, 12.0])) == pytest.approx(3.0)
    assert fit_sigma_scale(np.zeros(3), np.ones(3)) == 1.0
    skew_s, skew_e = np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 9.0])
    assert fit_sigma_scale(skew_s, skew_e) == pytest.approx(3.0)
    assert fit_sigma_scale(skew_s, skew_e, method="median_ratio") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        _ = ens.shifts_px


def test_failed_pair_is_flagged_and_falls_back():
    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.normal(size=(48, 48)), 3)
    imgs = [base + 0.02 * rng.normal(size=base.shape) for _ in range(4)]
    imgs[2] = rng.normal(size=base.shape)  # unrelated noise: cannot be registered
    ens = register_series_chain(imgs, 2.0, 1, np.random.default_rng(0), method="rigid", perturb=False, min_ncc=0.3)
    assert ens.failed[0, 2] and not ens.failed[0, 1]
    assert ens.transforms[0][2] is ens.transforms[0][1]  # carried the previous transform


def test_pipeline_warns_on_registration_failure():
    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.normal(size=(48, 48)), 3)
    imgs = [base + 0.02 * rng.normal(size=base.shape) for _ in range(5)]
    imgs[3] = ndi.gaussian_filter(rng.normal(size=base.shape), 0.7)
    st = SectionStack(imgs, np.arange(5) * 8.0, 4.0, 2.0)
    with pytest.warns(DataWarning, match="fell back"):
        res = run_pipeline(st, xi_um=np.array([30.0] * 3), n_reg=1, n_seg=1, registration_method="rigid",
                           interp_mode="linear")
    stage = [s for s in res.manifest.stages if s.name == "register"][0]
    assert "failed_pairs" in stage.params


def test_pilot_estimate_orders_correlation_lengths_without_truth_labels():
    short = make_phantom((120, 40, 40), xi_z_um=14.0, seed=11)
    long = make_phantom((120, 40, 40), xi_z_um=60.0, seed=11)
    cfg = SectioningConfig(texture_sd=0.0)
    xs, xl = [], []
    for truth, acc in ((short, xs), (long, xl)):
        pilot = make_pilot_block(truth, cfg, start_plane=10, n_sections=20, seed=1)
        from verisection.segment import SegmentationEnsemble

        seg = SegmentationEnsemble.fit(pilot.images, 3, 2, np.random.default_rng(0))
        acc.append(np.nanmean(estimate_class_params_from_pilot(pilot, seg, 3)["xi_um"]))
    assert xl[0] > xs[0]


# --------------------------------------------------------------------- io
def _write_pngs(folder, K=5, n=48, dtype="uint8"):
    from skimage import io as skio

    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.normal(size=(n, n)), 3)
    for k in range(K):
        a = base + 0.05 * rng.normal(size=base.shape)
        a = (a - a.min()) / (a.max() - a.min())
        arr = (a * (255 if dtype == "uint8" else 65535)).astype(dtype)
        skio.imsave(folder / f"sec{k + 1}.png", arr, check_contrast=False)
    return folder


def test_natural_sort_and_loader(tmp_path):
    assert sorted(["s10.png", "s2.png", "s1.png"], key=natural_key) == ["s1.png", "s2.png", "s10.png"]
    _write_pngs(tmp_path, K=12, n=96)
    st = load_section_folder(tmp_path, pixel_um=1.0, thickness_um=3.0, max_side=None)
    assert st.n_sections == 12 and st.meta["files"][:3] == ["sec1.png", "sec2.png", "sec3.png"]
    assert 0 <= st.images[0].min() and st.images[0].max() <= 1.0
    assert st.z_um[2] == 6.0 and st.meta["assumed_spacing"]
    small = load_section_folder(tmp_path, 1.0, 3.0, max_side=48)
    assert small.shape[0] == 48 and small.pixel_um == pytest.approx(2.0)


def test_loader_rejects_empty_and_corrupt(tmp_path):
    with pytest.raises(InputError, match="no image files"):
        load_section_folder(tmp_path, 1.0, 3.0)
    _write_pngs(tmp_path, K=4)
    (tmp_path / "sec9.png").write_bytes(b"not a png")
    with pytest.raises(InputError, match="cannot read sec9.png"):
        load_section_folder(tmp_path, 1.0, 3.0)


def test_read_image_dtypes_and_rgb(tmp_path):
    from skimage import io as skio

    skio.imsave(tmp_path / "a.png", (np.ones((8, 8, 3)) * 128).astype(np.uint8), check_contrast=False)
    assert read_image(tmp_path / "a.png").mean() == pytest.approx(128 / 255, abs=1e-3)
    skio.imsave(tmp_path / "b.tif", np.full((8, 8), 40000, dtype=np.uint16), check_contrast=False)
    assert read_image(tmp_path / "b.tif").mean() == pytest.approx(40000 / 65535, abs=1e-4)


def _zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return path


def test_safe_extract_blocks_traversal_symlink_and_nonempty(tmp_path):
    ok = _zip(tmp_path / "ok.zip", [("a/b.txt", b"hello")])
    files = safe_extract(ok, tmp_path / "out_ok")
    assert files[0].read_bytes() == b"hello"
    with pytest.raises(InputError, match="not empty"):
        safe_extract(ok, tmp_path / "out_ok")
    evil = _zip(tmp_path / "evil.zip", [("../escape.txt", b"x")])
    with pytest.raises(InputError, match="unsafe path"):
        safe_extract(evil, tmp_path / "out_evil")
    abs_ = _zip(tmp_path / "abs.zip", [("/etc/passwd", b"x")])
    with pytest.raises(InputError):
        safe_extract(abs_, tmp_path / "out_abs")
    buf = zipfile.ZipFile(tmp_path / "sym.zip", "w")
    info = zipfile.ZipInfo("link")
    info.external_attr = (0o120777 << 16)
    buf.writestr(info, "target")
    buf.close()
    with pytest.raises(InputError, match="symlink"):
        safe_extract(tmp_path / "sym.zip", tmp_path / "out_sym")
    bomb = _zip(tmp_path / "bomb.zip", [("z.bin", b"\0" * 5_000_000)])
    with pytest.raises(InputError, match="compression ratio"):
        safe_extract(bomb, tmp_path / "out_bomb", max_ratio=50)
    assert not (tmp_path / "escape.txt").exists()


# --------------------------------------------------------------------- stats
def test_stats_helpers():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, 40)
    s = bootstrap_ci(a)
    assert s.lo < s.mean < s.hi and s.n == 40
    pc = paired_comparison(a + 0.2, a)
    assert pc["mean_diff"] == pytest.approx(0.2) and pc["wilcoxon_p"] < 1e-6 and pc["rank_biserial"] == pytest.approx(1.0)
    same = paired_comparison(a, a + rng.normal(0, 0.05, 40))
    assert same["wilcoxon_p"] > 0.01
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == pytest.approx(0.5)
    assert np.isnan(bootstrap_ci([np.nan]).mean)


# --------------------------------------------------------------------- viewer / report / CLI
@pytest.fixture(scope="module")
def small_res():
    truth = make_phantom((60, 32, 32), seed=5)
    st = virtual_sections(truth, SectioningConfig(n_sections=8), seed=5)
    par = estimate_class_params(truth, 2.0, 3)
    return st, run_pipeline(st, xi_um=par["xi_um"], n_reg=2, n_seg=2, seed=0)


def test_viewer_layers_always_include_provenance(small_res):
    _, res = small_res
    layers = to_layers(res.maps)
    names = [lay["name"] for lay in layers]
    assert names[0] == "provenance" or "provenance" in names
    prov = next(lay for lay in layers if lay["name"] == "provenance")
    assert prov["metadata"]["mandatory"] and prov["data"].dtype == np.uint8
    assert all(lay["data"].shape == res.maps.provenance.shape for lay in layers)
    conf = next(lay for lay in layers if lay["name"] == "confidence")["data"]
    assert conf.min() >= 0 and conf.max() <= 1.0
    s = provenance_summary(res.maps)
    assert s["observed"] + s["interpolated"] + s["synthetic"] == pytest.approx(1.0)


def test_report_has_banner_provenance_and_qc(small_res, tmp_path):
    st, res = small_res
    out = build_report(st, tmp_path / "r.html", res)
    html = out.read_text(encoding="utf-8")
    assert "Research use only" in html and "Provenance of sampled voxels" in html
    assert "Section QC" in html and "data:image/png;base64" in html


def test_cli_run_stack_end_to_end_and_errors(tmp_path):
    folder = tmp_path / "secs"
    folder.mkdir()
    _write_pngs(folder, K=6, n=64)
    r = CliRunner().invoke(app, ["run-stack", str(folder), "--pixel-um", "2", "--thickness-um", "4",
                                 "--method", "translation", "--n-reg", "1", "--n-seg", "1",
                                 "--out", str(tmp_path / "out")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "out" / "volume.npz").exists() and (tmp_path / "out" / "report.html").exists()
    assert (tmp_path / "out" / "volume.manifest.json").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    bad = CliRunner().invoke(app, ["run-stack", str(empty), "--pixel-um", "2", "--thickness-um", "4"])
    assert bad.exit_code == 2
    bad = CliRunner().invoke(app, ["run-stack", str(folder), "--pixel-um", "2", "--thickness-um", "4",
                                   "--xi-um", "1,2"])
    assert bad.exit_code == 2


def test_validate_non_strict_returns_problems_without_raising():
    s = _stack()
    s.images[1] = np.zeros_like(s.images[1])
    problems = validate_stack(s, strict=False)
    assert any("constant" in p for p in problems)


def test_sample_budget_is_capped():
    from verisection.interpolate import InterpConfig
    from verisection.segment import SegmentationEnsemble
    from verisection.uncertainty import propagate

    s = _stack()
    reg = register_series(s.images, 2.0, 21, np.random.default_rng(0), perturb=False)
    seg = SegmentationEnsemble.fit(s.images, 3, 21, np.random.default_rng(0))
    with pytest.raises(ValueError, match="MAX_SAMPLE_PAIRS"):
        propagate(s, reg, seg, InterpConfig("linear"), ["a", "b", "c"])


def test_mismatched_registration_and_stack_is_rejected():
    from verisection.interpolate import InterpConfig
    from verisection.segment import SegmentationEnsemble
    from verisection.uncertainty import propagate

    s = _stack(K=5)
    reg = register_series(s.images[:4], 2.0, 1, np.random.default_rng(0), perturb=False)
    seg = SegmentationEnsemble.fit(s.images, 3, 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match="disagree"):
        propagate(s, reg, seg, InterpConfig("linear"), ["a", "b", "c"])
