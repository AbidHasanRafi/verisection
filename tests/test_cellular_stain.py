import numpy as np
import pytest

from verisection import cellular as C
from verisection import stain


# ------------------------------------------------------------------ nuclei
def _field(seed=0, box=(160.0, 160.0, 160.0), spacing=4.0, t=4.0, jitter=0.0, miss=0.0):
    rng = np.random.default_rng(seed)
    nuc = C.simulate_nuclei(box, seed=seed)
    z0 = 20.0 + np.arange(int((box[0] - 40) // spacing)) * spacing
    m = 15.0
    profs = []
    for k, z in enumerate(z0):
        p = C.cut_section(nuc, z, t, k, rng, centroid_jitter_um=jitter, miss_rate=miss)
        sel = ((p.xy[:, 0] > m) & (p.xy[:, 0] < box[1] - m) & (p.xy[:, 1] > m) & (p.xy[:, 1] < box[2] - m))
        profs.append(C.Profiles(p.section, p.xy[sel], p.radius[sel], p.ids[sel]))
    c = nuc.centers
    lo, hi = z0[0] - spacing / 2, z0[-1] + spacing / 2
    truth = int(((c[:, 0] >= lo) & (c[:, 0] < hi) & (c[:, 1] > m) & (c[:, 1] < box[1] - m)
                 & (c[:, 2] > m) & (c[:, 2] < box[2] - m)).sum())
    return nuc, profs, truth, (box[1] - 2 * m) * (box[2] - 2 * m)


def test_simulation_is_deterministic_and_profiles_follow_geometry():
    a = C.simulate_nuclei(seed=3)
    b = C.simulate_nuclei(seed=3)
    assert np.array_equal(a.centers, b.centers) and a.centers.shape[1] == 3
    rng = np.random.default_rng(0)
    nuc = C.Nuclei(np.array([[50.0, 20.0, 20.0]]), np.array([[3.0, 4.0, 4.0]]))
    mid = C.cut_section(nuc, 50.0, 4.0, 0, rng, centroid_jitter_um=0.0, miss_rate=0.0)
    assert len(mid.xy) == 1 and mid.radius[0] == pytest.approx(4.0, rel=0.1)  # widest section
    far = C.cut_section(nuc, 70.0, 4.0, 1, rng, miss_rate=0.0)
    assert len(far.xy) == 0
    cap = C.cut_section(nuc, 50.0 + 3.0 + 2.0 - 0.01, 4.0, 2, rng, centroid_jitter_um=0.0, miss_rate=0.0)
    assert len(cap.xy) == 0  # a barely-clipped cap is below the minimum visible radius


def test_perfect_linking_without_noise_and_gating():
    nuc, profs, truth, area = _field(seed=1, jitter=0.0)
    tracks = C.link_series(profs, tolerance_um=0.3)
    sc = C.linking_scores(tracks, profs)
    assert sc["pair_precision"] > 0.9 and sc["pair_recall"] > 0.85
    a = C.Profiles(0, np.array([[10.0, 10.0]]), np.array([4.0]), np.array([0]))
    far = C.Profiles(1, np.array([[60.0, 60.0]]), np.array([4.0]), np.array([0]))
    assert C.link_pair(a, far, 0.3) == []
    big = C.Profiles(1, np.array([[10.0, 10.5]]), np.array([20.0]), np.array([0]))
    assert C.link_pair(a, big, 0.3) == []  # size ratio gate
    near = C.Profiles(1, np.array([[10.0, 10.5]]), np.array([4.1]), np.array([0]))
    (i, j, p), = C.link_pair(a, near, 0.3)
    assert (i, j) == (0, 0) and 0.5 < p <= 1.0
    assert C.link_pair(C.Profiles(0, np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)), near, 0.3) == []


def test_linking_degrades_gracefully_with_misregistration():
    f1 = []
    for sigma in (0.3, 3.0):
        _, profs, _, _ = _field(seed=2, jitter=0.0)
        rng = np.random.default_rng(0)
        for p in profs:
            p.xy = p.xy + rng.normal(0, sigma, 2)
        f1.append(C.linking_scores(C.link_series(profs, np.sqrt(0.09 + sigma**2)), profs)["pair_f1"])
    assert f1[0] > f1[1] > 0.0


def test_count_estimators_against_simulated_truth():
    errs = {"naive": [], "abercrombie": [], "tracks_ht": []}
    for seed in range(4):
        nuc, profs, truth, area = _field(seed=seed + 10, jitter=0.1, miss=0.02)
        est = C.estimate_counts(profs, 4.0, 4.0, area, 0.35, caliper_um=2 * nuc.semi_axes[:, 0].mean())
        for k in errs:
            errs[k].append(est[k] / truth - 1)
    assert np.mean(errs["naive"]) > 1.0  # profiles over-count by more than 2x
    assert abs(np.mean(errs["abercrombie"])) < 0.12
    assert abs(np.mean(errs["tracks_ht"])) < 0.25


def test_abercrombie_formula_and_caliper_estimate():
    assert C.abercrombie_count(100, 4.0, 6.0) == pytest.approx(40.0)
    p = [C.Profiles(0, np.zeros((2, 2)), np.array([3.0, 5.0]), np.array([0, 1]))]
    assert C.estimate_caliper_from_profiles(p) == pytest.approx(8.0)
    assert C.estimate_caliper_from_profiles([C.Profiles(0, np.zeros((0, 2)), np.zeros(0), np.zeros(0, int))]) == 0.0


# -------------------------------------------------------------------- stain
def _he_image(h=24, w=24, seed=0):
    rng = np.random.default_rng(seed)
    conc = np.zeros((h, w, 3))
    conc[..., 0] = rng.uniform(0.2, 1.2, (h, w))  # hematoxylin
    conc[..., 1] = rng.uniform(0.2, 1.2, (h, w))  # eosin
    od = conc @ stain.RUIFROK_HE
    return conc, od


def test_deconvolution_recovers_known_concentrations():
    conc, od = _he_image()
    est = stain.deconvolve(od, stain.RUIFROK_HE)
    assert np.allclose(est[..., :2], conc[..., :2], atol=1e-6)
    assert stain.reconstruction_residual(od, stain.RUIFROK_HE, est) < 1e-6


def test_od_roundtrip_and_validation():
    rgb = np.random.default_rng(0).uniform(0.2, 1.0, (8, 8, 3))
    od = stain.rgb_to_od(rgb, i0=1.0)
    assert np.allclose(stain.od_to_rgb(od, 1.0), rgb, atol=1e-9)
    with pytest.raises(ValueError):
        stain.rgb_to_od(np.zeros((8, 8)))


def test_macenko_estimate_close_to_true_vectors_and_rejects_blank_slides():
    conc, od = _he_image(64, 64, seed=1)
    V, diag = stain.estimate_stain_vectors(od)
    assert diag["angle_to_ruifrok_h_deg"] < 9.0 and diag["angle_to_ruifrok_e_deg"] < 9.0
    assert V.shape == (3, 3) and np.allclose(np.linalg.norm(V, axis=1), 1.0)
    with pytest.raises(ValueError, match="too few tissue pixels"):
        stain.estimate_stain_vectors(np.zeros((32, 32, 3)))


def test_stain_passport_and_cli(tmp_path):
    from skimage import io as skio
    from typer.testing import CliRunner

    from verisection.cli import app
    from verisection.io import read_rgb
    from verisection.qc import stain_passport

    conc, od = _he_image(64, 64, seed=3)
    conc[..., 0] += 0.3
    od = conc @ stain.RUIFROK_HE
    rgb = np.clip(stain.od_to_rgb(od), 0, 1)
    pp = stain_passport(rgb)
    assert pp["ok"] and pp["residual_rms"] < 0.05 and 0.1 < pp["tissue_fraction"] <= 1.0
    blank = stain_passport(np.ones((64, 64, 3)))
    assert not blank["ok"] and "too few" in blank["reason"]
    for k in range(4):
        skio.imsave(tmp_path / f"s{k}.png", (rgb * 255).astype(np.uint8), check_contrast=False)
    back = read_rgb(tmp_path / "s0.png", max_side=32)
    assert back.shape == (32, 32, 3) and 0 <= back.min() and back.max() <= 1
    r = CliRunner().invoke(app, ["stain-qc", str(tmp_path)])
    assert r.exit_code == 0 and "4/4 sections" in r.output
    empty = tmp_path / "empty"
    empty.mkdir()
    assert CliRunner().invoke(app, ["stain-qc", str(empty)]).exit_code == 2
    skio.imsave(tmp_path / "gray.png", np.zeros((16, 16), np.uint8), check_contrast=False)
    with pytest.raises(Exception, match="not an RGB"):
        read_rgb(tmp_path / "gray.png")
