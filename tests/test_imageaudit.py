"""Image audit: sampling rules, per-image descriptors, the quality gate, the payload, the CLI and the workspace in a real browser."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi
from typer.testing import CliRunner

from verisection import imageaudit as ia
from verisection import workstation as W
from verisection.cli import app

CLASSES = ("colon_n", "colon_aca", "breast_benign", "breast_malignant")
N_PER_CLASS = 8


def _he_like(rng, size=160, tint=(0.0, 0.0, 0.0), blur=1.5):
    """Smooth random field mapped to a pink/purple range, with a bright background corner (a stand-in for H&E)."""
    f = ndi.gaussian_filter(rng.random((size, size)), blur)
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    rgb = np.stack([0.95 - 0.45 * f, 0.80 - 0.55 * f, 0.92 - 0.35 * f], axis=-1) + np.array(tint)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    rng = np.random.default_rng(0)
    for c in CLASSES:
        d = root / c
        d.mkdir()
        for i in range(N_PER_CLASS):
            Image.fromarray(_he_like(rng, tint=(0.03 * (c.endswith("aca") or c.endswith("malignant")), 0, 0))).save(d / f"orig_{i:04d}.jpg", quality=95)
        for i in range(5):  # augmented copies must never be drawn
            Image.fromarray(_he_like(rng)).save(d / f"aug_{i:04d}.jpg")
    pad = _he_like(rng)
    pad[:60] = 0                                           # letterbox: 37% black
    Image.fromarray(pad).save(root / "colon_n" / "orig_9999.jpg", quality=95)
    return root


def test_sampling_never_draws_augmented_copies_and_is_seeded(dataset):
    a = ia.sample_dataset(dataset, per_class=6, seed=3)
    b = ia.sample_dataset(dataset, per_class=6, seed=3)
    assert [s.path for s in a] == [s.path for s in b]
    assert len(a) == 6 * len(CLASSES)
    assert not any(s.path.name.startswith("aug_") for s in a)
    assert {s.cls for s in a} == set(CLASSES)


def test_describe_image_returns_stain_quality_and_texture_descriptors():
    rng = np.random.default_rng(1)
    d = ia.describe_image(_he_like(rng) / 255.0)
    for k in ("mean_h", "mean_e", "residual_rms", "focus_lapvar", "noise_sigma", "black_fraction", "glcm_contrast_2px", "glcm_entropy_8px"):
        assert k in d and np.isfinite(d[k]), k
    assert d["black_fraction"] == 0.0 and 0.0 < d["tissue_fraction"] <= 1.0


def test_texture_is_computed_in_pixels_with_a_fixed_white_point():
    """Same texture, brighter or darker overall: fixed bins mean the descriptor changes (it is not rescaled per image)."""
    rng = np.random.default_rng(2)
    rgb = _he_like(rng) / 255.0
    f0 = ia.glcm_scale_features(ia.od_map(rgb))
    f1 = ia.glcm_scale_features(ia.od_map(np.clip(rgb * 0.8, 0, 1)))
    assert f0["glcm_contrast_2px"] != pytest.approx(f1["glcm_contrast_2px"])
    assert f0["glcm_homogeneity_1px"] >= f0["glcm_homogeneity_8px"]      # larger offsets are less alike


def test_quality_gate_excludes_padding_and_gating_keeps_the_rest(dataset):
    samples = ia.sample_dataset(dataset, per_class=N_PER_CLASS + 1, seed=0)
    padded = [i for i, s in enumerate(samples) if s.path.name == "orig_9999.jpg"]
    assert padded, "the padded file should be sampled when every original is drawn"
    df = ia.audit_samples(samples)
    v = ia.qc_verdicts(df)
    assert v.loc[padded[0], "verdict"] == "exclude" and "black padding" in v.loc[padded[0], "reasons"]
    assert (v["verdict"] == "accept").sum() >= len(df) - 3                 # ordinary patches pass
    res = ia.analyse(samples, df, seed=0, slow=False)
    assert res["n_gated"] >= 1 and len(res["df_ok"]) == len(df) - res["n_gated"]
    assert not (res["df_ok"]["black_fraction"] > ia.BLACK_EXCLUDE).any()


def test_run_audit_payload_is_complete_and_json_serialisable(dataset):
    p = ia.run_audit(dataset, per_class=6, seed=0, slow=True)
    assert p["kind"] == "image-audit" and len(p["rows"]) == 6 * len(CLASSES)
    assert {"separability", "domain_shift", "stability", "interpolation", "pca", "qc_rates"} <= set(p["analysis"])
    assert p["analysis"]["stability"] and p["analysis"]["interpolation"]
    ds = p["analysis"]["domain_shift"]
    assert {r["setting"] for r in ds} == {"in-source (5-fold)", "leave-one-source-out"}
    assert all(0.0 <= r["auroc"] <= 1.0 for r in ds)
    assert any(t for t in p["thumbs"].values())
    json.dumps(p, default=float)                                           # no numpy types left in the payload
    assert any("not serial" in n or "no 3D" in n for n in p["notes"])      # the limit is stated in the payload itself


def test_interpolation_bias_is_zero_at_observed_planes_and_negative_for_contrast_at_midpoint(dataset):
    s = ia.sample_dataset(dataset, per_class=6, seed=0)
    b = ia.interpolation_bias(s, n_pairs=8, seed=0)
    obs = b[b["alpha"] == 0.0]
    assert np.allclose(obs["relative_bias"].dropna(), 0.0, atol=1e-9)
    mid = b[(b["alpha"] == 0.5) & (b["scenario"] == "independent") & (b["feature"] == "contrast")]
    assert mid["relative_bias"].mean() < 0                                 # blending two independent planes smooths contrast


def test_stability_icc_has_the_documented_columns_and_bounds(dataset):
    s = ia.sample_dataset(dataset, per_class=6, seed=0)
    wins, meta = ia.sample_windows(s, 24, size=96, seed=0)
    t = ia.stability_table(wins, seed=0, offsets=(2,))
    assert {"feature", "perturbation", "icc", "icc_lower95", "certified"} <= set(t.columns)
    assert (t["icc_lower95"] <= t["icc"] + 1e-9).all() and len(meta) == 24


def test_stain_passport_default_behaviour_is_unchanged_and_sum_rule_keeps_pale_tissue():
    from verisection import qc, stain

    rng = np.random.default_rng(4)
    rgb = _he_like(rng) / 255.0
    base = qc.stain_passport(rgb)                                          # per-channel 99th percentile white point, 'all' rule
    assert "tissue_fraction" in base
    od = stain.rgb_to_od(rgb, i0=1.0)
    assert stain.tissue_mask(od, 0.15, "sum").mean() >= stain.tissue_mask(od, 0.15, "all").mean()
    with pytest.raises(ValueError):
        stain.tissue_mask(od, 0.15, "nonsense")


def test_cli_audit_writes_a_standalone_workstation_with_the_audit_embedded(dataset, tmp_path):
    out = tmp_path / "audit.html"
    r = CliRunner().invoke(app, ["audit", str(dataset), "--out", str(out), "--per-class", "6", "--fast", "--no-open", "--json-out", str(tmp_path / "a.json")])
    assert r.exit_code == 0, r.output
    html = out.read_text(encoding="utf-8")
    assert "window.VS_AUDIT={" in html and "window.VS_PAYLOAD=null" in html
    assert json.loads((tmp_path / "a.json").read_text())["kind"] == "image-audit"
    bad = CliRunner().invoke(app, ["audit", str(tmp_path / "missing"), "--no-open"])
    assert bad.exit_code == 2


def test_banner_is_only_the_centred_research_use_only_text():
    web = Path(W.__file__).parent / "web"
    html = (web / "workstation.html").read_text(encoding="utf-8")
    assert re.search(r'<div id="banner">RESEARCH USE ONLY</div>', html)
    css = (web / "workstation.css").read_text(encoding="utf-8")
    assert re.search(r"#banner\{[^}]*text-align:center", css)


def test_workspace_files_are_registered_in_order():
    assert W.JS_FILES.index("analysis.js") < W.JS_FILES.index("audit.js") < W.JS_FILES.index("tour.js") < W.JS_FILES.index("app.js")


def test_feature_groups_partition_the_combined_descriptors():
    cols = ia.descriptor_columns(__import__("pandas").DataFrame(columns=[f"glcm_{k}_{o}px" for k in ia.GLCM_FEATURES for o in ia.OFFSETS_PX] +
                                                              ["mean_h", "mean_e", "residual_rms", "angle_h_deg", "angle_e_deg", "angle_between_deg", "od_mean", "od_sd", "mean_r", "mean_g", "mean_b"]), "combined")
    groups = ia.feature_groups(cols)
    assert sorted(i for v in groups.values() for i in v) == list(range(len(cols)))      # every descriptor in exactly one group
    assert len(groups) == 11 and all(g for g in groups.values())


def test_grouped_permutation_importance_and_tumor_explanation(dataset):
    pytest.importorskip("shap")
    samples = ia.sample_dataset(dataset, per_class=8, seed=0)
    df = ia.audit_samples(samples)
    df = df[(ia.qc_verdicts(df)["verdict"] == "accept").to_numpy()].reset_index(drop=True)
    imp = ia.grouped_permutation_importance(df, "group", seed=0, n_splits=3, n_perm=2)
    assert {"fold", "group", "drop", "baseline"} <= set(imp.columns) and imp["group"].nunique() == 11
    ex = ia.explain_tumor(df, seed=0)
    assert ex is not None and set(ex["patches"]) == set(int(i) for i in df["id"])
    assert abs(sum(g["share"] for g in ex["groups"]) - 1.0) < 0.01
    one = next(iter(ex["patches"].values()))
    assert 0.0 <= one["p"] <= 1.0 and len(one["top"]) == 3 and abs(one["top"][0][1]) >= abs(one["top"][2][1])
    json.dumps(ex)


def test_audit_payload_carries_the_explanation_when_given(dataset):
    pytest.importorskip("shap")
    samples = ia.sample_dataset(dataset, per_class=8, seed=0)
    df = ia.audit_samples(samples)
    v = ia.qc_verdicts(df)
    ok = df[(v["verdict"] == "accept").to_numpy()].reset_index(drop=True)
    ex = ia.explain_tumor(ok)
    p = ia.audit_payload(samples, df, v, {}, explain=ex)
    assert p["analysis"]["explain"]["groups"] and any(r.get("xtop") for r in p["rows"])
    assert all(r.get("xtop") is None for r in p["rows"] if r["verdict"] != "accept")    # excluded patches get no explanation
    json.dumps(p, default=float)


EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
@pytest.mark.parametrize("frag,needles", [
    ("#ws=audit&tab=Overview", ["Samples by class", "Image inspector", "Collections"]),
    ("#ws=audit&tab=Stain", ["Stain concentration (H vs E)", "js-plotly-plot"]),
    ("#ws=audit&tab=Separability", ["Domain shift: in-source vs leave-one-source-out", "not for diagnosis"]),
    ("#ws=audit&tab=Method%20%26%20data", ["Source datasets", "Borkowski"]),
    ("#ws=audit&tour=3", ["Guided tour", "Collections"]),
])
def test_audit_workspace_renders_in_a_real_browser_without_script_errors(dataset, tmp_path, frag, needles):
    payload = ia.run_audit(dataset, per_class=6, seed=0, slow=True)
    html = W.write_audit_standalone(payload, tmp_path / "audit.html")
    exe = next(p for p in EDGE if Path(p).exists())
    cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1600,950",
           "--virtual-time-budget=8000", "--dump-dom", html.resolve().as_uri() + frag]
    dom = subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8", errors="replace").stdout
    assert 'id="errbox"' not in dom, "a script error was reported in the page"
    assert 'class="tile' in dom and 'id="banner">RESEARCH USE ONLY<' in dom
    for n in needles:
        assert n in dom, n


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
def test_explanation_tab_and_inspector_card_render_without_script_errors(dataset, tmp_path):
    pytest.importorskip("shap")
    payload = ia.run_audit(dataset, per_class=8, seed=0, slow=False)
    assert payload["analysis"].get("explain"), "an audit of 30 or more accepted patches should carry an explanation"
    sel = next(r["id"] for r in payload["rows"] if r.get("xtop"))
    html = W.write_audit_standalone(payload, tmp_path / "audit.html")
    exe = next(p for p in EDGE if Path(p).exists())
    for frag, needles in (("#ws=audit&tab=Explanation", ["What drives the tumor score (TreeSHAP share)", "not for diagnosis", "Descriptor groups"]),
                          (f"#ws=audit&tab=Overview&sel={sel}", ["Explanation (tumor score)", "forest score"])):
        cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1600,950",
               "--virtual-time-budget=8000", "--dump-dom", html.resolve().as_uri() + frag]
        dom = subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8", errors="replace").stdout
        assert 'id="errbox"' not in dom, "a script error was reported in the page"
        for n in needles:
            assert n in dom, n


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
@pytest.mark.parametrize("size", [(1600, 950), (1280, 800)])
def test_audit_workspace_layout_panels_do_not_touch_or_clip(dataset, tmp_path, size):
    payload = ia.run_audit(dataset, per_class=6, seed=0, slow=False)
    html = W.write_audit_standalone(payload, tmp_path / "audit.html")
    exe = next(p for p in EDGE if Path(p).exists())
    cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", f"--window-size={size[0]},{size[1]}",
           "--virtual-time-budget=9000", "--dump-dom", html.resolve().as_uri() + "#ws=audit&audit=1"]
    dom = subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8", errors="replace").stdout
    found = re.findall(r'<pre id="audit" style="display:\s*none;?">(\{.*?\})</pre>', dom, re.S)
    assert found, "layout audit element missing"
    a = json.loads(found[-1].replace("&quot;", '"').replace("&amp;", "&"))
    assert a["minGapPx"] is not None and a["minGapPx"] >= 6, a
    assert not a["tooClose"] and not a["offscreen"] and not a["clipped"] and a["sliders"] == 0, a
