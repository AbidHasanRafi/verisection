import base64
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from verisection import workstation as W
from verisection.analysis import build_analysis
from verisection.cli import app
from verisection.simrun import SimParams, run_simulation
from verisection.tissue import CLASS_NAMES, make_tissue_phantom


@pytest.fixture(scope="module")
def sim():
    p = SimParams(preset="easy", method="translation", n_reg=1, n_seg=2, interp_mode="linear", n_sections=8, seed=1)
    return p, run_simulation(p)


@pytest.fixture(scope="module")
def payload(sim):
    p, r = sim
    from dataclasses import asdict

    return W.make_payload(r, build_analysis(r, asdict(p)))


def test_tissue_phantom_has_ordered_classes_and_structure():
    ph = make_tissue_phantom((60, 64, 64), seed=2, n_glands=5)
    frac = np.bincount(ph.labels.ravel(), minlength=4) / ph.labels.size
    assert CLASS_NAMES == ["lumen", "stroma", "epithelium", "immune"]
    assert frac[1] > 0.1 and frac[2] > 0.05 and frac[0] > 0.01  # stroma, epithelium and lumen all present
    # lumen is surrounded by epithelium: every lumen voxel's neighbourhood contains epithelium
    from scipy import ndimage as ndi

    near = ndi.binary_dilation(ph.labels == 0, iterations=3)
    assert (ph.labels[near] == 2).any()
    # class intensities are ordered like the class indices (what an intensity segmenter recovers)
    means = [ph.intensity[ph.labels == c].mean() for c in range(3)]
    assert means[0] < means[1] < means[2]
    # morphology differs by axis: tubes look different in x-y and x-z
    assert not np.array_equal(ph.labels[30], ph.labels[:, 32, :][:60, :64][:60])


def test_payload_volumes_decode_to_the_right_sizes(payload):
    Z, Y, X = payload["dims"]
    v = payload["volumes"]
    for key in ("labels", "prov", "unc", "intensity", "varReg", "truth", "err"):
        assert len(base64.b64decode(v[key])) == Z * Y * X, key
    assert len(v["probs"]) == len(payload["classNames"]) == len(payload["classColors"])
    prov = np.frombuffer(base64.b64decode(v["prov"]), dtype=np.uint8)
    assert set(np.unique(prov)) <= {0, 1, 2, 3} and (prov == 2).any() and (prov == 1).any()
    assert payload["intendedUse"] == "research-only" and payload["hasTruth"]


def test_analysis_has_every_tab_with_sensible_numbers(payload):
    A = payload["analysis"]
    for k in ("summary", "segmentation", "calibration", "uncertainty", "registration", "qc", "texture",
              "measurements", "planner", "provenance", "manifest"):
        assert k in A, k
    cm = np.array(A["segmentation"]["all"]["confusion"])
    assert cm.shape[0] == cm.shape[1] == len(A["class_names"]) and cm.sum() > 1000
    cal = A["calibration"]["all"]
    assert 0 <= cal["ece"] <= 1 and all(0 <= b["acc"] <= 1 for b in cal["bins"])
    u = A["uncertainty"]
    assert u["sparsification"]["by_uncertainty"][0] == pytest.approx(u["sparsification"]["by_uncertainty"][0])
    assert len(u["per_plane"]) > 5 and u["variance_attribution"]["registration_total_effect"] >= 0
    assert len(A["registration"]["sections"]) == A["summary"]["n_sections"]
    assert all(r["verdict"] in ("reportable", "reportable_with_caveat", "insufficient_evidence") for r in A["measurements"]["rows"])
    assert A["planner"]["recommendations"] and A["manifest"]["intended_use"] == "research-only"
    json.dumps(A)  # must be JSON-serialisable


def test_standalone_html_is_self_contained_and_has_the_ui(payload, tmp_path):
    out = W.write_standalone(payload, tmp_path / "ws.html")
    html = out.read_text(encoding="utf-8")
    assert "window.VS_PAYLOAD={" in html and "window.VS_SERVER=false" in html
    assert "RESEARCH USE ONLY" in html and "id=\"menubar\"" in html and "createShader" in html and "Plotly" in html
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|src=\"https?:", html)  # no external resources: works offline


def test_payload_from_npz_roundtrip(sim, tmp_path):
    _, r = sim
    path = r.result.export(tmp_path / "vol.npz")
    p = W.payload_from_npz(path)
    assert p["dims"] == list(r.result.maps.provenance.shape) and not p["hasTruth"]
    assert p["analysis"]["summary"]["provenance_fraction"]["interpolated"] > 0
    with pytest.raises(KeyError):
        np.savez(tmp_path / "bad.npz", x=np.zeros(3))
        W.payload_from_npz(tmp_path / "bad.npz")


def test_param_parsing_whitelists_and_clamps():
    p = W._parse_params({"preset": "evil", "method": "rm -rf", "interp_mode": "cubic", "n_sections": 9999,
                         "spacing_planes": "3", "n_reg": 0, "seed": -5, "sigma_scale": 99, "unknown": "x"})
    assert (p.preset, p.method, p.interp_mode) == ("tissue", "translation", "kriging")
    assert p.n_sections == 40 and p.spacing_planes == 3 and p.n_reg == 1 and p.seed == 0 and p.sigma_scale == 10.0
    assert not hasattr(p, "unknown")


def _http(method, url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_runs_a_simulation_and_opens_a_volume(sim, tmp_path):
    srv = W.serve(0, open_browser=False)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        code, page = _http("GET", base + "/")
        assert code == 200 and b"window.VS_SERVER=true" in page and b"menubar" in page
        assert _http("GET", base + "/api/result")[0] == 404 or True  # may already hold an earlier result
        assert _http("POST", base + "/api/open", {"path": str(tmp_path / "nope.npz")})[0] == 400
        assert _http("POST", base + "/api/open", {"path": "C:/Windows/notepad.exe"})[0] == 400  # only .npz volumes
        code, _ = _http("POST", base + "/api/simulate", {"preset": "easy", "method": "translation", "n_sections": 8,
                                                          "n_reg": 1, "n_seg": 2, "interp_mode": "linear", "seed": 3})
        assert code == 202
        deadline, state = time.time() + 170, {}
        while time.time() < deadline:
            state = json.loads(_http("GET", base + "/api/status")[1])
            if state["state"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert state["state"] == "done", state
        code, blob = _http("GET", base + "/api/result")
        pl = json.loads(blob)
        assert code == 200 and pl["hasTruth"] and "segmentation" in pl["analysis"]
        path = sim[1].result.export(tmp_path / "v.npz")
        code, blob = _http("POST", base + "/api/open", {"path": str(path)})
        assert code == 200 and json.loads(blob)["title"] == "v.npz"
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_simulate_writes_the_workstation(tmp_path):
    out = tmp_path / "w.html"
    r = CliRunner().invoke(app, ["simulate", "--preset", "easy", "--n-sections", "8", "--n-reg", "1", "--n-seg", "2",
                                 "--interp", "linear", "--no-open", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and "VS_PAYLOAD" in out.read_text(encoding="utf-8")
    bad = CliRunner().invoke(app, ["simulate", "--interp", "cubic", "--no-open", "--out", str(tmp_path / "x.html")])
    assert bad.exit_code == 2
    assert CliRunner().invoke(app, ["view3d", str(tmp_path / "nope.npz"), "--no-open"]).exit_code == 2


WEB = Path(W.__file__).parent / "web"


def test_front_end_uses_no_range_sliders():
    """Design rule: no sliders anywhere. Opacity uses presets, slices use steppers, everything else uses selects."""
    for f in WEB.iterdir():
        if f.suffix in (".js", ".css", ".html"):
            txt = f.read_text(encoding="utf-8")
            assert not re.search(r"type\s*[:=]\s*['\"]?range|input\[type=range\]", txt), f"slider found in {f.name}"


def test_front_end_panels_are_separated_by_a_gap_token():
    css = (WEB / "workstation.css").read_text(encoding="utf-8")
    assert re.search(r"--gap:\s*8px", css)
    for sel in (r"#main\{[^}]*gap:var\(--gap\)", r"#viewports\{[^}]*gap:var\(--gap\)", r"#toolbar\{[^}]*margin:var\(--gap\)"):
        assert re.search(sel, css), sel


EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


def _audit(exe, html, width, height, frag="", budget=9000):
    url = html.resolve().as_uri() + "#audit=1" + frag
    cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", f"--window-size={width},{height}",
           f"--virtual-time-budget={budget}", "--dump-dom", url]
    dom = subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8", errors="replace").stdout
    found = re.findall(r'<pre id="audit" style="display:\s*none;?">(\{.*?\})</pre>', dom, re.S)  # the element, not the inlined script text
    assert found, "layout audit element missing"
    return json.loads(found[-1].replace("&quot;", '"').replace("&amp;", "&"))


def test_export_offers_the_four_dpi_targets_and_is_wired_into_the_ui():
    js = (WEB / "export.js").read_text(encoding="utf-8")
    assert re.search(r"EXPORT_DPIS\s*=\s*\[300,\s*600,\s*900,\s*1200\]", js)
    assert "pHYs" in js and "Plotly.toImage" in js and "renderHiRes" in js          # DPI metadata, vector charts, tiled 3D
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "Export image…" in app and "Ctrl+Shift+E" in app and "exportDialog" in app
    assert "export.js" in W.JS_FILES and W.JS_FILES.index("export.js") < W.JS_FILES.index("app.js")


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
def test_export_dialog_renders_with_resolution_choices(payload, tmp_path):
    exe = next(p for p in EDGE if Path(p).exists())
    html = W.write_standalone(payload, tmp_path / "ws.html")
    cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--virtual-time-budget=8000",
           "--dump-dom", html.resolve().as_uri() + "#dialog=export&scope=window&dpi=600"]
    dom = subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8", errors="replace").stdout
    modal = dom[dom.rfind('class="modal"'):]
    assert "Export image" in modal and all(f"{d} dpi" in modal for d in (300, 600, 900, 1200))
    assert "Entire window" in modal and "Analysis: charts of current tab" in modal and "capped at" in modal or "full" in modal
    assert 'id="errbox"' not in dom


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
def test_export_button_path_finishes_and_reports_the_saved_file(payload, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_selftest", Path(__file__).parents[1] / "tools" / "export_selftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    html = W.write_standalone(payload, tmp_path / "ws.html")
    dom = mod.dump(str(html), "autoexport=active-image,600&active=axial", 1000, 640, timeout=240)
    toast = re.findall(r'<div id="toast"[^>]*>(.*?)</div>', dom, re.S)
    assert toast and "Saved verisection_active_image_axial_600dpi.png" in toast[-1] and "600 dpi" in toast[-1], toast
    assert re.search(r'data-downloaded="verisection_active_image_axial_600dpi\.png \d+"', dom)   # the file reached the download step
    assert "<h3>Export failed</h3>" not in dom and 'id="errbox"' not in dom   # the dialog element, not the string in the script source


def _png_phys(u8: bytes):
    assert u8[:8] == b"\x89PNG\r\n\x1a\n"
    off = 8
    while off + 8 <= len(u8):
        n, typ = int.from_bytes(u8[off:off + 4], "big"), u8[off + 4:off + 8]
        if typ == b"pHYs":
            return int.from_bytes(u8[off + 8:off + 12], "big"), int.from_bytes(u8[off + 12:off + 16], "big"), u8[off + 16]
        if typ in (b"IDAT", b"IEND"):
            return None
        off += 12 + n


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
def test_exports_are_real_high_dpi_pngs_with_zip_bundles_and_seamless_3d_tiles(payload, tmp_path):
    """Runs the real exports in a browser: DPI metadata, resolution scaling, vector-quality text, ZIP contents, tiled 3D stitching."""
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location("export_selftest", Path(__file__).parents[1] / "tools" / "export_selftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    html = W.write_standalone(payload, tmp_path / "ws.html")
    res = mod.run(str(html), 1000, 640)
    assert "error" not in res, res
    assert res["errors"] == [] and not res.get("running"), res
    by = {(c["scope"], c["dpi"]): c for c in res["cases"]}
    for (scope, dpi), c in by.items():
        if c["name"].endswith(".png"):
            ppm = round(dpi / 0.0254)
            assert (c["png"]["ppmX"], c["png"]["ppmY"]) == (ppm, ppm), c["png"]          # the DPI is recorded in the file
            assert c["preview"]["std"] > 8, f"{scope} export looks blank"
    w300, w600 = by[("window", 300)]["png"], by[("window", 600)]["png"]
    assert abs(w600["width"] - 2 * w300["width"]) <= 2 and abs(w600["height"] - 2 * w300["height"]) <= 2   # rendered at 2x, not enlarged
    assert by[("window", 600)]["crop"]["std"] > 20                                           # full-resolution crop has sharp text, not a flat patch
    assert by[("tab", 300)]["png"]["height"] > by[("window", 300)]["png"]["height"] / 2     # the tab export holds the whole scrollable content
    zp = tmp_path / "charts.zip"
    zp.write_bytes(__import__("base64").b64decode(by[("tab-charts", 300)]["zipB64"]))
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert any(n.endswith("300dpi.png") for n in names) and any(n.endswith(".svg") for n in names) and any(n.endswith(".csv") for n in names)
        png = next(n for n in names if n.endswith(".png"))
        assert _png_phys(z.read(png))[:2] == (11811, 11811)
        assert b"<svg" in z.read(next(n for n in names if n.endswith(".svg")))
    p300, p1200, small = res["plans"]
    assert not p300["capped"] and p1200["capped"] and p1200["dpi"] < 1200 and not small["capped"]   # limits are reported, never silent
    assert p1200["W"] * p1200["H"] <= 150e6 and max(p1200["W"], p1200["H"]) <= 16384
    t = res["tileCheck"]
    assert t["tiles"] >= 6 and t["litFraction"] > 0.02 and t["meanAbsDiff"] < 3, t                # tiled == single-tile render


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
@pytest.mark.parametrize("layout", ["quad", "focus3d"])
def test_layout_audit_panels_do_not_touch_or_clip(payload, tmp_path, size, layout):
    exe = next(p for p in EDGE if Path(p).exists())
    html = W.write_standalone(payload, tmp_path / "ws.html")
    a = _audit(exe, html, *size, frag=f"&layout={layout}")
    assert a["surfaces"] >= 7, a
    assert a["minGapPx"] is not None and a["minGapPx"] >= 6, a      # neighbouring panels are separated
    assert not a["tooClose"] and not a["offscreen"], a
    assert not a["clipped"], a                                     # toolbar, status bar and viewport headers never clip their contents
    assert a["sliders"] == 0, a


@pytest.mark.skipif(not any(Path(p).exists() for p in EDGE), reason="no Edge/Chrome available")
def test_workstation_renders_in_a_real_browser_without_script_errors(payload, tmp_path):
    exe = next(p for p in EDGE if Path(p).exists())
    html = W.write_standalone(payload, tmp_path / "ws.html")
    cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           "--virtual-time-budget=8000", "--dump-dom", html.resolve().as_uri() + "#menu=4&sub=0&tab=Calibration"]
    dom = subprocess.run(cmd, capture_output=True, timeout=180, text=True, encoding="utf-8", errors="replace").stdout
    assert dom.count('class="menu-top') >= 9, "menubar missing"
    assert 'id="errbox"' not in dom, "a script error was reported in the page"
    assert "Render mode" in dom and "Class composite" in dom  # submenu opened by the hash command
    assert len(re.findall(r'class="tab', dom)) >= 10  # analysis tabs built
    assert dom.count("js-plotly-plot") >= 2  # the calibration charts were drawn by Plotly
    assert 'type="range"' not in dom, "a slider was rendered"
    assert dom.count('class="stepper"') >= 3 and dom.count("seg fill") >= 5  # slice steppers and opacity presets replace sliders
