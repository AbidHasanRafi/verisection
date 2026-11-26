"""VeriSection Workstation: payload encoding, standalone HTML and the local server.

The front end lives in ``web/`` (HTML, CSS, JavaScript; WebGL2 ray-marched 3D view, linked orthogonal
slice views, menubar, dockable panels, multi-tab analysis with Plotly charts). Data are sent to the page
as one JSON payload with volumes encoded as base64 uint8 arrays.

* ``write_standalone`` embeds a payload in a single HTML file (opens offline, no server).
* ``serve`` starts a local server (127.0.0.1 only) where *File > New simulation* runs the pipeline.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .provenance import INTENDED_USE, Provenance
from .tissue import CLASS_NAMES as TISSUE_CLASSES

WEB = Path(__file__).with_name("web")
CLASS_COLORS_TISSUE = ["#f4f4f4", "#f2a7c3", "#8e4ab8", "#1f2a8a"]  # lumen, stroma, epithelium, immune
CLASS_COLORS_GENERIC = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def registered_intensity(res) -> np.ndarray:
    """Registered section images on observed planes; model-derived fill (posterior-mean intensity) in gaps."""
    maps, stack = res.maps, res.stack
    Z, Y, X = maps.provenance.shape
    mean_t = res.reg.point().transforms[0]
    vol = np.zeros((Z, Y, X), dtype=np.float32)
    obs = np.zeros(Z, dtype=bool)
    for k in range(Z):
        if (maps.provenance[k] == Provenance.OBSERVED).mean() > 0.5:
            s = int(np.argmin(np.abs(stack.z_um - maps.z_um[k])))
            img, _ = mean_t[s].apply(stack.images[s].astype(np.float64), order=1, cval=0.0)
            vol[k] = img
            obs[k] = True
    lo, hi = np.percentile(vol[obs], [1, 99]) if obs.any() else (0.0, 1.0)
    vol = np.clip((vol - lo) / max(hi - lo, 1e-9), 0, 1)
    pred = maps.map_labels
    C = maps.p_hat.shape[0]
    means = np.array([vol[obs][pred[obs] == c].mean() if (pred[obs] == c).any() else 0.5 for c in range(C)])
    fill = np.tensordot(means, maps.p_hat, axes=(0, 0)).astype(np.float32)
    vol[~obs] = fill[~obs]
    return vol


def make_payload(sim, analysis: dict | None = None, title: str | None = None) -> dict:
    """Encode a ``SimResult`` (with truth) as the workstation payload."""
    res, v = sim.result, sim.view
    maps = res.maps
    C = maps.p_hat.shape[0]
    Z, Y, X = maps.provenance.shape
    names = list(v.class_names)
    colors = CLASS_COLORS_TISSUE if names == TISSUE_CLASSES else CLASS_COLORS_GENERIC
    valid = maps.provenance > 0
    unc = np.clip(maps.u_total / max(np.log2(C), 1e-9), 0, 1)
    vr = maps.var_reg
    vr_scale = float(np.percentile(vr[valid], 99)) if valid.any() else 1.0
    labels = np.where(valid, maps.map_labels + 1, 0)
    truth = None if v.truth is None else np.where(valid, v.truth + 1, 0)
    err = None if v.truth is None else (valid & (maps.map_labels != v.truth)).astype(np.uint8)
    vols = {
        "labels": _b64(labels), "prov": _b64(maps.provenance), "unc": _b64(unc * 255),
        "intensity": _b64(registered_intensity(res) * 255), "varReg": _b64(np.clip(vr / max(vr_scale, 1e-12), 0, 1) * 255),
        "probs": [_b64(np.clip(maps.p_hat[c], 0, 1) * 255) for c in range(C)],
        "truth": None if truth is None else _b64(truth), "err": None if err is None else _b64(err),
    }
    return {
        "version": 1, "title": title or v.title, "intendedUse": INTENDED_USE, "dims": [Z, Y, X],
        "zUm": [round(float(z), 3) for z in maps.z_um], "pixelUm": float(maps.pixel_um), "dzUm": float(maps.dz_um),
        "classNames": names, "classColors": colors[:C], "volumes": vols, "uncScale": float(np.log2(C)),
        "varRegScale": vr_scale, "sectionZ": [round(float(z), 3) for z in res.stack.z_um],
        "thicknessUm": float(res.stack.thickness_um), "hasTruth": v.truth is not None,
        "analysis": analysis or {}, "notes": list(v.notes),
    }


def payload_from_npz(path: str | Path) -> dict:
    """Payload for an exported ``volume.npz`` (no truth; analysis limited to what the file contains)."""
    with np.load(path, allow_pickle=False) as z:
        p = z["label_prob"].astype(np.float32)
        prov = z["provenance"]
        zum, px, dz = z["z_um"], float(z["pixel_um"]), float(z["dz_um"])
        names = [str(n) for n in z["class_names"]]
    C, Z, Y, X = p.shape
    valid = prov > 0
    ent = -(p * np.log2(np.clip(p, 1e-12, 1))).sum(axis=0)
    labels = np.where(valid, p.argmax(axis=0) + 1, 0)
    gray = np.tensordot(np.linspace(0.15, 0.85, C), p, axes=(0, 0))
    summary = {"dims": [Z, Y, X], "pixel_um": px, "grid_dz_um": dz, "class_names": names, "mean_entropy_bits": float(ent[valid].mean()),
               "provenance_fraction": {Provenance(i).name.lower(): float((prov == i).sum() / max(valid.sum(), 1)) for i in (1, 2, 3)},
               "predicted_class_fraction": [float((labels[valid] == c + 1).mean()) for c in range(C)]}
    return {
        "version": 1, "title": Path(path).name, "intendedUse": INTENDED_USE, "dims": [Z, Y, X],
        "zUm": [round(float(a), 3) for a in zum], "pixelUm": px, "dzUm": dz, "classNames": names,
        "classColors": (CLASS_COLORS_TISSUE if names == TISSUE_CLASSES else CLASS_COLORS_GENERIC)[:C],
        "volumes": {"labels": _b64(labels), "prov": _b64(prov), "unc": _b64(np.clip(ent / max(np.log2(C), 1e-9), 0, 1) * 255),
                    "intensity": _b64(gray * 255), "varReg": _b64(np.zeros_like(prov)), "probs": [_b64(np.clip(p[c], 0, 1) * 255) for c in range(C)],
                    "truth": None, "err": None},
        "uncScale": float(np.log2(C)), "varRegScale": 1.0, "sectionZ": [], "thicknessUm": 0.0, "hasTruth": False,
        "analysis": {"class_names": names, "summary": summary, "disclaimer": "Exported volume: no ground truth, so truth-based analysis tabs are empty."},
        "notes": ["Opened from an exported volume; interpolated voxels are model inference."],
    }


JS_FILES = ("core.js", "slices.js", "gl3d.js", "analysis.js", "audit.js", "tour.js", "export.js", "app.js")


def _assemble(payload: dict | None, audit: dict | None = None) -> str:
    """Inline CSS, plotly.js, the app scripts (one scope) and the payload into a single HTML string."""
    html = (WEB / "workstation.html").read_text(encoding="utf-8")
    css = (WEB / "workstation.css").read_text(encoding="utf-8")
    js = "(function(){'use strict';\n" + "\n".join((WEB / f).read_text(encoding="utf-8") for f in JS_FILES) + "\n})();"
    import plotly.offline as po

    data = "null" if payload is None else json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    adata = "null" if audit is None else json.dumps(audit, separators=(",", ":"), default=float).replace("</", "<\\/")
    return (html.replace("/*__CSS__*/", css).replace("/*__PLOTLY__*/", po.get_plotlyjs())
            .replace("/*__APPJS__*/", js).replace("/*__PAYLOAD__*/", f"window.VS_PAYLOAD={data};")
            .replace("/*__AUDITDATA__*/", f"window.VS_AUDIT={adata};")
            .replace("/*__SERVER__*/", "window.VS_SERVER=false;"))


def write_standalone(payload: dict, path: str | Path) -> Path:
    """One self-contained HTML file (works offline; no simulation menu)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_assemble(payload), encoding="utf-8")
    return path


def write_audit_standalone(audit: dict, path: str | Path, payload: dict | None = None) -> Path:
    """One self-contained HTML file that opens in the Image audit workspace (offline; optionally with a volume too)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_assemble(payload, audit), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------- server
class Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error
        self.progress = 0.0
        self.message = ""
        self.payload_json: bytes | None = None
        self.error = ""
        self.started = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "progress": self.progress, "message": self.message, "error": self.error,
                    "elapsed": time.time() - self.started if self.started else 0.0}


JOB = Job()


def _parse_params(raw: dict):
    """Whitelist and clamp simulation parameters from the browser."""
    from .simrun import SimParams

    allowed = {f.name: f.type for f in fields(SimParams)}
    p = SimParams()
    for k, v in raw.items():
        if k not in allowed or k in ("extended",):
            continue
        cur = getattr(p, k)
        if v is None or v == "":
            setattr(p, k, None if cur is None else cur)
            continue
        if isinstance(cur, bool):
            setattr(p, k, bool(v))
        elif isinstance(cur, int):
            setattr(p, k, int(v))
        elif isinstance(cur, float) or cur is None:
            setattr(p, k, float(v))
        else:
            setattr(p, k, str(v))
    p.preset = p.preset if p.preset in ("tissue", "hard", "easy") else "tissue"
    p.method = p.method if p.method in ("translation", "rigid", "nonrigid") else "translation"
    p.interp_mode = p.interp_mode if p.interp_mode in ("kriging", "linear", "nearest") else "kriging"
    p.n_sections = int(np.clip(p.n_sections, 6, 40))
    p.spacing_planes = int(np.clip(p.spacing_planes, 2, 10))
    p.n_reg = int(np.clip(p.n_reg, 1, 6))
    p.n_seg = int(np.clip(p.n_seg, 1, 6))
    p.seed = int(np.clip(p.seed, 0, 100000))
    p.sigma_scale = float(np.clip(p.sigma_scale, 1.0, 10.0))
    return p


def _run_job(raw: dict) -> None:
    from .analysis import build_analysis
    from .simrun import run_simulation

    def prog(msg: str, f: float) -> None:
        with JOB.lock:
            JOB.message, JOB.progress = msg, min(float(f), 0.99)

    try:
        extended = bool(raw.get("extended", False))
        p = _parse_params(raw)
        sim = run_simulation(p, progress=prog)
        prog("running numerical analysis", 0.92)
        an = build_analysis(sim, asdict(p), extended=extended, progress=prog)
        payload = make_payload(sim, an)
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with JOB.lock:
            JOB.payload_json, JOB.state, JOB.progress, JOB.message = blob, "done", 1.0, "done"
    except Exception as exc:  # report to the UI instead of dying silently
        with JOB.lock:
            JOB.state, JOB.error = "error", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


def _run_audit_job(raw: dict) -> None:
    from .imageaudit import run_audit

    def prog(msg: str, f: float) -> None:
        with JOB.lock:
            JOB.message, JOB.progress = msg, min(float(f), 0.99)

    try:
        root = Path(str(raw.get("root", "datasets"))).expanduser()
        if not root.is_dir():
            raise ValueError(f"folder not found: {root}")
        per_class = int(np.clip(int(raw.get("per_class", 16)), 4, 120))
        seed = int(np.clip(int(raw.get("seed", 0)), 0, 100000))
        payload = run_audit(root, per_class, seed, slow=bool(raw.get("slow", True)), workers=1, progress=prog)
        blob = json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8")
        with JOB.lock:
            JOB.payload_json, JOB.state, JOB.progress, JOB.message = blob, "done", 1.0, "done"
    except Exception as exc:  # report to the UI instead of dying silently
        with JOB.lock:
            JOB.state, JOB.error = "error", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


class Handler(BaseHTTPRequestHandler):
    server_version = "VeriSection"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = _assemble(None).replace("window.VS_SERVER=false;", "window.VS_SERVER=true;")
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, json.dumps(JOB.snapshot()).encode())
        elif path == "/api/result":
            with JOB.lock:
                blob = JOB.payload_json
            if blob is None:
                self._send(404, b'{"error":"no result yet"}')
            else:
                self._send(200, blob)
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid JSON"}')
            return
        if self.path == "/api/simulate":
            with JOB.lock:
                if JOB.state == "running":
                    self._send(409, b'{"error":"a job is already running"}')
                    return
                JOB.state, JOB.progress, JOB.message, JOB.error, JOB.started = "running", 0.0, "starting", "", time.time()
            threading.Thread(target=_run_job, args=(body,), daemon=True).start()
            self._send(202, b'{"ok":true}')
        elif self.path == "/api/audit":
            with JOB.lock:
                if JOB.state == "running":
                    self._send(409, b'{"error":"a job is already running"}')
                    return
                JOB.state, JOB.progress, JOB.message, JOB.error, JOB.started = "running", 0.0, "starting", "", time.time()
            threading.Thread(target=_run_audit_job, args=(body,), daemon=True).start()
            self._send(202, b'{"ok":true}')
        elif self.path == "/api/open":
            p = Path(str(body.get("path", ""))).expanduser()
            if p.suffix.lower() != ".npz" or not p.is_file():
                self._send(400, b'{"error":"give the path of an existing volume.npz"}')
                return
            try:
                blob = json.dumps(payload_from_npz(p), separators=(",", ":")).encode("utf-8")
            except (KeyError, ValueError, OSError) as exc:
                self._send(400, json.dumps({"error": f"cannot read volume: {exc}"}).encode())
                return
            self._send(200, blob)
        else:
            self._send(404, b'{"error":"not found"}')


def serve(port: int = 8765, open_browser: bool = True) -> ThreadingHTTPServer:
    """Start the workstation server on 127.0.0.1 (blocking in ``serve_forever``); returns the server object."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    return srv
