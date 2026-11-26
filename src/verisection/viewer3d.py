"""Interactive 3D / slice viewer built on Plotly (self-contained HTML, no GUI toolkit needed).

What you get
------------
* a rotatable, zoomable 3D volume rendering with a channel switch (uncertainty, class posterior,
  confidence, provenance) and outlines of the *observed* section slabs, so gaps are visible;
* x–y slice explorer (slider over z) and x–z slice explorer (slider over y) showing label map,
  uncertainty and provenance side by side (and the simulation truth when available);
* a permanent research-use banner and a provenance legend: interpolated voxels are never shown
  without their provenance next to them.

``ViewData`` is a plain container so the same viewer works for in-memory pipeline results and for
exported ``volume.npz`` files.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .provenance import INTENDED_USE, Provenance

BANNER = (f"Research use only ({INTENDED_USE}). Not a medical device. Not for diagnosis, prognosis or treatment "
          "decisions. Interpolated voxels are model inference, not observation.")
PROV_COLORS = ["#d9d9d9", "#2b7bba", "#f0a030", "#b04ab0"]  # unsampled, observed, interpolated, synthetic
CLASS_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


@dataclass
class ViewData:
    p_hat: np.ndarray  # (C, Z, Y, X)
    provenance: np.ndarray  # (Z, Y, X)
    z_um: np.ndarray  # (Z,)
    pixel_um: float
    dz_um: float
    class_names: list[str]
    u_total: np.ndarray  # (Z, Y, X) entropy in bits
    var_reg: np.ndarray | None = None
    truth: np.ndarray | None = None  # (Z, Y, X) simulation truth on the same grid
    title: str = "VeriSection"
    notes: list[str] = field(default_factory=list)

    @property
    def map_labels(self) -> np.ndarray:
        return self.p_hat.argmax(axis=0)

    @property
    def confidence(self) -> np.ndarray:
        c = self.p_hat.shape[0]
        return np.where(self.provenance > 0, 1.0 - self.u_total / np.log2(c), 0.0)


def view_data_from_maps(maps, truth: np.ndarray | None = None, title: str = "VeriSection simulation",
                        notes: list[str] | None = None) -> ViewData:
    return ViewData(maps.p_hat, maps.provenance, maps.z_um, maps.pixel_um, maps.dz_um, list(maps.class_names),
                    maps.u_total, getattr(maps, "var_reg", None), truth, title, notes or [])


def view_data_from_npz(path, title: str | None = None) -> ViewData:
    """Load a volume written by ``PipelineResult.export`` (needs ``label_prob`` and ``provenance``)."""
    name = getattr(path, "name", None) if hasattr(path, "read") else Path(path).name
    with np.load(path, allow_pickle=False) as z:
        p = z["label_prob"].astype(np.float32)
        prov = z["provenance"]
        zum = z["z_um"]
        px, dz = float(z["pixel_um"]), float(z["dz_um"])
        names = [str(n) for n in z["class_names"]]
    u = -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum(axis=0)
    return ViewData(p, prov, zum, px, dz, names, u, None, None, title or str(name or "volume"))


def _prov_counts(v: ViewData) -> dict[str, float]:
    s = max(int((v.provenance > 0).sum()), 1)
    return {Provenance(i).name.lower(): float((v.provenance == i).sum() / s) for i in (1, 2, 3)}


def _coords(v: ViewData, step: int):
    z, y, x = v.provenance.shape
    zz = np.asarray(v.z_um)
    ys = np.arange(0, y, step) * v.pixel_um
    xs = np.arange(0, x, step) * v.pixel_um
    return zz, ys, xs


def figure_3d(v: ViewData, step: int = 2, surface_count: int = 14):
    """3D volume rendering with a channel dropdown and observed-slab outlines."""
    import plotly.graph_objects as go

    zz, ys, xs = _coords(v, step)
    Z, Y, X = v.provenance.shape
    sl = (slice(None), slice(0, Y, step), slice(0, X, step))
    gz, gy, gx = (g.astype(np.float32) for g in np.meshgrid(zz, ys, xs, indexing="ij"))
    valid = (v.provenance[sl] > 0)

    channels: dict[str, tuple[np.ndarray, str, tuple[float, float]]] = {
        "uncertainty (bits)": (np.where(valid, v.u_total[sl], 0), "Magma", (0.0, float(np.log2(v.p_hat.shape[0])))),
        "confidence": (v.confidence[sl], "Viridis", (0.0, 1.0)),
        "provenance (1 obs, 2 interp)": (v.provenance[sl].astype(float), "Cividis", (0.0, 2.0)),
    }
    for c, name in enumerate(v.class_names):
        channels[f"P({name})"] = (np.where(valid, v.p_hat[c][sl], 0), "Viridis", (0.0, 1.0))
    fig = go.Figure()
    for k, (name, (vals, cs, (lo, hi))) in enumerate(channels.items()):
        fig.add_trace(go.Volume(
            x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=np.asarray(vals, dtype=np.float32).ravel(), isomin=lo + 0.05 * (hi - lo), isomax=hi,
            opacity=0.12, surface_count=surface_count, colorscale=cs, name=name, visible=(k == 0),
            colorbar=dict(title=name, len=0.6), caps=dict(x_show=False, y_show=False, z_show=False),
        ))
    # outlines of observed section slabs (planes where provenance == observed)
    obs_planes = [i for i in range(Z) if (v.provenance[i] == Provenance.OBSERVED).mean() > 0.3]
    x0, x1, y0, y1 = 0, (X - 1) * v.pixel_um, 0, (Y - 1) * v.pixel_um
    lx, ly, lz = [], [], []
    for i in obs_planes:
        lx += [x0, x1, x1, x0, x0, None]
        ly += [y0, y0, y1, y1, y0, None]
        lz += [zz[i]] * 5 + [None]
    n_ch = len(channels)
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#2b7bba", width=2),
                               name="observed sections", hoverinfo="skip"))
    buttons = []
    for k, name in enumerate(channels):
        vis = [j == k for j in range(n_ch)] + [True]
        buttons.append(dict(label=name, method="update", args=[{"visible": vis}]))
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.12, xanchor="left")],
        scene=dict(xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="z (µm)", aspectmode="data",
                   zaxis=dict(autorange="reversed")),
        margin=dict(l=0, r=0, t=30, b=0), height=620, title=dict(text="3D view (drag to rotate, scroll to zoom)", x=0.5),
        legend=dict(orientation="h", y=-0.02),
    )
    return fig


def _label_colorscale(n: int):
    cols = CLASS_COLORS[:n]
    return [[i / max(n - 1, 1) if n > 1 else 0.0, c] for i, c in enumerate(cols)] if n > 1 else [[0, cols[0]], [1, cols[0]]]


def _slice_panels(v: ViewData, which: str):
    """Yield (frame_index, [label, uncertainty, provenance(, truth)]) 2D arrays for each slider position."""
    n = v.provenance.shape[0] if which == "z" else v.provenance.shape[1]
    for k in range(n):
        if which == "z":
            lab, unc, prov = v.map_labels[k], v.u_total[k], v.provenance[k]
            tr = None if v.truth is None else v.truth[k]
        else:  # x–z section at row y = k: arrays have z along rows
            lab, unc, prov = v.map_labels[:, k, :], v.u_total[:, k, :], v.provenance[:, k, :]
            tr = None if v.truth is None else v.truth[:, k, :]
        yield k, [lab, unc, prov] + ([tr] if tr is not None else [])


def figure_slices(v: ViewData, which: str = "z"):
    """Slice explorer: 'z' = x–y slices (slider over z), 'y' = x–z slices (slider over y)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    c = v.p_hat.shape[0]
    has_truth = v.truth is not None
    titles = ["label map", "uncertainty (bits)", "provenance"] + (["truth (simulation)"] if has_truth else [])
    ncol = len(titles)
    fig = make_subplots(rows=1, cols=ncol, subplot_titles=titles, horizontal_spacing=0.04)
    frames_data = list(_slice_panels(v, which))
    px = v.pixel_um
    if which == "z":
        xs, ys = np.arange(v.provenance.shape[2]) * px, np.arange(v.provenance.shape[1]) * px
        ytitle, labels = "y (µm)", [f"z = {v.z_um[k]:.0f} µm" for k, _ in frames_data]
    else:
        xs, ys = np.arange(v.provenance.shape[2]) * px, np.asarray(v.z_um)
        ytitle, labels = "z (µm)", [f"y = {k * px:.0f} µm" for k, _ in frames_data]

    def traces(arrs, axes=True):
        a = [np.asarray(x).astype(np.float32 if i == 1 else np.int8) for i, x in enumerate(arrs)]  # labels/provenance as int8
        ax = dict(x=xs.astype(np.float32), y=np.asarray(ys, dtype=np.float32)) if axes else {}
        out = [go.Heatmap(z=a[0], colorscale=_label_colorscale(c), zmin=0, zmax=max(c - 1, 1), showscale=False,
                          hovertemplate="class %{z}<extra></extra>", **ax),
               go.Heatmap(z=a[1], colorscale="Magma", zmin=0, zmax=float(np.log2(c)),
                          colorbar=dict(title="bits", len=0.5, x=0.46 if ncol == 4 else 0.64), **ax),
               go.Heatmap(z=a[2], colorscale=[[0, PROV_COLORS[0]], [1 / 3, PROV_COLORS[1]],
                                              [2 / 3, PROV_COLORS[2]], [1, PROV_COLORS[3]]],
                          zmin=0, zmax=3, showscale=False,
                          hovertemplate="provenance %{z} (1 obs, 2 interp)<extra></extra>", **ax)]
        if has_truth:
            out.append(go.Heatmap(z=a[3], colorscale=_label_colorscale(c), zmin=0, zmax=max(c - 1, 1),
                                  showscale=False, hovertemplate="true class %{z}<extra></extra>", **ax))
        return out

    first = traces(frames_data[0][1])
    for j, t in enumerate(first):
        fig.add_trace(t, row=1, col=j + 1)
    fig.frames = [go.Frame(data=traces(arrs, axes=False), name=str(k)) for k, arrs in frames_data]
    fig.update_layout(
        sliders=[dict(active=0, pad=dict(t=40), currentvalue=dict(prefix="slice: "),
                      steps=[dict(method="animate", label=lab, args=[[str(k)], dict(mode="immediate",
                             frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                             for (k, _), lab in zip(frames_data, labels)])],
        height=420, margin=dict(l=40, r=10, t=50, b=0),
        title=dict(text=("x–y slices through z" if which == "z" else "x–z slices through y (gaps show as interpolated bands)"), x=0.5),
    )
    for j in range(ncol):
        fig.update_yaxes(title_text=ytitle if j == 0 else None, row=1, col=j + 1,
                         autorange="reversed" if which == "y" else True)
        fig.update_xaxes(title_text="x (µm)", row=1, col=j + 1, scaleanchor=f"y{j + 1 if j else ''}" if which == "z" else None)
    return fig


def build_html(v: ViewData, path: str | Path, step: int | None = None, offline: bool = True) -> Path:
    """Write one HTML file with the 3D view and both slice explorers.

    ``offline=True`` embeds plotly.js (about 5 MB, works without internet); ``False`` loads it from a CDN.
    """
    if step is None:
        step = max(1, int(np.ceil(max(v.provenance.shape[1:]) / 48)))  # keep the volume trace light
    counts = _prov_counts(v)
    legend = "".join(
        f"<span class='sw' style='background:{PROV_COLORS[i]}'></span>{name} " for i, name in
        ((0, "unsampled"), (1, "observed"), (2, "interpolated"), (3, "synthetic")))
    parts = [f"<div class='banner'>{html.escape(BANNER)}</div>", f"<h2>{html.escape(v.title)}</h2>",
             f"<p>{v.provenance.shape[0]}×{v.provenance.shape[1]}×{v.provenance.shape[2]} voxels "
             f"(z steps {v.dz_um:g} µm, {v.pixel_um:g} µm/px). Provenance of sampled voxels: "
             + ", ".join(f"{k} {p:.0%}" for k, p in counts.items()) + f". {legend}</p>"]
    if v.notes:
        parts.append("<ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in v.notes) + "</ul>")
    figs = [figure_3d(v, step=step), figure_slices(v, "z"), figure_slices(v, "y")]
    for i, f in enumerate(figs):
        parts.append(f.to_html(full_html=False, include_plotlyjs=((True if offline else "cdn") if i == 0 else False), config={"displaylogo": False}))
    css = ("body{font-family:system-ui,sans-serif;max-width:1250px;margin:1rem auto;padding:0 1rem}"
           ".banner{background:#fff3cd;border:1px solid #e0c36a;padding:8px;margin-bottom:1rem;position:sticky;top:0;z-index:5}"
           ".sw{display:inline-block;width:12px;height:12px;margin:0 4px 0 12px;border:1px solid #888}")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(v.title)}</title>"
           f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


def figure_slice_at(v: ViewData, which: str, k: int):
    """A single static slice figure (used by the Streamlit app, where the slider is a Streamlit widget)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    c = v.p_hat.shape[0]
    arrs = dict(_slice_panels_at(v, which, k))
    titles = ["label map", "uncertainty (bits)", "provenance"] + (["truth (simulation)"] if v.truth is not None else [])
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles, horizontal_spacing=0.04)
    px = v.pixel_um
    xs = np.arange(v.provenance.shape[2]) * px
    ys = np.arange(v.provenance.shape[1]) * px if which == "z" else np.asarray(v.z_um)
    prov_cs = [[0, PROV_COLORS[0]], [1 / 3, PROV_COLORS[1]], [2 / 3, PROV_COLORS[2]], [1, PROV_COLORS[3]]]
    spec = [("label", _label_colorscale(c), 0, max(c - 1, 1)), ("unc", "Magma", 0, float(np.log2(c))),
            ("prov", prov_cs, 0, 3), ("truth", _label_colorscale(c), 0, max(c - 1, 1))]
    for j in range(len(titles)):
        key, cs, lo, hi = spec[j]
        fig.add_trace(go.Heatmap(z=arrs[key], x=xs, y=ys, colorscale=cs, zmin=lo, zmax=hi, showscale=(key == "unc"),
                                 colorbar=dict(title="bits", len=0.8) if key == "unc" else None), row=1, col=j + 1)
        fig.update_xaxes(title_text="x (µm)", row=1, col=j + 1)
        fig.update_yaxes(title_text=("y (µm)" if which == "z" else "z (µm)") if j == 0 else None, row=1, col=j + 1,
                         autorange="reversed" if which == "y" else True)
    fig.update_layout(height=380, margin=dict(l=40, r=10, t=40, b=30))
    return fig


def _slice_panels_at(v: ViewData, which: str, k: int):
    if which == "z":
        out = {"label": v.map_labels[k], "unc": v.u_total[k], "prov": v.provenance[k]}
        if v.truth is not None:
            out["truth"] = v.truth[k]
    else:
        out = {"label": v.map_labels[:, k, :], "unc": v.u_total[:, k, :], "prov": v.provenance[:, k, :]}
        if v.truth is not None:
            out["truth"] = v.truth[:, k, :]
    return out.items()
