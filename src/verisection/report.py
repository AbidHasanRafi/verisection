"""Self-contained HTML report for a section stack and (optionally) a pipeline result.

No GUI needed: figures are rendered with matplotlib and embedded as base64 PNG. The report
leads with the intended-use banner, shows QC diagnostics and any registration failures, and
shows provenance next to every reconstructed view.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np

from .pipeline import PipelineResult
from .provenance import INTENDED_USE
from .qc import passport, qc_decisions
from .viewer import provenance_summary
from .volume import SectionStack


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _thumbs(stack: SectionStack, n: int = 8) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.linspace(0, stack.n_sections - 1, min(n, stack.n_sections)).astype(int)
    fig, ax = plt.subplots(1, len(idx), figsize=(2.0 * len(idx), 2.2))
    for a, k in zip(np.atleast_1d(ax), idx):
        a.imshow(stack.images[k], cmap="gray")
        a.set_title(f"#{k}  z={stack.z_um[k]:.0f} µm", fontsize=8)
        a.axis("off")
    return _png(fig)


def _recon_views(res: PipelineResult) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = res.maps
    y = m.p_hat.shape[2] // 2
    ext = [0, m.p_hat.shape[3] * m.pixel_um, m.z_um[-1], m.z_um[0]]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    ax[0].imshow(m.map_labels[:, y], cmap="viridis", aspect="auto", extent=ext, interpolation="nearest")
    ax[0].set_title("label map (x–z slice)")
    im = ax[1].imshow(m.u_total[:, y], cmap="magma", aspect="auto", extent=ext, interpolation="nearest")
    ax[1].set_title("uncertainty (bits)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(m.provenance[:, y], cmap="Greys", vmin=0, vmax=3, aspect="auto", extent=ext, interpolation="nearest")
    ax[2].set_title("provenance (dark = observed)")
    v = m.var_reg[:, y] + 0 * m.var_seg[:, y]
    im2 = ax[3].imshow(v, cmap="cividis", aspect="auto", extent=ext, interpolation="nearest")
    ax[3].set_title("registration variance")
    fig.colorbar(im2, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_xlabel("x (µm)")
    ax[0].set_ylabel("z (µm)")
    return _png(fig)


def build_report(
    stack: SectionStack, path: str | Path, result: PipelineResult | None = None, title: str = "VeriSection report"
) -> Path:
    """Write the HTML report and return its path."""
    pp = [passport(im, stack.images[k - 1] if k else None) for k, im in enumerate(stack.images)]
    dec = qc_decisions(pp)
    rows = "".join(
        f"<tr><td>{k}</td><td>{stack.z_um[k]:.1f}</td><td>{p['focus_lapvar']:.3g}</td><td>{p['noise_sigma']:.3g}</td>"
        f"<td>{p.get('continuity_wasserstein', float('nan')):.3g}</td>"
        f"<td class='{d.verdict}'>{d.verdict}</td><td>{html.escape('; '.join(d.reasons))}</td></tr>"
        for k, (p, d) in enumerate(zip(pp, dec))
    )
    parts = [
        f"<div class='banner'>Research use only ({INTENDED_USE}). Not a medical device. Not for diagnosis, prognosis or treatment decisions.</div>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>{stack.n_sections} sections, {stack.shape[0]}×{stack.shape[1]} px at {stack.pixel_um:.3g} µm/px, "
        f"thickness {stack.thickness_um:g} µm, spacing {np.median(stack.spacing_um):g} µm "
        f"{'(assumed, not read from the files)' if stack.meta.get('assumed_spacing') else ''}.</p>",
        f"<img src='{_thumbs(stack)}' alt='section thumbnails'>",
        "<h2>Section QC (rule-based v0)</h2>",
        "<table><tr><th>#</th><th>z µm</th><th>focus</th><th>noise σ</th><th>continuity</th><th>verdict</th><th>reasons</th></tr>"
        + rows + "</table>",
    ]
    if result is not None:
        r = result
        prov = provenance_summary(r.maps)
        parts += [
            "<h2>Registration</h2>",
            f"<p>method <b>{r.reg.method}</b>, {r.reg.n_samples} ensemble samples. "
            + (f"Median post-alignment NCC {np.median(r.reg.ncc[:, 1:]):.2f}. " if r.reg.ncc is not None else "")
            + (f"<b>Registration fell back to the previous transform for {int(r.reg.failed.sum())} section-sample pairs.</b>"
               if r.reg.failed is not None and r.reg.failed.any() else "No registration failures flagged.")
            + "</p>",
            "<h2>Reconstruction</h2>",
            "<p>Provenance of sampled voxels: "
            + ", ".join(f"{k} {v:.0%}" for k, v in prov.items())
            + ". Interpolated voxels are model inference, not observation.</p>",
            f"<img src='{_recon_views(r)}' alt='reconstruction views'>",
            "<h2>Manifest</h2>",
            f"<pre>{html.escape(r.manifest.finalize().to_json())}</pre>",
        ]
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
           "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 8px;font-size:13px}"
           ".banner{background:#fff3cd;border:1px solid #e0c36a;padding:8px;margin-bottom:1rem}"
           ".flag{background:#fff3cd}.exclude{background:#f8d7da}.accept{background:#e7f5e8}"
           "pre{background:#f6f6f6;padding:8px;overflow:auto;font-size:12px}img{max-width:100%}")
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
