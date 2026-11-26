"""Command line interface: ``verisection <command>`` (research use only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from ._version import __version__
from .provenance import INTENDED_USE

app = typer.Typer(add_completion=False, help="VeriSection — uncertainty-aware 3D histopathology (research use only).")


@app.command()
def version() -> None:
    """Print the version and the intended-use statement."""
    typer.echo(f"verisection {__version__}  [{INTENDED_USE}; not a medical device]")


@app.command()
def demo(
    out: Path = typer.Option(Path("runs/demo"), help="Output directory for results.json and figure.png"),
    seed: int = typer.Option(0, help="Random seed"),
    fast: bool = typer.Option(False, help="Smaller sample counts (about 1 minute)"),
) -> None:
    """Run the synthetic end-to-end smoke experiments (no real data involved)."""
    from .demo import run_all

    res = run_all(out, seed=seed, fast=fast)
    u = res["uncertainty"]["variants"]
    typer.echo("Synthetic phantom; machinery check only, not evidence about real tissue.\n")
    typer.echo("Uncertainty vs. error (all sampled voxels; lower AUSE/ECE is better):")
    for name, sc in u.items():
        a = sc["all"]
        typer.echo(f"  {name:24s} acc={a['accuracy']:.3f} ECE={a['ece']:.3f} AUSE={a['ause']:.3f} rho={a['spearman_u_err']:.2f}")
    c = res["conformal"]
    typer.echo(f"\nConformal (LOSO): nominal {c['nominal']:.2f}, empirical {c['coverage_mean']:.3f}, set size {c['mean_set_size']:.2f}")
    for key, title in (
        ("verdicts_assumed_pass", "calibration assumed to pass"),
        ("verdicts", f"calibration measured: {res['calibration_status_measured']}"),
    ):
        v = res[key]
        typer.echo(
            f"\nVerdict policies, mean over {v['n_splits']} random cal/test splits "
            f"({title}; raw interval coverage {v['interval_coverage_all_units']:.2f}, "
            f"target false-claim {v['alpha']:.2f}):"
        )
        for name, ev in v["policies"].items():
            typer.echo(
                f"  {name:28s} false-claim={ev['false_claim_rate']:.3f} "
                f"abstain={ev['abstention_rate']:.2f} lam={ev['lam_hat_mean']:.2f}"
            )
        ra = v["report_everything"]
        typer.echo(f"  {'report everything':28s} false-claim={ra['false_claim_rate']:.3f} abstain={ra['abstention_rate']:.2f}")
    typer.echo(f"\nWrote {Path(out) / 'results.json'} ({res['runtime_s']} s)")


@app.command()
def experiments(
    out: Path = typer.Option(Path("runs/exp"), help="Output directory for experiments.json"),
    n_cal: int = typer.Option(8, help="Calibration replicates (registration-spread recalibration)"),
    n_test: int = typer.Option(16, help="Test replicates (independent phantoms)"),
    workers: int = typer.Option(4, help="Parallel worker processes"),
    preset: str = typer.Option("hard", help="Simulator preset: 'hard' or 'easy'"),
    method: str = typer.Option("nonrigid", help="Registration: translation | rigid | nonrigid"),
) -> None:
    """Multi-replicate synthetic experiments with CIs and paired tests (several minutes)."""
    from .experiments import run_all_experiments

    r = run_all_experiments(out, n_cal, n_test, workers, preset, method)
    m = r["multiseed"]
    typer.echo("Synthetic phantoms; machinery checks only, not evidence about real tissue.\n")
    typer.echo(f"preset={m['preset']} method={m['method']} test replicates={m['n_test']} "
               f"spread multiplier: magnitude-matched={m['sigma_scale']:.2f}, ECE-selected={m['sigma_scale_ece_selected']:.1f}")
    typer.echo(f"registration: mean TRE {m['test_registration']['mean_tre_um']:.1f} um; spread/error raw "
               f"{m['test_registration']['raw_sigma_to_error']:.2f} -> calibrated {m['test_registration']['calibrated_sigma_to_error']:.2f}")
    typer.echo("\nvariant (mean [95% CI over replicates]):  ECE | AUSE")
    for v, d in m["variants"].items():
        typer.echo(f"  {v:15s} ECE {d['ece']}   AUSE {d['ause']}")
    typer.echo("\npaired comparisons (a - b; negative = a better):")
    for k, c in m["comparisons"].items():
        typer.echo(f"  {k:48s} diff {c['mean_diff']:+.3f}  p_holm={c.get('holm_p', float('nan')):.3g}")
    typer.echo(f"\nWrote {Path(out) / 'experiments.json'} ({m['runtime_s']} s)")


@app.command()
def plan(
    domain_um: float = typer.Option(240.0, help="Tissue extent along z (µm)"),
    prior: float = typer.Option(0.3, help="Class fraction of the target compartment"),
    xi_um: float = typer.Option(36.0, help="Correlation length along z (µm)"),
    target_sd: float = typer.Option(0.02, help="Target sd of the compartment-fraction estimate"),
    n_eff_xy: float = typer.Option(1.0, help="Effective number of independent in-plane columns"),
) -> None:
    """Prospective planner: smallest uniform design meeting a target uncertainty."""
    from .planner import FieldModel, plan_uniform_spacing

    m = FieldModel(prior=prior, xi_um=xi_um, n_eff_xy=n_eff_xy)
    r = plan_uniform_spacing((0.0, domain_um), m, target_sd)
    typer.echo(json.dumps(r, indent=2))
    if not r["feasible"]:
        typer.echo("Target not reachable within the section limit; relax the target or add information.")


@app.command()
def qc(
    npy: Path = typer.Argument(..., help=".npy file with a (K, H, W) stack of section images"),
) -> None:
    """Rule-based section QC (v0) on a stack saved as .npy."""
    from .qc import passport, qc_decisions

    arr = np.load(npy)
    pp = [passport(arr[k], arr[k - 1] if k else None) for k in range(arr.shape[0])]
    for d in qc_decisions(pp):
        typer.echo(f"section {d.section:3d}: {d.verdict:8s} {'; '.join(d.reasons)}")


@app.command("run-stack")
def run_stack(
    folder: Path = typer.Argument(..., help="Folder with one image per section (TIFF/PNG/JPEG)"),
    pixel_um: float = typer.Option(..., help="Pixel size of the input images (µm)"),
    thickness_um: float = typer.Option(..., help="Section thickness (µm)"),
    spacing_um: float = typer.Option(None, help="Section spacing (µm); defaults to the thickness (consecutive sections)"),
    max_side: int = typer.Option(256, help="Downsample so the longest side is at most this many pixels"),
    limit: int = typer.Option(None, help="Use only the first N sections"),
    method: str = typer.Option("nonrigid", help="translation | rigid | nonrigid"),
    n_reg: int = typer.Option(3, help="Registration ensemble size"),
    n_seg: int = typer.Option(3, help="Segmentation ensemble size"),
    classes: int = typer.Option(3, help="Number of intensity classes"),
    xi_um: str = typer.Option(None, help="Comma-separated class correlation lengths (µm); estimated from the stack if it is dense"),
    out: Path = typer.Option(Path("runs/real"), help="Output directory"),
) -> None:
    """Reconstruct a folder of section images and write volume, manifest and HTML report."""
    from .io import load_section_folder
    from .pipeline import run_pipeline
    from .register import register_series, register_series_chain
    from .report import build_report
    from .segment import SegmentationEnsemble
    from .validate import InputError
    from .variogram import conservative_xi, estimate_class_params_from_pilot

    try:
        stack = load_section_folder(folder, pixel_um, thickness_um, spacing_um, max_side=max_side, limit=limit)
        rng = np.random.default_rng(0)
        reg = (register_series(stack.images, stack.pixel_um, n_reg, rng) if method == "translation"
               else register_series_chain(stack.images, stack.pixel_um, n_reg, rng, method=method))
        seg = SegmentationEnsemble.fit(stack.images, classes, n_seg, rng)
        if xi_um:
            xi = np.array([float(x) for x in xi_um.split(",")])
            if len(xi) != classes:
                raise InputError(f"--xi-um needs {classes} values, got {len(xi)}")
            mode = "kriging"
        elif float(np.median(stack.spacing_um)) <= 2.0 * stack.thickness_um:
            par = estimate_class_params_from_pilot(stack, seg, classes, registration=reg.point(), rng=rng)
            xi = conservative_xi(par)
            mode = "kriging"
            typer.echo(f"correlation lengths from the dense stack (um): {np.round(par['xi_um'], 1).tolist()}; "
                       f"reliable={par['reliable'].tolist()}; used (clamped to the {float(par['max_lag_um']):.0f} um "
                       f"sampled extent when unreliable): {np.round(xi, 1).tolist()}")
        else:
            xi, mode = None, "linear"
            typer.echo("sparse stack and no --xi-um given: using linear interpolation (uncertainty in gaps is NOT kriging-based)")
        res = run_pipeline(stack, n_classes=classes, interp_mode=mode, xi_um=xi, reg=reg, seg=seg, validate=False)
    except InputError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    out.mkdir(parents=True, exist_ok=True)
    res.export(out / "volume.npz")
    rep = build_report(stack, out / "report.html", res)
    from .workstation import payload_from_npz, write_standalone

    ws = write_standalone(payload_from_npz(out / "volume.npz"), out / "workstation.html")
    typer.echo(f"wrote {out / 'volume.npz'}, manifest, {rep} and {ws} (interactive 3D workstation)")


@app.command("real-conformal")
def real_conformal(
    folder: Path = typer.Argument(..., help="Folder with a DENSE stack (spacing ~ thickness)"),
    pixel_um: float = typer.Option(...), thickness_um: float = typer.Option(...),
    step: int = typer.Option(4), max_side: int = typer.Option(192), limit: int = typer.Option(None),
    method: str = typer.Option("nonrigid"), alpha: float = typer.Option(0.1),
    out: Path = typer.Option(Path("runs/real_conformal")),
) -> None:
    """Blocked split-conformal coverage check on a real dense stack (proxy truth)."""
    import json

    from .io import load_section_folder
    from .real_eval import real_conformal_experiment
    from .validate import InputError

    try:
        stack = load_section_folder(folder, pixel_um, thickness_um, None, max_side=max_side, limit=limit)
        r = real_conformal_experiment(stack, step=step, alpha=alpha, method=method)
    except (InputError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_conformal.json").write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
    for name, d in r["directions"].items():
        typer.echo(f"  {name}: coverage {d['coverage_mean']:.3f} (nominal {r['nominal']:.2f}), "
                   f"min section {d['coverage_min_section']:.3f}, set size {d['mean_set_size']:.2f}")
    typer.echo(r["note"])


@app.command("real-eval")
def real_eval(
    folder: Path = typer.Argument(..., help="Folder with a DENSE stack (spacing ~ thickness)"),
    pixel_um: float = typer.Option(...), thickness_um: float = typer.Option(...),
    step: int = typer.Option(4, help="Keep every N-th section; hold out the rest"),
    max_side: int = typer.Option(192), limit: int = typer.Option(None),
    method: str = typer.Option("nonrigid"), classes: int = typer.Option(3),
    out: Path = typer.Option(Path("runs/real_eval")),
) -> None:
    """Hold-out interpolation test on a real dense stack (proxy truth = held-out section's own segmentation)."""
    import json

    from .io import load_section_folder
    from .real_eval import real_subsample_experiment
    from .validate import InputError

    try:
        stack = load_section_folder(folder, pixel_um, thickness_um, None, max_side=max_side, limit=limit)
        r = real_subsample_experiment(stack, step=step, n_classes=classes, method=method)
    except (InputError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_eval.json").write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
    typer.echo(f"{r['n_sections']} sections, keep every {step} ({r['n_kept']} kept, {r['n_heldout']} held out)")
    for name, v in r["variants"].items():
        typer.echo(f"  {name:8s} agreement={v['agreement_with_heldout_segmentation']:.3f} ECE={v['ece']:.3f} "
                   f"AUSE={v['ause']:.3f} by gap={ {k: round(x, 3) for k, x in v['by_distance_to_nearest_kept_section'].items()} }")
    typer.echo(r["note"])


@app.command("stain-qc")
def stain_qc(
    folder: Path = typer.Argument(..., help="Folder with RGB section images"),
    max_side: int = typer.Option(512, help="Downsample before estimation"),
    limit: int = typer.Option(None, help="Use only the first N sections"),
) -> None:
    """Per-section stain-vector diagnostics (angles to default H&E, mean H/E, residual) and drift flags."""
    from .io import IMAGE_SUFFIXES, natural_key, read_rgb
    from .qc import robust_z, stain_passport
    from .validate import InputError

    files = sorted((p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: natural_key(p.name))
    if limit:
        files = files[:limit]
    if not files:
        typer.secho(f"no images under {folder}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    rows = []
    for f in files:
        try:
            rows.append((f.name, stain_passport(read_rgb(f, max_side))))
        except InputError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
    ok = [r for _, r in rows if r["ok"]]
    zs = {}
    for key in ("angle_default_h_deg", "mean_h", "mean_e"):
        vals = np.array([r.get(key, np.nan) if r["ok"] else np.nan for _, r in rows])
        zs[key] = robust_z(vals) if np.isfinite(vals).sum() >= 3 else np.zeros(len(rows))
    typer.echo(f"{'section':16s} {'tissue':>6s} {'angH':>6s} {'angE':>6s} {'meanH':>6s} {'meanE':>6s} {'resid':>6s}  flag")
    for k, (name, r) in enumerate(rows):
        if not r["ok"]:
            typer.echo(f"{name:16s} {r['tissue_fraction']:6.2f}  not enough tissue for stain estimation")
            continue
        flag = "DRIFT" if max(abs(zs[key][k]) for key in zs) > 3.5 else ""
        typer.echo(f"{name:16s} {r['tissue_fraction']:6.2f} {r['angle_to_ruifrok_h_deg']:6.1f} {r['angle_to_ruifrok_e_deg']:6.1f} "
                   f"{r['mean_h']:6.2f} {r['mean_e']:6.2f} {r['residual_rms']:6.3f}  {flag}")
    typer.echo(f"{len(ok)}/{len(rows)} sections had enough tissue; angles are to the Ruifrok H&E defaults.")


@app.command()
def ui(
    port: int = typer.Option(8765, help="Port for the local workstation server (127.0.0.1 only)"),
    no_browser: bool = typer.Option(False, help="Do not open a browser tab automatically"),
) -> None:
    """Start the VeriSection Workstation: menus, linked slice views, WebGL 3D view, analysis tabs."""
    from .workstation import serve

    try:
        srv = serve(port, open_browser=not no_browser)
    except OSError as exc:
        typer.secho(f"cannot start the server on port {port}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"VeriSection Workstation at http://127.0.0.1:{port}/  (Ctrl+C to stop; local access only)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        srv.server_close()


@app.command()
def simulate(
    out: Path = typer.Option(Path("runs/workstation.html"), help="Standalone workstation HTML to write"),
    preset: str = typer.Option("tissue", help="'tissue' (glands, stroma, immune cells), 'hard' or 'easy'"),
    seed: int = typer.Option(0), n_sections: int = typer.Option(20),
    spacing_um: int = typer.Option(6, help="Section spacing in µm (4, 6, 8, 12, 16)"),
    method: str = typer.Option("translation", help="translation | rigid | nonrigid"),
    interp: str = typer.Option("kriging", help="kriging | linear | nearest"),
    n_reg: int = typer.Option(2), n_seg: int = typer.Option(3),
    extended: bool = typer.Option(False, help="Add leave-one-section-out conformal analysis (slower)"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the result in your browser"),
) -> None:
    """Run one simulation and write a standalone workstation (opens offline, no server needed)."""
    import webbrowser
    from dataclasses import asdict

    from .analysis import build_analysis
    from .simrun import SimParams, run_simulation
    from .workstation import make_payload, write_standalone

    try:
        p = SimParams(preset=preset, seed=seed, n_sections=n_sections, spacing_planes=max(spacing_um // 2, 1),
                      method=method, interp_mode=interp, n_reg=n_reg, n_seg=n_seg)
        res = run_simulation(p, progress=lambda m, f: typer.echo(f"  [{f:4.0%}] {m}"))
        an = build_analysis(res, asdict(p), extended=extended, progress=lambda m, f: typer.echo(f"  [{f:4.0%}] {m}"))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    path = write_standalone(make_payload(res, an), out)
    a = res.metrics["all"]
    typer.echo(f"accuracy vs truth {a['accuracy']:.2f}, ECE {a['ece']:.3f}, mean TRE "
               f"{res.metrics['registration']['mean_tre_um']:.1f} um")
    typer.echo(f"wrote {path}")
    if open_browser:
        webbrowser.open(Path(path).resolve().as_uri())


@app.command()
def view3d(
    volume: Path = typer.Argument(..., help="volume.npz written by run-stack / PipelineResult.export"),
    out: Path = typer.Option(Path("runs/view.html")),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Open an exported volume (with provenance) in the workstation (standalone HTML)."""
    import webbrowser

    from .workstation import payload_from_npz, write_standalone

    try:
        payload = payload_from_npz(volume)
    except (OSError, KeyError, ValueError) as exc:
        typer.secho(f"cannot read {volume}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    path = write_standalone(payload, out)
    typer.echo(f"wrote {path}")
    if open_browser:
        webbrowser.open(Path(path).resolve().as_uri())


@app.command()
def audit(
    root: Path = typer.Argument(..., help="Folder of class sub-folders of 2D patches (e.g. colon_aca/, lung_n/, ...)"),
    out: Path = typer.Option(Path("runs/audit.html"), help="Standalone workstation HTML (Image audit workspace)"),
    per_class: int = typer.Option(16, help="Patches sampled per class (never files named aug_*)"),
    seed: int = typer.Option(0),
    fast: bool = typer.Option(False, help="Skip the stability and interpolation experiments"),
    json_out: Path = typer.Option(None, help="Also write the audit payload as JSON"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the result in your browser"),
) -> None:
    """Audit independent 2D histology patches: stain, quality, texture, descriptor stability, interpolation bias.

    These are not serial sections, so nothing here is a 3D reconstruction and nothing is a diagnostic claim.
    """
    import json
    import webbrowser

    from .imageaudit import run_audit
    from .workstation import write_audit_standalone

    if not Path(root).is_dir():
        typer.secho(f"not a folder: {root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        payload = run_audit(root, per_class, seed, slow=not fast, workers=1, progress=lambda m, f: typer.echo(f"  [{f:4.0%}] {m}"))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    path = write_audit_standalone(payload, out)
    if json_out:
        Path(json_out).write_text(json.dumps(payload, separators=(",", ":"), default=float), encoding="utf-8")
    flagged = sum(r["verdict"] != "accept" for r in payload["rows"])
    typer.echo(f"{len(payload['rows'])} patches, {flagged} flagged or excluded by the pooled QC rule")
    typer.echo(f"wrote {path}")
    if open_browser:
        webbrowser.open(Path(path).resolve().as_uri())
