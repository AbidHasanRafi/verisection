import io

import numpy as np
import pytest

from verisection import viewer3d as V
from verisection.simrun import SimParams, run_simulation


@pytest.fixture(scope="module")
def sim():
    return run_simulation(SimParams(preset="easy", method="translation", n_reg=2, n_seg=2, interp_mode="linear",
                                    n_sections=8, seed=1))


def test_simulation_result_has_view_metrics_verdict_and_plan(sim):
    m = sim.metrics
    assert 0.3 < m["all"]["accuracy"] <= 1.0 and m["registration"]["mean_tre_um"] >= 0
    assert m["measurement"]["verdict"] in ("reportable", "reportable_with_caveat", "insufficient_evidence")
    assert "calibration untested in this run" in m["measurement"]["reasons"]
    assert len(m["planner_next_sections_um"]) == 5
    v = sim.view
    assert v.p_hat.shape[1:] == v.provenance.shape == v.u_total.shape == v.truth.shape
    assert any("Synthetic phantom" in n for n in v.notes)


def test_simulation_rejects_bad_interpolation_and_clamps_section_count():
    with pytest.raises(ValueError, match="interp_mode"):
        run_simulation(SimParams(interp_mode="cubic"))
    r = run_simulation(SimParams(preset="easy", method="translation", n_reg=1, n_seg=1, interp_mode="linear",
                                 n_sections=40, spacing_planes=10))
    assert r.metrics["n_sections"] <= 12 and any("reduced" in line for line in r.log)


def test_viewer_figures_and_html_contain_provenance_and_banner(sim, tmp_path):
    f3 = V.figure_3d(sim.view, step=3)
    names = [t.name for t in f3.data]
    assert "observed sections" in names and any(n.startswith("uncertainty") for n in names)
    assert sum(1 for t in f3.data if t.type == "volume") == 3 + sim.view.p_hat.shape[0]
    fz, fy = V.figure_slices(sim.view, "z"), V.figure_slices(sim.view, "y")
    assert len(fz.frames) == sim.view.provenance.shape[0] and len(fy.frames) == sim.view.provenance.shape[1]
    assert len(fz.data) == 4  # label, uncertainty, provenance, truth
    assert len(V.figure_slice_at(sim.view, "z", 3).data) == 4
    out = V.build_html(sim.view, tmp_path / "v.html", offline=False)
    html = out.read_text(encoding="utf-8")
    assert "Research use only" in html and "interpolated" in html and html.count("Plotly.newPlot") == 3


def test_view_data_from_exported_npz_roundtrip(sim, tmp_path):
    path = sim.result.export(tmp_path / "vol.npz")
    v = V.view_data_from_npz(path)
    assert v.truth is None and v.p_hat.shape == sim.view.p_hat.shape
    assert np.array_equal(v.provenance, sim.view.provenance)
    v2 = V.view_data_from_npz(io.BytesIO(path.read_bytes()), title="uploaded")
    assert v2.title == "uploaded" and np.allclose(v2.u_total, v.u_total)
    assert len(V.figure_slices(v, "z").data) == 3  # no truth panel for exported volumes
