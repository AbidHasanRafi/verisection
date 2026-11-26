"""Viewer adapter: turns pipeline outputs into layer descriptions for a 3D viewer (e.g. napari).

The layer builder is pure NumPy so it can be tested without a GUI. ``launch_napari`` imports
napari lazily and raises a clear error if it is not installed. The provenance layer is always
included and cannot be omitted: interpolated and synthetic voxels must never be
visually indistinguishable from observed ones.
"""
from __future__ import annotations

import numpy as np

from .provenance import LEGEND, Provenance
from .uncertainty import UncertaintyMaps


def to_layers(maps: UncertaintyMaps, intensity: np.ndarray | None = None) -> list[dict]:
    """Layer dicts: name, kind ('image'|'labels'), data (Z, Y, X), scale (dz, dy, dx) in µm, and metadata."""
    scale = (float(maps.dz_um), float(maps.pixel_um), float(maps.pixel_um))
    valid = maps.provenance > 0
    prov = maps.provenance.astype(np.uint8)
    # opacity-friendly confidence: 1 - entropy / log2(C), zero where unsampled
    conf = np.where(valid, 1.0 - maps.u_total / np.log2(maps.p_hat.shape[0]), 0.0).astype(np.float32)
    layers = [
        {"name": "provenance", "kind": "labels", "data": prov, "scale": scale,
         "metadata": {"legend": dict(LEGEND), "mandatory": True}},
        {"name": "label_map", "kind": "labels", "data": np.where(valid, maps.map_labels + 1, 0).astype(np.uint8),
         "scale": scale, "metadata": {"classes": list(maps.class_names)}},
        {"name": "confidence", "kind": "image", "data": conf, "scale": scale, "colormap": "viridis",
         "metadata": {"definition": "1 - entropy/log2(C)"}},
        {"name": "uncertainty_bits", "kind": "image", "data": maps.u_total.astype(np.float32), "scale": scale,
         "colormap": "magma", "metadata": {"unit": "bits"}},
        {"name": "interp_uncertainty_bits", "kind": "image", "data": maps.u_interp.astype(np.float32),
         "scale": scale, "colormap": "magma", "metadata": {"unit": "bits", "support": "interpolated voxels only"}},
        {"name": "var_registration", "kind": "image", "data": maps.var_reg.astype(np.float32), "scale": scale,
         "colormap": "cividis", "metadata": {"unit": "variance of MAP-class probability"}},
        {"name": "var_segmentation", "kind": "image", "data": maps.var_seg.astype(np.float32), "scale": scale,
         "colormap": "cividis", "metadata": {"unit": "variance of MAP-class probability"}},
    ]
    if intensity is not None:
        layers.insert(0, {"name": "intensity", "kind": "image", "data": intensity, "scale": scale, "colormap": "gray",
                          "metadata": {"note": "observed planes only; gap planes are not image data"}})
    names = [lay["name"] for lay in layers]
    assert "provenance" in names, "provenance layer is mandatory"
    return layers


def launch_napari(maps: UncertaintyMaps, intensity: np.ndarray | None = None):  # pragma: no cover - GUI
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is not installed; `pip install napari[all]` or use `verisection report`") from exc
    viewer = napari.Viewer(ndisplay=3)
    for lay in to_layers(maps, intensity):
        add = viewer.add_labels if lay["kind"] == "labels" else viewer.add_image
        kw = {"name": lay["name"], "scale": lay["scale"], "metadata": lay.get("metadata", {})}
        if lay["kind"] == "image" and "colormap" in lay:
            kw["colormap"] = lay["colormap"]
        add(lay["data"], **kw)
    return viewer


def provenance_summary(maps: UncertaintyMaps) -> dict[str, float]:
    """Fraction of sampled voxels per provenance label (shown in any viewer header)."""
    p = maps.provenance
    sampled = (p > 0).sum()
    return {Provenance(i).name.lower(): float((p == i).sum() / sampled) if sampled else 0.0 for i in (1, 2, 3)}
