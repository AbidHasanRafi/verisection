"""VeriSection: uncertainty-aware 3D histopathology (research use only).

This package implements the v0 core: provenance,
simulation with known ground truth, registration/segmentation ensembles, gap-aware
interpolation, uncertainty propagation, calibration, texture, QC, evidence-gated
verdicts and the section planner.
"""
from ._version import __version__
from .provenance import INTENDED_USE, Manifest, Provenance, ProvenanceError
from .volume import ProbVolume, SectionStack, export_volume

__all__ = [
    "__version__",
    "INTENDED_USE",
    "Manifest",
    "Provenance",
    "ProvenanceError",
    "ProbVolume",
    "SectionStack",
    "export_volume",
]
