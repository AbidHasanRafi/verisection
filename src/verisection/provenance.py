"""Provenance labels and run manifests.

Every voxel carries one of four provenance labels. Exports without provenance and
a manifest are refused by construction (see ``volume.export_volume``).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ._version import __version__

SCHEMA_VERSION = "0.1"
INTENDED_USE = "research-only"


class Provenance(IntEnum):
    """Epistemic status of a voxel."""

    UNSAMPLED = 0  # outside the tissue/field of view covered by any section
    OBSERVED = 1  # lies inside a physical section slab
    INTERPOLATED = 2  # lies in a gap between slabs; inferred with quantified uncertainty
    SYNTHETIC = 3  # generated (Mode B); illustrative only


LEGEND: dict[str, str] = {str(int(p)): p.name.capitalize() for p in Provenance}


class ProvenanceError(ValueError):
    """Raised when provenance is missing or invalid."""


def validate_provenance(arr: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return ``arr`` if it is a valid provenance array, else raise ``ProvenanceError``."""
    if arr is None:
        raise ProvenanceError("provenance array is required")
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        raise ProvenanceError(f"provenance must be uint8, got {arr.dtype}")
    if arr.size and int(arr.max()) > int(max(Provenance)):
        raise ProvenanceError("provenance contains labels outside the legend")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ProvenanceError(f"provenance shape {arr.shape} != expected {shape}")
    return arr


def _software_versions() -> dict[str, str]:
    return {
        "verisection": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 of a JSON-serialisable configuration."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


class StageRecord(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Manifest(BaseModel):
    """Run manifest: everything needed to trace an output back to its computation."""

    schema_version: str = SCHEMA_VERSION
    created: str = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )
    software: dict[str, str] = Field(default_factory=_software_versions)
    seed: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    provenance_legend: dict[str, str] = Field(default_factory=lambda: dict(LEGEND))
    intended_use: str = INTENDED_USE

    def add_stage(self, name: str, **params: Any) -> "Manifest":
        self.stages.append(StageRecord(name=name, params=params))
        return self

    def add_decision(self, **record: Any) -> "Manifest":
        self.decisions.append(record)
        return self

    def finalize(self) -> "Manifest":
        self.config_hash = config_hash(self.config)
        return self

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.finalize().to_json(), encoding="utf-8")
        return path
