import json

import numpy as np
import pytest

from verisection import Manifest, Provenance, ProvenanceError, SectionStack
from verisection.provenance import config_hash, validate_provenance
from verisection.volume import ProbVolume, export_volume, make_grid


def test_provenance_labels_are_stable():
    assert [int(p) for p in Provenance] == [0, 1, 2, 3]
    assert Provenance.OBSERVED.name == "OBSERVED"


def test_validate_provenance_rejects_bad_input():
    with pytest.raises(ProvenanceError):
        validate_provenance(None)
    with pytest.raises(ProvenanceError):
        validate_provenance(np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ProvenanceError):
        validate_provenance(np.full((2, 2), 9, dtype=np.uint8))
    with pytest.raises(ProvenanceError):
        validate_provenance(np.zeros((2, 2), dtype=np.uint8), shape=(3, 3))


def test_manifest_roundtrip_and_hash(tmp_path):
    m = Manifest(seed=1, config={"a": 1, "b": [1, 2]})
    m.add_stage("register", n=3).add_decision(id="DB1", verdict="reportable")
    path = m.write(tmp_path / "m.json")
    data = json.loads(path.read_text())
    assert data["intended_use"] == "research-only"
    assert data["config_hash"] == config_hash({"b": [1, 2], "a": 1})  # key order irrelevant
    assert data["stages"][0]["name"] == "register"
    assert data["provenance_legend"]["2"] == "Interpolated"


def _vol(prov=None):
    C, Z, Y, X = 3, 4, 5, 5
    p = np.full((C, Z, Y, X), 1.0 / C, dtype=np.float32)
    prov = np.ones((Z, Y, X), dtype=np.uint8) if prov is None else prov
    return ProbVolume(p, prov, np.arange(Z) * 4.0, 2.0, 4.0, np.zeros(Z), ["a", "b", "c"])


def test_probvolume_validates_probabilities_and_shapes():
    _vol()
    bad = np.full((3, 4, 5, 5), 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        ProbVolume(bad, np.ones((4, 5, 5), np.uint8), np.arange(4.0), 2.0, 4.0, np.zeros(4), ["a", "b", "c"])
    with pytest.raises(ProvenanceError):
        _vol(prov=np.ones((4, 5, 5), dtype=np.float32))


def test_export_requires_manifest_and_writes_provenance(tmp_path):
    v = _vol()
    with pytest.raises(ProvenanceError):
        export_volume(v, tmp_path / "v.npz", manifest=None)  # type: ignore[arg-type]
    out = export_volume(v, tmp_path / "v.npz", Manifest(seed=0))
    assert out.exists() and (tmp_path / "v.manifest.json").exists()
    with np.load(out) as z:
        assert "provenance" in z.files and z["provenance"].dtype == np.uint8


def test_observed_fraction():
    prov = np.ones((4, 5, 5), dtype=np.uint8)
    prov[1:3] = Provenance.INTERPOLATED
    prov[3] = Provenance.UNSAMPLED
    v = _vol(prov)
    assert v.observed_fraction() == pytest.approx(25 / 75)


def test_section_stack_validation():
    imgs = [np.zeros((4, 4)) for _ in range(3)]
    SectionStack(imgs, [0, 8, 16], 4.0, 1.0)
    with pytest.raises(ValueError):
        SectionStack(imgs, [0, 3, 16], 4.0, 1.0)  # slabs overlap
    with pytest.raises(ValueError):
        SectionStack(imgs, [0, 8, 8], 4.0, 1.0)  # not increasing
    with pytest.raises(ValueError):
        SectionStack(imgs[:1], [0], 4.0, 1.0)


def test_make_grid_marks_slabs_and_gaps():
    z = np.array([0.0, 12.0, 24.0])
    centers, plane_section, gap = make_grid(z, 4.0, 4.0)
    assert np.allclose(centers, np.arange(0, 25, 4.0))
    assert plane_section.tolist() == [0, -1, -1, 1, -1, -1, 2]
    assert gap[0] == 0 and gap[1] == 2 and gap[3] == 0
