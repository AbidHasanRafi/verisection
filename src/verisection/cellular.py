"""3D nucleus linking and count correction.

Nuclei (5–15 µm across) are cut by 3–5 µm sections, so one nucleus usually shows up as 2D profiles
in one to several consecutive sections. Summing profiles over-counts; linking profiles across
sections and correcting for the probability that a nucleus is cut at all gives an object count.

This module works on 2D profile tables (centroid, radius) such as those produced by a nucleus
segmenter, plus a registration uncertainty that is used to gate and score matches. It includes a
simulator with known 3D truth, because there is no public dataset here with both cellular
resolution and 3D ground truth.

Assumptions that must hold for the estimators (and are tested only in simulation): roughly
spherical-to-ellipsoidal nuclei, a mean z-caliper ``D`` that is known or estimated from the 2D
size distribution, sections not overlapping, and a correctly registered stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------------ simulation
@dataclass
class Nuclei:
    centers: np.ndarray  # (N, 3) z, y, x in µm
    semi_axes: np.ndarray  # (N, 3) z, y, x semi-axes in µm


def simulate_nuclei(
    box_um: tuple[float, float, float] = (140.0, 200.0, 200.0),
    density_per_um3: float = 1.0e-4,
    semi_axes_um: tuple[float, float, float] = (3.2, 4.2, 4.2),
    size_cv: float = 0.2,
    seed: int = 0,
) -> Nuclei:
    """Poisson nuclei with log-normally varying ellipsoid sizes in a box (no overlap constraint)."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(density_per_um3 * np.prod(box_um))
    centers = rng.uniform(0, 1, (n, 3)) * np.asarray(box_um)
    scale = rng.lognormal(0.0, size_cv, (n, 1))
    axes = np.asarray(semi_axes_um)[None] * scale * rng.lognormal(0.0, 0.08, (n, 3))
    return Nuclei(centers, axes)


@dataclass
class Profiles:
    """2D nucleus profiles of one section."""

    section: int
    xy: np.ndarray  # (n, 2) y, x centroids (µm)
    radius: np.ndarray  # (n,) equivalent radius (µm)
    ids: np.ndarray  # (n,) true nucleus id (simulation) or -1


def cut_section(
    nuclei: Nuclei,
    z_center_um: float,
    thickness_um: float,
    section: int,
    rng: np.random.Generator,
    min_radius_um: float = 1.2,
    centroid_jitter_um: float = 0.3,
    miss_rate: float = 0.05,
    false_positive_rate_per_1000um2: float = 0.0,
    area_um2: float = 0.0,
) -> Profiles:
    """Profiles seen in one slab: the widest cross-section of each cut ellipsoid within the slab."""
    z0, z1 = z_center_um - thickness_um / 2, z_center_um + thickness_um / 2
    zc, a = nuclei.centers[:, 0], nuclei.semi_axes[:, 0]
    cut = (zc + a > z0) & (zc - a < z1)
    z_star = np.clip(zc, z0, z1)  # slab level where the cross-section is widest
    frac = 1.0 - ((z_star - zc) / a) ** 2
    frac = np.clip(frac, 0.0, 1.0)
    r_eq = np.sqrt(nuclei.semi_axes[:, 1] * nuclei.semi_axes[:, 2]) * np.sqrt(frac)
    keep = cut & (r_eq >= min_radius_um) & (rng.random(len(zc)) > miss_rate)
    idx = np.flatnonzero(keep)
    xy = nuclei.centers[idx][:, 1:] + rng.normal(0, centroid_jitter_um, (len(idx), 2))
    ids = idx.astype(int)
    radius = r_eq[idx] * rng.lognormal(0, 0.05, len(idx))
    if false_positive_rate_per_1000um2 > 0 and area_um2 > 0:
        nfp = rng.poisson(false_positive_rate_per_1000um2 * area_um2 / 1000.0)
        if nfp:
            xy = np.vstack([xy, rng.uniform(0, 1, (nfp, 2)) * np.sqrt(area_um2)])
            radius = np.concatenate([radius, rng.uniform(min_radius_um, 3.0, nfp)])
            ids = np.concatenate([ids, -np.ones(nfp, int)])
    return Profiles(section, xy, radius, ids)


# ----------------------------------------------------------------------------- linking
def link_pair(
    a: Profiles, b: Profiles, tolerance_um: float, max_size_ratio: float = 2.5, p_min: float = 0.05
) -> list[tuple[int, int, float]]:
    """Match profiles of adjacent sections; returns (i, j, match_probability) triples.

    Cost is the centroid distance in units of the tolerance (registration sigma and centroid jitter
    combined with the nuclear radius, since a nucleus's profile moves less than its size between
    adjacent sections), gated by a size-ratio test; the assignment is solved with the Hungarian
    algorithm so each profile matches at most one profile.
    """
    if len(a.xy) == 0 or len(b.xy) == 0:
        return []
    d = np.linalg.norm(a.xy[:, None, :] - b.xy[None, :, :], axis=2)
    r_max = np.maximum(a.radius[:, None], b.radius[None, :])
    r_min = np.minimum(a.radius[:, None], b.radius[None, :])
    sigma = np.sqrt(tolerance_um**2 + (0.35 * r_max) ** 2)  # allowed centroid shift between sections
    z = d / sigma
    prob = np.exp(-0.5 * z**2)
    prob[(r_max / np.maximum(r_min, 1e-9)) > max_size_ratio] = 0.0
    cost = -np.log(np.maximum(prob, 1e-12))
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(prob[i, j])) for i, j in zip(rows, cols) if prob[i, j] >= p_min]


def link_series(profiles: list[Profiles], tolerance_um: float, p_min: float = 0.05) -> list[list[tuple[int, int]]]:
    """Link adjacent sections and merge matches into tracks (3D objects) with union-find.

    Returns tracks as lists of (section, profile_index).
    """
    offsets = np.concatenate([[0], np.cumsum([len(p.xy) for p in profiles])])
    parent = list(range(int(offsets[-1])))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in range(len(profiles) - 1):
        for i, j, _ in link_pair(profiles[s], profiles[s + 1], tolerance_um, p_min=p_min):
            ra, rb = find(int(offsets[s]) + i), find(int(offsets[s + 1]) + j)
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list[tuple[int, int]]] = {}
    for s, p in enumerate(profiles):
        for i in range(len(p.xy)):
            groups.setdefault(find(int(offsets[s]) + i), []).append((s, i))
    return list(groups.values())


def linking_scores(tracks: list[list[tuple[int, int]]], profiles: list[Profiles]) -> dict:
    """Pairwise precision/recall of the tracks against true nucleus ids (simulation only)."""
    lab_of = {}
    for t, tr in enumerate(tracks):
        for s, i in tr:
            lab_of[(s, i)] = t
    truth: dict[int, list[tuple[int, int]]] = {}
    for s, p in enumerate(profiles):
        for i, nid in enumerate(p.ids):
            if nid >= 0:
                truth.setdefault(int(nid), []).append((s, i))
    tp = fp = fn = 0
    for tr in tracks:
        ids = [int(profiles[s].ids[i]) for s, i in tr if profiles[s].ids[i] >= 0]
        n = len(tr)
        pairs_pred = n * (n - 1) // 2
        same = sum(c * (c - 1) // 2 for c in np.bincount(ids)) if ids else 0
        tp += same
        fp += pairs_pred - same
    for members in truth.values():
        n = len(members)
        pairs_true = n * (n - 1) // 2
        by_track = np.bincount([lab_of[m] for m in members]) if members else np.array([0])
        same = sum(c * (c - 1) // 2 for c in by_track)
        fn += pairs_true - same
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"pair_precision": prec, "pair_recall": rec,
            "pair_f1": 2 * prec * rec / max(prec + rec, 1e-12), "n_tracks": len(tracks)}


# ------------------------------------------------------------------------------ counts
def abercrombie_count(profile_count: float, thickness_um: float, mean_caliper_um: float) -> float:
    """Abercrombie (1946) correction of the number of nuclear profiles per section.

    ``N = n * t / (t + D)`` with ``D`` the mean nuclear caliper along z. Assumes ``D`` is known; with
    only 2D data it must be estimated (``estimate_caliper_from_profiles`` assumes isotropy).
    """
    return profile_count * thickness_um / (thickness_um + mean_caliper_um)


def estimate_caliper_from_profiles(profiles: list[Profiles]) -> float:
    """Isotropic guess of the z-caliper: mean 2D equivalent diameter. Biased for flattened nuclei."""
    radii = [p.radius for p in profiles if len(p.radius)]
    if not radii:  # no profiles at all: nothing to estimate from
        return 0.0
    return float(2.0 * np.concatenate(radii).mean())


def estimate_counts(
    profiles: list[Profiles],
    thickness_um: float,
    spacing_um: float,
    area_um2: float,
    tolerance_um: float,
    caliper_um: float | None = None,
) -> dict:
    """Estimated number of nuclei in the sampled volume ``area x (n_sections * spacing)``.

    With ``t`` the section thickness, ``s`` the spacing and ``D`` the mean nuclear z-caliper, a slab
    cuts every nucleus whose centre lies within ``t + D`` of the section plane, so

    * ``naive``       = n_profiles * s / t       treats every profile as one nucleus (biased high by (t+D)/t);
    * ``abercrombie`` = n_profiles * s / (t + D) the classic correction;
    * ``tracks_ht``   = n_tracks / min(1, (t+D)/s)  counts linked objects and divides by the chance that a
      nucleus is cut by any section (Horvitz-Thompson style); needs good linking, and is only informative
      when nuclei span several sections.
    """
    D = estimate_caliper_from_profiles(profiles) if caliper_um is None else caliper_um
    n_prof = float(sum(len(p.xy) for p in profiles))
    tracks = link_series(profiles, tolerance_um)
    p_cut = min(1.0, (thickness_um + D) / spacing_um)
    return {
        "n_profiles": n_prof, "n_sections": len(profiles), "caliper_um": D, "n_tracks": len(tracks),
        "naive": n_prof * spacing_um / thickness_um,
        "abercrombie": abercrombie_count(n_prof, thickness_um, D) * spacing_um / thickness_um,
        "tracks_ht": len(tracks) / p_cut,
        "area_um2": area_um2,
    }
