"""Uncertainty-guided section planner.

Model: for a target that is a linear functional of a class-indicator field along z
(e.g. a compartment volume fraction), the simple-kriging estimation variance
depends only on *where* sections are and on the variogram, not on the observed
values. Expected value of information of an extra section is the reduction of
that variance; the planner ranks candidate positions by reduction per cost.

Assumptions (checked in tests, and limits stated in the plan, R20): stationary
exponential covariance, class indicator treated as one random field, in-plane
columns averaged with an effective independent-column count ``n_eff_xy``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldModel:
    prior: float  # class fraction pi
    xi_um: float  # e-folding correlation length along z
    nugget: float = 0.05  # observation noise as a fraction of the sill
    n_eff_xy: float = 1.0  # effective number of independent in-plane columns

    @property
    def sill(self) -> float:
        return max(self.prior * (1.0 - self.prior), 1e-6)


def _cov(a: np.ndarray, b: np.ndarray, m: FieldModel) -> np.ndarray:
    return m.sill * np.exp(-np.abs(a[:, None] - b[None, :]) / m.xi_um)


def mean_estimation_variance(
    z_sections: np.ndarray, domain_um: tuple[float, float], m: FieldModel, n_grid: int = 300
) -> float:
    """Variance of the kriging estimate of the domain-mean fraction."""
    zg = np.linspace(domain_um[0], domain_um[1], n_grid)
    Cgg = _cov(zg, zg, m)
    if len(z_sections) == 0:
        err = Cgg
    else:
        zs = np.asarray(z_sections, dtype=float)
        K = _cov(zs, zs, m) + m.nugget * m.sill * np.eye(len(zs))
        Cgs = _cov(zg, zs, m)
        err = Cgg - Cgs @ np.linalg.solve(K, Cgs.T)
    return float(err.mean() / max(m.n_eff_xy, 1.0))


def evoi(
    z_sections: np.ndarray,
    candidates_um: np.ndarray,
    domain_um: tuple[float, float],
    m: FieldModel,
    n_grid: int = 300,
) -> np.ndarray:
    """Variance reduction from adding each candidate section."""
    base = mean_estimation_variance(z_sections, domain_um, m, n_grid)
    out = np.empty(len(candidates_um))
    for i, c in enumerate(candidates_um):
        out[i] = base - mean_estimation_variance(np.append(z_sections, c), domain_um, m, n_grid)
    return out


def recommend(
    z_sections: np.ndarray,
    candidates_um: np.ndarray,
    domain_um: tuple[float, float],
    m: FieldModel,
    costs: np.ndarray | None = None,
    top_k: int = 5,
    min_separation_um: float = 0.0,
) -> list[dict]:
    """Rank candidate positions by expected variance reduction per cost."""
    candidates_um = np.asarray(candidates_um, dtype=float)
    keep = np.ones(len(candidates_um), bool)
    if min_separation_um > 0 and len(z_sections):
        keep = np.abs(candidates_um[:, None] - np.asarray(z_sections)[None]).min(1) >= min_separation_um
    cand = candidates_um[keep]
    cost = np.ones(len(cand)) if costs is None else np.asarray(costs, float)[keep]
    gain = evoi(np.asarray(z_sections, float), cand, domain_um, m)
    base = mean_estimation_variance(np.asarray(z_sections, float), domain_um, m)
    order = np.argsort(-(gain / cost))[:top_k]
    return [
        {
            "rank": r + 1,
            "z_um": float(cand[i]),
            "variance_reduction": float(gain[i]),
            "relative_sd_reduction": float(1.0 - np.sqrt(max(base - gain[i], 0.0) / base)) if base > 0 else 0.0,
            "cost": float(cost[i]),
            "evoi_per_cost": float(gain[i] / cost[i]),
            "method": "kriging",
        }
        for r, i in enumerate(order)
    ]


def plan_uniform_spacing(
    domain_um: tuple[float, float],
    m: FieldModel,
    target_sd: float,
    max_sections: int = 200,
) -> dict:
    """Smallest uniform design whose estimation sd is at most ``target_sd`` (prospective mode)."""
    L = domain_um[1] - domain_um[0]
    for n in range(1, max_sections + 1):
        z = domain_um[0] + (np.arange(n) + 0.5) * L / n
        sd = float(np.sqrt(mean_estimation_variance(z, domain_um, m)))
        if sd <= target_sd:
            return {"n_sections": n, "spacing_um": L / n, "predicted_sd": sd, "feasible": True}
    return {"n_sections": max_sections, "spacing_um": L / max_sections, "predicted_sd": sd, "feasible": False}


def greedy_design(
    start_um: np.ndarray,
    candidates_um: np.ndarray,
    domain_um: tuple[float, float],
    m: FieldModel,
    n_add: int,
) -> np.ndarray:
    """Add ``n_add`` sections one by one by maximal expected variance reduction."""
    z = np.asarray(start_um, float).copy()
    cand = list(np.asarray(candidates_um, float))
    for _ in range(n_add):
        gains = evoi(z, np.array(cand), domain_um, m, n_grid=200)
        i = int(np.argmax(gains))
        z = np.append(z, cand.pop(i))
    return np.sort(z)


def kriged_domain_mean(
    z_sections: np.ndarray,
    observations: np.ndarray,
    domain_um: tuple[float, float],
    m: FieldModel,
    n_grid: int = 300,
) -> float:
    """Simple-kriging estimate of the domain-mean fraction from per-section class fractions."""
    zs = np.asarray(z_sections, float)
    zg = np.linspace(domain_um[0], domain_um[1], n_grid)
    K = _cov(zs, zs, m) + m.nugget * m.sill * np.eye(len(zs))
    w = np.linalg.solve(K, _cov(zg, zs, m).mean(axis=0))
    return float(m.prior + w @ (np.asarray(observations, float) - m.prior))
