"""Split-conformal prediction sets calibrated on held-out sections.

The exchangeable *unit* is the section (neighbouring voxels are strongly dependent),
so calibration scores come from held-out sections, optionally sub-sampled per
section to limit dependence, and coverage is evaluated on *other* sections.
Coverage claims hold only for the calibration regime that was actually tested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConformal:
    alpha: float = 0.1
    qhat: float = float("nan")
    n_cal: int = 0

    def calibrate(
        self,
        records: list[dict],
        per_section: int | None = 400,
        rng: np.random.Generator | None = None,
    ) -> "SplitConformal":
        """Calibrate from LOSO records (each with ``p`` (C, N) and ``y`` (N,))."""
        rng = rng or np.random.default_rng(0)
        scores = []
        for r in records:
            s = 1.0 - r["p"][r["y"], np.arange(len(r["y"]))]
            if per_section is not None and len(s) > per_section:
                s = rng.choice(s, size=per_section, replace=False)
            scores.append(s)
        s = np.concatenate(scores)
        n = len(s)
        level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.qhat = float(np.quantile(s, level, method="higher"))
        self.n_cal = n
        return self

    def predict_sets(self, p: np.ndarray, non_empty: bool = False) -> np.ndarray:
        """Boolean (C, N) membership of each class in the prediction set.

        Standard split conformal can return *empty* sets when the model is less confident than the
        calibration threshold; they count as misses in ``evaluate``. ``non_empty=True`` always adds the
        top class (coverage can only increase, the guarantee is preserved).
        """
        sets = (1.0 - p) <= self.qhat
        if non_empty:
            sets[p.argmax(axis=0), np.arange(p.shape[1])] = True
        return sets

    def evaluate(self, records: list[dict]) -> dict:
        """Empirical coverage and set size, pooled and per section."""
        per_cov, per_size = [], []
        for r in records:
            sets = self.predict_sets(r["p"])
            per_cov.append(float(sets[r["y"], np.arange(len(r["y"]))].mean()))
            per_size.append(float(sets.sum(axis=0).mean()))
        return {
            "alpha": self.alpha,
            "nominal": 1.0 - self.alpha,
            "coverage_mean": float(np.mean(per_cov)),
            "coverage_min_section": float(np.min(per_cov)),
            "coverage_sd_sections": float(np.std(per_cov)),
            "mean_set_size": float(np.mean(per_size)),
            "n_sections": len(records),
        }
