"""Measurements on reconstructed volumes with propagated uncertainty.

A measurement is computed on *every* Monte-Carlo sample. Its interval combines the
registration/segmentation spread with the sectioning (kriging) variance of the
estimator, because a plain MC spread over soft posteriors cannot see the variance
that comes from tissue lying in unsampled gaps. The combination is a v0
Gaussian approximation whose coverage must be checked in simulation (it is, in
``demo``), not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .provenance import Provenance


def class_volume_fraction(p: np.ndarray, provenance: np.ndarray, c: int) -> float:
    """Mean posterior of class ``c`` over sampled voxels."""
    valid = provenance > 0
    return float(p[c][valid].mean())


def observed_support_fraction(provenance: np.ndarray) -> float:
    valid = provenance > 0
    if not valid.any():
        return 0.0
    return float((provenance == Provenance.OBSERVED).sum() / valid.sum())


@dataclass
class MeasurementInterval:
    estimate: float
    lo: float
    hi: float
    halfwidth: float
    sd_mc: float
    sd_krig: float
    observed_fraction: float
    level: float

    def contains(self, truth: float) -> bool:
        return self.lo <= truth <= self.hi

    @property
    def relative_halfwidth(self) -> float:
        return self.halfwidth / max(abs(self.estimate), 1e-9)


def interval_from_samples(
    samples: np.ndarray, krig_var: float, observed_fraction: float, level: float = 0.9
) -> MeasurementInterval:
    """Gaussian interval from MC samples (M, N) plus sectioning variance."""
    est = float(np.mean(samples))
    sd_mc = float(np.std(samples))
    sd_k = float(np.sqrt(max(krig_var, 0.0)))
    sd = float(np.sqrt(sd_mc**2 + sd_k**2))
    hw = float(norm.ppf(0.5 + level / 2.0) * sd)
    return MeasurementInterval(est, est - hw, est + hw, hw, sd_mc, sd_k, observed_fraction, level)
