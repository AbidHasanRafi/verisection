"""Indicator variograms along z and correlation-length estimation.

The class correlation length ``xi_c`` is the e-folding length of an exponential
model ``gamma(h) = sill * (1 - exp(-h / xi))``. It needs a densely sampled
sub-block (or a reference 3D volume); otherwise it must be flagged as transferred.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def indicator_variogram_z(
    labels: np.ndarray, dz_um: float, class_id: int, max_lag_planes: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical indicator semivariogram along z for ``class_id``.

    ``labels`` has shape (Z, Y, X). Returns ``(lags_um, gamma)``.
    """
    ind = (labels == class_id).astype(np.float64)
    n_z = ind.shape[0]
    max_lag = (n_z - 1) if max_lag_planes is None else min(max_lag_planes, n_z - 1)
    lags = np.arange(1, max_lag + 1)
    gamma = np.array([0.5 * np.mean((ind[h:] - ind[:-h]) ** 2) for h in lags])
    return lags * dz_um, gamma


def fit_exponential_range(
    lags_um: np.ndarray, gamma: np.ndarray, sill: float, with_nugget: bool = False
) -> float:
    """Fit the e-folding length ``xi`` of ``c0 + (sill - c0) * (1 - exp(-h / xi))``.

    ``with_nugget=True`` also fits the nugget ``c0`` in [0, 0.8*sill]; use it when the labels come
    from a noisy segmentation (label noise adds a nugget and, if ignored, shortens ``xi``).
    """
    lo_b, hi_b = float(lags_um[0]) * 0.1, float(lags_um[-1]) * 20
    p0_xi = max(float(lags_um[len(lags_um) // 4]), float(lags_um[0]))
    try:
        if with_nugget:
            def model(h, xi, c0):
                return c0 + (sill - c0) * (1.0 - np.exp(-h / xi))

            popt, _ = curve_fit(model, lags_um, gamma, p0=[p0_xi, 0.05 * sill],
                                bounds=([lo_b, 0.0], [hi_b, 0.8 * sill]))
        else:
            def model(h, xi):
                return sill * (1.0 - np.exp(-h / xi))

            popt, _ = curve_fit(model, lags_um, gamma, p0=[p0_xi], bounds=(lo_b, hi_b))
        return float(popt[0])
    except (RuntimeError, ValueError):
        return float("nan")


def estimate_class_params(
    labels: np.ndarray, dz_um: float, n_classes: int, max_lag_planes: int = 40
) -> dict[str, np.ndarray]:
    """Per-class prior fractions and correlation lengths from a dense label volume."""
    priors = np.array([(labels == c).mean() for c in range(n_classes)])
    xi = np.full(n_classes, np.nan)
    for c in range(n_classes):
        sill = max(priors[c] * (1.0 - priors[c]), 1e-6)
        lags, gamma = indicator_variogram_z(labels, dz_um, c, max_lag_planes)
        xi[c] = fit_exponential_range(lags, gamma, sill)
    return {"priors": priors, "xi_um": xi}


def estimate_class_params_from_pilot(
    pilot_stack,
    seg,
    n_classes: int,
    registration=None,
    rng: np.random.Generator | None = None,
    max_lag_planes: int | None = None,
) -> dict[str, np.ndarray]:
    """Correlation lengths from a *dense pilot block* using the pipeline's own segmentation.

    No ground truth is used: the pilot sections are registered (translation by default),
    segmented (MAP of the mean ensemble posterior), stacked as consecutive planes and their indicator
    variograms fitted. The pilot must be dense (spacing near the thickness); otherwise ``xi`` is
    biased and this should be flagged as *transferred*, not estimated.
    """
    from .register import register_series

    rng = rng or np.random.default_rng(0)
    if max_lag_planes is None:  # every available lag, at most 40
        max_lag_planes = int(min(pilot_stack.n_sections - 1, 40))
    reg = registration or register_series(pilot_stack.images, pilot_stack.pixel_um, 1, rng, perturb=False)
    transforms = reg.transforms[0]
    labels = []
    for k, img in enumerate(pilot_stack.images):
        p = np.mean([seg.probs(img, n) for n in range(seg.n_members)], axis=0)
        w, valid = transforms[k].apply(p, order=1, cval=0.0)
        lab = w.argmax(axis=0)
        lab = np.where(valid, lab, -1)
        labels.append(lab)
    vol = np.stack(labels)  # (K, H, W); -1 = invalid
    dz = float(np.median(np.diff(pilot_stack.z_um)))
    priors = np.array([(vol == c).sum() / max((vol >= 0).sum(), 1) for c in range(n_classes)])
    xi = np.full(n_classes, np.nan)
    max_lag_um = 0.0
    for c in range(n_classes):
        sill = max(priors[c] * (1 - priors[c]), 1e-6)
        ind = (vol == c).astype(float)
        ok = vol >= 0
        lags, gam = [], []
        for h in range(1, min(max_lag_planes, vol.shape[0] - 1) + 1):
            m = ok[h:] & ok[:-h]
            if m.sum() > 100:
                gam.append(0.5 * np.mean((ind[h:][m] - ind[:-h][m]) ** 2))
                lags.append(h * dz)
        if len(lags) >= 3:
            xi[c] = fit_exponential_range(np.array(lags), np.array(gam), sill, with_nugget=True)
            max_lag_um = max(max_lag_um, lags[-1])
    # a correlation length longer than the sampled extent is an extrapolation, not an estimate
    reliable = np.isfinite(xi) & (xi <= max_lag_um)
    if not reliable.all():
        import warnings

        from .validate import DataWarning

        warnings.warn(
            f"correlation length(s) {np.round(xi[~reliable], 0).tolist()} um exceed the pilot extent "
            f"({max_lag_um:.0f} um) or could not be fitted; the pilot block is too short to estimate them. "
            "Supply xi_um from a longer dense block or accept that gap uncertainty is under-constrained.",
            DataWarning, stacklevel=2,
        )
    return {"priors": priors, "xi_um": xi, "reliable": reliable, "max_lag_um": np.array(max_lag_um),
            "source": np.array("pilot")}


def conservative_xi(par: dict[str, np.ndarray]) -> np.ndarray:
    """Usable correlation lengths from ``estimate_class_params_from_pilot``.

    Unreliable entries (fitted length beyond the sampled extent, or not fitted) are clamped to the
    sampled extent, which is conservative: gap posteriors revert toward the class prior sooner.
    If nothing could be fitted at all, raises ``ValueError``.
    """
    xi = np.asarray(par["xi_um"], dtype=float).copy()
    max_lag = float(par["max_lag_um"])
    reliable = np.asarray(par["reliable"], dtype=bool)
    if max_lag <= 0:
        raise ValueError("no correlation length could be estimated; the pilot block is too short or uniform")
    xi = np.where(np.isfinite(xi), xi, max_lag)
    return np.where(reliable, xi, np.minimum(xi, max_lag))
