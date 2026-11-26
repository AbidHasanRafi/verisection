"""Series registration with an ensemble of transforms.

Methods
-------
``translation``  first-neighbour chain of translations (cross-correlation + robust pose graph);
``rigid``        sequential rotation + translation;
``nonrigid``     rigid followed by a smooth dense displacement (iterative Lucas-Kanade flow).

Rigid/non-rigid registration is *sequential*: section ``k`` is aligned to the already
registered section ``k-1``. Every pairwise result is checked (post-alignment NCC); a
non-rigid refinement that does not improve NCC is discarded, and a pair whose alignment
stays poor is flagged and falls back to the previous transform instead of injecting garbage.

The ensemble spread (``sigma_um``) is a *ranking* signal until it has been recalibrated
against measured error with ``fit_sigma_scale`` (published evidence and our own simulations
both show raw registration-ensemble spread under-states real error).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import optical_flow_ilk, phase_cross_correlation

from .transform import Transform, landmark_grid, mean_transform, sample_coords


# --------------------------------------------------------------------------- ensemble
@dataclass
class RegistrationEnsemble:
    transforms: list[list[Transform]]  # [M][K]
    pixel_um: float
    shape: tuple[int, int]
    sigma_scale: float = 1.0
    ncc: np.ndarray | None = None  # (M, K) post-alignment NCC of each section vs previous
    failed: np.ndarray | None = None  # (M, K) bool: pair failed and fell back
    method: str = "translation"

    @property
    def n_samples(self) -> int:
        return len(self.transforms)

    @property
    def n_sections(self) -> int:
        return len(self.transforms[0])

    # -- translation-only conveniences (kept for backward compatibility) --
    @property
    def shifts_px(self) -> np.ndarray:
        if not all(t.is_pure_translation for row in self.transforms for t in row):
            raise ValueError("shifts_px is only defined for pure-translation ensembles")
        return np.array([[t.t for t in row] for row in self.transforms])

    def mean_shifts(self) -> np.ndarray:
        return self.shifts_px.mean(axis=0)

    # -- spread / calibration --
    def landmark_positions(self, n: int = 6) -> np.ndarray:
        """Raw-section coordinates of fixed common-frame landmarks: (M, K, 2, L)."""
        pts = landmark_grid(self.shape, n)
        return np.array(
            [[sample_coords(t, self.shape, pts) for t in row] for row in self.transforms]
        )

    def sigma_um(self) -> np.ndarray:
        """Per-section spread of landmark positions across samples (µm), times ``sigma_scale``."""
        pos = self.landmark_positions()
        dev = pos - pos.mean(axis=0, keepdims=True)
        per_landmark = np.sqrt((dev**2).sum(axis=2).mean(axis=0))  # (K, L)
        return per_landmark.mean(axis=1) * self.pixel_um * self.sigma_scale

    def point(self) -> "RegistrationEnsemble":
        if all(t.is_pure_translation for row in self.transforms for t in row):
            mean = [Transform.translation(m) for m in self.mean_shifts()]
        else:
            mean = [
                mean_transform([row[k] for row in self.transforms], self.shape)
                for k in range(self.n_sections)
            ]
        return RegistrationEnsemble([mean], self.pixel_um, self.shape, 1.0, None, None, self.method)

    def subset(self, idx: list[int]) -> "RegistrationEnsemble":
        return RegistrationEnsemble(
            [[row[i] for i in idx] for row in self.transforms], self.pixel_um, self.shape,
            self.sigma_scale,
            None if self.ncc is None else self.ncc[:, idx],
            None if self.failed is None else self.failed[:, idx],
            self.method,
        )

    def calibrated(self, scale: float | None = None) -> "RegistrationEnsemble":
        """Return an ensemble whose samples are pulled away from (or toward) the mean by ``scale``.

        This makes the *propagated* spread match the recalibrated sigma (``fit_sigma_scale``).
        """
        scale = self.sigma_scale if scale is None else scale
        if scale == 1.0:
            return self
        mean = [
            mean_transform([row[k] for row in self.transforms], self.shape).coords(self.shape)
            for k in range(self.n_sections)
        ]
        out = []
        for row in self.transforms:
            out.append([
                Transform.from_coords(mean[k] + scale * (t.coords(self.shape) - mean[k]))
                for k, t in enumerate(row)
            ])
        return RegistrationEnsemble(out, self.pixel_um, self.shape, 1.0, self.ncc, self.failed, self.method)


def fit_sigma_scale(
    sigma_um: np.ndarray, error_um: np.ndarray, min_sigma_um: float = 1e-3, method: str = "ratio_of_means"
) -> float:
    """Multiplier mapping the ensemble spread to the measured error.

    ``ratio_of_means`` (default) matches the average magnitude: mean(error)/mean(sigma).
    ``median_ratio`` uses the median of per-section ratios; it is more robust to outliers but
    over-corrects when the ratios are right-skewed (measured: it left the spread ~2x too wide on
    held-out replicates). Section 0 (the gauge) is excluded by the caller. Returns 1.0 if unusable.
    """
    ok = sigma_um > min_sigma_um
    if not ok.any():
        return 1.0
    if method == "median_ratio":
        return float(np.median(error_um[ok] / sigma_um[ok]))
    if method != "ratio_of_means":
        raise ValueError("method must be 'ratio_of_means' or 'median_ratio'")
    return float(error_um[ok].mean() / sigma_um[ok].mean())


# ------------------------------------------------------------------ translation baseline
def _prep(img: np.ndarray, blur: float) -> np.ndarray:
    """Smooth, z-score and zero-pad by half the size (suppresses circular wrap-around)."""
    x = ndi.gaussian_filter(img.astype(np.float64), blur) if blur > 0 else img.astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    return np.pad(x, ((x.shape[0] // 2,) * 2, (x.shape[1] // 2,) * 2))


def pairwise_shift(
    ref: np.ndarray, mov: np.ndarray, upsample: int = 10, blur: float = 1.0, n_iter: int = 3
) -> np.ndarray:
    """Shift to apply to ``mov`` so it aligns with ``ref`` (estimate of ``a_ref - a_mov``).

    Plain (non-phase-normalised) cross-correlation on smoothed, z-scored images: phase
    normalisation returned zero shifts on noisy, partly decorrelated sections. A single
    correlation peak is biased toward zero for larger shifts, so the estimate is refined
    iteratively: ``mov`` is warped by the current estimate and the (smaller) residual is
    re-estimated, which removes most of that bias on the simulated stacks.
    """
    mov = np.asarray(mov, dtype=np.float64)
    total = np.zeros(2)
    for _ in range(n_iter):
        warped = ndi.shift(mov, total, order=1, mode="nearest")
        d, _, _ = phase_cross_correlation(
            _prep(ref, blur), _prep(warped, blur), upsample_factor=upsample, normalization=None
        )
        total = total + np.asarray(d, dtype=float)
    return total


def solve_pose_graph(
    pairs: dict[tuple[int, int], np.ndarray],
    weights: dict[tuple[int, int], float],
    K: int,
    n_irls: int = 12,
    huber_c: float = 2.0,
    tukey_c: float = 4.685,
) -> np.ndarray:
    """Least-squares positions ``a`` (K, 2) from pairwise estimates of ``a_k - a_l``; ``a_0 = 0``."""
    rows, rhs, w0 = [], [], []
    for (k, j), d in pairs.items():
        r = np.zeros(K - 1)
        if k > 0:
            r[k - 1] += 1.0
        if j > 0:
            r[j - 1] -= 1.0
        rows.append(r)
        rhs.append(d)
        w0.append(weights[(k, j)])
    A0, b0, w0 = np.array(rows), np.array(rhs), np.array(w0)
    w = w0.copy()
    sol = np.zeros((K - 1, 2))
    n_huber = max(n_irls // 2, 1)
    for it in range(n_irls):
        # IRLS: Huber first (bounded influence), then Tukey biweight (gross outliers -> weight 0,
        # i.e. failed correlations are rejected rather than merely down-weighted).
        sw = np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(A0 * sw, b0 * sw, rcond=None)
        res = np.linalg.norm(A0 @ sol - b0, axis=1)
        scale = max(1.4826 * np.median(res), 0.25)
        if it < n_huber:
            w = w0 * np.minimum(1.0, huber_c * scale / np.maximum(res, 1e-12))
        else:
            u = res / (tukey_c * scale)
            w = w0 * np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
        if w.sum() < 1e-9:
            break
    return np.vstack([np.zeros((1, 2)), sol])


def _perturbed(images, rng, perturb):
    H, W = images[0].shape
    if not perturb:
        return 0, 0, H, W, 1.0, 0.0
    f = rng.uniform(0.8, 1.0)
    h, w = int(H * f), int(W * f)
    return (
        int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1)), h, w,
        float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.0, 0.10)),
    )


def register_series(
    images: list[np.ndarray],
    pixel_um: float,
    n_samples: int,
    rng: np.random.Generator,
    neighbor_orders: tuple[int, ...] = (1,),
    order_weights: tuple[float, ...] = (1.0,),
    perturb: bool = True,
) -> RegistrationEnsemble:
    """Translation-only registration ensemble (first-neighbour chain by default)."""
    K = len(images)
    samples = []
    for _ in range(n_samples):
        y0, x0, h, w_, blur, noise = _perturbed(images, rng, perturb)
        crops = []
        for im in images:
            c = im[y0 : y0 + h, x0 : x0 + w_].astype(np.float64)
            if noise > 0:
                c = c + rng.normal(0, noise * (c.std() + 1e-9), c.shape)
            crops.append(c)
        pairs, weights = {}, {}
        for order, wt in zip(neighbor_orders, order_weights):
            for k in range(K - order):
                pairs[(k, k + order)] = pairwise_shift(crops[k], crops[k + order], blur=blur)
                weights[(k, k + order)] = wt
        a = solve_pose_graph(pairs, weights, K)
        samples.append([Transform.translation(a[k]) for k in range(K)])
    return RegistrationEnsemble(samples, pixel_um, images[0].shape, method="translation")


# ------------------------------------------------------------- rigid / non-rigid chain
def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size < 25:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / den) if den > 0 else -1.0


def _rigid_pair(ref, raw, blur, angles_deg):
    """Best rotation+translation Transform aligning ``raw`` to ``ref``; returns (ncc, Transform)."""
    fill = float(raw.mean())

    def score(theta):
        rot = Transform(theta, np.zeros(2))
        img, _ = rot.apply(raw, order=1, cval=fill)
        d = pairwise_shift(ref, img, blur=blur, n_iter=2)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        tr = Transform(theta, -R @ d)
        aligned, inside = tr.apply(raw, order=1, cval=fill)
        return ncc(ref, aligned, inside), tr

    best = max((score(np.deg2rad(a)) for a in angles_deg), key=lambda s: s[0])
    step = np.deg2rad(np.ptp(angles_deg) / max(len(angles_deg) - 1, 1)) / 2.0
    cand = [best] + [score(best[1].theta + o * step) for o in (-1.0, -0.5, 0.5, 1.0)]
    return max(cand, key=lambda s: s[0])


def _flow_refine(ref, raw, rigid, radius, num_warp, smooth, max_disp):
    """Smooth dense displacement f (2,H,W) refining ``rigid``; returns (ncc, Transform)."""
    fill = float(raw.mean())
    aligned, _ = rigid.apply(raw, order=1, cval=fill)
    a = (ref - ref.mean()) / (ref.std() + 1e-9)
    b = (aligned - aligned.mean()) / (aligned.std() + 1e-9)
    flow = optical_flow_ilk(a.astype(np.float32), b.astype(np.float32), radius=radius, num_warp=num_warp)
    flow = np.stack([ndi.gaussian_filter(f, smooth) for f in flow]).astype(np.float64)
    mag = np.sqrt((flow**2).sum(axis=0))
    flow *= np.minimum(1.0, max_disp / np.maximum(mag, 1e-12))
    tr = Transform(rigid.theta, rigid.t, disp=flow)
    out, inside = tr.apply(raw, order=1, cval=fill)
    return ncc(ref, out, inside), tr


def register_series_chain(
    images: list[np.ndarray],
    pixel_um: float,
    n_samples: int,
    rng: np.random.Generator,
    method: str = "nonrigid",
    max_rot_deg: float = 6.0,
    min_ncc: float = 0.25,
    perturb: bool = True,
    max_disp_px: float = 4.0,
    rel_ncc: float = 0.5,
) -> RegistrationEnsemble:
    """Sequential rigid / non-rigid registration ensemble with failure detection."""
    if method not in ("rigid", "nonrigid"):
        raise ValueError("method must be 'rigid' or 'nonrigid'")
    K = len(images)
    shape = images[0].shape
    angles = np.linspace(-max_rot_deg, max_rot_deg, 9)
    chains, nccs, fails = [], np.zeros((n_samples, K)), np.zeros((n_samples, K), bool)
    for m in range(n_samples):
        blur = float(rng.uniform(0.8, 2.0)) if perturb else 1.2
        noise = float(rng.uniform(0.0, 0.08)) if perturb else 0.0
        radius = int(rng.choice([11, 15, 19])) if perturb else 15
        num_warp = int(rng.choice([4, 6])) if perturb else 6
        smooth = float(rng.uniform(6.0, 10.0)) if perturb else 8.0
        raws = []
        for im in images:
            x = im.astype(np.float64)
            if noise:
                x = x + rng.normal(0, noise * (x.std() + 1e-9), x.shape)
            raws.append(x)
        chain = [Transform()]
        reg_prev = raws[0]
        nccs[m, 0] = 1.0
        for k in range(1, K):
            sm_prev = ndi.gaussian_filter(reg_prev, blur)
            sm_raw = ndi.gaussian_filter(raws[k], blur)
            sc, tr = _rigid_pair(sm_prev, sm_raw, blur, angles)
            if method == "nonrigid":
                sc2, tr2 = _flow_refine(sm_prev, sm_raw, tr, radius, num_warp, smooth, max_disp_px)
                if sc2 > sc + 0.01:  # keep the refinement only if it measurably helps
                    sc, tr = sc2, tr2
            # Failure test. A match must clear an absolute floor and must not collapse relative to the
            # pairs accepted so far (a section whose NCC is under half the running median is suspect).
            # NCC against a chance level was tried and rejected: the null of an unrelated pair is not
            # separable from a poor-but-correct match on smooth images.
            accepted = nccs[m, 1:k][~fails[m, 1:k]] if k > 1 else np.array([])
            rel_floor = rel_ncc * float(np.median(accepted)) if accepted.size else 0.0
            if sc < max(min_ncc, rel_floor):  # failed pair: do not trust it; carry the previous transform
                fails[m, k] = True
                tr = chain[-1]
                img, inside = tr.apply(sm_raw, order=1, cval=float(sm_raw.mean()))
                sc = ncc(sm_prev, img, inside)
            nccs[m, k] = sc
            chain.append(tr)
            if not fails[m, k]:  # a failed section never becomes the reference for the next one
                reg_prev = tr.apply(raws[k], order=1, cval=float(raws[k].mean()))[0]
        chains.append(chain)
    return RegistrationEnsemble(chains, pixel_um, shape, ncc=nccs, failed=fails, method=method)


# ------------------------------------------------------------------------- evaluation
def registration_error_px(est: np.ndarray, true_shifts: np.ndarray) -> np.ndarray:
    """Per-section error of estimated translations vs simulation truth (relative to section 0)."""
    t = true_shifts - true_shifts[0]
    e = est - est[0]
    return np.linalg.norm(e - t, axis=1)


def registration_tre_px(
    transforms: list[Transform], true_h: list[np.ndarray], shape: tuple[int, int], n_landmarks: int = 8
) -> np.ndarray:
    """Per-section target registration error in the common frame (pixels).

    ``true_h[k]`` (2, H, W) maps raw pixel coordinates of section ``k`` to common coordinates.
    A common-frame landmark ``x`` is sent to its estimated raw location ``c_k(x)`` and back
    through the true forward map; the distance to ``x`` is the error.
    """
    pts = landmark_grid(shape, n_landmarks)
    out = np.zeros(len(transforms))
    for k, (t, h) in enumerate(zip(transforms, true_h)):
        y = sample_coords(t, shape, pts)
        back = np.stack([ndi.map_coordinates(h[i], y, order=1, mode="nearest") for i in range(2)])
        out[k] = np.linalg.norm(back - pts, axis=0).mean()
    return out
