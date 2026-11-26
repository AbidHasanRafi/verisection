"""Section-to-common-frame transforms: rotation + translation + smooth non-rigid displacement.

Convention (backward mapping, used for resampling): a transform gives, for every pixel ``x``
of the *common frame*, the coordinates ``c(x)`` in the *raw section* that show the same tissue:

    c(x) = R(theta) (x + f(x) - o) + o + t,        o = image centre

so the registered image is ``raw(c(x))``. ``f`` is a smooth displacement field in the common
frame (None for rigid transforms). Pure translation by ``a`` is ``theta=0, t=a`` (matching the
earlier ``shift(img, -a)`` convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


def _grid(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(shape[0], dtype=np.float64), np.arange(shape[1], dtype=np.float64), indexing="ij")
    return np.stack([yy, xx])


@dataclass
class Transform:
    theta: float = 0.0  # radians, rotation about the image centre
    t: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (row, col) translation
    disp: np.ndarray | None = None  # (2, H, W) displacement f(x) in the common frame
    _coords: np.ndarray | None = None  # explicit coordinates (used for ensemble means)

    def coords(self, shape: tuple[int, int]) -> np.ndarray:
        """Raw-section coordinates (2, H, W) for each common-frame pixel."""
        if self._coords is not None:
            return self._coords
        g = _grid(shape)
        if self.disp is not None:
            g = g + self.disp
        o = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])[:, None, None]
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        rel = (g - o).reshape(2, -1)
        out = (R @ rel).reshape(g.shape) + o + np.asarray(self.t, dtype=float)[:, None, None]
        return out

    def apply(
        self, img: np.ndarray, order: int = 1, cval: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample ``img`` into the common frame; returns (image, fully-valid mask)."""
        shape = img.shape[-2:]
        co = self.coords(shape)
        if img.ndim == 2:
            out = ndi.map_coordinates(img, co, order=order, mode="constant", cval=cval)
        else:
            out = np.stack([ndi.map_coordinates(ch, co, order=order, mode="constant", cval=cval) for ch in img])
        inside = (
            (co[0] >= 0) & (co[0] <= shape[0] - 1) & (co[1] >= 0) & (co[1] <= shape[1] - 1)
        )
        return out, inside

    @staticmethod
    def translation(a: np.ndarray) -> "Transform":
        return Transform(0.0, np.asarray(a, dtype=float))

    @staticmethod
    def from_coords(coords: np.ndarray) -> "Transform":
        return Transform(_coords=np.asarray(coords, dtype=float))

    @property
    def is_pure_translation(self) -> bool:
        return self._coords is None and self.disp is None and abs(self.theta) < 1e-12


def mean_transform(transforms: list[Transform], shape: tuple[int, int]) -> Transform:
    """Pointwise mean of the coordinate maps (a non-parametric 'mean' transform)."""
    return Transform.from_coords(np.mean([t.coords(shape) for t in transforms], axis=0))


def landmark_grid(shape: tuple[int, int], n: int = 8, margin: float = 0.2) -> np.ndarray:
    """Regular interior landmarks (2, n*n) in the common frame for TRE/spread evaluation."""
    ys = np.linspace(margin * shape[0], (1 - margin) * shape[0] - 1, n)
    xs = np.linspace(margin * shape[1], (1 - margin) * shape[1] - 1, n)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()])


def sample_coords(transform: Transform, shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Raw-section coordinates of common-frame points ``pts`` (2, N) (bilinear on the coord map)."""
    co = transform.coords(shape)
    return np.stack([ndi.map_coordinates(co[i], pts, order=1, mode="nearest") for i in range(2)])


def smooth_random_field(
    shape: tuple[int, int], amp_px: float, sigma_px: float, rng: np.random.Generator
) -> np.ndarray:
    """Smooth random displacement field with the requested RMS amplitude, (2, H, W)."""
    f = np.stack([ndi.gaussian_filter(rng.normal(size=shape), sigma_px, mode="reflect") for _ in range(2)])
    rms = np.sqrt((f**2).sum(axis=0).mean())
    return f * (amp_px / max(rms, 1e-12))
