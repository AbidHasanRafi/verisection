"""Input validation: fail early with actionable messages instead of producing silent garbage."""
from __future__ import annotations

import warnings

import numpy as np

from .volume import SectionStack


class InputError(ValueError):
    """The input data cannot be processed safely; the message says what to fix."""


class DataWarning(UserWarning):
    """The input is processable but suspicious; results should be read with care."""


def validate_stack(
    stack: SectionStack,
    min_sections: int = 3,
    min_size_px: int = 32,
    min_contrast: float = 1e-6,
    strict: bool = True,
) -> list[str]:
    """Check a ``SectionStack``. Raises ``InputError`` on fatal problems; returns warnings otherwise.

    Fatal: non-finite pixels, constant (blank) sections, images too small, fewer than
    ``min_sections`` sections, non-positive pixel size or thickness.
    Warnings: very different contrast between sections, spacing far larger than thickness,
    tiny size, extreme aspect ratio.
    """
    problems: list[str] = []
    notes: list[str] = []
    if stack.n_sections < min_sections:
        problems.append(f"only {stack.n_sections} sections; at least {min_sections} are needed")
    if not (stack.pixel_um > 0 and np.isfinite(stack.pixel_um)):
        problems.append(f"pixel size must be a positive number of µm, got {stack.pixel_um!r}")
    if not (stack.thickness_um > 0 and np.isfinite(stack.thickness_um)):
        problems.append(f"section thickness must be positive, got {stack.thickness_um!r}")
    H, W = stack.shape
    if min(H, W) < min_size_px:
        problems.append(f"image size {H}x{W} is smaller than {min_size_px} px")
    sds = []
    for k, im in enumerate(stack.images):
        if not np.issubdtype(im.dtype, np.number):
            problems.append(f"section {k}: non-numeric dtype {im.dtype}")
            continue
        if not np.isfinite(im).all():
            bad = int((~np.isfinite(im)).sum())
            problems.append(f"section {k}: {bad} non-finite pixels (NaN/inf); inpaint or mask them first")
            continue
        sd = float(im.std())
        sds.append(sd)
        if sd <= min_contrast:
            problems.append(f"section {k}: constant/blank image (sd={sd:.2g}); remove or replace it")
    if sds and len(sds) > 2:
        ratio = max(sds) / max(min(sds), 1e-12)
        if ratio > 5:
            notes.append(f"contrast differs {ratio:.1f}x between sections; stain/exposure normalisation advised")
    sp = stack.spacing_um
    if len(sp) and sp.max() > 8 * stack.thickness_um:
        notes.append("section spacing exceeds 8x thickness; interpolation will be uncertain over most of the volume")
    if len(sp) > 2 and sp.max() > 3 * np.median(sp):
        notes.append("uneven section spacing (missing sections?); gaps vary in size")
    if max(H, W) / max(min(H, W), 1) > 4:
        notes.append("extreme aspect ratio; check cropping")
    if problems and strict:
        raise InputError("invalid input:\n  - " + "\n  - ".join(problems))
    for n in notes:
        warnings.warn(n, DataWarning, stacklevel=2)
    return problems + notes
