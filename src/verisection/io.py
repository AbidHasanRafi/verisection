"""Reading real section images into a ``SectionStack``.

Supported now: single-plane 2D image files (TIFF, PNG, JPEG) in a folder, one file per section.
Not yet: whole-slide formats (SVS/NDPI/MRXS via OpenSlide), OME-Zarr pyramids, DICOM-WSI.

Everything read here is untrusted data. Archives are extracted with ``safe_extract`` (path
traversal, absolute paths, symlinks and decompression-bomb limits are checked) into a new empty
directory, and nothing from the data is ever executed.
"""
from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
from skimage.transform import resize

from .validate import DataWarning, InputError, validate_stack
from .volume import SectionStack

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(name: str) -> list:
    """Sort key so that ``sec2`` < ``sec10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def safe_extract(
    zip_path: str | Path,
    dest: str | Path,
    max_total_bytes: int = 8 * 2**30,
    max_files: int = 20_000,
    max_ratio: float = 200.0,
) -> list[Path]:
    """Extract a zip into an *empty or new* directory, refusing unsafe archives."""
    zip_path, dest = Path(zip_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise InputError(f"destination {dest} is not empty; extract each archive into its own new directory")
    root = dest.resolve()
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        if len(infos) > max_files:
            raise InputError(f"archive has {len(infos)} entries (limit {max_files})")
        total = sum(i.file_size for i in infos)
        comp = sum(i.compress_size for i in infos) or 1
        if total > max_total_bytes:
            raise InputError(f"archive expands to {total / 2**30:.1f} GiB (limit {max_total_bytes / 2**30:.1f})")
        if total / comp > max_ratio:
            raise InputError(f"suspicious compression ratio {total / comp:.0f}x (possible decompression bomb)")
        for i in infos:
            name = i.filename
            target = (root / name).resolve()
            if not str(target).startswith(str(root)) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise InputError(f"unsafe path in archive: {name!r}")
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise InputError(f"archive contains a symlink: {name!r}")
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)
    return out


def read_image(path: str | Path) -> np.ndarray:
    """Read one 2D image as float32 in [0, 1]; RGB(A) becomes the mean of the colour channels.

    uint8 / uint16 are divided by their full range; float images already in [0, 1] are kept,
    other float images are min-max scaled.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    else:
        from skimage import io as skio

        arr = skio.imread(path)
    arr = np.asarray(arr)
    orig = arr.dtype
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = arr[:3].mean(axis=0)
    elif arr.ndim != 2:
        raise InputError(f"{path.name}: expected a single 2D image, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    if orig == np.uint8:
        return arr / 255.0
    if orig == np.uint16:
        return arr / 65535.0
    lo, hi = float(arr.min()), float(arr.max())
    if 0.0 <= lo and hi <= 1.0 + 1e-6:
        return arr
    return (arr - lo) / (hi - lo) if hi > lo else arr * 0.0


def load_section_folder(
    folder: str | Path,
    pixel_um: float,
    thickness_um: float,
    spacing_um: float | None = None,
    z_um: list[float] | None = None,
    target_pixel_um: float | None = None,
    max_side: int | None = 512,
    limit: int | None = None,
    validate: bool = True,
) -> SectionStack:
    """Load one image per section (natural filename order) into a ``SectionStack``.

    ``spacing_um`` defaults to ``thickness_um`` (consecutive sections) and is *assumed*, not read
    from the files; pass ``z_um`` if positions are known. Images are resampled to
    ``target_pixel_um`` (or so that the longest side is at most ``max_side``), brought to a common
    size by centre-cropping, and scaled to [0, 1] per image dtype.
    """
    folder = Path(folder)
    files = sorted((p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: natural_key(p.name))
    if not files:
        raise InputError(f"no image files ({', '.join(sorted(IMAGE_SUFFIXES))}) found under {folder}")
    if limit:
        files = files[:limit]
    imgs = []
    for f in files:
        try:
            imgs.append(read_image(f))
        except Exception as exc:  # corrupt/truncated files must not abort silently
            raise InputError(f"cannot read {f.name}: {exc}") from exc
    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    if len({i.shape for i in imgs}) > 1:
        warnings.warn("sections differ in size; centre-cropping to the common size", DataWarning, stacklevel=2)
    out = []
    px = pixel_um
    for i in imgs:
        y0, x0 = (i.shape[0] - h) // 2, (i.shape[1] - w) // 2
        out.append(i[y0 : y0 + h, x0 : x0 + w])
    factor = 1.0
    if target_pixel_um is not None:
        factor = pixel_um / target_pixel_um
    elif max_side is not None and max(h, w) > max_side:
        factor = max_side / max(h, w)
    if factor < 1.0:
        nh, nw = max(int(round(h * factor)), 8), max(int(round(w * factor)), 8)
        out = [resize(i, (nh, nw), anti_aliasing=True, preserve_range=True).astype(np.float32) for i in out]
        px = pixel_um * (h / nh)
    spacing = thickness_um if spacing_um is None else spacing_um
    z = np.asarray(z_um, dtype=float) if z_um is not None else np.arange(len(out)) * spacing
    stack = SectionStack(out, z[: len(out)], thickness_um, px, meta={"files": [p.name for p in files], "assumed_spacing": z_um is None})
    if validate:
        validate_stack(stack)
    return stack


def read_rgb(path: str | Path, max_side: int | None = 512) -> np.ndarray:
    """Read an RGB(A) image as float32 (H, W, 3) in [0, 1], optionally downsampled (anti-aliased)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    else:
        from skimage import io as skio

        arr = skio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise InputError(f"{path.name}: not an RGB image (shape {arr.shape})")
    scale = 255.0 if arr.dtype == np.uint8 else 65535.0 if arr.dtype == np.uint16 else 1.0
    rgb = arr[..., :3].astype(np.float32) / scale
    if max_side and max(rgb.shape[:2]) > max_side:
        f = max_side / max(rgb.shape[:2])
        rgb = resize(rgb, (max(int(rgb.shape[0] * f), 8), max(int(rgb.shape[1] * f), 8), 3),
                     anti_aliasing=True, preserve_range=True).astype(np.float32)
    return rgb
