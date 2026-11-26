"""Image-level audit of real 2D histology patches.

The reconstruction layers of VeriSection need serial sections. Public patch collections (LC25000, BreakHis,
oral cavity, ovarian) are *not* serial: every patch is an independent field of view. They are therefore used
for the layers that only need single images:

* **stain / optical-density diagnostics** (``stain.py``): Macenko-vs-Ruifrok vectors, H/E concentration,
  model residual;
* **image quality** (``qc.py``): focus, noise, robust-z accept / flag / exclude;
* **physical-scale texture** (``texture.py``): fixed-bin GLCM descriptors at several offsets;
* **descriptor stability** (``texture.stability_suite``): ICC certification under acquisition perturbations;
* **interpolation bias** with *real* texture: what naive intensity interpolation does to texture when the
  two bracketing sections are decorrelated or mis-registered.

Nothing here is a diagnostic claim. Patch labels are used only to ask whether a descriptor carries
tissue-type signal and how much of that signal is specific to the source collection (domain shift).

Pixel size is not recorded in these collections, so every offset is expressed in pixels (``pixel_um = 1``).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import qc, texture

# --------------------------------------------------------------------------------------- registry
SOURCES = {
    "LC25000": "Borkowski et al., 2019 (lung and colon, LC25000)",
    "BreakHis": "Spanhol et al., 2016 (breast)",
    "Oral": "Rahman et al., 2020 (oral cavity)",
    "Ovarian": "Kasture et al., 2021 (ovarian)",
}
# folder -> (source, organ, display name, group)   group: "non-tumor" (normal/benign) or "tumor"
CLASSES: dict[str, tuple[str, str, str, str]] = {
    "colon_n": ("LC25000", "Colon", "Colon normal", "non-tumor"),
    "colon_aca": ("LC25000", "Colon", "Colon adenocarcinoma", "tumor"),
    "lung_n": ("LC25000", "Lung", "Lung normal", "non-tumor"),
    "lung_aca": ("LC25000", "Lung", "Lung adenocarcinoma", "tumor"),
    "lung_scc": ("LC25000", "Lung", "Lung squamous carcinoma", "tumor"),
    "breast_benign": ("BreakHis", "Breast", "Breast benign", "non-tumor"),
    "breast_malignant": ("BreakHis", "Breast", "Breast malignant", "tumor"),
    "oral_normal_oral_cavity": ("Oral", "Oral cavity", "Oral normal", "non-tumor"),
    "oral_oscc": ("Oral", "Oral cavity", "Oral squamous carcinoma", "tumor"),
    "ovarian_non_cancerous": ("Ovarian", "Ovary", "Ovarian non-cancerous", "non-tumor"),
    "ovarian_serous": ("Ovarian", "Ovary", "Ovarian serous", "tumor"),
    "ovarian_mucinous": ("Ovarian", "Ovary", "Ovarian mucinous", "tumor"),
    "ovarian_clear_cell": ("Ovarian", "Ovary", "Ovarian clear cell", "tumor"),
    "ovarian_endometri": ("Ovarian", "Ovary", "Ovarian endometrioid", "tumor"),
}
OFFSETS_PX = (1, 2, 4, 8)
GLCM_FEATURES = ("contrast", "dissimilarity", "homogeneity", "asm", "entropy", "correlation")
GLCM_BIN = 0.05
GLCM_LEVELS = 32
BLACK_LEVEL = 0.08       # luminance below this is "black": letterbox padding, pen marks, scanner background
BLACK_EXCLUDE = 0.02     # more black than this and the file is not a clean field of view


@dataclass(frozen=True)
class Sample:
    cls: str
    path: Path


def list_originals(root: str | Path, cls: str) -> list[Path]:
    """Files of one class folder that are not augmented copies (``aug_*``)."""
    d = Path(root) / cls
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff") and not p.name.startswith("aug_"))


def sample_dataset(root: str | Path, per_class: int = 48, seed: int = 0) -> list[Sample]:
    """Random sample of non-augmented images, ``per_class`` from every class folder that exists."""
    rng = np.random.default_rng(seed)
    out: list[Sample] = []
    for cls in CLASSES:
        if not (Path(root) / cls).is_dir():
            continue
        files = list_originals(root, cls)
        k = min(per_class, len(files))
        out.extend(Sample(cls, files[i]) for i in sorted(rng.choice(len(files), k, replace=False)))
    return out


# ------------------------------------------------------------------------------ per-image features
def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def od_map(rgb: np.ndarray) -> np.ndarray:
    """Scalar optical density of the luminance with a *fixed* white point (I0 = 1), so bins mean the same everywhere."""
    return texture.optical_density(luminance(rgb), i0=1.0, eps=1.0 / 255.0)


def glcm_scale_features(od: np.ndarray, offsets_px=OFFSETS_PX, mask: np.ndarray | None = None) -> dict[str, float]:
    mask = np.ones(od.shape, bool) if mask is None else mask
    out: dict[str, float] = {}
    for off in offsets_px:
        f = texture.window_features_2d(od, mask, pixel_um=1.0, offset_um=float(off), bin_width=GLCM_BIN, n_levels=GLCM_LEVELS)
        for k in GLCM_FEATURES:
            out[f"glcm_{k}_{off}px"] = f[k]
    return out


def describe_image(rgb: np.ndarray) -> dict[str, float]:
    """Stain, quality and texture descriptors of one RGB patch (values in [0, 1])."""
    sp = qc.stain_passport(rgb, i0=1.0, rule="sum")
    gray = luminance(rgb)
    od = od_map(rgb)
    row: dict[str, float] = {
        "tissue_fraction": sp["tissue_fraction"], "stain_ok": float(sp["ok"]),
        "angle_h_deg": sp.get("angle_to_ruifrok_h_deg", np.nan), "angle_e_deg": sp.get("angle_to_ruifrok_e_deg", np.nan),
        "angle_between_deg": sp.get("angle_between_stains_deg", np.nan),
        "mean_h": sp.get("mean_h", np.nan), "mean_e": sp.get("mean_e", np.nan), "residual_rms": sp.get("residual_rms", np.nan),
        "od_mean": float(od.mean()), "od_sd": float(od.std()),
        "focus_lapvar": qc.focus_laplacian_var(gray), "noise_sigma": qc.noise_sigma_haar(gray),
        "black_fraction": float((gray < BLACK_LEVEL).mean()), "white_fraction": float((gray > 0.92).mean()),
        "mean_r": float(rgb[..., 0].mean()), "mean_g": float(rgb[..., 1].mean()), "mean_b": float(rgb[..., 2].mean()),
    }
    row.update(glcm_scale_features(od))
    return row


def _describe_path(args):
    cls, path = args
    rgb = load_rgb(path)
    src, organ, name, group = CLASSES[cls]
    return {"class": cls, "class_name": name, "source": src, "organ": organ, "group": group, "file": Path(path).name,
            "height": rgb.shape[0], "width": rgb.shape[1]} | describe_image(rgb)


def audit_samples(samples: list[Sample], workers: int = 1) -> pd.DataFrame:
    jobs = [(s.cls, str(s.path)) for s in samples]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_describe_path, jobs, chunksize=4))
    else:
        rows = [_describe_path(j) for j in jobs]
    df = pd.DataFrame(rows)
    df.insert(0, "id", np.arange(len(df)))
    return df


# --------------------------------------------------------------------------- image-level QC verdicts
QC_COLUMNS = ("focus_lapvar", "noise_sigma")


def _robust_z_floored(x: np.ndarray, min_scale: float) -> np.ndarray:
    """Robust z with the scale floored (log units): a homogeneous sample must not turn ordinary variation into outliers."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return (x - med) / max(mad, min_scale)


def qc_verdicts(df: pd.DataFrame, z_flag: float = 3.5, z_exclude: float = 6.0, by: str | None = None,
                black_exclude: float = BLACK_EXCLUDE, min_scale: float = 0.15) -> pd.DataFrame:
    """Accept / flag / exclude by robust z of focus (low is bad), noise (high is bad) and stain-model residual (high is bad).

    ``by=None`` pools all images (so a source with a different look is flagged *relative to the pool*, which is
    the domain-shift reading); ``by='source'`` standardizes inside each source (batch-aware QC).

    The robust-z scale is floored at ``min_scale`` (0.15 in log units, i.e. about 15%).

    Rule 0 (hard): more than ``black_exclude`` of the pixels are black (letterbox padding or a pen blob): the file is not a
    clean field of view, it is excluded whatever its z-scores, because its stain and noise estimates are not about tissue.
    """
    cols = {"focus_lapvar": -1.0, "noise_sigma": 1.0, "residual_rms": 1.0}
    groups = [df.index] if by is None else [g.index for _, g in df.groupby(by)]
    verdict = pd.Series("accept", index=df.index)
    reason = pd.Series("", index=df.index)
    zmax = pd.Series(0.0, index=df.index)
    for idx in groups:
        sub = df.loc[idx]
        zs = {}
        for c, sign in cols.items():
            v = sub[c].to_numpy(float)
            zs[c] = sign * _robust_z_floored(np.log(np.clip(v, 1e-12, None)), min_scale) if np.isfinite(v).sum() >= 3 else np.zeros(len(sub))
        Z = np.vstack(list(zs.values()))
        worst = np.nanmax(Z, axis=0)
        zmax.loc[idx] = worst
        verdict.loc[idx] = np.where(worst > z_exclude, "exclude", np.where(worst > z_flag, "flag", "accept"))
        names = list(zs)
        reason.loc[idx] = ["; ".join(f"{names[j]} z={Z[j, i]:.1f}" for j in range(len(names)) if Z[j, i] > z_flag) for i in range(len(sub))]
    if "black_fraction" in df.columns:
        pad = (df["black_fraction"] > black_exclude).to_numpy()
        verdict[pad] = "exclude"
        reason[pad] = [f"black padding {100 * b:.0f}% of pixels" + (f"; {r}" if r else "") for b, r in zip(df["black_fraction"].to_numpy()[pad], reason[pad])]
    return pd.DataFrame({"id": df["id"], "verdict": verdict, "z_max": zmax, "reasons": reason})


# ------------------------------------------------------------------------------ descriptor sets
def descriptor_columns(df: pd.DataFrame, kind: str) -> list[str]:
    tex = [c for c in df.columns if c.startswith("glcm_")]
    sta = ["mean_h", "mean_e", "residual_rms", "angle_h_deg", "angle_e_deg", "angle_between_deg", "od_mean", "od_sd", "mean_r", "mean_g", "mean_b"]
    if kind == "texture":
        return tex
    if kind == "stain":
        return sta
    if kind == "combined":
        return sta + tex
    raise ValueError(kind)


def classwise_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Class means of each descriptor, standardized over the whole sample (heat-map data)."""
    m = df.groupby("class_name")[cols].mean()
    return (m - df[cols].mean()) / df[cols].std(ddof=0).replace(0, np.nan)


# ------------------------------------------------------------------------------ separability
def _clf(seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)


def _boot_ci(stat, n: int, rng: np.random.Generator, n_boot: int = 400, q=(2.5, 97.5)):
    vals = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = [v for v in vals if np.isfinite(v)]
    return tuple(np.percentile(vals, q)) if vals else (np.nan, np.nan)


def separability(df: pd.DataFrame, seed: int = 0, n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold accuracy for several label sets and descriptor sets (in-distribution; no augmented copies)."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rng = np.random.default_rng(seed)
    tasks = {"14 classes": "class", "5 organs": "organ", "4 sources": "source", "tumor vs non-tumor": "group"}
    rows = []
    for tname, tcol in tasks.items():
        y = df[tcol].to_numpy()
        for kind in ("stain", "texture", "combined"):
            X = df[descriptor_columns(df, kind)].to_numpy(float)
            X = np.nan_to_num(X, nan=np.nanmedian(X))
            pred = cross_val_predict(_clf(seed), X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
            acc = float((pred == y).mean())
            bacc = float(balanced_accuracy_score(y, pred))
            lo, hi = _boot_ci(lambda i, pred=pred, y=y: float((pred[i] == y[i]).mean()), len(y), rng)
            chance = float(max(pd.Series(y).value_counts(normalize=True)))
            rows.append({"task": tname, "descriptors": kind, "n_images": len(y), "n_classes": len(set(y)), "accuracy": acc, "accuracy_lo95": lo,
                         "accuracy_hi95": hi, "balanced_accuracy": bacc, "majority_baseline": chance})
    return pd.DataFrame(rows)


def confusion_14(df: pd.DataFrame, seed: int = 0, n_splits: int = 5) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.nan_to_num(df[descriptor_columns(df, "combined")].to_numpy(float))
    y = df["class_name"].to_numpy()
    pred = cross_val_predict(_clf(seed), X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    names = [CLASSES[c][2] for c in CLASSES if CLASSES[c][2] in set(y)]
    cm = pd.crosstab(pd.Categorical(y, names), pd.Categorical(pred, names), dropna=False)
    cm.index.name, cm.columns.name = "true", "predicted"
    return cm


def domain_shift(df: pd.DataFrame, seed: int = 0, kinds=("stain", "texture", "combined")) -> pd.DataFrame:
    """Tumor vs non-tumor AUROC: in-source 5-fold vs leave-one-source-out (train on three sources, test on the fourth)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rng = np.random.default_rng(seed)
    y_all = (df["group"] == "tumor").to_numpy().astype(int)
    rows = []
    for kind in kinds:
        X_all = np.nan_to_num(df[descriptor_columns(df, kind)].to_numpy(float))
        for src in sorted(df["source"].unique()):
            te = (df["source"] == src).to_numpy()
            if len(set(y_all[te])) < 2:
                continue
            clf = _clf(seed).fit(X_all[~te], y_all[~te])
            p_loso = clf.predict_proba(X_all[te])[:, 1]
            p_in = cross_val_predict(_clf(seed), X_all[te], y_all[te], cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
            yt = y_all[te]
            for setting, p in (("in-source (5-fold)", p_in), ("leave-one-source-out", p_loso)):
                auc = float(roc_auc_score(yt, p))
                lo, hi = _boot_ci(lambda i, p=p, yt=yt: float(roc_auc_score(yt[i], p[i])) if len(set(yt[i])) == 2 else np.nan, len(yt), rng, n_boot=300)
                rows.append({"descriptors": kind, "source": src, "setting": setting, "n_test": int(te.sum()), "auroc": auc, "auroc_lo95": lo,
                             "auroc_hi95": hi, "accuracy": float(((p > 0.5) == yt).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ explanation
def feature_groups(cols: list[str]) -> dict[str, list[int]]:
    """Descriptor groups used to explain a model: a group is permuted or summed as one unit (GLCM groups pool the four offsets)."""
    g: dict[str, list[str]] = {
        "H and E concentration": ["mean_h", "mean_e"], "Stain-vector angles": ["angle_h_deg", "angle_e_deg", "angle_between_deg"],
        "Stain-model residual": ["residual_rms"], "Optical density mean and SD": ["od_mean", "od_sd"], "Mean R, G, B": ["mean_r", "mean_g", "mean_b"],
    }
    for k in GLCM_FEATURES:
        g[f"GLCM {k}"] = [c for c in cols if c.startswith(f"glcm_{k}_")]
    return {name: [cols.index(c) for c in members if c in cols] for name, members in g.items()}


def grouped_permutation_importance(df: pd.DataFrame, label: str = "group", seed: int = 0, n_splits: int = 5, n_perm: int = 8) -> pd.DataFrame:
    """Drop in balanced accuracy on held-out folds when one descriptor group is shuffled across patches (random forest, combined descriptors)."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    cols = descriptor_columns(df, "combined")
    X = np.nan_to_num(df[cols].to_numpy(float))
    y = df[label].to_numpy()
    groups = feature_groups(cols)
    rng = np.random.default_rng(seed)
    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)):
        clf = _clf(seed).fit(X[tr], y[tr])
        base = balanced_accuracy_score(y[te], clf.predict(X[te]))
        for name, idx in groups.items():
            drops = []
            for _ in range(n_perm):
                Xp = X[te].copy()
                Xp[:, idx] = X[te][rng.permutation(len(te))][:, idx]
                drops.append(base - balanced_accuracy_score(y[te], clf.predict(Xp)))
            rows.append({"fold": fold, "group": name, "drop": float(np.mean(drops)), "baseline": float(base)})
    return pd.DataFrame(rows)


def explain_tumor(df: pd.DataFrame, seed: int = 0, top: int = 3) -> dict | None:
    """Per-patch TreeSHAP explanation of a tumor vs non-tumor random forest trained on the given (accepted) patches.

    Returns ``None`` when the optional ``shap`` package is missing. The score is the in-sample forest probability, so
    the explanation describes the model, not a held-out prediction. Research use only.
    """
    try:
        import shap
    except ImportError:
        return None
    cols = descriptor_columns(df, "combined")
    X = np.nan_to_num(df[cols].to_numpy(float))
    y = (df["group"] == "tumor").to_numpy().astype(int)
    if len(set(y)) < 2:
        return None
    clf = _clf(seed).fit(X, y)
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(X, check_additivity=False)
    sv = np.asarray(sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv))
    p = clf.predict_proba(X)[:, 1]
    groups = feature_groups(cols)
    mean_abs = {k: float(np.abs(sv[:, idx]).sum(1).mean()) for k, idx in groups.items()}
    tot = sum(mean_abs.values()) or 1.0
    order = np.argsort(-np.abs(sv), axis=1)[:, :top]
    patches = {int(i): {"p": round(float(p[r]), 3), "top": [[cols[j], round(float(sv[r, j]), 4), round(float(X[r, j]), 4)] for j in order[r]]}
               for r, i in enumerate(df["id"].to_numpy())}
    return {"method": "TreeSHAP of a random forest (300 trees) trained on the accepted patches, tumor vs non-tumor, combined descriptors",
            "base_value": round(float(np.ravel(ex.expected_value)[-1]), 4),
            "groups": [{"group": k, "share": round(v / tot, 4)} for k, v in sorted(mean_abs.items(), key=lambda kv: -kv[1])], "patches": patches}


def pca_embedding(df: pd.DataFrame, kind: str = "texture", n: int = 2):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(df[descriptor_columns(df, kind)].to_numpy(float))
    Z = StandardScaler().fit_transform(X)
    p = PCA(n_components=n, random_state=0).fit(Z)
    E = p.transform(Z)
    emb = pd.DataFrame(E, columns=[f"PC{i + 1}" for i in range(n)])
    emb.insert(0, "id", df["id"].to_numpy())
    return emb, p.explained_variance_ratio_


# ------------------------------------------------------------------------------ stability
FEATURE_FN = {k: (lambda w, k=k, off=2: texture.window_features_2d(w, np.ones(w.shape, bool), 1.0, float(off), GLCM_BIN, GLCM_LEVELS)[k]) for k in GLCM_FEATURES}
PERTURBATIONS = {
    "noise": [("noise", 0.05), ("noise", 0.10)],
    "blur": [("blur", 0.7), ("blur", 1.5)],
    "OD scaling": [("od_scale", 0.10), ("od_scale", -0.10)],
    "OD offset": [("offset", 0.05), ("offset", -0.05)],
}


def sample_windows(samples: list[Sample], n: int, size: int = 128, seed: int = 0) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Random OD windows from random sampled images (one window per image)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), min(n, len(samples)), replace=False)
    wins, meta = [], []
    for i in idx:
        od = od_map(load_rgb(samples[i].path))
        y, x = rng.integers(0, od.shape[0] - size + 1), rng.integers(0, od.shape[1] - size + 1)
        wins.append(od[y:y + size, x:x + size])
        meta.append({"window": len(wins) - 1, "class": samples[i].cls, "source": CLASSES[samples[i].cls][0], "file": samples[i].path.name, "y0": int(y), "x0": int(x)})
    return wins, pd.DataFrame(meta)


def stability_table(windows: list[np.ndarray], seed: int = 0, threshold: float = 0.75, offsets=(1, 2, 4, 8)) -> pd.DataFrame:
    rows = []
    for off in offsets:
        for feat in GLCM_FEATURES:
            fn = lambda w, feat=feat, off=off: texture.window_features_2d(w, np.ones(w.shape, bool), 1.0, float(off), GLCM_BIN, GLCM_LEVELS)[feat]  # noqa: E731
            fams = dict(PERTURBATIONS) | {"all": [c for v in PERTURBATIONS.values() for c in v]}
            for fam, conds in fams.items():
                if off != 2 and fam != "all":
                    continue
                r = texture.stability_suite(fn, windows, conds, np.random.default_rng(seed), feat, threshold)
                rows.append({"feature": feat, "offset_px": off, "perturbation": fam, "icc": r.icc, "icc_lower95": r.icc_lower, "certified": r.certified,
                             "n_windows": r.n_windows, "threshold": threshold})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ interpolation bias
def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def interpolation_bias(samples: list[Sample], n_pairs: int = 24, size: int = 160, seed: int = 0, offset_px: int = 2,
                       alphas=(0.0, 0.125, 0.25, 0.375, 0.5)) -> pd.DataFrame:
    """Texture bias of naive linear intensity interpolation between two bracketing sections.

    Scenario *independent*: the two sections are different fields of view (z-decorrelated: spacing >> correlation length).
    Scenarios *shift-k*: the second section is the first displaced by k px (a registration error).
    Bias is relative to the mean of the two observed windows' features; alpha = 0 is an observed plane (no bias by
    construction); alpha = 0.5 is the plane farthest from both neighbors. Real texture, simulated sectioning.
    """
    rng = np.random.default_rng(seed)
    by_cls: dict[str, list[Sample]] = {}
    for s in samples:
        by_cls.setdefault(s.cls, []).append(s)
    classes = [c for c, v in by_cls.items() if len(v) >= 2]
    rows = []
    for pair in range(n_pairs):
        c = classes[pair % len(classes)]
        a_s, b_s = (by_cls[c][i] for i in rng.choice(len(by_cls[c]), 2, replace=False))
        A_full, B_full = od_map(load_rgb(a_s.path)), od_map(load_rgb(b_s.path))
        # patches differ in size (194-512 px): one crop side per pair, 8 px margin so shifted copies stay inside the image
        side = int(min(size, *(d - 16 for d in A_full.shape), *(d - 16 for d in B_full.shape)))
        if side < 96:
            continue
        ya, xa = (int(rng.integers(0, d - side - 16 + 1)) + 8 for d in A_full.shape)
        yb, xb = (int(rng.integers(0, d - side - 16 + 1)) + 8 for d in B_full.shape)
        A = A_full[ya:ya + side, xa:xa + side]
        scenarios = {"independent": B_full[yb:yb + side, xb:xb + side]}
        for k in (4, 12):
            scenarios[f"shift {k} px"] = _shift(A_full, k, k)[ya:ya + side, xa:xa + side]
        mask = np.ones(A.shape, bool)
        fa = texture.window_features_2d(A, mask, 1.0, float(offset_px), GLCM_BIN, GLCM_LEVELS)
        for scen, B in scenarios.items():
            fb = texture.window_features_2d(B, mask, 1.0, float(offset_px), GLCM_BIN, GLCM_LEVELS)
            for al in alphas:
                f = texture.window_features_2d((1 - al) * A + al * B, mask, 1.0, float(offset_px), GLCM_BIN, GLCM_LEVELS)
                for k in GLCM_FEATURES:
                    ref = 0.5 * (fa[k] + fb[k]) if al > 0 else fa[k]
                    rows.append({"pair": pair, "class": c, "source": CLASSES[c][0], "scenario": scen, "alpha": al, "feature": k, "crop_px": side,
                                 "observed_mean": ref, "interpolated": f[k], "relative_bias": (f[k] - ref) / ref if abs(ref) > 1e-12 else np.nan,
                                 "interpolation_contamination_index": float(min(al, 1 - al) * 2)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ UI payload
def thumbnail_b64(path: str | Path, side: int = 128, quality: int = 82) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def audit_payload(samples: list[Sample], df: pd.DataFrame, verdicts: pd.DataFrame, analysis: dict, title: str = "Image audit",
                  thumbs_per_class: int = 12, explain: dict | None = None) -> dict:
    """Compact JSON-able payload for the workstation's *Image audit* workspace."""
    cols = ["id", "class", "class_name", "source", "organ", "group", "file", "tissue_fraction", "mean_h", "mean_e", "residual_rms", "angle_h_deg",
            "od_mean", "od_sd", "focus_lapvar", "noise_sigma", "black_fraction", "glcm_contrast_2px", "glcm_homogeneity_2px", "glcm_entropy_2px", "glcm_correlation_2px",
            "glcm_contrast_8px"]
    d = df[cols].merge(verdicts[["id", "verdict", "z_max", "reasons"]], on="id")
    if explain:
        pm = explain["patches"]
        d["xp"] = [pm[int(i)]["p"] if int(i) in pm else None for i in d["id"]]
        d["xtop"] = [pm[int(i)]["top"] if int(i) in pm else None for i in d["id"]]
        analysis = {**analysis, "explain": {k: explain[k] for k in ("method", "base_value", "groups") if k in explain}}
    seen: dict[str, int] = {}
    thumbs = {}
    for s, i in zip(samples, df["id"]):
        n = seen.get(s.cls, 0)
        if n < thumbs_per_class:
            thumbs[int(i)] = thumbnail_b64(s.path)
            seen[s.cls] = n + 1
    return {
        "version": 1, "title": title, "kind": "image-audit", "sources": SOURCES,
        "classes": [{"key": k, "source": v[0], "organ": v[1], "name": v[2], "group": v[3]} for k, v in CLASSES.items() if k in set(df["class"])],
        "rows": [{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()} for r in d.to_dict("records")],
        "thumbs": thumbs, "analysis": analysis,
        "notes": ["Independent 2D patches (not serial sections): only stain, quality and texture layers apply; no 3D reconstruction is claimed.",
                  "Augmented files (aug_*) were excluded from sampling; LC25000 files are themselves augmented copies of fewer originals.",
                  "Pixel size is not recorded: all scales are in pixels."],
    }


# ------------------------------------------------------------------------------ one call for everything
def analyse(samples: list[Sample], df: pd.DataFrame, seed: int = 0, n_windows: int = 84, n_pairs: int = 84, slow: bool = True,
            progress=None) -> dict:
    """All experiments on an audited sample. Returns DataFrames keyed by name (one table per experiment).

    ``slow=False`` skips the stability and interpolation experiments (they dominate the run time).
    """
    say = progress or (lambda msg, f: None)
    out: dict = {}
    say("image-quality verdicts", 0.05)
    out["qc_pooled"] = qc_verdicts(df)
    out["qc_within_source"] = qc_verdicts(df, by="source")
    # evidence gate: only accepted patches feed the texture experiments (the others are reported, not silently dropped)
    keep = (out["qc_pooled"]["verdict"] == "accept").to_numpy()
    out["n_gated"] = int((~keep).sum())
    samples = [s for s, k in zip(samples, keep) if k]
    df = df[keep].reset_index(drop=True)
    out["df_ok"], out["samples_ok"] = df, samples
    say("texture embedding", 0.1)
    out["pca"], out["pca_var"] = pca_embedding(df, "texture")
    say("descriptor separability", 0.2)
    out["separability"] = separability(df, seed)
    say("domain shift", 0.4)
    out["domain_shift"] = domain_shift(df, seed)
    if slow:
        say("stability certification", 0.55)
        wins, wmeta = sample_windows(samples, n_windows, 128, seed)
        out["windows"], out["stability"] = wmeta, stability_table(wins, seed)
        say("interpolation bias", 0.8)
        out["interpolation_pairs"] = interpolation_bias(samples, n_pairs=n_pairs, seed=seed)
    return out


def summarize_bias(pairs: pd.DataFrame, by: tuple[str, ...] = ("scenario", "alpha", "feature")) -> pd.DataFrame:
    """Mean relative bias with t-based 95% intervals over pairs."""
    from scipy import stats

    rows = []
    for key, g in pairs.groupby(list(by)):
        x = g["relative_bias"].dropna().to_numpy(float)
        m = float(x.mean()) if len(x) else np.nan
        hw = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,))) | {
            "n_pairs": len(x), "relative_bias_mean": m, "relative_bias_lo95": m - hw, "relative_bias_hi95": m + hw,
            "interpolation_contamination_index": float(g["interpolation_contamination_index"].iloc[0])})
    return pd.DataFrame(rows)


def qc_rates(df: pd.DataFrame, pooled: pd.DataFrame, within: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, v in (("pooled", pooled), ("within_source", within)):
        vv = v.merge(df[["id", "source"]], on="id")
        for src, g in vv.groupby("source"):
            for verdict in ("accept", "flag", "exclude"):
                rows.append({"scope": scope, "source": src, "verdict": verdict, "n": int((g["verdict"] == verdict).sum()), "n_images": len(g),
                             "fraction": float((g["verdict"] == verdict).mean())})
    return pd.DataFrame(rows)


def run_audit(root: str | Path, per_class: int = 16, seed: int = 0, slow: bool = True, workers: int = 1, progress=None) -> dict:
    """Sample, audit and analyse a folder of class sub-folders; returns the workstation's audit payload."""
    say = progress or (lambda msg, f: None)
    say("sampling patches", 0.0)
    samples = sample_dataset(root, per_class, seed)
    if not samples:
        raise ValueError(f"no class folders found under {root}; expected sub-folders such as {', '.join(list(CLASSES)[:3])}")
    say(f"measuring {len(samples)} patches", 0.02)
    df = audit_samples(samples, workers=workers)
    res = analyse(samples, df, seed, n_windows=min(84, len(samples)), n_pairs=min(84, len(samples)), slow=slow,
                  progress=lambda m, f: say(m, 0.1 + 0.8 * f))
    say("building the audit", 0.95)
    analysis = {
        "separability": res["separability"].to_dict("records"), "domain_shift": res["domain_shift"].to_dict("records"),
        "stability": res["stability"].to_dict("records") if "stability" in res else [],
        "interpolation": summarize_bias(res["interpolation_pairs"]).to_dict("records") if "interpolation_pairs" in res else [],
        "pca": res["pca"].merge(df[["id", "class_name", "source"]], on="id").round(4).to_dict("records"), "pca_var": [float(x) for x in res["pca_var"]],
        "qc_rates": qc_rates(df, res["qc_pooled"], res["qc_within_source"]).to_dict("records"), "n_gated": res["n_gated"],
    }
    explain = explain_tumor(res["df_ok"]) if len(res["df_ok"]) >= 30 else None
    return audit_payload(samples, df, res["qc_pooled"], analysis, title=f"Image audit · {len(samples)} patches", explain=explain)
