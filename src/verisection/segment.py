"""Segmentation ensembles.

The baseline is an intensity mixture model fitted by k-means on bootstrap
resamples of *sections*. Each ensemble member gives a different parameter set,
which captures epistemic uncertainty from stain/section variation. Real models
(HoVer-Net, StarDist, CellViT, ...) plug in behind the same ``probs(image, n)`` API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import KMeans


class SegmentationSamples(Protocol):
    n_members: int
    n_classes: int

    def probs(self, image: np.ndarray, n: int) -> np.ndarray:  # (C, H, W)
        ...


@dataclass
class SegmentationEnsemble:
    means: np.ndarray  # (N, C)
    sigmas: np.ndarray  # (N, C)
    priors: np.ndarray  # (N, C)

    @property
    def n_members(self) -> int:
        return self.means.shape[0]

    @property
    def n_classes(self) -> int:
        return self.means.shape[1]

    @classmethod
    def fit(
        cls,
        images: list[np.ndarray],
        n_classes: int,
        n_members: int,
        rng: np.random.Generator,
        subsample: int = 20000,
    ) -> "SegmentationEnsemble":
        means, sigmas, priors = [], [], []
        K = len(images)
        for _ in range(n_members):
            pick = rng.integers(0, K, size=K)  # bootstrap over sections
            pix = np.concatenate([images[k].ravel() for k in pick])
            if pix.size > subsample:
                pix = rng.choice(pix, size=subsample, replace=False)
            km = KMeans(
                n_clusters=n_classes, n_init=3, random_state=int(rng.integers(0, 2**31 - 1))
            ).fit(pix.reshape(-1, 1))
            order = np.argsort(km.cluster_centers_.ravel())
            centers = km.cluster_centers_.ravel()[order]
            lab = np.argsort(order)[km.labels_]
            sig = np.array([max(pix[lab == c].std(), 1e-3) for c in range(n_classes)])
            pri = np.array([(lab == c).mean() for c in range(n_classes)])
            means.append(centers)
            sigmas.append(sig)
            priors.append(pri)
        return cls(np.array(means), np.array(sigmas), np.array(priors))

    def probs(self, image: np.ndarray, n: int) -> np.ndarray:
        mu = self.means[n][:, None, None]
        sg = self.sigmas[n][:, None, None]
        logp = np.log(self.priors[n][:, None, None] + 1e-12) - 0.5 * ((image[None] - mu) / sg) ** 2 - np.log(sg)
        logp -= logp.max(axis=0, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=0, keepdims=True)
        return p.astype(np.float32)

    def point(self) -> "SegmentationEnsemble":
        """Single-member ensemble at the mean parameters (for single-source baselines)."""
        return SegmentationEnsemble(
            self.means.mean(0, keepdims=True),
            self.sigmas.mean(0, keepdims=True),
            self.priors.mean(0, keepdims=True),
        )


def _features(image: np.ndarray) -> np.ndarray:
    """(F, H, W): intensity at three scales and local standard deviation (a small texture cue)."""
    from scipy import ndimage as ndi

    x = image.astype(np.float32)
    g1, g3 = ndi.gaussian_filter(x, 1.2), ndi.gaussian_filter(x, 3.0)
    m = ndi.uniform_filter(x, 5)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(x * x, 5) - m * m, 0.0))
    return np.stack([x, g1, g3, sd])


@dataclass
class SupervisedSegmentationEnsemble:
    """Pixel classifier trained on labelled sections *from other specimens* (a stand-in for a trained model).

    Each ensemble member is a quadratic discriminant fitted on a bootstrap resample of the training pixels,
    using multi-scale intensity and local-variance features, so members disagree where the training data
    constrain the classes poorly (epistemic uncertainty) while each posterior also carries class overlap
    (aleatoric uncertainty).
    """

    models: list
    n_classes: int

    @property
    def n_members(self) -> int:
        return len(self.models)

    @classmethod
    def fit(
        cls,
        images: list[np.ndarray],
        labels: list[np.ndarray],
        n_classes: int,
        n_members: int,
        rng: np.random.Generator,
        max_pixels: int = 60000,
    ) -> "SupervisedSegmentationEnsemble":
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        X = np.concatenate([_features(im).reshape(4, -1).T for im in images])
        y = np.concatenate([np.asarray(lb).ravel() for lb in labels]).astype(int)
        models = []
        for _ in range(n_members):
            pick = rng.choice(len(y), size=min(max_pixels, len(y)), replace=True)
            # guarantee every class is present so predict_proba has a column per class
            for c in range(n_classes):
                if not (y[pick] == c).any():
                    pick[: 5] = np.flatnonzero(y == c)[:5]
            qda = QuadraticDiscriminantAnalysis(reg_param=0.05).fit(X[pick], y[pick])
            models.append(qda)
        return cls(models, n_classes)

    def probs(self, image: np.ndarray, n: int) -> np.ndarray:
        f = _features(image)
        P = self.models[n].predict_proba(f.reshape(f.shape[0], -1).T)
        full = np.full((P.shape[0], self.n_classes), 1e-6, dtype=np.float64)
        full[:, self.models[n].classes_] = P
        full /= full.sum(axis=1, keepdims=True)
        return full.T.reshape((self.n_classes,) + image.shape).astype(np.float32)

    def point(self) -> "SupervisedSegmentationEnsemble":
        """Single-member ensemble (the first member) for single-source baselines."""
        return SupervisedSegmentationEnsemble(self.models[:1], self.n_classes)
