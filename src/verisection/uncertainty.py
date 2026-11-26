"""Monte-Carlo uncertainty propagation and source decomposition.

Registration samples (M) and segmentation samples (N) are crossed; each pair gives
a full per-voxel posterior (including the interpolation model). From the
(M, N, C, Z, Y, X) stack we derive

* the aggregated posterior ``p_hat`` and its entropy ``u_total``;
* the entropy split ``expected_entropy`` (per-sample, aleatoric-type) +
  ``mutual_information`` (disagreement across samples, epistemic-type);
* variance attribution of the MAP-class probability to registration vs
  segmentation (total and first-order effects, plus interaction).

Units differ across sources (entropy in bits, variance dimensionless); they are
reported separately and never summed into one ad-hoc number.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .interpolate import InterpConfig
from .provenance import Provenance
from .reconstruct import build_volume, section_priors
from .register import RegistrationEnsemble
from .segment import SegmentationEnsemble
from .volume import SectionStack

MAX_SAMPLE_PAIRS = 400


def entropy_bits(p: np.ndarray, axis: int = 0) -> np.ndarray:
    return -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum(axis=axis)


@dataclass
class UncertaintyMaps:
    p_hat: np.ndarray  # (C, Z, Y, X)
    provenance: np.ndarray  # (Z, Y, X)
    z_um: np.ndarray
    dz_um: float
    pixel_um: float
    gap_distance_um: np.ndarray
    class_names: list[str]
    u_total: np.ndarray  # entropy of p_hat (bits)
    expected_entropy: np.ndarray
    mutual_information: np.ndarray
    var_total: np.ndarray
    var_reg: np.ndarray  # total effect of registration: E_n Var_m
    var_seg: np.ndarray  # total effect of segmentation: E_m Var_n
    var_reg_first: np.ndarray  # first-order: Var_m E_n
    var_seg_first: np.ndarray  # first-order: Var_n E_m
    var_interaction: np.ndarray
    measurements: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (M, N)
    n_reg: int = 0
    n_seg: int = 0

    @property
    def map_labels(self) -> np.ndarray:
        return self.p_hat.argmax(axis=0).astype(np.int8)

    @property
    def u_interp(self) -> np.ndarray:
        """Per-sample (expected) entropy inside gaps: the interpolation model's own uncertainty."""
        return np.where(self.provenance == Provenance.INTERPOLATED, self.expected_entropy, 0.0)


def propagate(
    stack: SectionStack,
    reg: RegistrationEnsemble,
    seg: SegmentationEnsemble,
    interp: InterpConfig,
    class_names: list[str],
    measure_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> UncertaintyMaps:
    """Propagate registration x segmentation samples through reconstruction."""
    M, N = reg.n_samples, seg.n_members
    if reg.n_sections != stack.n_sections:
        raise ValueError("registration ensemble and stack disagree on the number of sections")
    if M * N > MAX_SAMPLE_PAIRS:
        raise ValueError(f"M*N={M * N} exceeds MAX_SAMPLE_PAIRS={MAX_SAMPLE_PAIRS}")
    section_probs_by_n = [
        [seg.probs(img, n) for img in stack.images] for n in range(N)
    ]
    priors = np.mean([section_priors(sp) for sp in section_probs_by_n], axis=0)

    all_p, provs = [], []
    for m in range(M):
        row = []
        for n in range(N):
            vol = build_volume(
                stack, reg.transforms[m], section_probs_by_n[n], interp, class_names, priors
            )
            row.append(vol.label_prob)
            provs.append(vol.provenance)
            last = vol
        all_p.append(np.stack(row))
    p_all = np.stack(all_p)  # (M, N, C, Z, Y, X)
    prov_stack = np.stack(provs)
    prov = np.where((prov_stack == 0).any(axis=0), 0, prov_stack[0]).astype(np.uint8)

    p_hat = p_all.mean(axis=(0, 1))
    h_total = entropy_bits(p_hat)
    exp_h = entropy_bits(p_all, axis=2).mean(axis=(0, 1))
    mi = np.maximum(h_total - exp_h, 0.0)

    cstar = p_hat.argmax(axis=0)
    idx = np.broadcast_to(cstar[None, None, None], (M, N, 1) + cstar.shape)
    s = np.take_along_axis(p_all, idx, axis=2)[:, :, 0]  # (M, N, Z, Y, X)
    var_total = s.reshape((M * N,) + s.shape[2:]).var(axis=0)
    var_reg = s.var(axis=0).mean(axis=0)
    var_seg = s.var(axis=1).mean(axis=0)
    first_reg = s.mean(axis=1).var(axis=0)
    first_seg = s.mean(axis=0).var(axis=0)
    inter = var_total - first_reg - first_seg

    meas: dict[str, np.ndarray] = {}
    if measure_fns:
        for name, fn in measure_fns.items():
            meas[name] = np.array(
                [[fn(p_all[m, n], prov) for n in range(N)] for m in range(M)]
            )
    return UncertaintyMaps(
        p_hat=p_hat.astype(np.float32),
        provenance=prov,
        z_um=last.z_um,
        dz_um=last.dz_um,
        pixel_um=last.pixel_um,
        gap_distance_um=last.gap_distance_um,
        class_names=class_names,
        u_total=h_total,
        expected_entropy=exp_h,
        mutual_information=mi,
        var_total=var_total,
        var_reg=var_reg,
        var_seg=var_seg,
        var_reg_first=first_reg,
        var_seg_first=first_seg,
        var_interaction=inter,
        measurements=meas,
        n_reg=M,
        n_seg=N,
    )
