"""Evidence-gated verdicts with conformal risk control.

Decisions here concern the *analysis* (is this number reportable?), for researchers,
with reasons and override logging. They are not clinical decisions.

A measurement is summarised by an evidence vector e = (o, w, s, c, u):
  o  observed fraction of the support,
  w  interval half-width relative to the estimate,
  s  stability (lower CI bound of feature ICC; 1.0 for features with no fragility concern),
  c  calibration status of the uncertainty used ('pass' | 'fail' | 'untested'),
  u  out-of-distribution flag.
One strictness parameter ``lam`` in [0, 1] tightens all thresholds together, so the
false-claim loss is non-increasing in ``lam`` (a requirement of conformal risk control).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Verdict(str, Enum):
    REPORTABLE = "reportable"
    CAVEAT = "reportable_with_caveat"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class EvidenceVector:
    o: float
    w: float
    s: float = 1.0
    c: str = "pass"
    u: bool = False


@dataclass
class VerdictPolicy:
    w_cap: float = 1.0  # relative half-width allowed at lam=0
    caveat_margin: float = 0.10  # soft band below the threshold
    use_o: bool = True
    use_w: bool = True
    use_s: bool = True

    def thresholds(self, lam: float) -> dict[str, float]:
        return {"o": lam, "w": self.w_cap * (1.0 - lam), "s": lam}

    def decide(self, e: EvidenceVector, lam: float) -> tuple[Verdict, list[str]]:
        reasons: list[str] = []
        if e.c == "fail":
            return Verdict.INSUFFICIENT, ["calibration failed for this stratum"]
        if e.u:
            return Verdict.INSUFFICIENT, ["out-of-distribution input"]
        t = self.thresholds(lam)
        hard, soft = [], []
        if self.use_o and e.o < t["o"]:
            (soft if e.o >= t["o"] - self.caveat_margin else hard).append(
                f"observed fraction {e.o:.2f} < {t['o']:.2f}"
            )
        if self.use_w and e.w > t["w"]:
            (soft if e.w <= t["w"] * 1.5 + 1e-12 else hard).append(
                f"relative half-width {e.w:.2f} > {t['w']:.2f}"
            )
        if self.use_s and e.s < t["s"]:
            (soft if e.s >= t["s"] - self.caveat_margin else hard).append(
                f"stability {e.s:.2f} < {t['s']:.2f}"
            )
        if hard:
            return Verdict.INSUFFICIENT, hard
        if soft or e.c == "untested":
            reasons = soft + (["calibration untested"] if e.c == "untested" else [])
            return Verdict.CAVEAT, reasons
        return Verdict.REPORTABLE, []

    def reportable(self, e: EvidenceVector, lam: float) -> bool:
        """Strict check used for risk control: Reportable only (caveats count as not reported)."""
        return self.decide(e, lam)[0] is Verdict.REPORTABLE


LAMBDAS = np.linspace(0.0, 1.0, 101)


def false_claim_losses(
    units: list[tuple[EvidenceVector, bool]], policy: VerdictPolicy, lambdas: np.ndarray = LAMBDAS
) -> np.ndarray:
    """Loss matrix (n_units, n_lambdas): 1 if reported and the interval missed the truth."""
    L = np.zeros((len(units), len(lambdas)))
    for i, (e, covered) in enumerate(units):
        if covered:
            continue
        for j, lam in enumerate(lambdas):
            L[i, j] = 1.0 if policy.reportable(e, lam) else 0.0
    return L


def is_monotone_nonincreasing(losses: np.ndarray) -> bool:
    return bool(np.all(np.diff(losses, axis=1) <= 1e-12))


def conformal_risk_control(
    losses: np.ndarray, alpha: float, lambdas: np.ndarray = LAMBDAS, B: float = 1.0
) -> dict:
    """Pick the least strict ``lam`` whose adjusted empirical risk is <= ``alpha``.

    Conformal risk control (Angelopoulos et al., ICLR 2024) requires a bounded loss that is
    non-increasing in ``lam``; both are checked. ``feasible`` is False when even the strictest
    setting cannot certify ``alpha`` with this many calibration units (then the policy abstains).
    """
    if not is_monotone_nonincreasing(losses):
        raise ValueError("loss is not non-increasing in lambda; risk control does not apply")
    n = losses.shape[0]
    risk = losses.mean(axis=0)
    adjusted = n / (n + 1.0) * risk + B / (n + 1.0)
    ok = np.flatnonzero(adjusted <= alpha)
    if ok.size == 0:
        return {"lam_hat": float(lambdas[-1]), "feasible": False, "n": n, "risk": risk, "adjusted": adjusted}
    return {"lam_hat": float(lambdas[ok[0]]), "feasible": True, "n": n, "risk": risk, "adjusted": adjusted}


def evaluate_policy(
    units: list[tuple[EvidenceVector, bool]], policy: VerdictPolicy, lam: float
) -> dict:
    """False-claim rate and abstention on test units at a given strictness."""
    rep = np.array([policy.reportable(e, lam) for e, _ in units])
    cov = np.array([c for _, c in units])
    n = len(units)
    return {
        "n": n,
        "reportable_rate": float(rep.mean()),
        "abstention_rate": float(1.0 - rep.mean()),
        "false_claim_rate": float((rep & ~cov).sum() / n),
        "false_claims_among_reported": float((rep & ~cov).sum() / max(rep.sum(), 1)),
    }
