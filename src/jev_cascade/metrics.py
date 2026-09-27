"""Evaluation metrics for the cascade.

Headline numbers to report (the paper's framing):
  - accuracy retained: cascade accuracy as a fraction of the fallback alone
  - fee ratio: total cascade cost relative to running the fallback on everything
  - escalation rate: fraction of items sent to the fallback
  - AUROC of q against first-pass correctness: does confidence separate
    right from wrong? (Near 0.5 means the cascade cannot help.)
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from .cascade import CascadeResult
from .judges import Verdict


def accuracy(verdicts: Sequence[Verdict], golds: Sequence[str]) -> float:
    return sum(v.label == g for v, g in zip(verdicts, golds)) / len(golds)


def escalation_rate(results: Sequence[CascadeResult]) -> float:
    return sum(r.escalated for r in results) / len(results)


def cascade_cost(results: Sequence[CascadeResult]) -> float:
    total = 0.0
    for r in results:
        total += r.first_pass.cost_usd
        if r.escalated:
            total += r.final.cost_usd
    return total


def fee_ratio(results: Sequence[CascadeResult],
              fallback_verdicts: Sequence[Verdict]) -> float:
    """Total cascade cost / cost of running the fallback on every item."""
    denom = sum(v.cost_usd for v in fallback_verdicts)
    return cascade_cost(results) / denom if denom else float("nan")


def paired_bootstrap_ci(a_correct: Sequence[bool], b_correct: Sequence[bool],
                        n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI for (accuracy_b - accuracy_a) via paired cluster bootstrap."""
    rng = random.Random(seed)
    n = len(a_correct)
    deltas = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_correct[i] for i in idx) / n
        db = sum(b_correct[i] for i in idx) / n
        deltas.append(db - da)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[int(0.975 * n_resamples) - 1]
    return lo, hi


def report(results: Sequence[CascadeResult],
           fallback_verdicts: Sequence[Verdict],
           golds: Sequence[str], tau: float,
           auroc_q: float) -> str:
    """One printable summary of a frozen-cascade evaluation."""
    casc_acc = accuracy([r.final for r in results], golds)
    fb_acc = accuracy(fallback_verdicts, golds)
    first_acc = accuracy([r.first_pass for r in results], golds)
    retained = casc_acc / fb_acc if fb_acc else float("nan")
    lo, hi = paired_bootstrap_ci(
        [v.label == g for v, g in zip(fallback_verdicts, golds)],
        [r.final.label == g for r, g in zip(results, golds)])
    lines = [
        "Frozen cascade evaluation",
        f"  tau (frozen)               : {tau}",
        f"  items                      : {len(results)}",
        f"  escalation rate            : {escalation_rate(results):.1%}",
        f"  first-pass accuracy        : {first_acc:.1%}",
        f"  fallback accuracy          : {fb_acc:.1%}",
        f"  cascade accuracy           : {casc_acc:.1%}",
        f"  accuracy retained          : {retained:.1%} of fallback",
        f"  cascade-minus-fallback 95% CI: [{lo:+.2%}, {hi:+.2%}]",
        f"  fee ratio vs fallback alone: {fee_ratio(results, fallback_verdicts):.3f}",
        f"  AUROC(q vs first-pass correct): {auroc_q:.3f}",
    ]
    return "\n".join(lines)
