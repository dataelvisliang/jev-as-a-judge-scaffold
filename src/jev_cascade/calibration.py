"""Calibration: fit the cascade's tau on a SELECTION set, then freeze it.

Protocol (mirrors the paper):
  1. Split labeled data into a selection set and a test set BEFORE any fitting.
  2. Optionally temperature-scale the first-pass probabilities on the
     selection set (Guo et al., 2017).
  3. Sweep tau on the selection set; pick the SMALLEST tau (least escalation,
     lowest cost) that retains `target_retention` of the fallback's accuracy.
  4. Freeze tau. Evaluate once on the test set. Never refit on test.

Thresholds do not transfer across tasks or model versions. A tau fitted
here is valid only for this judge, this task, this data distribution.
"""
from __future__ import annotations

import math
from typing import Sequence


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC of `scores` against binary `labels` (Mann-Whitney U).

    Used as: auroc(q, first_pass_correct). Well above 0.5 means confidence
    separates right from wrong and a cascade is worth running. Near 0.5
    (the paper's reference-free prose case) means no threshold helps.
    """
    n_pos = sum(1 for l in labels if l)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = sum(rank for rank, i in enumerate(order, start=1) if labels[i])
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_temperature(prob_list: Sequence[dict[str, float]],
                    gold_labels: Sequence[str],
                    grid: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
                    ) -> float:
    """Grid-search a temperature T minimizing NLL on the SELECTION set.

    Scaling: p_i -> p_i^(1/T), renormalized. T < 1 sharpens, T > 1 softens.
    Fit on the selection set only, then freeze alongside tau.
    """
    best_t, best_nll = 1.0, math.inf
    for t in grid:
        nll = 0.0
        for probs, gold in zip(prob_list, gold_labels):
            scaled = {l: p ** (1.0 / t) for l, p in probs.items()}
            z = sum(scaled.values()) or 1.0
            nll -= math.log(scaled.get(gold, 1e-12) / z + 1e-12)
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def apply_temperature(probs: dict[str, float], t: float) -> dict[str, float]:
    scaled = {l: p ** (1.0 / t) for l, p in probs.items()}
    z = sum(scaled.values()) or 1.0
    return {l: v / z for l, v in scaled.items()}


def sweep_tau(q: Sequence[float],
              first_correct: Sequence[bool],
              fallback_correct: Sequence[bool],
              taus: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
              ) -> list[dict]:
    """Score every tau on the SELECTION set. Returns rows with tau,
    escalation_rate, cascade accuracy, and accuracy retained vs fallback."""
    n = len(q)
    fb_acc = sum(fallback_correct) / n
    rows = []
    for tau in taus:
        correct = [(fc if qi < tau else vc)
                   for qi, vc, fc in zip(q, first_correct, fallback_correct)]
        acc = sum(correct) / n
        rows.append({
            "tau": tau,
            "escalation_rate": sum(1 for qi in q if qi < tau) / n,
            "cascade_accuracy": acc,
            "retained": acc / fb_acc if fb_acc else float("nan"),
        })
    return rows


def fit_tau(q: Sequence[float],
            first_correct: Sequence[bool],
            fallback_correct: Sequence[bool],
            target_retention: float = 0.99,
            taus: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
            ) -> float:
    """Pick the SMALLEST tau (least escalation, lowest cost) retaining
    `target_retention` of the fallback's accuracy on the SELECTION set.

    Larger tau means more escalation and higher cost, so the cheapest
    policy meeting the accuracy target uses the smallest qualifying tau.
    Freeze the returned tau. Do not refit on the test set.
    """
    fb_acc = sum(fallback_correct) / len(fallback_correct)
    target = fb_acc * target_retention
    for tau in sorted(taus):  # ascending: cheapest first
        acc = sum((fc if qi < tau else vc)
                  for qi, vc, fc in zip(q, first_correct, fallback_correct)
                  ) / len(q)
        if acc >= target:
            return tau
    return max(taus)  # target unreachable: escalate as much as possible
