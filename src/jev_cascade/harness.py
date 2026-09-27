"""Harness: run judges and cascades over a list of items."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .cascade import CascadeResult, FrozenCascade
from .judges import Judge, Question, Verdict


@dataclass(frozen=True)
class Judgment:
    state: dict[str, Any]
    question: Question
    verdict: Verdict


def run_judge(judge: Judge, items: Sequence[dict[str, Any]],
              question: Question) -> list[Judgment]:
    return [Judgment(state=item, question=question,
                     verdict=judge.judge(item, question))
            for item in items]


def run_cascade(cascade: FrozenCascade, items: Sequence[dict[str, Any]],
                question: Question) -> list[CascadeResult]:
    return [cascade.decide(item, question) for item in items]


def split_selection_test(items: Sequence[dict[str, Any]],
                         selection_frac: float = 0.4,
                         seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split BEFORE any fitting. The paper used 40/60 on its pilot.

    Fit temperatures and tau on the selection split only. Evaluate the
    frozen policy once on the test split.
    """
    import random
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    k = int(len(items) * selection_frac)
    return items[:k], items[k:]
