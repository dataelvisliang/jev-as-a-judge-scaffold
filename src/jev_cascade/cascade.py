"""The frozen confidence cascade.

Rule: accept the first-pass judge's verdict when q >= tau, otherwise
escalate to the fallback judge and take its verdict.

tau is fitted on a SELECTION set (see calibration.fit_tau) and FROZEN
before evaluation. The paper's headline policy is tau = 0.9 with a strong
LLM fallback. Thresholds do not transfer across tasks: refit locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .judges import Question, Verdict


@dataclass(frozen=True)
class CascadeResult:
    state: dict[str, Any]
    question: Question
    final: Verdict        # the verdict the pipeline consumes
    first_pass: Verdict   # what the cheap judge said (kept for analysis)
    escalated: bool


@dataclass(frozen=True)
class FrozenCascade:
    first_pass: Any  # Judge
    fallback: Any    # Judge
    tau: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")

    def decide(self, state: dict[str, Any], question: Question) -> CascadeResult:
        v1 = self.first_pass.judge(state, question)
        if v1.confidence >= self.tau:
            return CascadeResult(state=state, question=question,
                                 final=v1, first_pass=v1, escalated=False)
        v2 = self.fallback.judge(state, question)
        return CascadeResult(state=state, question=question,
                             final=v2, first_pass=v1, escalated=True)
