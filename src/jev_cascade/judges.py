"""Judge interfaces.

A Judge answers a typed Question about a state dict and returns a Verdict:
a label plus a probability distribution over labels, produced in ONE
forward pass. Confidence q = max_k p_k comes from that distribution, not
from repeated sampling (this is the whole point of a decision-only model).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Questions: the typed interface a decision-only judge exposes.
# Mirrors Jev's three question types: Choice, Noul, Score.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """Pick one label from a fixed set."""
    instructions: str
    criteria: dict[str, str | None]  # label -> what the label means


@dataclass(frozen=True)
class Noul:
    """Yes/no question; the answer is P(yes)."""
    instructions: str


@dataclass(frozen=True)
class Score:
    """Rate the input on ordered rubric levels (worst -> best)."""
    instructions: str
    criteria: list[str]


Question = Choice | Noul | Score


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    label: str
    probabilities: dict[str, float]
    confidence: float  # q = max_k p_k, or the judge's native confidence
    cost_usd: float = 0.0
    judge_name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class Judge(Protocol):
    name: str

    def judge(self, state: dict[str, Any], question: Question) -> Verdict:
        """Return a verdict for `question` about `state` in one call."""
        ...


# ---------------------------------------------------------------------------
# JevJudge: first-pass judge backed by TypeSafe Jev (hosted, proprietary).
# ---------------------------------------------------------------------------

class JevJudge:
    """Cheap decision-only first pass.

    Requires TYPESAFE_API_KEY in the environment and `pip install typesafe-sdk`.
    The model version is PINNED (default jev-1.13.0, the version evaluated in
    the paper). Do not use the `jev-latest` alias for a frozen policy.
    """

    PRICE_PER_MTOK_USD = 0.042  # input tokens; output tokens are free

    def __init__(self, model: str = "jev-1.13.0") -> None:
        try:
            from typesafe_sdk import TypeSafeClient  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "typesafe-sdk is not installed. Run `pip install typesafe-sdk`, "
                "or use MockJevJudge for offline work."
            ) from exc
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise OSError("TYPESAFE_API_KEY is not set.")
        self.model = model

    @property
    def name(self) -> str:
        return f"jev/{self.model}"

    def judge(self, state: dict[str, Any], question: Question) -> Verdict:
        from typesafe_sdk import Choice as SdkChoice
        from typesafe_sdk import Noul as SdkNoul
        from typesafe_sdk import Score as SdkScore
        from typesafe_sdk import TypeSafeClient

        if isinstance(question, Choice):
            sdk_q: Any = SdkChoice(instructions=question.instructions,
                                   criteria=question.criteria)
        elif isinstance(question, Noul):
            sdk_q = SdkNoul(instructions=question.instructions)
        elif isinstance(question, Score):
            sdk_q = SdkScore(instructions=question.instructions,
                             criteria=question.criteria)
        else:
            raise TypeError(f"unknown question type: {type(question)}")

        with TypeSafeClient() as client:
            response = client.system_one(state=state,
                                         questions={"q": sdk_q},
                                         model=self.model)
        answer = response.answers["q"]
        label, probs = self._to_label_probs(answer, question)
        confidence = float(getattr(answer, "confidence", max(probs.values())))
        return Verdict(label=label, probabilities=probs, confidence=confidence,
                       cost_usd=self._estimate_cost(state, question),
                       judge_name=self.name)

    @staticmethod
    def _to_label_probs(answer: Any, question: Question) -> tuple[str, dict[str, float]]:
        if isinstance(question, Choice):
            probs = dict(answer.probabilities)
            return str(answer.choice), probs
        if isinstance(question, Noul):
            p_yes = float(answer.noul)
            return ("yes" if p_yes >= 0.5 else "no",
                    {"yes": p_yes, "no": 1.0 - p_yes})
        # Score: label is the argmax rubric level.
        probs = {str(k): float(v) for k, v in dict(answer.probabilities).items()}
        label = max(probs, key=lambda k: probs[k])
        return label, probs

    def _estimate_cost(self, state: dict[str, Any], question: Question) -> float:
        # Rough estimate: ~4 chars per token. Prefer reported usage when
        # available; this is only for fee-ratio bookkeeping.
        text = json.dumps(state, default=str) + question.instructions
        tokens = max(1, len(text) // 4)
        return tokens / 1_000_000 * self.PRICE_PER_MTOK_USD


# ---------------------------------------------------------------------------
# VerbalizedJudge: plug any strong LLM in as the fallback.
# ---------------------------------------------------------------------------

class VerbalizedJudge:
    """Fallback judge built on any callable returning (label, probs, confidence).

    The paper's generative baselines were given the same decision-and-probability
    contract as Jev (verdict + verbalized label probabilities, no rationale).
    Wire your strongest judge here, e.g. an OpenAI/Anthropic/Google call.
    """

    def __init__(self, name: str,
                 fn: Callable[[dict[str, Any], Question],
                              tuple[str, dict[str, float], float]],
                 cost_per_call_usd: float = 0.0) -> None:
        self._name = name
        self._fn = fn
        self._cost = cost_per_call_usd

    @property
    def name(self) -> str:
        return self._name

    def judge(self, state: dict[str, Any], question: Question) -> Verdict:
        label, probs, confidence = self._fn(state, question)
        return Verdict(label=label, probabilities=dict(probs),
                       confidence=confidence, cost_usd=self._cost,
                       judge_name=self.name)


# ---------------------------------------------------------------------------
# Mock judges: deterministic stand-ins for offline development and tests.
# Items must carry "_gold" with the true label.
# ---------------------------------------------------------------------------

def _labels_of(question: Question) -> list[str]:
    if isinstance(question, Choice):
        return list(question.criteria)
    if isinstance(question, Noul):
        return ["yes", "no"]
    return [str(i) for i in range(len(question.criteria))]


class MockJevJudge:
    """Simulates a cheap judge whose confidence separates right from wrong.

    Correct verdicts get high q (Beta(8, 2)), wrong ones low q (Beta(2, 5)),
    so the AUROC of q against correctness is well above 0.5 and a cascade
    is worth running. Tune `accuracy` to explore harsher regimes.
    """

    def __init__(self, accuracy: float = 0.86, seed: int = 42,
                 cost_per_call_usd: float = 2e-5,
                 name: str = "mock-jev") -> None:
        self.accuracy = accuracy
        self.cost_per_call_usd = cost_per_call_usd
        self._name = name
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def judge(self, state: dict[str, Any], question: Question) -> Verdict:
        gold = str(state["_gold"])
        labels = _labels_of(question)
        if self._rng.random() < self.accuracy:
            label = gold
            q = self._rng.betavariate(8, 2)
        else:
            label = self._rng.choice([l for l in labels if l != gold] or [gold])
            q = self._rng.betavariate(2, 5)
        rest = (1.0 - q) / max(1, len(labels) - 1)
        probs = {l: (q if l == label else rest) for l in labels}
        return Verdict(label=label, probabilities=probs, confidence=q,
                       cost_usd=self.cost_per_call_usd, judge_name=self.name)


class MockStrongJudge:
    """Simulates the expensive fallback: accurate, confident, pricey."""

    def __init__(self, accuracy: float = 0.93, seed: int = 7,
                 cost_per_call_usd: float = 0.006,
                 name: str = "mock-strong") -> None:
        self.accuracy = accuracy
        self.cost_per_call_usd = cost_per_call_usd
        self._name = name
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def judge(self, state: dict[str, Any], question: Question) -> Verdict:
        gold = str(state["_gold"])
        labels = _labels_of(question)
        label = gold if self._rng.random() < self.accuracy else \
            self._rng.choice([l for l in labels if l != gold] or [gold])
        q = self._rng.betavariate(9, 1)
        rest = (1.0 - q) / max(1, len(labels) - 1)
        probs = {l: (q if l == label else rest) for l in labels}
        return Verdict(label=label, probabilities=probs, confidence=q,
                       cost_usd=self.cost_per_call_usd, judge_name=self.name)
