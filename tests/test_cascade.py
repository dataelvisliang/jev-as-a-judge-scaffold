"""Unit tests for the cascade logic. Run with: pytest tests/"""
import sys

sys.path.insert(0, "src")

import pytest

from jev_cascade import (Choice, FrozenCascade, MockJevJudge, MockStrongJudge,
                         auroc, fit_tau)
from jev_cascade.judges import Verdict


QUESTION = Choice(instructions="pick one",
                  criteria={"a": "first", "b": "second"})


def _item(gold="a"):
    return {"text": "x", "_gold": gold}


class FixedJudge:
    """Returns a canned verdict regardless of input."""

    def __init__(self, label, confidence, cost=0.0, name="fixed"):
        self._v = Verdict(label=label,
                          probabilities={label: confidence,
                                         "other": 1 - confidence},
                          confidence=confidence, cost_usd=cost,
                          judge_name=name)
        self._name = name

    @property
    def name(self):
        return self._name

    def judge(self, state, question):
        return self._v


def test_accept_when_confident():
    cascade = FrozenCascade(FixedJudge("a", 0.95), FixedJudge("b", 0.99), tau=0.9)
    r = cascade.decide(_item(), QUESTION)
    assert not r.escalated
    assert r.final.label == "a"


def test_escalate_when_unsure():
    cascade = FrozenCascade(FixedJudge("a", 0.5), FixedJudge("b", 0.99), tau=0.9)
    r = cascade.decide(_item(), QUESTION)
    assert r.escalated
    assert r.final.label == "b"          # fallback's verdict is consumed
    assert r.first_pass.label == "a"    # first pass kept for analysis


def test_tau_boundaries():
    c_all = FrozenCascade(FixedJudge("a", 0.0), FixedJudge("b", 0.99), tau=1.0)
    assert c_all.decide(_item(), QUESTION).escalated
    c_none = FrozenCascade(FixedJudge("a", 0.0), FixedJudge("b", 0.99), tau=0.0)
    assert not c_none.decide(_item(), QUESTION).escalated


def test_tau_must_be_unit_interval():
    with pytest.raises(ValueError):
        FrozenCascade(FixedJudge("a", 0.5), FixedJudge("b", 0.5), tau=1.5)


def test_fit_tau_prefers_cheapest_qualifying():
    # First pass always right and confident: the smallest tau already
    # retains everything, so fit_tau picks the cheapest one.
    q = [0.95] * 20
    first_correct = [True] * 20
    fb_correct = [True] * 20
    assert fit_tau(q, first_correct, fb_correct) == 0.5


def test_auroc_perfect_separation():
    assert auroc([0.9, 0.95, 0.1, 0.2], [True, True, False, False]) == 1.0


def test_mock_judges_run():
    jev, strong = MockJevJudge(seed=1), MockStrongJudge(seed=1)
    cascade = FrozenCascade(jev, strong, tau=0.9)
    results = [cascade.decide(_item("a" if i % 2 == 0 else "b"), QUESTION)
               for i in range(50)]
    assert any(r.escalated for r in results)
    assert any(not r.escalated for r in results)
