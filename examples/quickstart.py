"""End-to-end demo of the frozen confidence cascade. No API key needed.

Mirrors the paper's protocol:
  1. Build synthetic labeled items and split 40/60 into selection / test.
  2. Fit tau on the SELECTION set (target: retain 99% of fallback accuracy).
  3. Freeze tau. Evaluate once on the TEST set and print the report.
"""
import random
import sys

sys.path.insert(0, "src")

from jev_cascade import (Choice, FrozenCascade, MockJevJudge, MockStrongJudge,
                         auroc, fit_tau, report, run_cascade, run_judge,
                         split_selection_test, sweep_tau)

LABELS = ["refund", "billing", "technical", "other"]

QUESTION = Choice(
    instructions="What is the customer's main request?",
    criteria={
        "refund": "The customer wants money returned.",
        "billing": "Payment or subscription issues.",
        "technical": "The customer needs a bug or integration fixed.",
        "other": "None of the other options clearly fits.",
    },
)

TEMPLATES = [
    "I was charged twice and need the duplicate refunded today.",
    "My invoice shows the wrong amount for last month.",
    "The Stripe integration keeps failing with a 500 error.",
    "Where can I find the API documentation?",
]


def make_items(n: int = 400, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        gold = LABELS[i % len(LABELS)]  # balanced classes
        items.append({"text": rng.choice(TEMPLATES), "_gold": gold})
    return items


def main() -> None:
    items = make_items()
    selection, test = split_selection_test(items, selection_frac=0.4, seed=0)
    print(f"selection: {len(selection)} items, test: {len(test)} items\n")

    first_pass = MockJevJudge(accuracy=0.90, seed=42)
    fallback = MockStrongJudge(accuracy=0.96, seed=7)

    # --- fit on SELECTION only -------------------------------------------
    sel_judgments = run_judge(first_pass, selection, QUESTION)
    sel_fallback = run_judge(fallback, selection, QUESTION)
    q = [j.verdict.confidence for j in sel_judgments]
    first_correct = [j.verdict.label == s["_gold"]
                     for j, s in zip(sel_judgments, selection)]
    fb_correct = [j.verdict.label == s["_gold"]
                  for j, s in zip(sel_fallback, selection)]

    print(f"AUROC(q vs first-pass correct) on selection: "
          f"{auroc(q, first_correct):.3f}\n")
    print("tau sweep on selection:")
    for row in sweep_tau(q, first_correct, fb_correct):
        print(f"  tau={row['tau']:<5} escalate={row['escalation_rate']:.1%} "
              f"cascade_acc={row['cascade_accuracy']:.1%} "
              f"retained={row['retained']:.1%}")

    tau = fit_tau(q, first_correct, fb_correct, target_retention=0.99)
    print(f"\nFROZEN tau = {tau}  (do not refit on test)\n")

    # --- evaluate once on TEST -------------------------------------------
    cascade = FrozenCascade(first_pass=first_pass, fallback=fallback, tau=tau)
    results = run_cascade(cascade, test, QUESTION)
    fb_test = run_judge(fallback, test, QUESTION)
    golds = [t["_gold"] for t in test]
    test_q = [r.first_pass.confidence for r in results]
    test_correct = [r.first_pass.label == g for r, g in zip(results, golds)]
    print(report(results, [j.verdict for j in fb_test], golds,
                 tau=tau, auroc_q=auroc(test_q, test_correct)))


if __name__ == "__main__":
    main()
