# jev-as-a-judge-scaffold

A minimal, runnable scaffold of the cascade idea from
**"JEV-as-a-Judge: Accept When Confident, Escalate When Unsure"**
(Yubo Li, Yidi Miao, Ramayya Krishnan, Rema Padman, Carnegie Mellon University,
arXiv:2609.26550, September 2026).

This repo does **not** reproduce the paper's experiments. It captures the
paper's core idea as clean, reusable code: a cheap decision-only judge with a
confidence signal, plus a frozen confidence cascade that escalates uncertain
cases to a strong (expensive) judge.

## The idea in 30 seconds

1. A cheap **decision-only judge** (TypeSafe Jev) answers typed questions
   (`Choice` / `Noul` / `Score`) about an agent's output and returns label
   probabilities in **one forward pass**. No text generation, no sampling.
2. Its **confidence** `q = max_k p_k` (the top label probability) tells you
   when to trust it. One call gives you both the verdict and the confidence.
3. A **frozen cascade** accepts Jev's verdict when `q >= tau` and escalates
   the rest to a strong LLM judge.
4. `tau` is fitted on a **selection set** and frozen before touching the test
   set. Thresholds do not transfer across tasks: refit locally.
5. Paper headline (preference + evidence-grounded factuality): Jev lands
   within 3pp of the strongest LLM judge at 0.36% of its fee; the cascade
   retains ~99% of the strong judge's accuracy at roughly half the cost.

See [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md) for the full notes.

## Layout

```
config/default.yaml        frozen policy: pinned model versions, tau
src/jev_cascade/
    judges.py              Judge protocol, typed questions, Jev/LLM/mock judges
    cascade.py             FrozenCascade: accept if q >= tau, else escalate
    calibration.py         fit tau on a selection set, temperature scaling, AUROC
    metrics.py             accuracy retained, fee ratio, escalation rate, CIs
    harness.py             run_cascade / run_judge over a list of items
examples/quickstart.py     end-to-end demo with mock judges (no API key needed)
tests/test_cascade.py      unit tests for the cascade logic
docs/PAPER_NOTES.md        paper summary: setup, results, warnings
```

## Quickstart (no API key needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # or: PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src python examples/quickstart.py
```

The demo builds 400 synthetic items, splits them 40/60 into a selection set
and a test set (mirroring the paper's protocol), fits `tau` on the selection
set, freezes it, then evaluates the frozen cascade on the test set.

## Using it on your own data

1. Collect labeled items: `{"text": ..., "_gold": <label>}` plus a `Choice`
   question describing the judgment.
2. Split into a **selection set** (fit here) and a **test set** (evaluate here).
   Never tune on the test set.
3. Fit: `tau = fit_tau(q, first_correct, fallback_correct)` in
   `calibration.py`. Optionally temperature-scale Jev's probabilities first
   with `fit_temperature`.
4. Freeze `tau` into `config/default.yaml` and evaluate with `harness.py`.
5. Check the report: escalation rate, accuracy retained vs the fallback alone,
   fee ratio, and the AUROC of `q` against correctness. If the AUROC is near
   0.5 on your task, confidence routing will not help: escalate everything or
   pick a different first-pass judge.

## Warnings (from the paper, encoded as comments in the code)

- **Thresholds do not transfer.** A `tau` fitted on one task/dataset fails on
  another. Always refit locally.
- **Confidently wrong exists.** On hard pairs (elaborately written wrong
  answers) Jev is wrong on a third of the pairs it scores in [0.9, 0.95).
  Validate the cascade on the kind of items it will actually meet.
- **Reference-free prose breaks it.** AUROC of `q` near 0.5 means no
  threshold helps.
- **Pin the model version** (`jev-1.13.0`, not `jev-latest`). The alias moves
  on release and silently invalidates a frozen `tau`.

## Citation

```bibtex
@misc{li2026jevasajudge,
  title={JEV-as-a-Judge: Accept When Confident, Escalate When Unsure},
  author={Yubo Li and Yidi Miao and Ramayya Krishnan and Rema Padman},
  year={2026}, eprint={2609.26550}, archivePrefix={arXiv}
}
```
