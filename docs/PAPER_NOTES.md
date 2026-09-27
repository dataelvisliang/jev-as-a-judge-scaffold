# Paper notes: JEV-as-a-Judge (arXiv:2609.26550)

**Title:** JEV-as-a-Judge: Accept When Confident, Escalate When Unsure
**Authors:** Yubo Li, Yidi Miao, Ramayya Krishnan, Rema Padman (Carnegie Mellon University)
**Date:** submitted 22 Sep 2026. **Note:** Jev itself is built by TypeSafe AI;
this paper is an independent evaluation, not a model release. No official
code was published with the paper.

## What Jev is

A hosted "System One" decision model. It does not generate text. You send a
`state` (the thing to judge) plus typed questions, and it returns label
probabilities as JSON in one parallel forward pass:

- `Choice`: pick one label from a fixed set, with per-label probabilities
- `Noul`: yes/no question, returns P(yes)
- `Score`: rate against ordered rubric levels, returns level probabilities

Published operating point (jev-1.13.0): $0.042 per million input tokens,
output tokens free, 70-500ms end to end. Access via `POST
https://api.typesafe.ai/v1/systemone` (API key), Cloudflare Workers AI
(`typesafe/jev`), or Vercel AI Gateway (`typesafe-ai/jev`).

## Experimental setup

- Compared jev-as-a-judge against **16 generative and reward-model judges**
  (GPT-4.1/5.x/6, Claude Sonnet 5, Gemini 3.x, Qwen, Skywork-Reward-V2,
  PairRM, ...), with blinded human adjudication of disagreements.
- Benchmarks: RewardBench (400 pairs), JudgeBench (350 pairs), HaluEval
  (240 evidence-grounded judgments), plus adjudication/control sets.
- Generative baselines got the same decision-and-probability contract as Jev
  (verdict + verbalized probabilities, no rationale).
- Protocol discipline: a 642-item pilot was frozen before any inference and
  split 40/60 into a selection set (fit temperatures and routing thresholds
  only) and a test set; a 670-item extension was frozen before pilot accuracy
  was inspected. "Frozen" policies below are pre-specified, not post hoc.

## Headline results

| Finding | Detail |
|---|---|
| Ordinary preference + evidence-grounded factuality | Within 3pp of the strongest LLM judge at **0.36% of its fee** |
| Where it breaks | Checking a derivation; resisting elaborately written wrong answers |
| Confidence signal | `q = max_k p_k`; Spearman 0.97+ with Jev's native confidence |
| Cascade (tau=0.9, GPT-6 fallback, pooled) | Escalates 34%, retains **99.6%** of GPT-6 accuracy at **47%** of its fee (~$6.3 / 1k judgments) |
| Frozen two-order policy (tau=0.9, 510 pairs) | Accepts 53.7%, 92.5% vs 93.1% accuracy, 56.8% of fallback fee |
| RewardBench specifically | Cascade **beats** GPT-6 alone (94.0% vs 93.5%) at 22% of fee: the two judges err on different items |

## The cascade rule (what this repo implements)

```
verdict, q = first_pass.judge(state, question)   # one call
if q >= tau:   return verdict                     # accept
else:          return fallback.judge(state, question)  # escalate
```

- `tau` is fitted on the selection set (largest tau retaining the target
  fraction of fallback accuracy) and frozen before test evaluation.
- Random escalation at the same budget does far worse, so `q` carries real
  signal; a label-aware oracle does better still, so the signal is imperfect.

## Warnings (load-bearing, not footnotes)

1. **Thresholds do not transfer.** Fit on RewardBench does not carry to
   JudgeBench-style items, let alone your own task. Always refit locally.
2. **Confidently wrong is real.** On hard RM-Bench pairs (rejected answer is
   the more elaborately written one), Jev is wrong on ~1/3 of pairs it scores
   in [0.9, 0.95); tau=0.9 retains only 96.5% there vs 99.6% on normal pairs.
3. **Reference-free prose: AUROC of q is 0.518.** No threshold helps; the
   cascade cannot fix a first pass that is incompetent-but-confident.
4. **Pin the model version.** `jev-latest` moves on release; a frozen tau is
   only valid for the exact model build it was fitted on.

## What to steal for your own eval

- The split of labor: code checks what the agent *did* (tool calls, steps,
  exact-match fields); the judge only reads what the agent *wrote*.
- The operating pattern: cheap typed judge as default, strong judge only on
  low-confidence items, humans only on the residual.
- The methodology that makes the numbers honest: frozen pilot, selection/test
  split, pre-specified thresholds, paired bootstrap CIs, reported fee ratios
  with conservative bounds.
