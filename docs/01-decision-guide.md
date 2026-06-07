# Decision Guide

Use this tree to pick the right mode for your situation.

---

## Q1 — What is your input format?

```
├─ SKILL.md + evals/evals.json  ──────────────────────────────── skill mode
│   "Did adding this skill improve outputs?"
│   Measures uplift (treatment − control).
│   → examples/config_skill_minimal.yaml  (single judge, fastest)
│   → examples/config_skill.yaml          (multi-judge panel + agreement)
│
└─ JSONL dataset  ─────────────────────────────────────────────── dataset mode
    "How good are these outputs? Did prompt B beat prompt A?"
    Measures absolute quality + A/B between system prompt variants.
    → examples/config_dataset.yaml         (diverse jury)
    → examples/config_dataset_panel.yaml   (homogeneous panel, verify rubric)
```

---

## Q2 — What do you want from multiple judges?

```
├─ "Are my judges consistent with each other?"  ──────────────── JudgePanel
│   All judges share the same rubric.
│   Krippendorff α → 1.0: rubric is unambiguous; scores are trustworthy.
│   Low α is actionable: add ICL calibration examples or tighten wording.
│   → examples/config_skill.yaml
│   → examples/config_dataset_panel.yaml
│
└─ "I want diverse perspectives, not consensus."  ────────────── JudgeJury
    Different judges cover different quality dimensions.
    Low α is expected and healthy — diversity is the point.
    Aggregate with weighted_mean to prioritise trusted judges.
    → examples/config_dataset.yaml
```

---

## Q3 — Do you need to detect judge bias?

```
├─ Positional bias ("does first position always win?")
│   Runs A/B swap test per case. Adds 2× calls for the positional judge.
│   Rate > 15%: the judge is position-biased; scores are unreliable.
│   → add positional_judge: section to any config
│   → examples/config_regulatory.yaml
│
└─ Verbosity bias ("do longer outputs score higher?")
    Spearman ρ > threshold AND p < 0.05 → flag.
    Default threshold: 0.30; stricter for regulatory use: 0.25.
    → verbosity_bias_threshold: 0.25
    → examples/config_regulatory.yaml
```

---

## Q4 — How many systems are you comparing?

```
├─ Exactly 2  ────────────────────────────────────────────────── Pairwise
│   Want preference rates ("judges preferred B 68% of cases")?
│   → evaluate_pairwise_dataset() → PairwiseReport
│   → examples/config_pairwise.yaml
│   See: docs/03-pairwise-tournament.md
│
├─ 3–6 systems  ──────────────────────────────────────────────── Tournament
│   Full round-robin: every pair compared; produces Elo leaderboard.
│   Cost: N(N−1) × M judge calls (with bias detection).
│   → TournamentEvaluator
│   See: docs/03-pairwise-tournament.md
│
└─ 7+ systems  ───────────────────────────────────────────────── Champion-challenger
    Run each challenger only against the current champion.
    Linear cost O(N) instead of quadratic O(N²).
    See: docs/03-pairwise-tournament.md § Champion-challenger pattern
```

---

## Q5 — Do your judges come from different providers?

```
Anthropic + OpenRouter + vLLM + Ollama in one panel?
→ Use api_key_env per backend — each resolves its own key independently.
→ examples/config_mixed_panel.yaml
→ See: docs/05-key-management.md
```

---

## Quick reference

| Goal | Mode | Config example |
|---|---|---|
| Skill evaluation (minimal) | skill + assertion + 1 judge | `config_skill_minimal.yaml` |
| Skill + rubric clarity check | skill + assertion + panel | `config_skill.yaml` |
| RAG / LLM quality scoring | dataset + rubric + jury | `config_dataset.yaml` |
| Rubric consistency check | dataset + rubric + panel | `config_dataset_panel.yaml` |
| Compare two live endpoints | endpoints + assertion + panel | `config_endpoints.yaml` |
| Score pre-recorded outputs | prerecorded + rubric + panel | `config_prerecorded.yaml` |
| A/B preference rates | pairwise | `config_pairwise.yaml` |
| N-system leaderboard | tournament | Python API |
| Regulatory evidence | skill/dataset + regulatory config | `config_regulatory.yaml` |
| Mixed providers (4 backends) | any panel/jury | `config_mixed_panel.yaml` |
