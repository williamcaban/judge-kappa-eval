# Supported Evaluation Target Types

judge-kappa has six entry points covering every way you might have system outputs.

---

## Entry point summary

| Entry point | Evaluation target | Generation model called? | Output type |
|---|---|---|---|
| `evaluate_skill` | SKILL.md + evals.json assertions | Yes — both variants | `EvalReport` (uplift) |
| `evaluate_dataset` | Python callable / pipeline function | Yes — via `predict_fn(inputs)` | `EvalReport` (uplift) |
| `evaluate_endpoints` | Two live OpenAI-compatible endpoints | Yes — one call per variant per case | `EvalReport` (uplift) |
| `evaluate_prerecorded` | Outputs already in the dataset | **No** | `EvalReport` (uplift) |
| `evaluate_pairwise_dataset` | Pre-recorded A/B pairs | **No** | `PairwiseReport` (preference rates) |
| `TournamentEvaluator` | N ≤ 6 systems (callables or endpoints) | Only if using `run_endpoints` | `TournamentReport` (Elo leaderboard) |
| `RankJudge.rank()` | N ≥ 7 systems, listwise (v0.2) | Caller generates; judge ranks | `dict[system → JudgeVerdict]` |

---

## When to use each

### `evaluate_skill` — skill injection A/B
You have a `SKILL.md` skill and want to measure whether injecting it improves outputs.  
Uses assertion-based scoring from `evals/evals.json`.

```python
report = evaluator.evaluate_skill(
    skill_dir="./my-skill",
    control_variant=Variant(name="control"),
    treatment_variant=Variant(name="treatment"),  # SKILL.md auto-injected
)
```

---

### `evaluate_dataset` — callable pipeline A/B
You have a callable (RAG pipeline, LangChain chain, API wrapper) and a dataset.  
Calls your function per case. Both variants call the same function by default — pass separate functions to compare two pipeline versions.

```python
def rag_v1(inputs: dict) -> str: ...
def rag_v2(inputs: dict) -> str: ...

report = evaluator.evaluate_dataset(
    data=my_dataset,
    predict_fn=rag_v1,
    control_variant=Variant(name="v1",  predict_fn=rag_v1),
    treatment_variant=Variant(name="v2", predict_fn=rag_v2),
)
```

**Dataset row format:**
```json
{
  "id": "q001",
  "inputs":       { "question": "What is RAG?", "context": "..." },
  "expectations": { "answer": "Retrieval-Augmented Generation..." }
}
```

---

### `evaluate_endpoints` — live endpoint A/B
You want to compare two model endpoints head-to-head.  
Each `Variant` carries its own `generation_backend`. No shared backend required.

```python
ctrl = Variant(name="gpt-4o-mini", generation_backend=OpenAIBackend("gpt-4o-mini"))
trt  = Variant(name="gpt-4o",      generation_backend=OpenAIBackend("gpt-4o"))

report = evaluator.evaluate_endpoints(cases, ctrl, trt)
```

Use cases:
- GPT-4o vs GPT-4o-mini on the same eval
- Fine-tuned vLLM model vs base model
- Two providers (Anthropic vs OpenAI) on identical prompts
- Prompt variant A vs prompt variant B on the same model

---

### `evaluate_prerecorded` — no generation model
You already have both outputs (from production, from another pipeline, or a prior run).  
**Only judge calls are made** — no generation model is called. Fast and cheap.

```python
data = [
    {
        "id": "q1",
        "inputs": {"question": "What is RAG?"},
        "output_control":   "RAG stands for Retrieval-Augmented Generation...",
        "output_treatment": "RAG combines retrieval with generation...",
    },
]

report = evaluator.evaluate_prerecorded(
    data,
    control_output_field="output_control",
    treatment_output_field="output_treatment",
)
```

Also supports a single `output` field when both variants share the same pre-recorded output (useful when you want to audit what a single system produced):
```json
{ "id": "q1", "inputs": {"question": "..."}, "output": "pre-recorded answer" }
```

---

### `evaluate_pairwise_dataset` — preference rates, not uplift
You want preference rates rather than a score delta.  
A `PairwiseJudge` compares both outputs in one prompt and returns which was preferred.

```python
report = evaluator.evaluate_pairwise_dataset(
    data=pairwise_data,
    pairwise_judge=PairwiseJudge("judge", AnthropicBackend("claude-sonnet-4-6")),
    output_a_field="output_a",
    output_b_field="output_b",
    label_a="System A",
    label_b="System B",
)
```

**Dataset row format:**
```json
{ "id": "pair-001", "prompt": "Explain attention.", "output_a": "...", "output_b": "..." }
```

See [docs/03-pairwise-tournament.md](03-pairwise-tournament.md) for full detail.

---

### `TournamentEvaluator` — N-system Elo leaderboard (N ≤ 6)
You have 3–6 systems to rank. Runs every pair through `PairwiseJudge` (round-robin) and derives Elo ratings.

```python
from judge_kappa import TournamentEvaluator

tournament = TournamentEvaluator(pairwise_judge=judge)
report = tournament.run_dataset(
    data=my_data,
    systems={
        "rag-v1": lambda inputs: rag_v1(inputs["question"]),
        "rag-v2": lambda inputs: rag_v2(inputs["question"]),
        "rag-v3": lambda inputs: rag_v3(inputs["question"]),
    },
)
report.print_leaderboard()
```

See [docs/03-pairwise-tournament.md](03-pairwise-tournament.md) for full detail.

---

### `RankJudge.rank()` — listwise ranking for N ≥ 7 (v0.2)
You have 7 or more systems and pairwise cost is prohibitive. `RankJudge` presents all N outputs to the judge in one prompt per case — O(N) calls instead of O(N²).

```python
from judge_kappa import RankJudge, AnthropicBackend
from collections import defaultdict

judge = RankJudge("rank-j", AnthropicBackend("claude-sonnet-4-6"), max_systems=12)

scores: dict[str, list[float]] = defaultdict(list)
for case in eval_cases:
    outputs = {sys: generate(sys, case.prompt) for sys in systems}
    for sys, verdict in judge.rank(case, outputs).items():
        scores[sys].append(verdict.score)

leaderboard = sorted(scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
```

See [docs/03-pairwise-tournament.md](03-pairwise-tournament.md) § RankJudge for full detail, cost tables, and the recommended 3-step workflow.

---

## EvalReport vs PairwiseReport vs TournamentReport

### `EvalReport` — A/B uplift modes
Produced by: `evaluate_skill`, `evaluate_dataset`, `evaluate_endpoints`, `evaluate_prerecorded`

Key fields:
```
mean_uplift              : +0.234    signed delta (treatment − control)
mean_control_score       : 0.512
mean_treatment_score     : 0.746
agreement.alpha          : 0.812     Krippendorff's α across judges and cases
agreement.alpha_ci_low   : 0.701     bootstrap 95% CI lower bound (v0.2)
agreement.alpha_ci_high  : 0.889     bootstrap 95% CI upper bound (v0.2)
agreement.icc            : 0.831     ICC(2,k) absolute agreement (v0.2)
agreement.kappa          : 0.791     Cohen's κ (mean pairwise)
significance.p_value     : 0.009     McNemar p-value (v0.2)
significance.significant : true      p < 0.05 (v0.2)
significance.uplift_ci_low  : 0.120  bootstrap CI for mean_uplift (v0.2)
significance.uplift_ci_high : 0.348
judge_fit[]              : per-judge outfit MNSQ t-statistic (v0.2)
bias.positional_bias_rate    : 0.083
bias.verbosity_bias_rho      : 0.091
cases[]                  : per-case results with per-judge verdicts
```

Use when you need a continuous quality difference you can track over time.

---

### `PairwiseReport` — preference rates
Produced by: `evaluate_pairwise_dataset`

Key fields:
```
preference_rate_a : 0.27     System A preferred in 27% of cases
preference_rate_b : 0.68     System B preferred in 68% of cases
tie_rate          : 0.05
mean_score_a      : 0.612    absolute score assigned to A by the judge
mean_score_b      : 0.821
cases[]           : per-case scores, preferred label, rationale
```

Use when you need human-interpretable preference data (RLHF, UX research, regulatory evidence).

---

### `TournamentReport` — Elo leaderboard
Produced by: `TournamentEvaluator`

Key fields:
```
standings[]:
  system     : "gpt-4o"
  elo        : 1087.3
  wins       : 45
  losses     : 15
  tie_rate   : 0.0
  win_rate   : 0.75
win_matrix   : { "gpt-4o": { "llama-70b": 0.65 } }
pair_results[]: per-pair preference rates and positional bias rates
```

Use when you need a ranked leaderboard across N systems.
