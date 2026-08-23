# Pairwise, Tournament, and Listwise Ranking Modes

All three modes use a judge that compares outputs. They differ in scope, cost, and how many systems they handle.

---

## Mode overview

| Mode | Systems | Judge calls / case | Output | Best for |
|---|---|---|---|---|
| **Pairwise** | Exactly 2 | 2 (with bias detection) | `PairwiseReport` | A/B preference; RLHF; regulatory evidence |
| **Tournament** | 3–6 | N(N−1) (with bias detection) | `TournamentReport` | Leaderboards; model selection |
| **RankJudge** | 7–12 | 1 | Per-case `dict[system → JudgeVerdict]` | Large N, cost-constrained; first-pass ranking |
| **Champion-challenger** | Any N | 2(N−1) | Per-pair `PairwiseReport` | Incremental model selection; very large N |

All modes operate on **pre-recorded or generated outputs**. Use `evaluate_endpoints` or `run_endpoints` first if you need to generate outputs from live endpoints, then compare.

---

## Pairwise mode

### When to use

**Use pairwise when:**
- You have exactly two systems (A and B) and want preference rates
- You are building RLHF preference data from model outputs
- You need a human-interpretable result: "judges preferred B in 73% of cases"
- You are submitting comparative evidence to an audit or regulatory body

**Do NOT use pairwise when:**
- You have 3 or more systems to rank → use Tournament or RankJudge
- You need a continuous quality score, not a preference rate → use `evaluate_prerecorded` with `EvalReport`

### Dataset format

```jsonl
{"id": "pair-001", "prompt": "Explain attention mechanisms.", "output_a": "System A output...", "output_b": "System B output..."}
{"id": "pair-002", "prompt": "What is RAG?",                 "output_a": "System A output...", "output_b": "System B output..."}
```

Field names are configurable via `output_a_field` and `output_b_field`.

### Python API

```python
from judge_kappa import JuryEvaluator, PairwiseJudge, AnthropicBackend

evaluator = JuryEvaluator(panel=my_panel, generation_backend=my_backend)
judge = PairwiseJudge("claude-pairwise", AnthropicBackend("claude-sonnet-4-6"))

report = evaluator.evaluate_pairwise_dataset(
    data=my_data,
    pairwise_judge=judge,
    output_a_field="output_a",
    output_b_field="output_b",
    label_a="System A",
    label_b="System B",
)

print(f"System A preferred: {report.preference_rate_a:.1%}")
print(f"System B preferred: {report.preference_rate_b:.1%}")
print(f"Ties:               {report.tie_rate:.1%}")
```

### Interpreting the output

```
preference_rate_a : 0.27    System A preferred in 27% of cases
preference_rate_b : 0.68    System B preferred in 68% of cases
tie_rate          : 0.05    5% of cases — judge could not distinguish
mean_score_a      : 0.612   absolute score assigned to A (0.0–1.0)
mean_score_b      : 0.821   absolute score assigned to B (0.0–1.0)
```

**Significance thresholds:**
- `preference_rate_b > 0.60` — meaningful signal; System B wins
- `preference_rate_b` between `0.45–0.55` — statistical tie; do not claim a winner
- `tie_rate > 0.30` — systems are very similar or prompts are ambiguous

### Positional bias in pairwise mode

The evaluator runs each case in both orders:
- Round 1: judge sees (A, B)
- Round 2: judge sees (B, A) — swapped

A positionally biased judge always prefers whichever output appears **first**, regardless of content. A positional flip is flagged when the first-positioned output wins in **both** orderings (not just when the winner changes).

```python
# Per-case breakdown
for case in report.cases:
    print(f"{case.case_id}: preferred={case.preferred}  "
          f"score_a={case.score_a:.3f}  score_b={case.score_b:.3f}")
```

**If `positional_bias_rate > 0.15`:** the preference rates are unreliable. Options:
1. Use a different judge model (some are more position-stable than others)
2. Average scores from both orderings (built in by default)
3. Flag the result and do not cite it as regulatory evidence

---

## Tournament mode (3–6 systems)

### When to use

**Use tournament when:**
- You have 3–6 systems to rank
- You want an Elo leaderboard derived from head-to-head pairwise comparisons
- You need win rates for every pair, not just a global rank
- You are making a model selection decision ("which of these 4 candidates should we promote?")

**Do NOT use tournament when:**
- You have 7+ systems and cost is a concern → use RankJudge (first pass) or champion-challenger (incremental)
- You only have 2 systems → use pairwise (same result, cleaner API)
- You need absolute quality scores, not relative rankings → use `evaluate_prerecorded` with a rubric panel

### Entry points

**Comparing callables (RAG pipelines, LangChain chains, etc.):**
```python
from judge_kappa import TournamentEvaluator, PairwiseJudge, AnthropicBackend

tournament = TournamentEvaluator(
    pairwise_judge=PairwiseJudge("judge", AnthropicBackend("claude-sonnet-4-6")),
    detect_positional_bias=True,
)

report = tournament.run_dataset(
    data=my_eval_dataset,
    systems={
        "rag-v1":       lambda inputs: rag_v1(inputs["question"]),
        "rag-v2":       lambda inputs: rag_v2(inputs["question"]),
        "rag-v3":       lambda inputs: rag_v3(inputs["question"]),
        "rag-baseline": lambda inputs: baseline(inputs["question"]),
    },
)
```

**Comparing live endpoints:**
```python
from judge_kappa import OpenAIBackend, Variant

report = tournament.run_endpoints(
    cases=eval_cases,
    variants=[
        Variant(name="gpt-4o-mini",  generation_backend=OpenAIBackend("gpt-4o-mini")),
        Variant(name="gpt-4o",       generation_backend=OpenAIBackend("gpt-4o")),
        Variant(name="llama-70b",    generation_backend=OpenAIBackend(
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            base_url="http://localhost:8000/v1", api_key="no-key",
        )),
        Variant(name="claude-haiku", generation_backend=AnthropicBackend("claude-haiku-4-5-20251001")),
    ],
)
```

**Using pre-recorded outputs from a dataset:**
```python
report = tournament.run(
    cases=eval_cases,
    systems={
        "system-a": lambda inputs: inputs["output_a"],   # pre-recorded in dataset
        "system-b": lambda inputs: inputs["output_b"],
        "system-c": lambda inputs: inputs["output_c"],
    },
)
```

### Reading the results

```python
report.print_leaderboard()
```

```
Rank  System                          Elo    Win%    W    L    T
─────────────────────────────────────────────────────────────────
1     gpt-4o                       1087.3  75.0%   45   15    0
2     llama-70b                    1023.8  58.3%   35   25    0
3     claude-haiku                  987.4  41.7%   25   35    0
4     gpt-4o-mini                   901.5  25.0%   15   45    0
```

```python
# Win matrix: rate at which row beat column
rate = report.win_matrix["gpt-4o"]["llama-70b"]   # → 0.65

# Per-pair results
for pair in report.pair_results:
    print(f"{pair.system_a} vs {pair.system_b}: "
          f"A={pair.preference_rate_a:.1%}  B={pair.preference_rate_b:.1%}  "
          f"pos_bias={pair.positional_bias_rate:.1%}")

# Full standings
for s in report.standings:
    print(f"{s.system}: Elo={s.elo:.0f}  W={s.wins}  L={s.losses}  T={s.ties}")
```

### Interpreting Elo

- Elo is anchored at 1000 (starting value). The absolute number is less meaningful than the gap.
- **100 Elo gap ≈ 64% win rate** for the higher-rated system
- **200 Elo gap ≈ 76% win rate** — decisive
- **< 50 Elo gap** — practical tie; use more cases before claiming a ranking

### When positional bias invalidates tournament results

If any pair has `positional_bias_rate > 0.15`, that pair's preference data is unreliable. The Elo derived from it may be driven by position, not quality.

Options:
1. Switch to a different judge model for that pair
2. Scores from both orderings are already averaged by default — check if the averaged result is still biased
3. Exclude that pair from Elo calculation and report separately

---

## RankJudge — listwise ranking for N ≥ 7

`RankJudge` presents all N outputs to the judge in a single prompt and receives a ranking JSON. This is O(N) in judge calls vs O(N²) for `TournamentEvaluator`.

### When to use

**Use RankJudge when:**
- You have 7–12 systems and pairwise cost is prohibitive
- You want a first-pass ranking before running targeted pairwise comparisons on the top-k
- Context window permits fitting all N outputs in one prompt

**Do NOT use RankJudge when:**
- You need per-pair positional bias rates (no A/B swap is run)
- N × mean_output_length exceeds the judge model's context window
- You need the accuracy of head-to-head comparisons (pairwise is more precise)

### Cost comparison

| N systems | M cases | TournamentEvaluator (N(N−1)×M×2) | RankJudge (M×1) | Reduction |
|---|---|---|---|---|
| 7 | 20 | 1680 calls | 20 calls | **98.8%** |
| 8 | 20 | 2240 calls | 20 calls | **99.1%** |
| 10 | 20 | 3600 calls | 20 calls | **99.4%** |
| 12 | 20 | 5280 calls | 20 calls | **99.6%** |

### Python API

```python
from judge_kappa import RankJudge, AnthropicBackend
from judge_kappa.models import EvalCase
from collections import defaultdict

judge = RankJudge(
    "rank-judge",
    AnthropicBackend("claude-sonnet-4-6"),
    max_systems=12,     # hard cap; raise only if context window allows
)

# Score one case across all systems in a single call
system_outputs = {
    "system-a": "Output from system A...",
    "system-b": "Output from system B...",
    # ... up to max_systems
}
verdicts = judge.rank(case, system_outputs)
# verdicts: dict[system_name → JudgeVerdict]
# scores: rank 1 → 1.0, rank N → 0.0 (normalised)

# Build a leaderboard across many cases
scores_by_system: dict[str, list[float]] = defaultdict(list)
for case in eval_cases:
    outputs = {sys: generate(sys, case) for sys in systems}
    for sys, verdict in judge.rank(case, outputs).items():
        scores_by_system[sys].append(verdict.score)

leaderboard = sorted(
    [(sys, sum(s) / len(s)) for sys, s in scores_by_system.items()],
    key=lambda x: -x[1],
)
for rank, (sys, score) in enumerate(leaderboard, 1):
    print(f"{rank}. {sys}  mean_score={score:.4f}")
```

### Rank-to-score conversion

Rank 1 (best) → score 1.0; rank N (worst) → score 0.0.

```
score_i = (N - rank_i) / (N - 1)
```

This preserves ordinal information while producing `JudgeVerdict` scores compatible with the standard `AgreementResult` / `EvalReport` pipeline.

### Trade-offs vs. TournamentEvaluator

| Aspect | RankJudge | TournamentEvaluator |
|---|---|---|
| Judge calls per case | 1 | N(N−1) / 2 × 2 (with bias detection) |
| Positional bias detection | None — no swap test | Per-pair bias rate reported |
| Context window pressure | Grows with N × output length | Fixed (2 outputs per call) |
| Accuracy | Slightly lower (one listwise judgment) | Higher (independent pairwise votes) |
| Best for | N ≥ 7, first-pass screening | N ≤ 6, final model selection |

### Recommended workflow for N ≥ 7

1. **RankJudge** all N systems → identify top-k (k = 3–5)
2. **TournamentEvaluator** on top-k only → precise Elo within the finalists
3. **Pairwise** on champion vs. runner-up → regulatory-grade evidence

---

## Champion-challenger pattern (any N, incremental)

Full round-robin becomes expensive past N=6. Instead, run each challenger only against the current champion. Best for incremental model selection where a new candidate is evaluated against the current best without re-running all prior comparisons.

```python
from judge_kappa import JuryEvaluator, PairwiseJudge, AnthropicBackend

evaluator = JuryEvaluator(panel=my_panel, generation_backend=my_backend)
pairwise_judge = PairwiseJudge("judge", AnthropicBackend("claude-sonnet-4-6"))

champion = "gpt-4o"
challengers = ["model-a", "model-b", "model-c", "model-d", "model-e", "model-f"]

for challenger in challengers:
    report = evaluator.evaluate_pairwise_dataset(
        data=data,
        pairwise_judge=pairwise_judge,
        output_a_field=f"output_{challenger}",
        output_b_field=f"output_{champion}",
        label_a=challenger,
        label_b=champion,
    )
    if report.preference_rate_a > 0.55:
        print(f"{challenger} beats {champion} ({report.preference_rate_a:.1%}) — promoting")
        champion = challenger
    else:
        print(f"{challenger} loses to {champion} ({report.preference_rate_a:.1%}) — eliminated")

print(f"Final champion: {champion}")
```

**Cost:** `(N−1) × 2M` judge calls — linear in N, not quadratic.

**Trade-off:** If system rankings are not transitive (A beats B, B beats C, but C beats A), the champion-challenger design can miss the true best system. This is rare in practice but worth knowing.

---

## Cost tables

**Variables:**
- **N** — number of systems being compared
- **M** — number of evaluation cases
- **b** — bias detection: ×2 if `detect_positional_bias=True`, ×1 if False
- **G** — generation calls per system per case (only if outputs are not pre-recorded)

### Judge calls (evaluation only, not generation)

| Mode | Formula | N=2, M=20 | N=4, M=20 | N=6, M=20 | N=8, M=20 | N=10, M=20 |
|---|---|---|---|---|---|---|
| Pairwise, no bias detection | M | 20 | — | — | — | — |
| Pairwise, with bias detection | 2M | **40** | — | — | — | — |
| Tournament, no bias detection | C(N,2) × M | — | 120 | 300 | 560 | 900 |
| Tournament, with bias detection | N(N−1) × M | — | **240** | **600** | **1120** | **1800** |
| RankJudge (listwise) | M | — | **20** | **20** | **20** | **20** |
| Champion-challenger, with bias | 2(N−1) × M | — | 120 | 200 | 280 | 360 |

### Total calls including output generation

Only relevant when using `run_endpoints` (live generation). Pre-recorded modes have no generation calls.

| Mode | Formula | N=2, M=20 | N=4, M=20 | N=6, M=20 |
|---|---|---|---|---|
| Pairwise (generate + judge, with bias) | 2M (gen) + 2M (judge) | **80** | — | — |
| Tournament (generate + judge, with bias) | N×M (gen) + N(N−1)×M (judge) | — | **320** | **720** |
| RankJudge (generate + judge) | N×M (gen) + M (judge) | — | **100** | **140** |
| Champion-challenger (generate + judge, with bias) | N×M (gen) + 2(N−1)×M (judge) | — | **200** | **320** |

### Cost reduction strategies

| Strategy | How | Savings | Trade-off |
|---|---|---|---|
| Skip bias detection | `detect_positional_bias=False` | 50% judge calls | Positional bias undetected |
| Pre-record outputs first | Run generation separately, reuse | Avoid re-generating per run | Outputs may become stale |
| RankJudge first pass | Score all N in one call, then pairwise top-k | O(N) first pass | Less precision than full pairwise |
| Champion-challenger | Only (challenger vs champion) per round | O(N) instead of O(N²) | May miss non-transitive rankings |
| Fewer cases | Use representative subset | Linear reduction | Wider confidence intervals |
| Cheaper judge model | `claude-haiku` or `gpt-4o-mini` | 60–80% cost reduction | Potentially lower quality |

---

## Mode selection summary

```
How many systems?
│
├─ 2  → Pairwise (evaluate_pairwise_dataset)
│        Want score delta instead of preference? → evaluate_prerecorded (EvalReport)
│
├─ 3–6  → Tournament (TournamentEvaluator)
│          Full round-robin: all C(N,2) pairs compared
│          Cost: N(N−1) × M × 2 judge calls
│
└─ 7+  → Choose by situation:
           │
           ├─ Quick first-pass, cost-sensitive
           │   → RankJudge (listwise, 1 judge call per case)
           │   → Then pairwise on top-k finalists
           │
           └─ Incremental evaluation (new model vs current best)
               → Champion-challenger (N−1 pairwise comparisons)
               → Cost: (N−1) × 2M judge calls (linear)
```
