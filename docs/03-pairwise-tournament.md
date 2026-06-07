# Pairwise and Tournament Modes

Both modes use `PairwiseJudge` — a judge that scores two outputs in a single prompt and returns which was preferred. They differ in scope and cost.

---

## Mode overview

| Mode | Systems | Judge calls / case | Output | Best for |
|---|---|---|---|---|
| **Pairwise** | Exactly 2 | 2 (with bias detection) | `PairwiseReport` | A/B preference; RLHF; regulatory evidence |
| **Tournament** | N ≥ 3 | N(N−1) (with bias detection) | `TournamentReport` | Leaderboards; model selection |
| **Champion-challenger** | N ≥ 7 | 2(N−1) | Per-pair `PairwiseReport` | Large-scale ranking, cost-constrained |

Both modes operate on **pre-recorded outputs** — outputs already written to the dataset. Neither calls a generation model during comparison. Use `evaluate_endpoints` first if you need to generate outputs from live endpoints, then pass the results.

---

## Pairwise mode

### When to use

**Use pairwise when:**
- You have exactly two systems (A and B) and want preference rates
- You are building RLHF preference data from model outputs
- You need a human-interpretable result: "judges preferred B in 73% of cases"
- You are submitting comparative evidence to an audit or regulatory body

**Do NOT use pairwise when:**
- You have 3 or more systems to rank → use Tournament
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

A positionally biased judge always prefers whichever output appears first, regardless of content. Check `positional_bias_rate` across all cases.

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

## Tournament mode

### When to use

**Use tournament when:**
- You have 3–6 systems to rank
- You want an Elo leaderboard derived from head-to-head pairwise comparisons
- You need win rates for every pair, not just a global rank
- You are making a model selection decision ("which of these 4 candidates should we promote?")

**Do NOT use tournament when:**
- You have 7+ systems and cost is a concern → use champion-challenger
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
        "rag-v1":      lambda inputs: rag_v1(inputs["question"]),
        "rag-v2":      lambda inputs: rag_v2(inputs["question"]),
        "rag-v3":      lambda inputs: rag_v3(inputs["question"]),
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

## Champion-challenger pattern (N ≥ 7)

Full round-robin becomes expensive past N=6. Instead, run each challenger only against the current champion.

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

| Mode | Formula | N=2, M=20 | N=3, M=20 | N=4, M=20 | N=5, M=20 | N=6, M=20 |
|---|---|---|---|---|---|---|
| Pairwise, no bias detection | M | 20 | — | — | — | — |
| Pairwise, with bias detection | 2M | **40** | — | — | — | — |
| Tournament, no bias detection | C(N,2) × M | — | 60 | 120 | 200 | 300 |
| Tournament, with bias detection | N(N−1) × M | — | **120** | **240** | **400** | **600** |
| Champion-challenger, with bias | 2(N−1) × M | — | 80 | 120 | 160 | 200 |

### Total calls including output generation

Only relevant when using `run_endpoints` (live generation). Pre-recorded modes have no generation calls.

| Mode | Formula | N=2, M=20 | N=4, M=20 | N=6, M=20 |
|---|---|---|---|---|
| Pairwise (generate + judge, with bias) | 2M (gen) + 2M (judge) | **80** | — | — |
| Tournament (generate + judge, with bias) | N×M (gen) + N(N−1)×M (judge) | — | **320** | **720** |
| Champion-challenger (generate + judge, with bias) | N×M (gen) + 2(N−1)×M (judge) | — | **200** | **320** |

### Cost reduction strategies

| Strategy | How | Savings | Trade-off |
|---|---|---|---|
| Skip bias detection | `detect_positional_bias=False` | 50% judge calls | Positional bias undetected |
| Pre-record outputs first | Run generation separately, reuse | Avoid re-generating per run | Outputs may become stale |
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
└─ 7+  → Champion-challenger
           Each challenger vs current champion only
           Cost: (N−1) × 2M judge calls (linear)
```
