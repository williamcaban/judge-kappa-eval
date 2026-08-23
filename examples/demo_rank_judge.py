"""
Demo: RankJudge — listwise ranking for N ≥ 7 systems.

Compares 8 system variants using a single prompt-per-case rather than
C(8,2) = 28 pairwise calls. Shows:
  - Cost comparison: RankJudge vs. TournamentEvaluator
  - How to build a leaderboard from rank verdicts
  - Spearman ρ between rank order and ground-truth quality order

No API key required — uses a MockRankBackend that simulates a realistic ranking LLM.

Run:
    python examples/demo_rank_judge.py

When to use RankJudge vs. TournamentEvaluator:
  N ≤ 6 systems   → TournamentEvaluator (pairwise, most accurate, tractable cost)
  N ≥ 7 systems   → RankJudge (listwise, O(N) calls, context pressure trade-off)
  N ≥ 15 systems  → Champion-challenger design (rank first, then pairwise top-k)
"""

from __future__ import annotations

import json
from collections import defaultdict

from judge_kappa.judges.rank import RankJudge
from judge_kappa.models import EvalCase


# ── Systems under evaluation ──────────────────────────────────────────────────
# Ground-truth quality order (best to worst) — not known to the judge
SYSTEMS = [
    "gpt-4o",           # rank 1 (best)
    "claude-sonnet",    # rank 2
    "gemini-1.5-pro",   # rank 3
    "llama-3-70b",      # rank 4
    "gpt-4o-mini",      # rank 5
    "claude-haiku",     # rank 6
    "llama-3-8b",       # rank 7
    "mistral-7b",       # rank 8 (worst)
]

# Simulated outputs per system for 5 eval cases
OUTPUTS: dict[str, list[str]] = {
    sys: [f"{sys} answer to case {i}" for i in range(5)]
    for sys in SYSTEMS
}


# ── Mock backend that returns realistic rankings ──────────────────────────────

class MockRankBackend:
    """
    Simulates a ranking LLM. In production, replace with AnthropicBackend or OpenAIBackend.

    This mock produces rankings close to the ground-truth quality order with
    slight noise (±1 rank swap per case) to simulate real-world variance.
    """

    def __init__(self, noise_seed: int = 42) -> None:
        import random
        self._rng = random.Random(noise_seed)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        # Extract the label letters from the user prompt
        import re
        labels = re.findall(r"## Output ([A-Z])", user)
        n = len(labels)
        if not n:
            return json.dumps({"rankings": {}, "rationale": "mock"})

        # Ground-truth order: labels appear in system-quality order
        # (because OUTPUTS dict preserves insertion order and we iterate SYSTEMS)
        ranks = {label: i + 1 for i, label in enumerate(labels)}

        # Add noise: swap 1-2 adjacent ranks
        swap_count = self._rng.randint(1, 2)
        items = list(ranks.items())
        for _ in range(swap_count):
            i = self._rng.randint(0, n - 2)
            items[i], items[i + 1] = items[i + 1], items[i]
        # Re-assign sequential ranks after swap
        swapped = {label: i + 1 for i, (label, _) in enumerate(items)}

        return json.dumps({"rankings": swapped, "rationale": "quality-based ranking"})


# ── Cost comparison ───────────────────────────────────────────────────────────

n_systems = len(SYSTEMS)
n_cases   = 5
pairwise_calls = (n_systems * (n_systems - 1) // 2) * n_cases * 2  # ×2 for positional swap
rank_calls = n_cases  # one call per case

print("=" * 60)
print(f"Cost comparison: {n_systems} systems × {n_cases} cases")
print("=" * 60)
print(f"  TournamentEvaluator (pairwise):  {pairwise_calls:4d} judge calls")
print(f"  RankJudge (listwise):            {rank_calls:4d} judge calls")
print(f"  Reduction:                       {1 - rank_calls/pairwise_calls:.0%}")
print()

# ── Run RankJudge ─────────────────────────────────────────────────────────────

cases = [EvalCase(id=f"case-{i}", prompt=f"Eval question {i}") for i in range(n_cases)]
judge = RankJudge("rank-judge-1", MockRankBackend(), max_systems=12)

# Accumulate scores across cases per system
accumulated: dict[str, list[float]] = defaultdict(list)
call_count = 0

for case in cases:
    system_outputs = {sys: OUTPUTS[sys][int(case.id.split("-")[1])] for sys in SYSTEMS}
    verdicts = judge.rank(case, system_outputs)
    call_count += 1
    for sys, verdict in verdicts.items():
        accumulated[sys].append(verdict.score)

# ── Leaderboard ───────────────────────────────────────────────────────────────

mean_scores = {sys: sum(scores) / len(scores) for sys, scores in accumulated.items()}
ranked = sorted(mean_scores.items(), key=lambda x: -x[1])

print("=" * 60)
print("RankJudge Leaderboard")
print("=" * 60)
print(f"  {'Rank':>4}  {'System':<20}  {'Mean score':>10}  {'GT rank':>7}")
print(f"  {'-'*4}  {'-'*20}  {'-'*10}  {'-'*7}")

predicted_order = [sys for sys, _ in ranked]
for rank, (sys, score) in enumerate(ranked, 1):
    gt_rank = SYSTEMS.index(sys) + 1
    delta = abs(rank - gt_rank)
    marker = " ✓" if delta == 0 else (f" (±{delta})" if delta <= 2 else f" ⚠ (±{delta})")
    print(f"  {rank:>4}  {sys:<20}  {score:>10.4f}  {gt_rank:>7}{marker}")

print()

# Spearman ρ between predicted and ground-truth rank order
from scipy.stats import spearmanr
gt_ranks = [SYSTEMS.index(sys) + 1 for sys in predicted_order]
pr_ranks  = list(range(1, n_systems + 1))
rho, p = spearmanr(pr_ranks, gt_ranks)

print(f"  Spearman ρ with ground truth: {rho:.4f}  (p={p:.4f})")
print(f"  Judge calls used:             {call_count}  (vs. {pairwise_calls} pairwise)")
print()

# ── Production usage with real backends ──────────────────────────────────────

print("=" * 60)
print("Production usage with a real backend")
print("=" * 60)
print("""
  from judge_kappa import AnthropicBackend
  from judge_kappa.judges import RankJudge
  from judge_kappa.models import EvalCase

  judge = RankJudge(
      "rank-claude",
      AnthropicBackend("claude-sonnet-4-6"),
      max_systems=12,          # raise to 15 for very long contexts
  )

  for case in eval_cases:
      system_outputs = {
          "system-a": pipeline_a(case.prompt),
          "system-b": pipeline_b(case.prompt),
          # ... up to max_systems
      }
      verdicts = judge.rank(case, system_outputs)
      # verdicts: dict[system_name → JudgeVerdict] with score 0–1

  # Aggregate across cases to build a leaderboard:
  # scores[system] = mean(verdict.score for each case)
""")

print("  Required: ANTHROPIC_API_KEY")
print("  Cost:     1 judge call per case (vs. C(N,2)×2 for TournamentEvaluator)")
