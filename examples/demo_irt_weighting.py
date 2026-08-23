"""
Demo: IRT-based judge weighting (IRTJudgeWeighter).

Fits a 2-Parameter Logistic (2PL) IRT model on judge responses to
human-validated calibration examples. Derives per-judge reliability
weights that replace arbitrary hand-tuned panel weights.

No API key required — uses synthetic calibration data.

Run:
    python examples/demo_irt_weighting.py

Background (Fonseca Rivera et al. 2026, arXiv:2608.05086):
  P(judge scores item correctly | θ, a, b) = 1 / (1 + exp(-a(θ - b)))
  θ = judge reliability (latent ability)
  a = item discrimination (how sharply an item separates reliable from unreliable judges)
  b = item difficulty (how hard it is for any judge to score correctly)
"""

from __future__ import annotations

from judge_kappa.calibration import IRTJudgeWeighter

# ── Calibration data ──────────────────────────────────────────────────────────
# Human experts scored these 12 (prompt, output) pairs.
# Scores represent ground-truth quality; judges are evaluated against these.

human_scores = [
    # Easy items (clear-cut quality signals): judges should score these correctly
    0.95,  # item-0: excellent output — all reliable judges should agree
    0.10,  # item-1: clearly bad — all reliable judges should flag
    0.90,  # item-2: very good
    0.15,  # item-3: poor

    # Medium items (nuanced quality): only well-calibrated judges get these right
    0.65,  # item-4: good-but-not-great
    0.50,  # item-5: borderline
    0.70,  # item-6: above average
    0.45,  # item-7: below average

    # Hard items (subtle quality signals): even good judges may struggle
    0.55,  # item-8: requires domain expertise to judge
    0.60,  # item-9: requires reasoning about trade-offs
    0.48,  # item-10: edge case
    0.52,  # item-11: another edge case
]

# Simulated judge scores on the same 12 items
# "reliable" judges track human scores; "noisy" and "biased" do not
judge_scores = {
    "claude-reliable": [
        0.93, 0.12, 0.88, 0.18,   # easy items — near-perfect
        0.62, 0.52, 0.68, 0.48,   # medium items — good
        0.57, 0.63, 0.50, 0.55,   # hard items — reasonable
    ],
    "gpt4-reliable": [
        0.91, 0.14, 0.86, 0.20,   # easy items
        0.58, 0.55, 0.65, 0.52,   # medium items
        0.60, 0.58, 0.45, 0.58,   # hard items
    ],
    "gpt4o-moderate": [
        0.88, 0.20, 0.82, 0.25,   # easy: slightly off
        0.55, 0.58, 0.60, 0.55,   # medium: OK
        0.65, 0.50, 0.40, 0.60,   # hard: more errors
    ],
    "small-noisy": [
        0.75, 0.35, 0.70, 0.40,   # easy: many errors on clear-cut cases
        0.60, 0.48, 0.55, 0.60,   # medium: noisy
        0.50, 0.65, 0.55, 0.45,   # hard: near-random
    ],
    "lenient-biased": [
        0.95, 0.60, 0.92, 0.65,   # easy: misses bad outputs (always lenient)
        0.80, 0.75, 0.85, 0.72,   # medium: always inflates
        0.78, 0.80, 0.74, 0.78,   # hard: always inflates
    ],
}

# ── Fit the IRT model ─────────────────────────────────────────────────────────

weighter = IRTJudgeWeighter(
    tolerance=0.20,      # |judge_score - human_score| ≤ 0.20 counts as correct
    prior_a_sigma=1.0,   # regularises discrimination (prevents overfitting)
    prior_b_sigma=2.0,   # regularises difficulty
    max_iter=1000,
)

weighter.fit(judge_scores, human_scores)

# ── Results ───────────────────────────────────────────────────────────────────

theta = weighter.theta()
weights = weighter.weights()
item_params = weighter.item_parameters()

print("=" * 60)
print("IRT Judge Weighting — Results")
print("=" * 60)
print()
print(f"{'Judge':<22}  {'θ (ability)':>12}  {'Weight':>8}")
print(f"{'-'*22}  {'-'*12}  {'-'*8}")
for name in sorted(theta, key=lambda k: -theta[k]):
    print(f"  {name:<20}  {theta[name]:>12.4f}  {weights[name]:>8.4f}")
print()
print("  Higher θ = more reliable agreement with human-validated scores.")
print("  Weights are softmax(θ) — sum to 1.0, positive, grounded in reliability.")
print()
print("  Note: equal weights for the three reliable judges is expected — they all")
print("  score within tolerance=0.20 for all items (indistinguishable by this criterion).")
print("  Use tolerance=0.08 to differentiate fine-grained accuracy levels.")
print()

# Item parameters
print("Item parameters (discrimination a, difficulty b):")
print(f"  {'Item':>6}  {'Human score':>12}  {'a (discrim)':>12}  {'b (difficulty)':>14}")
print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*14}")
for i, (p, hs) in enumerate(zip(item_params, human_scores)):
    print(f"  {i:>6}  {hs:>12.2f}  {p['a']:>12.4f}  {p['b']:>14.4f}")
print()
print("  High a → item sharply separates reliable from unreliable judges")
print("  High b → item is difficult (even reliable judges may miss it)")
print()

# ── Using weights in a JudgePanel ─────────────────────────────────────────────

print("=" * 60)
print("Integrating IRT weights into a JudgePanel")
print("=" * 60)
print()
print("  # After fitting, pass weights to JudgePanel:")
print()
print("  from judge_kappa import JudgePanel, AssertionJudge, AnthropicBackend, OpenAIBackend")
print()
print("  w = weighter.weights()")
print("  panel = JudgePanel(")
print("      judges=[")
print("          AssertionJudge('claude-reliable', AnthropicBackend('claude-sonnet-4-6')),")
print("          AssertionJudge('gpt4-reliable',   OpenAIBackend('gpt-4o')),")
print("          AssertionJudge('gpt4o-moderate',  OpenAIBackend('gpt-4o-mini')),")
print("          AssertionJudge('small-noisy',     OpenAIBackend('gpt-3.5-turbo')),")
print("          AssertionJudge('lenient-biased',  OpenAIBackend('gpt-4o-mini')),")
print("      ],")
print("      strategy=AggregationStrategy.WEIGHTED_MEAN,")
print("      weights=[w['claude-reliable'], w['gpt4-reliable'], w['gpt4o-moderate'],")
print("               w['small-noisy'], w['lenient-biased']],")
print("  )")

# Show the concrete weight values
print()
print(f"  # Concrete weights from this run:")
for name in judge_scores:
    print(f"  #   {name}: {weights[name]:.4f}")
