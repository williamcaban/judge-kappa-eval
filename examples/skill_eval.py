"""
Example 1 — Skill evaluation (agent-skills-eval compatible).

Runs a 3-judge panel on a SKILL.md skill directory, measuring uplift,
inter-rater agreement (Krippendorff's α), and positional bias.

Install: pip install 'jury-eval[all]'
Run:     python examples/skill_eval.py
"""

from jury_eval import (
    AnthropicBackend,
    AssertionJudge,
    CalibrationExample,
    JudgePanel,
    JuryEvaluator,
    OpenAIBackend,
    PairwiseJudge,
    ScaleType,
    Variant,
    AggregationStrategy,
)

# ── ICL calibration anchors ──────────────────────────────────────────────────
# Ground all judges on the same human-validated reference points.
# More anchors → lower inter-judge score drift → higher Krippendorff's α.

CALIBRATION = [
    CalibrationExample(
        prompt="List the top 3 revenue months from this CSV.",
        output="January, March, July.",
        score=0.3,
        rationale="Correct months but missing the revenue figures the assertion requires.",
    ),
    CalibrationExample(
        prompt="List the top 3 revenue months from this CSV.",
        output="January ($4.2M), March ($3.8M), July ($3.1M) — all other months were below $3M.",
        score=0.95,
        rationale="Identifies months AND figures; context sentence is a bonus.",
    ),
]

# ── LLM backends ────────────────────────────────────────────────────────────
# Mix providers deliberately: if all three judges agree despite different
# training distributions, the signal is more trustworthy.

claude_backend  = AnthropicBackend("claude-sonnet-4-6")
gpt4_backend    = OpenAIBackend("gpt-4o")
llama_backend   = OpenAIBackend(           # vLLM local serving
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# ── Judge Panel (homogeneous rubric) ─────────────────────────────────────────
# All judges use AssertionJudge with the same calibration.
# JudgePanel measures Krippendorff's α across judges.

panel = JudgePanel(
    judges=[
        AssertionJudge("claude",  claude_backend,  calibration_examples=CALIBRATION),
        AssertionJudge("gpt4",    gpt4_backend,    calibration_examples=CALIBRATION),
        AssertionJudge("llama70", llama_backend,   calibration_examples=CALIBRATION),
    ],
    strategy=AggregationStrategy.MEAN,
)

# ── Pairwise judge for positional bias detection ──────────────────────────────
# Uses a separate (trusted) judge to swap A/B positions and detect
# whether scores flip purely due to presentation order.

pairwise_judge = PairwiseJudge("pairwise-claude", claude_backend, calibration_examples=CALIBRATION)

# ── Evaluator ────────────────────────────────────────────────────────────────

evaluator = JuryEvaluator(
    panel=panel,
    generation_backend=claude_backend,   # backend used to run control/treatment variants
    scale_type=ScaleType.ORDINAL,
    positional_judge=pairwise_judge,
    verbosity_bias_threshold=0.30,
)

# ── Variants ─────────────────────────────────────────────────────────────────
# control:   same model, no skill context
# treatment: same model, SKILL.md injected as system context

control   = Variant(name="control",   model="claude-sonnet-4-6")
treatment = Variant(name="treatment", model="claude-sonnet-4-6")
# skill_context is auto-loaded from SKILL.md by evaluate_skill()

# ── Run ──────────────────────────────────────────────────────────────────────

report = evaluator.evaluate_skill(
    skill_dir="./my-skill",   # must contain SKILL.md and evals/evals.json
    control_variant=control,
    treatment_variant=treatment,
)

# ── Results ──────────────────────────────────────────────────────────────────

print(f"\n=== JuryEval Report ===")
print(f"Cases evaluated:      {len(report.cases)}")
print(f"Mean uplift:          {report.mean_uplift:+.3f}  (treatment - control)")
print(f"Control score:        {report.mean_control_score:.3f}")
print(f"Treatment score:      {report.mean_treatment_score:.3f}")
print()
print(f"Inter-rater agreement (corpus):")
print(f"  Krippendorff's α:   {report.agreement.alpha:.3f}  → {report.agreement.alpha_interpretation}")
print(f"  Cohen's κ (mean):   {report.agreement.kappa:.3f}")
print(f"  Expected P(chance): {report.agreement.expected_chance_agreement:.3f}")
print()
print(f"Bias diagnostics:")
print(f"  Positional bias:    {report.bias.positional_bias_rate:.1%} of cases flipped")
print(f"  Verbosity bias ρ:   {report.bias.verbosity_bias_rho:.3f}  (p={report.bias.verbosity_bias_p:.3f})")
print(f"  Verbosity flagged:  {report.bias.verbosity_biased}")
print()
print("Per-case uplift:")
for c in report.cases:
    flag = " ⚠ positional" if c.positional_flip else ""
    print(f"  {c.case_id:20s}  uplift={c.uplift:+.3f}  α={c.agreement.alpha or 'n/a'}{flag}")
