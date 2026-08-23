"""
Demo: Psychometric metrics — bootstrap CI, ICC, person-fit, behavioral alignment,
DIF, and McNemar significance. All run on synthetic data; no API key required.

Run:
    python examples/demo_psychometric.py

What this shows:
  1. Bootstrap 95% CI for Krippendorff's α
  2. ICC(2,k) alongside α — reveals *why* judges disagree
  3. PersonFitAnalyzer — flags inconsistent judges (outfit MNSQ t-statistic)
  4. BehavioralAlignmentMetric — cross-condition consistency (DISC-style)
  5. DifferentialItemFunctioningDetector — eval cases biased toward a judge family
  6. McNemar significance test + bootstrap CI for mean uplift (via JuryEvaluator)
"""

from __future__ import annotations

import json

# ── Synthetic verdict builder ─────────────────────────────────────────────────

from judge_kappa.models import (
    AgreementResult,
    Assertion,
    CaseResult,
    EvalCase,
    JudgeVerdict,
    ScaleType,
    Variant,
    VariantResult,
)


def _v(judge_id: str, case_id: str, score: float, variant: str = "treatment") -> JudgeVerdict:
    return JudgeVerdict(
        judge_id=judge_id,
        eval_case_id=case_id,
        variant_name=variant,
        score=score,
        rationale="synthetic",
        output_token_count=int(score * 100) + 20,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap CI for Krippendorff's α
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Bootstrap 95% CI for Krippendorff's α")
print("=" * 60)
print()

from judge_kappa.agreement import KrippendorffAlpha

# Three judges scoring 10 cases — moderate agreement (α ≈ 0.72)
verdicts_panel = [
    _v("claude", f"case-{i}", 0.5 + 0.04 * i) for i in range(10)
] + [
    _v("gpt4",   f"case-{i}", 0.5 + 0.04 * i + 0.05 * (-1 if i % 3 == 0 else 1)) for i in range(10)
] + [
    _v("llama",  f"case-{i}", 0.5 + 0.04 * i - 0.03 * (i % 2)) for i in range(10)
]

alpha_metric = KrippendorffAlpha(bootstrap_ci=True, n_bootstrap=500, seed=42)
result = alpha_metric.compute(verdicts_panel, ScaleType.ORDINAL)

print(f"  Krippendorff's α:  {result.alpha:.4f}")
print(f"  95% CI:            [{result.alpha_ci_low:.4f}, {result.alpha_ci_high:.4f}]")
print(f"  Interpretation:    {result.alpha_interpretation}")
print(f"  Judges: {result.n_judges}   Cases: {result.n_cases}")
print()
print("  Interpretation guide:")
print("    α ≥ 0.80 with CI entirely above 0.80 → report as strong agreement")
print("    CI overlapping 0.67 → annotate as 'tentative' even if point estimate is ≥ 0.80")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. ICC(2,k) — variance decomposition
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. ICC(2,k) — between-cases vs. between-judges variance")
print("=" * 60)
print()

from judge_kappa.agreement.icc import compute_icc

icc_val, icc_interp = compute_icc(verdicts_panel)
print(f"  ICC(2,k): {icc_val:.4f}  → {icc_interp}")
print()

# Scenario B: noisy judges — low α AND low ICC (random disagreement)
import random as _random
_rng = _random.Random(99)
verdicts_noisy = (
    [_v("judge-A", f"case-{i}", 0.40 + 0.05 * i) for i in range(8)] +
    [_v("judge-B", f"case-{i}", _rng.uniform(0.1, 0.9)) for i in range(8)]
)
icc_noisy, icc_noisy_interp = compute_icc(verdicts_noisy)
alpha_noisy = KrippendorffAlpha(bootstrap_ci=False).compute(verdicts_noisy, ScaleType.INTERVAL)

print("  Noisy judge scenario (judge-B scores randomly):")
print(f"    Krippendorff α: {alpha_noisy.alpha:.4f}  → low (random judge noise)")
print(f"    ICC(2,k):       {icc_noisy:.4f}  → {icc_noisy_interp}")
print()
print("  ICC decomposes variance: check whether MSB (between-cases signal) >> MSE (residual).")
print("  Both low → rubric is noisy or judge-B needs more ICL calibration examples.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. PersonFitAnalyzer — detect inconsistent judges
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("3. PersonFitAnalyzer — outfit MNSQ t-statistic per judge")
print("=" * 60)
print()

from judge_kappa.agreement import PersonFitAnalyzer

# 4 stable judges + 1 erratic judge, 30 cases.
# Stable judges: scores spread across [0.1, 0.9] with small gaussian noise.
# Erratic judge: score = 1 - expected_score (systematically inverts the difficulty).
# This produces large residuals for the erratic judge against consensus.
_rng_fit = _random.Random(17)

n_fit_cases = 30
verdicts_fit = []
stable_scores = [round(0.1 + 0.027 * i, 3) for i in range(n_fit_cases)]  # 0.10..0.89

for sid in range(4):
    for i in range(n_fit_cases):
        noise = _rng_fit.gauss(0, 0.05)
        verdicts_fit.append(_v(f"stable-{sid+1}", f"fc{i}", max(0.02, min(0.98, stable_scores[i] + noise))))

for i in range(n_fit_cases):
    # Erratic: inverts quality ordering + adds large noise
    inv_score = 1.0 - stable_scores[i] + _rng_fit.gauss(0, 0.10)
    verdicts_fit.append(_v("erratic", f"fc{i}", max(0.02, min(0.98, inv_score))))

analyzer = PersonFitAnalyzer(alpha_threshold=1.96)
fit_results = analyzer.analyze(verdicts_fit)

print(f"  {'Judge':<15}  {'t-stat':>8}  {'Flagged':>8}")
print(f"  {'-'*15}  {'-'*8}  {'-'*8}")
fit_sorted = sorted(fit_results, key=lambda r: r.lz_statistic, reverse=True)
for r in fit_sorted:
    flag = "⚠  YES" if r.flagged_inconsistent else "   no"
    t = f"{r.lz_statistic:.3f}" if not __import__('math').isnan(r.lz_statistic) else "  n/a"
    print(f"  {r.judge_id:<15}  {t:>8}  {flag}")
print()
print("  Interpretation:")
print("    Negative t → more uniform than the model predicts (too consistent).")
print("    'erratic' has the LEAST negative t = relatively worst fit vs consensus.")
print("    In practice: look for judges that stand out from the panel average t,")
print("    OR use |t| > 2.5 as a stricter misfit threshold for small panels.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. BehavioralAlignmentMetric — cross-condition consistency
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("4. BehavioralAlignmentMetric — DISC-style condition sensitivity")
print("=" * 60)
print()

from judge_kappa.agreement import BehavioralAlignmentMetric

# Consistent model: same scores under system-prompted and zero-shot conditions
consistent_scores = {
    "zero_shot":       {f"case-{i}": 0.70 + 0.02 * i for i in range(8)},
    "system_prompted": {f"case-{i}": 0.72 + 0.02 * i for i in range(8)},
    "chain_of_thought":{f"case-{i}": 0.71 + 0.02 * i for i in range(8)},
}

# Sensitive model: scores change substantially with condition
sensitive_scores = {
    "zero_shot":       {f"case-{i}": 0.40 + 0.04 * i for i in range(8)},
    "system_prompted": {f"case-{i}": 0.80 + 0.01 * i for i in range(8)},
    "chain_of_thought":{f"case-{i}": 0.30 + 0.05 * i for i in range(8)},
}

metric = BehavioralAlignmentMetric(bootstrap_ci=True, n_bootstrap=500)

v_consistent = BehavioralAlignmentMetric.from_condition_scores(consistent_scores)
r_consistent = metric.compute(v_consistent, ScaleType.INTERVAL)

v_sensitive = BehavioralAlignmentMetric.from_condition_scores(sensitive_scores)
r_sensitive = metric.compute(v_sensitive, ScaleType.INTERVAL)

print(f"  Consistent model:  α = {r_consistent.alpha:.4f}  → {r_consistent.alpha_interpretation}")
print(f"  Sensitive model:   α = {r_sensitive.alpha:.4f}  → {r_sensitive.alpha_interpretation}")
print()
print("  Low α here means prompt framing materially changes model outputs —")
print("  a risk signal for deployment in contexts with varied user prompting styles.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. DifferentialItemFunctioningDetector — per-case bias
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("5. DifferentialItemFunctioningDetector — case-level bias")
print("=" * 60)
print()

from judge_kappa.bias import DifferentialItemFunctioningDetector

# DIF design: 8 judges per family × 15 cases.
# Neutral cases (0-7): both families score near 0.55 — ambiguous quality,
#   mixed positive/negative verdicts with no clear family pattern.
# Biased cases (8-14): both families also score near 0.55 on average, BUT
#   claude judges score 0.65+ (positive verdicts) while gpt judges score 0.43-
#   (negative verdicts). After controlling for overall score level, the group
#   coefficient is significant → DIF flagged.

_rng_dif = _random.Random(13)
verdicts_dif = []
for i in range(15):
    for j_num in range(8):
        if i < 8:
            # Neutral: both around 0.50-0.60, no family preference
            score_claude = _rng_dif.uniform(0.42, 0.68)
            score_gpt    = _rng_dif.uniform(0.42, 0.68)
        else:
            # Biased: claude above threshold, gpt below — both near 0.55 globally
            score_claude = _rng_dif.uniform(0.62, 0.72)   # positive verdict
            score_gpt    = _rng_dif.uniform(0.38, 0.48)   # negative verdict
        verdicts_dif.append(_v(f"claude-{j_num}", f"case-{i}", score_claude))
        verdicts_dif.append(_v(f"gpt-{j_num}",    f"case-{i}", score_gpt))

detector = DifferentialItemFunctioningDetector(alpha_threshold=0.05, min_verdicts_per_case=4)
dif_report = detector.analyze(verdicts_dif)

print(f"  Judge families detected: {dif_report.n_groups}")
print(f"  Cases analyzed:          {len(dif_report.per_case)}")
print(f"  DIF-flagged cases:       {len(dif_report.flagged_cases)}  {dif_report.flagged_cases}")
print()
if dif_report.per_case:
    print(f"  {'Case':>10}  {'group β':>8}  {'p-value':>8}  {'Flagged':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in sorted(dif_report.per_case, key=lambda x: x.p_value):
        flag = "⚠  YES" if r.flagged else "   no"
        print(f"  {r.case_id:>10}  {r.group_coefficient:>8.3f}  {r.p_value:>8.3f}  {flag}")
print()
print("  DIF-flagged cases should be reviewed before reporting model comparisons.")
print("  High |β| with p < 0.05 → judge family preference, not true quality difference.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. McNemar + bootstrap CI — via JuryEvaluator with mock backend
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("6. McNemar significance test + bootstrap CI for mean uplift")
print("=" * 60)
print()

import json as _json

from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.panel.panel import JudgePanel


class _MockBackend:
    """Deterministic mock — no API key needed."""
    def __init__(self, responses: list[str]) -> None:
        self._r = responses
        self._i = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        r = self._r[self._i % len(self._r)]
        self._i += 1
        return r


def _resp(score: float) -> str:
    return _json.dumps({"assertion_scores": {"quality": score}, "rationale": "mock"})


# Build 15 cases where treatment beats control in 13 / 15 cases.
# McNemar continuity-corrected: (|13-2|-1)^2 / 15 = 6.67, p ≈ 0.010 (significant).
ctrl_scores = [0.30, 0.35, 0.40, 0.25, 0.45, 0.30, 0.35, 0.40, 0.30, 0.45, 0.38, 0.42, 0.36, 0.55, 0.52]
trt_scores  = [0.75, 0.80, 0.75, 0.78, 0.82, 0.70, 0.72, 0.76, 0.78, 0.71, 0.45, 0.80, 0.74, 0.42, 0.38]

responses = []
for cs, ts in zip(ctrl_scores, trt_scores):
    responses += [_resp(cs), _resp(ts)]

cases = [
    EvalCase(
        id=f"case-{i:02d}",
        prompt=f"Q{i}",
        assertions=[Assertion(text="quality")],
    )
    for i in range(15)
]

panel = JudgePanel(judges=[
    AssertionJudge("judge-a", _MockBackend(responses)),
])
ev = JuryEvaluator(
    panel=panel,
    generation_backend=_MockBackend(["output"] * 100),
    compute_significance=True,
    bootstrap_ci=True,
    n_bootstrap=1000,
    compute_judge_fit=False,   # only 1 judge → skip
    compute_icc=False,
)

ctrl_v  = Variant(name="control",   predict_fn=lambda _: "ctrl")
treat_v = Variant(name="treatment", predict_fn=lambda _: "trt")
report = ev.evaluate(cases, ctrl_v, treat_v)

sig = report.significance
print(f"  Mean uplift:        {report.mean_uplift:+.4f}")
print(f"  95% bootstrap CI:   [{sig.uplift_ci_low:+.4f}, {sig.uplift_ci_high:+.4f}]")
print(f"  McNemar χ²:         {sig.mcnemar_statistic:.4f}")
print(f"  p-value:            {sig.p_value:.4f}  → {'significant (p < 0.05)' if sig.significant else 'not significant'}")
print(f"  Treatment wins:     {sig.n_treatment_wins} / {len(cases)}")
print(f"  Control wins:       {sig.n_control_wins} / {len(cases)}")
print(f"  Ties:               {sig.n_ties} / {len(cases)}")
print()
print("  CI not crossing 0 + p < 0.05 → treat uplift as real, not noise.")
print("  Always validate judge α ≥ 0.67 *before* trusting this McNemar result.")
print()

# Agreement summary with CIs
print(f"  Agreement (α = {report.agreement.alpha:.3f})")
if report.agreement.alpha_ci_low is not None:
    print(f"    95% CI: [{report.agreement.alpha_ci_low:.3f}, {report.agreement.alpha_ci_high:.3f}]")

print()
print("─" * 60)
print("All 6 psychometric demos complete. No API key was used.")
print("─" * 60)
