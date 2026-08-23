# Psychometric Methods

judge-kappa v0.2 adds nine psychometrically grounded capabilities. This document explains each one, when to use it, and how to run the demos.

---

## Quick reference

| Capability | Key class / function | Demo |
|---|---|---|
| Bootstrap CI for α | `KrippendorffAlpha(bootstrap_ci=True)` | `demo_psychometric.py` §1 |
| ICC(2,k) variance decomposition | `compute_icc()` | `demo_psychometric.py` §2 |
| Person-fit (inconsistent judge detection) | `PersonFitAnalyzer` | `demo_psychometric.py` §3 |
| Behavioral alignment (DISC-style) | `BehavioralAlignmentMetric` | `demo_psychometric.py` §4 |
| Differential item functioning | `DifferentialItemFunctioningDetector` | `demo_psychometric.py` §5 |
| McNemar + bootstrap CI for uplift | `JuryEvaluator` (`compute_significance=True`) | `demo_psychometric.py` §6 |
| IRT-based judge weighting | `IRTJudgeWeighter` | `demo_irt_weighting.py` |
| Listwise ranking for N ≥ 7 | `RankJudge` | `demo_rank_judge.py` |
| Accurate verbosity tokens | `pip install judge-kappa[tiktoken]` | (automatic) |

Run all demos without an API key:

```bash
cd judge-kappa-eval
python examples/demo_psychometric.py    # sections 1–6: analytics metrics
python examples/demo_irt_weighting.py   # section 7: IRT judge weighting
python examples/demo_rank_judge.py      # section 8: RankJudge leaderboard
```

---

## 1 — Bootstrap CI for Krippendorff's α

### The problem

A point estimate of α = 0.79 sits just below the "strong agreement" threshold (0.80). Is the true α above or below 0.80? A 95% CI answers this.

### API

```python
from judge_kappa.agreement import KrippendorffAlpha

alpha_metric = KrippendorffAlpha(
    bootstrap_ci=True,    # default True
    n_bootstrap=2000,     # default; reduce to 500 for speed
    seed=42,
)
result = alpha_metric.compute(verdicts, ScaleType.ORDINAL)

print(result.alpha)              # point estimate
print(result.alpha_ci_low)       # 2.5th percentile
print(result.alpha_ci_high)      # 97.5th percentile
print(result.alpha_interpretation)
```

`AgreementResult` fields added: `alpha_ci_low`, `alpha_ci_high`.

### When to report CIs

| Situation | Guidance |
|---|---|
| Paper / compliance report | Always report CI alongside point estimate |
| N_cases < 20 | CI is wide — report it to signal uncertainty |
| CI overlaps 0.67 threshold | Do not claim "strong agreement" even if point estimate ≥ 0.80 |
| Quick iteration / dev loop | Set `bootstrap_ci=False` to skip CI and save ~1s |

Bootstrap resamples cases (columns of the reliability matrix), not judges, which is the correct unit of observation for Krippendorff.

---

## 2 — ICC(2,k) variance decomposition

### The problem

α = 0.55 on a 2-judge panel. Does this mean:
- (a) The rubric is genuinely ambiguous (judges disagree randomly), or
- (b) One judge is systematically harsh and the other lenient?

These require different fixes. ICC(2,k) separates them.

### API

```python
from judge_kappa.agreement.icc import compute_icc, enrich_agreement_with_icc

# Standalone
icc_val, icc_interp = compute_icc(verdicts)

# Or enrich an existing AgreementResult in-place
result = alpha_metric.compute(verdicts, ScaleType.ORDINAL)
result = enrich_agreement_with_icc(result, verdicts)
print(result.icc, result.icc_interpretation)
```

`AgreementResult` fields added: `icc`, `icc_interpretation`.

### Interpreting ICC vs. α discrepancy

| Pattern | Diagnosis | Fix |
|---|---|---|
| ICC ≈ α (both low) | Rubric is ambiguous — random disagreement | Add ICL calibration anchors; tighten rubric wording |
| ICC >> α | Systematic leniency/harshness offset | Calibrate judge temperature; add shared anchor examples |
| ICC < α | Unusual; check for outlier cases | Inspect per-case verdicts; run DIF analysis |

ICC thresholds (Cicchetti 1994): ≥ 0.75 = excellent, 0.60–0.74 = good, 0.40–0.59 = fair, < 0.40 = poor.

ICC is computed automatically by `JuryEvaluator` when `compute_icc=True` (the default).

---

## 3 — Person-fit: inconsistent judge detection

### The problem

Aggregate α = 0.77 looks acceptable. But one judge is scoring erratically — reliable on easy cases but near-random on hard ones. The aggregate α masks this.

### API

```python
from judge_kappa.agreement import PersonFitAnalyzer

analyzer = PersonFitAnalyzer(alpha_threshold=1.96)  # |t| > 1.96 → p < 0.05
fit_results = analyzer.analyze(verdicts)

for r in fit_results:
    print(f"{r.judge_id}: t={r.lz_statistic:.3f}  flagged={r.flagged_inconsistent}")
```

`JudgeFitResult` fields: `judge_id`, `lz_statistic` (outfit MNSQ t-statistic), `flagged_inconsistent`.

`JuryEvaluator` populates `EvalReport.judge_fit` automatically when `compute_judge_fit=True` (default).

### Statistic details

Uses the outfit mean-square (MNSQ) t-transformation (Wright & Masters 1982):

- **MNSQ > 1**: judge is more erratic than the model predicts — score inconsistency
- **MNSQ < 1**: judge is suspiciously uniform — possible straight-line rating bias
- **t-statistic**: standardised version; |t| > 1.96 is the conventional misfit flag

This is a model-free approximation of the IRT lz statistic — no IRT model needs to be fitted separately.

### Action on flagged judges

1. Inspect the flagged judge's rationales on hard cases
2. Add more domain-specific ICL calibration examples
3. If the judge remains erratic after recalibration, exclude or down-weight it

---

## 4 — BehavioralAlignmentMetric (DISC-style)

### The problem

You want to know whether a model's output quality is consistent across different prompt framings (zero-shot vs. chain-of-thought vs. system-prompted). This is the cross-condition analogue of inter-rater reliability.

### API

```python
from judge_kappa.agreement import BehavioralAlignmentMetric

metric = BehavioralAlignmentMetric(bootstrap_ci=True, n_bootstrap=2000)

# Option A: from_condition_scores helper
condition_scores = {
    "zero_shot":        {"case-0": 0.72, "case-1": 0.68, ...},
    "system_prompted":  {"case-0": 0.74, "case-1": 0.70, ...},
    "chain_of_thought": {"case-0": 0.73, "case-1": 0.69, ...},
}
verdicts = BehavioralAlignmentMetric.from_condition_scores(condition_scores)
result = metric.compute(verdicts, ScaleType.INTERVAL)

# Option B: use JudgeVerdict directly — set judge_id = condition name
result = metric.compute(my_condition_verdicts, ScaleType.INTERVAL)

print(result.alpha)                  # behavioral consistency α
print(result.alpha_interpretation)   # condition-framed interpretation
```

### Interpretation

| α | Meaning | Action |
|---|---|---|
| ≥ 0.80 | Behaviorally consistent — framing doesn't matter | Safe to use any prompt format |
| 0.67–0.80 | Modest sensitivity | Document which format was used; consider prompt standardisation |
| < 0.67 | High condition sensitivity | Model outputs depend heavily on framing; standardise prompts before benchmarking |

**Connection to DISC paper**: this reuses Krippendorff α with conditions as "raters" — the same math, different semantic frame. Low α here does *not* mean judges disagree; it means the model disagrees with itself across conditions.

---

## 5 — Differential Item Functioning (DIF)

### The problem

Eval case #7 shows treatment winning 90% of the time with GPT judges but only 40% with Claude judges. Is treatment genuinely better, or is this a DIF artefact — case #7 happens to favour GPT judge style?

### API

```python
from judge_kappa.bias import DifferentialItemFunctioningDetector

detector = DifferentialItemFunctioningDetector(
    alpha_threshold=0.05,          # Wald test p-value threshold
    positive_threshold=0.5,        # score ≥ 0.5 = "positive verdict"
    min_verdicts_per_case=4,       # skip cases with fewer verdicts
    min_groups=2,                  # require ≥ 2 model families
)

report = detector.analyze(verdicts)

print(report.summary())            # quick summary string
print(report.flagged_cases)        # list of DIF case IDs
for r in report.per_case:
    print(f"{r.case_id}: β={r.group_coefficient:.3f}  p={r.p_value:.3f}  flagged={r.flagged}")
```

`DIFReport` fields: `flagged_cases`, `per_case` (list of `DIFCaseResult`), `n_groups`, `alpha_threshold`.

### How it works

For each case, a logistic regression predicts verdict (positive/negative) from:
1. **Mean score level** (controls for overall difficulty)
2. **Judge model family** (binary group indicator)

If the group coefficient is statistically significant (Wald p < α_threshold), the case has DIF — verdicts differ across model families beyond what the score level explains.

### Judge model family extraction

Judge family is extracted from the `judge_id` prefix before the first `-`, `_`, or `:`:
- `claude-3-opus-j1` → `claude`
- `gpt-4o-j2` → `gpt`
- `llama-3-j1` → `llama`

Use consistent judge ID naming across your panel.

### Action on DIF cases

- **Review** DIF-flagged cases for systematic formatting or style preferences
- **Stratify** reporting: report scores separately for flagged and clean cases
- **Replace** persistently DIF-flagged cases with reformulated versions

---

## 6 — McNemar test + bootstrap CI for mean uplift

### The problem

`mean_uplift = +0.042` — but is this real or noise? Without a significance test, you cannot distinguish a meaningful treatment effect from judge stochasticity.

### API

McNemar and bootstrap CI are computed automatically by `JuryEvaluator`:

```python
ev = JuryEvaluator(
    panel=panel,
    generation_backend=backend,
    compute_significance=True,   # default True
    bootstrap_ci=True,           # default True
    n_bootstrap=2000,
)
report = ev.evaluate(cases, control, treatment)

sig = report.significance
print(f"Mean uplift:  {report.mean_uplift:+.4f}")
print(f"95% CI:       [{sig.uplift_ci_low:+.4f}, {sig.uplift_ci_high:+.4f}]")
print(f"McNemar χ²:   {sig.mcnemar_statistic:.4f}")
print(f"p-value:      {sig.p_value:.4f}  significant={sig.significant}")
print(f"Treatment wins: {sig.n_treatment_wins} / {len(report.cases)}")
```

`UpliftSignificance` fields: `n_treatment_wins`, `n_control_wins`, `n_ties`, `mcnemar_statistic`, `p_value`, `significant`, `uplift_ci_low`, `uplift_ci_high`.

### Decision table

| Result | Conclusion |
|---|---|
| CI does not cross 0 AND p < 0.05 | Treatment effect is real — report the uplift |
| CI crosses 0 OR p ≥ 0.05 | Cannot distinguish treatment from noise — run more cases |
| p < 0.05 but CI is very narrow (e.g., [0.001, 0.008]) | Statistically significant but practically negligible |

### Two-step validation protocol

```
Step 1: Check judge reliability FIRST
  → EvalReport.agreement.alpha ≥ 0.67

Step 2: Only then interpret McNemar
  → EvalReport.significance.significant
```

Skipping step 1 means McNemar might be measuring *judge noise*, not model quality differences.

### Cost

Bootstrap CI adds ~1s for 2000 samples on 50 cases. Disable with `bootstrap_ci=False` or reduce `n_bootstrap=500` for faster iteration.

---

## 7 — IRT-based judge weighting

### The problem

Panel weights in `JudgePanel(weights=[1.0, 1.0, 1.0])` are arbitrary. `IRTJudgeWeighter` derives weights from how reliably each judge agrees with human-validated scores.

### Setup

**Step 1**: Collect calibration data — 10–30 (prompt, output) pairs with human-expert scores.

**Step 2**: Run each judge on the same pairs and record their scores.

**Step 3**: Fit the model.

```python
from judge_kappa.calibration import IRTJudgeWeighter

human_scores = [0.95, 0.10, 0.65, 0.50, 0.80, ...]   # ground truth

judge_scores = {
    "claude-j":  [0.92, 0.14, 0.63, 0.52, 0.78, ...],
    "gpt4-j":    [0.89, 0.18, 0.60, 0.55, 0.75, ...],
    "llama-j":   [0.75, 0.40, 0.58, 0.60, 0.65, ...],  # noisier judge
}

weighter = IRTJudgeWeighter(tolerance=0.20)
weighter.fit(judge_scores, human_scores)

theta   = weighter.theta()    # per-judge latent reliability
weights = weighter.weights()  # softmax(θ) → panel weights summing to 1.0
items   = weighter.item_parameters()  # per-item a (discrimination), b (difficulty)
```

**Step 4**: Pass weights to `JudgePanel`:

```python
from judge_kappa import JudgePanel, AggregationStrategy

panel = JudgePanel(
    judges=[claude_judge, gpt4_judge, llama_judge],
    strategy=AggregationStrategy.WEIGHTED_MEAN,
    weights=[weights["claude-j"], weights["gpt4-j"], weights["llama-j"]],
)
```

### Model details

The 2PL (2-Parameter Logistic) IRT model:

```
P(judge i scores item j correctly | θ_i, a_j, b_j)
    = 1 / (1 + exp(-a_j × (θ_i − b_j)))

θ_i = judge reliability (higher → more consistent with humans)
a_j = item discrimination (higher → sharper quality signal)
b_j = item difficulty (higher → harder for any judge)
```

A response is "correct" when `|judge_score − human_score| ≤ tolerance` (default 0.20).

Regularisation priors (Fonseca Rivera et al. 2026):
- Log-normal prior on a_j (σ=1.0): prevents discrimination estimates from collapsing
- Normal prior on b_j (σ=2.0): prevents difficulty from diverging

**Minimum requirements**: ≥ 5 calibration examples, ≥ 2 judges.

### Run the demo

```bash
python examples/demo_irt_weighting.py
```

No API key required. Uses synthetic calibration data with 5 judges and 12 items.

---

## 8 — RankJudge (listwise, N ≥ 7 systems)

### The problem

Comparing 8 systems with `TournamentEvaluator` requires C(8,2) × cases × 2 = 280 judge calls on 10 cases. `RankJudge` reduces this to 10 calls (one per case) by presenting all outputs simultaneously.

### Cost table

| N systems | TournamentEvaluator calls | RankJudge calls | Reduction |
|---|---|---|---|
| 6 | 150 | 10 | 93% |
| 8 | 280 | 10 | 96% |
| 10 | 450 | 10 | 98% |
| 12 | 660 | 10 | 98% |

(Assumes 10 cases × 2 orderings for positional bias detection.)

### API

```python
from judge_kappa.judges import RankJudge
from judge_kappa import AnthropicBackend

judge = RankJudge(
    "rank-j1",
    AnthropicBackend("claude-sonnet-4-6"),
    max_systems=12,    # hard cap; raise only if context window allows
)

# Score one case across all systems
system_outputs = {
    "system-a": "Output from system A...",
    "system-b": "Output from system B...",
    # ...up to max_systems
}
verdicts = judge.rank(case, system_outputs)
# verdicts: dict[system_name → JudgeVerdict]
# scores: rank 1 → 1.0, rank N → 0.0

# Aggregate across cases to build a leaderboard
scores_by_system: dict[str, list[float]] = defaultdict(list)
for case in eval_cases:
    outputs = {sys: generate(sys, case) for sys in systems}
    for sys, verdict in judge.rank(case, outputs).items():
        scores_by_system[sys].append(verdict.score)

leaderboard = sorted(
    [(sys, sum(s)/len(s)) for sys, s in scores_by_system.items()],
    key=lambda x: -x[1],
)
```

### Trade-offs vs. TournamentEvaluator

| Aspect | RankJudge | TournamentEvaluator |
|---|---|---|
| Judge calls | O(N) per case | O(N²) per case |
| Positional bias detection | None (no swap test) | Per-pair positional bias rate |
| Context window pressure | Grows with N × output_length | Fixed (2 outputs per call) |
| Accuracy | Slightly lower (1 judge, full context) | Higher (multiple pairwise verdicts) |
| Recommended N | ≥ 7 | ≤ 6 |

### Run the demo

```bash
python examples/demo_rank_judge.py
```

No API key required. Uses `MockRankBackend` with realistic noise.

---

## 9 — Accurate token counts for verbosity bias

### Why it matters

`VerbosityBiasDetector` measures whether judges reward longer outputs (Spearman ρ). The correlation depends on accurate token counts. The default whitespace-split approximation inflates counts for CJK text and deflates for code.

### Setup

```bash
pip install "judge-kappa[tiktoken]"
# or
pip install "judge-kappa[all]"   # includes tiktoken
```

Once installed, token counts are used automatically when judges set `output_token_count` in their verdicts. The `tiktoken_encoding` parameter selects the tokenizer:

```python
from judge_kappa.bias import VerbosityBiasDetector

detector = VerbosityBiasDetector(
    threshold=0.30,
    tiktoken_encoding="cl100k_base",   # GPT-4, Claude-3; use "o200k_base" for GPT-4o
)
result = detector.detect(verdicts=verdicts)
print(result.verbosity_bias_rho, result.verbosity_bias_p, result.verbosity_biased)
```

For judges that generate output, set `output_token_count` in `JudgeVerdict` using a proper tokenizer in your custom judge subclass.

---

## Complete evaluation pipeline with all psychometric features

```python
from judge_kappa import (
    AnthropicBackend, AssertionJudge, JudgePanel, JuryEvaluator,
    PairwiseJudge, ScaleType, AggregationStrategy,
)
from judge_kappa.calibration import IRTJudgeWeighter
from judge_kappa.bias import DifferentialItemFunctioningDetector

# Step 1: Fit IRT weights from calibration data
weighter = IRTJudgeWeighter().fit(judge_scores, human_scores)
weights = weighter.weights()

# Step 2: Build panel with IRT-derived weights
backend = AnthropicBackend("claude-sonnet-4-6")
panel = JudgePanel(
    judges=[AssertionJudge(f"j{i}", backend) for i in range(3)],
    strategy=AggregationStrategy.WEIGHTED_MEAN,
    weights=[weights[f"j{i}"] for i in range(3)],
)

# Step 3: Evaluate with all psychometric features enabled
ev = JuryEvaluator(
    panel=panel,
    generation_backend=backend,
    scale_type=ScaleType.ORDINAL,
    compute_significance=True,    # McNemar + bootstrap CI
    compute_icc=True,             # ICC(2,k)
    compute_judge_fit=True,       # person-fit per judge
    bootstrap_ci=True,            # CI for α
    n_bootstrap=2000,
)
report = ev.evaluate(cases, control, treatment)

# Step 4: Inspect results
print(f"Uplift: {report.mean_uplift:+.4f}  p={report.significance.p_value:.4f}")
print(f"α: {report.agreement.alpha:.4f}  [{report.agreement.alpha_ci_low:.4f}, {report.agreement.alpha_ci_high:.4f}]")
print(f"ICC(2,k): {report.agreement.icc:.4f}  → {report.agreement.icc_interpretation}")
for jf in report.judge_fit:
    if jf.flagged_inconsistent:
        print(f"⚠ Judge {jf.judge_id} flagged: t={jf.lz_statistic:.3f}")

# Step 5: DIF analysis on the collected verdicts
all_verdicts = [v for c in report.cases for v in c.verdicts]
dif = DifferentialItemFunctioningDetector().analyze(all_verdicts)
if dif.flagged_cases:
    print(f"⚠ DIF-flagged cases: {dif.flagged_cases}")
```
