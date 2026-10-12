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
| Order-aware BTL ranking + display-order effect | `fit_btl`, `bootstrap_btl`, `TournamentEvaluator`, `PositionalBiasDetector.estimate_order_effect` | `demo_order_aware_ranking.py` |
| Output length for verbosity bias | `VerbosityBiasDetector` (reads `JudgeVerdict.output_token_count`) | none (see §9) |

Run all demos without an API key:

```bash
cd judge-kappa-eval
python examples/demo_psychometric.py    # sections 1–6: analytics metrics
python examples/demo_irt_weighting.py   # section 7: IRT judge weighting
python examples/demo_rank_judge.py      # section 8: RankJudge leaderboard
python examples/demo_order_aware_ranking.py  # section 10: order-aware BTL ranking
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

### When agreement cannot be estimated

`agreement.alpha` and `agreement.kappa` are `None` (never `NaN`, and never a made-up `1.0`) when they are undefined, and a reason field says why:

| Field | Reason | Meaning |
|---|---|---|
| `alpha_not_estimated_reason` | `single_judge` | Fewer than 2 judges scored anything. Alpha compares values that different judges gave to the same case |
| | `no_pairable_values` | 2 or more judges, but no case was scored by 2 or more of them (disjoint coverage) |
| `kappa_not_estimated_reason` | `no_judge_pair` | No two judges share at least 2 cases |
| | `no_score_variation` | A compared pair used one and the same single score, so observed and expected agreement are both 1 and kappa is 0/0. `expected_chance_agreement` is still reported |

Two conventions are worth knowing:

- **Identical scores from 2 or more judges on shared cases give `alpha = 1.0`.** Krippendorff's formula is 0/0 there; the library treats it as perfect agreement, as before. Kappa is *not* given the same treatment, because "every judge said 1.0 on everything" shows no discrimination at all. Read an `alpha` of exactly `1.0` together with the score spread.
- **`kappa` is the mean over judge pairs.** If any compared pair is undefined the mean is undefined, so `kappa` is `None` even when other pairs are fine.

Earlier releases reported `alpha = 1.0` ("strong agreement") for a single judge and for judges with disjoint cases, and `kappa = NaN` in Python (`null` in JSON) for identical single scores. Code that formats these values (`f"{report.agreement.alpha:.3f}"`) now has to handle `None`. The CLI text report prints `n/a (<reason>)`.

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

## 9 — Output length for verbosity bias

### Why it matters

`VerbosityBiasDetector` measures whether judges reward longer outputs (Spearman ρ between output length and judge score). The result is only as meaningful as the length measure, so it is worth knowing exactly what is counted.

### What is counted

Length is `JudgeVerdict.output_token_count` of the **evaluated output** (not the judge's rationale). The built-in judges set it to the **whitespace-split word count** of that output. The unit is words, not model tokens, and no tokenizer is involved.

Spearman is rank-based, so word counts and token counts rank ordinary same-language prose almost identically, and the correlation is usually insensitive to the choice. Whitespace words are a weak proxy where it matters:

| Corpus | Effect of whitespace splitting |
|---|---|
| CJK, Thai and other scripts written without spaces | A whole sentence counts as a few "words": lengths are compressed and ties are common, so ρ is unreliable |
| Source code | Counts depend on formatting more than on content |
| Mixed-language corpora | Lengths are not comparable across languages |

For those corpora set `output_token_count` yourself, with the length measure you trust, in a custom judge or when building verdicts:

```python
import tiktoken  # your own dependency, not installed by judge-kappa

enc = tiktoken.get_encoding("o200k_base")
verdict.output_token_count = len(enc.encode(output))
```

### Verdicts without a length

`output_token_count == 0` means "not measured". Such verdicts are left out of the correlation, and the report says so:

```python
report.bias.verbosity_n_measured    # verdicts used
report.bias.verbosity_n_unmeasured  # verdicts left out
report.data_quality.warnings        # explains it when n_unmeasured > 0
```

An empty output has length 0 and is therefore also reported as unmeasured. Earlier releases silently used the word count of the judge's *rationale* for such verdicts, which measures the judge's own wordiness rather than the output's length.

### When the correlation cannot be estimated

A rank correlation is undefined when one of the two series does not vary, and is not reported on fewer than 3 points. In those cases `verbosity_bias_rho` and `verbosity_bias_p` are `None` (never `NaN`, and never `0.0`, which would assert "no association"), and `verbosity_not_estimated_reason` says why:

| `verbosity_not_estimated_reason` | Meaning | Typical cause |
|---|---|---|
| `too_few_measured` | Fewer than 3 verdicts with a measured length | Tiny runs; outputs with no recorded length |
| `constant_scores` | Every measured verdict has the same score | A judge at the ceiling on a pass/fail rubric |
| `constant_lengths` | Every measured output has the same length | Single-token or fixed-format answers |

`verbosity_biased` is `False` in every one of these cases, so **check `verbosity_bias_rho is None` before reading `False` as "no bias detected"**. If both series are constant, `constant_scores` is reported. The CLI text report prints `not estimated (<reason>)`. Only missing length data adds a `data_quality.warnings` entry; identical scores or lengths are a property of the data and a clean run stays clean.

### Deprecated: `tiktoken_encoding` and the `tiktoken` extra

`VerbosityBiasDetector(tiktoken_encoding=...)` was accepted but never used. It now emits a `DeprecationWarning` and is ignored. The `tiktoken` extra installs nothing and is no longer part of `[all]`; it remains only so existing `pip install "judge-kappa[tiktoken]"` commands keep working.

---

## 10 — Order-aware BTL ranking and the display-order effect

### Why it matters

A pairwise judge may prefer whichever response is shown first. The flip rate detects this but does not correct rankings, and it conflates position bias with a judge that is simply right in both orders. An order-aware Bradley-Terry model estimates the order effect *jointly* with item quality, so rankings are adjusted and the effect is reported with an interval.

### Model

For items i, j shown in order a (+1 = i first, -1 = j first):

```
logit P(i preferred over j) = s_i - s_j + a * b
```

`s` are centred latent scores (log-odds) and `b` is the judge's order effect (`b > 0` favours the first-shown response). `b` is fitted only when both display orders were observed; otherwise it is fixed at zero and reported as not identifiable. Scores are estimated by ridge-penalised maximum likelihood (`l2=1e-4`) so the fit stays finite under complete separation; `possibly_separated` flags such cases. Cases are the resampling unit for bootstrap intervals, which keeps pairs judged on the same prompt together.

### Usage

```python
from judge_kappa import ComparisonGame, fit_btl, bootstrap_btl

games = [
    # case id, item i, item j, order (+1: i shown first), outcome (1 i wins, 0 j wins, 0.5 tie)
    ComparisonGame(0, 0, 1, +1, 1.0),
    ComparisonGame(0, 0, 1, -1, 0.0),
    ...
]
fit = fit_btl(n_items=2, games=games)
print(fit.scores, fit.order_effect, fit.order_identifiable, fit.possibly_separated)
boot = bootstrap_btl(2, games, n_boot=500, seed=0)
print(boot.scores_low, boot.scores_high, boot.order_effect_low, boot.order_effect_high)
```

`TournamentEvaluator` and `PositionalBiasDetector` build the games for you and expose the results as `TournamentReport.order_effect` and `BiasResult.order_effect`.

### Interpretation and limits

- Outcomes are ordinal (win/tie/loss per game); score margins are not used by the BTL fit.
- A single constant order effect per judge is assumed. Effects that vary by pair or case are not modelled.
- Single judge only. Multi-judge structure (shared consensus plus disagreement) and calibration to human labels are described in the DIAL paper but not implemented here.
- Standard errors on `BTLFit` are Wald approximations; use `bootstrap_btl` for reporting.

### Validation

`tests/test_ranking_btl.py` checks a hand-derived two-item closed form, an independent scikit-learn logistic-regression oracle, truth recovery on simulated data, label-swap invariance, separation handling, and bootstrap behaviour.

### Demo

```bash
python examples/demo_order_aware_ranking.py
```

No API key required.

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
