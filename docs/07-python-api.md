# Python API

All public symbols are importable directly from `judge_kappa`.

---

## Core objects

```python
from judge_kappa import (
    # Evaluators
    JuryEvaluator,
    TournamentEvaluator,

    # Backends (LLM providers)
    AnthropicBackend,
    OpenAIBackend,

    # Judges
    AssertionJudge,
    RubricJudge,
    PairwiseJudge,
    RankJudge,                  # v0.2: listwise ranking for N ≥ 7 systems

    # Panels
    JudgePanel,    # homogeneous rubric — measures inter-rater agreement
    JudgeJury,     # diverse rubrics — aggregates diverse perspectives

    # Models (input)
    Variant,
    EvalCase,
    Assertion,
    RubricDimension,
    CalibrationExample,
    ScaleType,
    AggregationStrategy,

    # Models (output)
    EvalReport,
    PairwiseReport,
    AgreementResult,    # v0.2: + alpha_ci_low/high, icc, icc_interpretation
    BiasResult,
    CaseResult,
    VariantResult,
    JudgeVerdict,
    PairwiseCaseResult,
    UpliftSignificance,     # v0.2: McNemar + bootstrap CI for mean_uplift
    JudgeFitResult,         # v0.2: per-judge outfit MNSQ t-statistic

    # Agreement metrics
    KrippendorffAlpha,          # v0.2: + bootstrap_ci parameter
    CohenKappa,
    PersonFitAnalyzer,          # v0.2: judge consistency detection
    BehavioralAlignmentMetric,  # v0.2: cross-condition consistency (DISC-style)

    # Bias detectors
    PositionalBiasDetector,
    VerbosityBiasDetector,
    DifferentialItemFunctioningDetector,  # v0.2: per-case DIF analysis

    # Calibration
    IRTJudgeWeighter,   # v0.2: 2PL IRT-based panel weights from calibration data

    # Adapters
    SkillAdapter,
    DatasetAdapter,
)
```

---

## Pattern 1 — Skill evaluation (assertion-based A/B)

```python
from judge_kappa import (
    JuryEvaluator, JudgePanel, AssertionJudge,
    AnthropicBackend, CalibrationExample, Variant, ScaleType,
)

calibration = [
    CalibrationExample(
        prompt="Summarize the monthly revenue.",
        output="Revenue was high.",
        score=0.2,
        rationale="Too vague — omits specific figures.",
    ),
    CalibrationExample(
        prompt="Summarize the monthly revenue.",
        output="March: $5.0M (highest). Jan: $4.2M. Feb: $3.1M.",
        score=0.95,
        rationale="Names all months with exact figures.",
    ),
]

backend = AnthropicBackend("claude-sonnet-4-6")

panel = JudgePanel(
    judges=[
        AssertionJudge("judge-claude", backend, calibration_examples=calibration),
        AssertionJudge("judge-haiku",  AnthropicBackend("claude-haiku-4-5-20251001"),
                       calibration_examples=calibration),
    ],
)

evaluator = JuryEvaluator(
    panel=panel,
    generation_backend=backend,
    scale_type=ScaleType.ORDINAL,
)

report = evaluator.evaluate_skill(
    skill_dir="./my-skill",
    control_variant=Variant(name="control"),
    treatment_variant=Variant(name="treatment"),  # SKILL.md injected automatically
)

print(f"Uplift: {report.mean_uplift:+.3f}")
print(f"α:      {report.agreement.alpha:.3f}  — {report.agreement.alpha_interpretation}")
print(f"κ:      {report.agreement.kappa:.3f}")
```

---

## Pattern 2 — Dataset evaluation (rubric-based, MLflow-compatible)

```python
from judge_kappa import (
    JuryEvaluator, JudgeJury, RubricJudge, RubricDimension,
    AnthropicBackend, OpenAIBackend, AggregationStrategy,
)

rubric = [
    RubricDimension(name="faithfulness",     description="Claims grounded in context.", weight=2.0),
    RubricDimension(name="answer_relevance", description="Directly addresses the question.", weight=1.5),
    RubricDimension(name="no_hallucination", description="No fabricated facts.", weight=3.0),
]

jury = JudgeJury(
    jurors=[
        (RubricJudge("safety",  AnthropicBackend("claude-sonnet-4-6"), rubric=rubric), 3.0),
        (RubricJudge("quality", OpenAIBackend("gpt-4o"),               rubric=rubric), 1.0),
    ],
    strategy=AggregationStrategy.WEIGHTED_MEAN,
)

def my_rag_pipeline(inputs: dict) -> str:
    return call_your_rag_system(inputs["question"], inputs["context"])

data = [
    {
        "id": "rag-001",
        "inputs":       {"question": "What is RAG?", "context": "RAG stands for..."},
        "expectations": {"answer": "Retrieval-Augmented Generation..."},
    },
]

evaluator = JuryEvaluator(panel=jury, generation_backend=AnthropicBackend("claude-sonnet-4-6"))
report = evaluator.evaluate_dataset(data=data, predict_fn=my_rag_pipeline)

# Serialize — compatible with MLflow artifact logging
import json, mlflow
mlflow.log_dict(json.loads(report.model_dump_json()), "judge_kappa_report.json")
```

---

## Pattern 3 — Endpoint comparison

```python
from judge_kappa import OpenAIBackend, Variant

ctrl = Variant(name="gpt-4o-mini", generation_backend=OpenAIBackend("gpt-4o-mini"))
trt  = Variant(name="gpt-4o",      generation_backend=OpenAIBackend("gpt-4o"))

# Compare a local fine-tuned model vs the hosted base:
# ctrl = Variant(name="base",         generation_backend=OpenAIBackend("meta-llama/...", base_url="http://localhost:8000/v1", api_key="no-key"))
# trt  = Variant(name="fine-tuned",   generation_backend=OpenAIBackend("my-org/...", base_url="http://localhost:8001/v1", api_key="no-key"))

report = evaluator.evaluate_endpoints(cases, ctrl, trt)
```

---

## Pattern 4 — Pre-recorded outputs

```python
data = [
    {
        "id": "q1",
        "inputs": {"question": "What is RAG?"},
        "output_control":   "RAG stands for Retrieval-Augmented Generation...",
        "output_treatment": "RAG combines retrieval with generation to reduce hallucination...",
    },
]

report = evaluator.evaluate_prerecorded(
    data,
    control_output_field="output_control",
    treatment_output_field="output_treatment",
)
```

---

## Pattern 5 — Pairwise preference

```python
from judge_kappa import PairwiseJudge, AnthropicBackend

pairwise_judge = PairwiseJudge("pairwise", AnthropicBackend("claude-sonnet-4-6"))

report = evaluator.evaluate_pairwise_dataset(
    data=pairwise_data,
    pairwise_judge=pairwise_judge,
    output_a_field="output_a",
    output_b_field="output_b",
    label_a="System A",
    label_b="System B",
)

print(f"System A preferred: {report.preference_rate_a:.1%}")
print(f"System B preferred: {report.preference_rate_b:.1%}")
print(f"Ties:               {report.tie_rate:.1%}")
```

---

## Pattern 6 — Tournament (N systems)

```python
from judge_kappa import TournamentEvaluator, PairwiseJudge, AnthropicBackend, Variant, OpenAIBackend

tournament = TournamentEvaluator(
    pairwise_judge=PairwiseJudge("judge", AnthropicBackend("claude-sonnet-4-6")),
    detect_positional_bias=True,
)

# Option A: callables
report = tournament.run_dataset(
    data=my_eval_data,
    systems={
        "rag-v1": lambda inputs: rag_v1(inputs["question"]),
        "rag-v2": lambda inputs: rag_v2(inputs["question"]),
        "rag-v3": lambda inputs: rag_v3(inputs["question"]),
    },
)

# Option B: live endpoints
report = tournament.run_endpoints(
    cases=eval_cases,
    variants=[
        Variant(name="gpt-4o-mini",  generation_backend=OpenAIBackend("gpt-4o-mini")),
        Variant(name="gpt-4o",       generation_backend=OpenAIBackend("gpt-4o")),
        Variant(name="claude-haiku", generation_backend=AnthropicBackend("claude-haiku-4-5-20251001")),
    ],
)

report.print_leaderboard()
print(report.win_matrix["gpt-4o"]["gpt-4o-mini"])  # win rate of gpt-4o over gpt-4o-mini
```

---

## Pattern 7 — Adding bias detection

```python
from judge_kappa import PairwiseJudge

pairwise = PairwiseJudge("bias-detector", AnthropicBackend("claude-sonnet-4-6"))

evaluator = JuryEvaluator(
    panel=panel,
    generation_backend=backend,
    positional_judge=pairwise,            # enables A/B swap test per case
    verbosity_bias_threshold=0.25,        # stricter than default 0.30
)

report = evaluator.evaluate_skill("./my-skill")

if report.bias.positional_bias_rate > 0.15:
    print(f"WARNING: positional bias {report.bias.positional_bias_rate:.1%} exceeds 15%")
if report.bias.verbosity_biased:
    print(f"WARNING: verbosity bias detected (ρ={report.bias.verbosity_bias_rho:.3f}, p={report.bias.verbosity_bias_p:.3f})")
```

---

## Pattern 8 — Local models (vLLM / Ollama)

```python
from judge_kappa import OpenAIBackend, AssertionJudge

# vLLM — no auth
vllm_backend = OpenAIBackend(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="no-key",
)

# Ollama — no auth
ollama_backend = OpenAIBackend(
    model="mistral",
    base_url="http://localhost:11434/v1",
    api_key="no-key",
)

# OpenRouter
openrouter_backend = OpenAIBackend(
    model="openai/gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

---

## Pattern 9 — McNemar significance + bootstrap CI (v0.2)

```python
ev = JuryEvaluator(
    panel=panel,
    generation_backend=backend,
    compute_significance=True,   # default True
    bootstrap_ci=True,           # CI for α and uplift; default True
    n_bootstrap=2000,
)
report = ev.evaluate(cases, control, treatment)

sig = report.significance
print(f"Uplift:   {report.mean_uplift:+.4f}")
print(f"95% CI:   [{sig.uplift_ci_low:+.4f}, {sig.uplift_ci_high:+.4f}]")
print(f"McNemar:  χ²={sig.mcnemar_statistic:.3f}  p={sig.p_value:.4f}  {'✓ significant' if sig.significant else '✗ not significant'}")
print(f"Wins:     treatment={sig.n_treatment_wins}  control={sig.n_control_wins}  ties={sig.n_ties}")
```

---

## Pattern 10 — IRT-based judge weighting (v0.2)

```python
from judge_kappa import IRTJudgeWeighter, JudgePanel, AggregationStrategy

# Step 1: collect human-validated calibration scores
human_scores = [0.95, 0.10, 0.65, 0.50, 0.80, 0.20, 0.70, 0.45]
judge_responses = {
    "claude-j": [0.93, 0.12, 0.63, 0.52, 0.78, 0.22, 0.68, 0.47],
    "gpt4-j":   [0.89, 0.18, 0.60, 0.55, 0.75, 0.30, 0.65, 0.52],
    "llama-j":  [0.75, 0.40, 0.58, 0.60, 0.65, 0.45, 0.55, 0.60],
}

# Step 2: fit 2PL IRT model
weighter = IRTJudgeWeighter(tolerance=0.20)
weighter.fit(judge_responses, human_scores)

weights = weighter.weights()    # softmax(θ) — sums to 1.0
theta   = weighter.theta()      # latent reliability per judge

# Step 3: pass weights to JudgePanel
panel = JudgePanel(
    judges=[claude_judge, gpt4_judge, llama_judge],
    strategy=AggregationStrategy.WEIGHTED_MEAN,
    weights=[weights["claude-j"], weights["gpt4-j"], weights["llama-j"]],
)
```

---

## Pattern 11 — RankJudge for N ≥ 7 systems (v0.2)

```python
from judge_kappa import RankJudge, AnthropicBackend
from collections import defaultdict

judge = RankJudge("rank-j", AnthropicBackend("claude-sonnet-4-6"), max_systems=12)

scores: dict[str, list[float]] = defaultdict(list)
for case in eval_cases:
    outputs = {sys: generate(sys, case) for sys in systems}
    for sys, verdict in judge.rank(case, outputs).items():
        scores[sys].append(verdict.score)    # rank 1 → 1.0, rank N → 0.0

leaderboard = sorted(scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
for rank, (sys, s) in enumerate(leaderboard, 1):
    print(f"{rank}. {sys}  mean={sum(s)/len(s):.4f}")
```

---

## Pattern 12 — DIF analysis (v0.2)

```python
from judge_kappa import DifferentialItemFunctioningDetector

# After running evaluate(), collect all verdicts
all_verdicts = [v for c in report.cases for v in c.verdicts]

detector = DifferentialItemFunctioningDetector(
    alpha_threshold=0.05,
    min_verdicts_per_case=4,
    min_groups=2,
)
dif_report = detector.analyze(all_verdicts)

print(dif_report.summary())
for r in dif_report.per_case:
    if r.flagged:
        print(f"  DIF case: {r.case_id}  β={r.group_coefficient:.3f}  p={r.p_value:.4f}")
```

---

## Pattern 13 — Behavioral alignment metric (v0.2)

```python
from judge_kappa import BehavioralAlignmentMetric
from judge_kappa.models import ScaleType

# Score the same cases under different prompting conditions
condition_scores = {
    "zero_shot":        {f"case-{i}": score_0shot[i]  for i in range(n)},
    "system_prompted":  {f"case-{i}": score_system[i] for i in range(n)},
    "chain_of_thought": {f"case-{i}": score_cot[i]    for i in range(n)},
}

metric = BehavioralAlignmentMetric(bootstrap_ci=True)
verdicts = BehavioralAlignmentMetric.from_condition_scores(condition_scores)
result = metric.compute(verdicts, ScaleType.INTERVAL)

print(f"α = {result.alpha:.4f}  [{result.alpha_ci_low:.4f}, {result.alpha_ci_high:.4f}]")
print(result.alpha_interpretation)
# "behaviorally consistent across conditions (α ≥ 0.80)"
# or "high condition sensitivity — framing materially changes outputs (α < 0.67)"
```

---

## Accessing output report fields

```python
report = evaluator.evaluate_skill("./my-skill")

# Corpus-level agreement (v0.2: + CI and ICC)
report.mean_uplift                        # float: treatment − control
report.agreement.alpha                    # Krippendorff's α
report.agreement.alpha_ci_low            # 95% bootstrap CI lower bound (v0.2)
report.agreement.alpha_ci_high           # 95% bootstrap CI upper bound (v0.2)
report.agreement.alpha_interpretation     # "strong agreement (α ≥ 0.80)"
report.agreement.kappa                    # Cohen's κ (mean pairwise)
report.agreement.icc                      # ICC(2,k) absolute agreement (v0.2)
report.agreement.icc_interpretation       # "excellent reliability (ICC ≥ 0.75)" (v0.2)
report.agreement.expected_chance_agreement # P(e)

# Significance (v0.2)
report.significance.mcnemar_statistic     # McNemar χ²
report.significance.p_value              # Wald p-value
report.significance.significant          # bool: p < 0.05
report.significance.uplift_ci_low        # bootstrap 95% CI lower bound
report.significance.uplift_ci_high       # bootstrap 95% CI upper bound
report.significance.n_treatment_wins     # cases where treatment > control

# Per-judge fit (v0.2)
for jf in report.judge_fit:
    jf.judge_id
    jf.lz_statistic          # outfit MNSQ t-statistic
    jf.flagged_inconsistent  # bool: |t| > 1.96

# Bias
report.bias.positional_bias_rate          # fraction of cases with A/B flip
report.bias.verbosity_bias_rho            # Spearman ρ(length, score)
report.bias.verbosity_biased              # bool

# Per-case
for case in report.cases:
    case.case_id
    case.uplift
    case.control.score
    case.treatment.score
    case.agreement.alpha             # per-case α (requires ≥ 2 judges)
    case.positional_flip             # bool

# Per-judge per-case (requires --verdicts flag in CLI or include_verdicts: true in config)
for verdict in case.verdicts:
    verdict.judge_id
    verdict.score
    verdict.rationale
    verdict.assertion_scores         # {"assertion text": 0.0|1.0}
    verdict.dimension_scores         # {"dim name": 0.0–1.0}
    verdict.output_token_count

# Serialize
json_str = report.model_dump_json(indent=2)
dict_obj = report.model_dump()
```
