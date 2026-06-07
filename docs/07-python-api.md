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
    AgreementResult,
    BiasResult,
    CaseResult,
    VariantResult,
    JudgeVerdict,
    PairwiseCaseResult,

    # Agreement metrics
    KrippendorffAlpha,
    CohenKappa,

    # Bias detectors
    PositionalBiasDetector,
    VerbosityBiasDetector,

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

## Accessing output report fields

```python
report = evaluator.evaluate_skill("./my-skill")

# Corpus-level
report.mean_uplift                        # float: treatment − control
report.agreement.alpha                    # Krippendorff's α
report.agreement.alpha_interpretation     # "strong agreement (α ≥ 0.80)"
report.agreement.kappa                    # Cohen's κ (mean pairwise)
report.agreement.expected_chance_agreement # P(e)
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
