# Architecture and Extension Points

judge-kappa is built on abstract base classes (ABCs) and Protocols. Every component can be replaced or extended without modifying existing code.

---

## Package structure

```
src/judge_kappa/
├── models.py            # Pydantic domain models — stable schema contract
│                          (v0.2: + UpliftSignificance, JudgeFitResult;
│                                + AgreementResult.alpha_ci_*, icc, icc_interpretation)
├── llm/
│   ├── base.py          # LLMBackend Protocol (structural — duck typing)
│   ├── openai_backend.py
│   └── anthropic_backend.py
├── adapters/
│   ├── base.py          # InputAdapter ABC
│   ├── skill.py         # SKILL.md + evals.json → EvalCase list
│   └── dataset.py       # list[dict] + predict_fn → EvalCase list
├── judges/
│   ├── base.py          # LLMJudge ABC + ICL alignment mixin
│   ├── assertion.py     # AssertionJudge — PASS/FAIL per assertion
│   ├── rubric.py        # RubricJudge — 0.0–1.0 per named dimension
│   ├── pairwise.py      # PairwiseJudge — score A and B in one prompt
│   └── rank.py          # RankJudge — listwise ranking for N ≥ 7 (v0.2)
├── panel/
│   ├── base.py          # EvaluationPanel ABC
│   ├── panel.py         # JudgePanel — homogeneous rubric, measure agreement
│   └── jury.py          # JudgeJury — diverse rubrics, weighted aggregation
├── agreement/
│   ├── base.py          # AgreementMetric ABC
│   ├── alpha.py         # KrippendorffAlpha + bootstrap CI (v0.2)
│   ├── kappa.py         # CohenKappa + expected chance agreement
│   ├── icc.py           # ICC(2,k) absolute agreement (v0.2)
│   ├── personfit.py     # PersonFitAnalyzer — outfit MNSQ t-stat (v0.2)
│   └── behavioral.py    # BehavioralAlignmentMetric — DISC-style (v0.2)
├── bias/
│   ├── base.py          # BiasDetector ABC
│   ├── positional.py    # PositionalBiasDetector — A/B swap test
│   ├── verbosity.py     # VerbosityBiasDetector — Spearman ρ(length, score)
│   └── dif.py           # DifferentialItemFunctioningDetector (v0.2)
├── calibration/
│   └── irt.py           # IRTJudgeWeighter — 2PL MLE judge weights (v0.2)
├── cli/
│   ├── config_schema.py # Pydantic config models (YAML schema)
│   ├── builder.py       # Config → runtime objects
│   └── main.py          # CLI entry point (no external deps)
├── tournament.py         # TournamentEvaluator — round-robin pairwise
│                           (v0.2: positional bias logic corrected)
└── evaluator.py          # JuryEvaluator — main orchestrator
                            (v0.2: McNemar, ICC, PersonFit, bootstrap CI)
```

---

## Key design decisions

### `LLMBackend` is a Protocol (structural typing)

Any object with a `complete(system, user, temperature) -> str` method works as a backend. No inheritance required.

```python
from judge_kappa.llm.base import LLMBackend

class MyBackend:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return call_my_model(system, user)

# Works as a drop-in — no import from judge_kappa needed
judge = AssertionJudge("my-judge", MyBackend())
```

### All abstractions are ABCs

Every module has an ABC that defines the contract. Subclass it to add a new implementation:

| ABC / Class | What to subclass / extend | To add |
|---|---|---|
| `LLMBackend` (Protocol) | Implement `complete()` | New LLM provider |
| `InputAdapter` | Subclass + implement `load()` | New input format |
| `LLMJudge` | Subclass + implement `judge()` | New judge type (v0.2: also `RankJudge.rank()`) |
| `EvaluationPanel` | Subclass `evaluate()` + `aggregate_score()` | New aggregation strategy |
| `AgreementMetric` | Subclass + implement `compute()` | New agreement statistic |
| `BiasDetector` | Subclass + implement `detect()` | New bias type |
| `IRTJudgeWeighter` | Use as-is or subclass | Custom prior / tolerance / fitting |

### Pydantic models are the schema contract

`models.py` is the stable interface. All cross-module data passes through Pydantic models. Adding a field is backward compatible (use `Optional` with a default). Removing or renaming a field is breaking.

### The CLI is intentionally thin

`cli/main.py` has no business logic. It parses arguments, calls `cmd_run()`, `cmd_validate()`, or `cmd_schema()`. All logic is in `builder.py` (config → objects) and `evaluator.py` (evaluation). This makes the Python API and CLI equivalent — the CLI is just a thin wrapper over the Python API.

---

## Extension examples

### Add a new LLM provider

```python
# src/judge_kappa/llm/bedrock_backend.py
class BedrockBackend:
    def __init__(self, model_id: str, region: str = "us-east-1") -> None:
        import boto3
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        import json
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
        })
        resp = self._client.invoke_model(modelId=self._model_id, body=body)
        return json.loads(resp["body"].read())["content"][0]["text"]
```

No changes needed anywhere else — just pass it to any judge.

### Add a new judge type

```python
# src/judge_kappa/judges/checklist.py
from judge_kappa.judges.base import LLMJudge
from judge_kappa.models import EvalCase, JudgeVerdict

class ChecklistJudge(LLMJudge):
    """Evaluates output against a checklist of binary criteria."""

    def __init__(self, *args, checklist: list[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._checklist = checklist

    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        # build prompt, call backend, parse response
        ...
```

### Add a new agreement metric

```python
# src/judge_kappa/agreement/gwet.py
from judge_kappa.agreement.base import AgreementMetric
from judge_kappa.models import AgreementResult, JudgeVerdict, ScaleType

class GwetAC1(AgreementMetric):
    """Gwet's AC1 — more robust than κ when rater agreement is very high."""

    def compute(self, verdicts: list[JudgeVerdict], scale_type: ScaleType) -> AgreementResult:
        ...
        return AgreementResult(kappa=ac1, n_judges=n_judges, n_cases=n_cases)
```

Inject it into `JuryEvaluator`:
```python
from judge_kappa.evaluator import JuryEvaluator

evaluator = JuryEvaluator(panel=panel, generation_backend=backend)
evaluator._kappa_metric = GwetAC1()   # override default CohenKappa
```

### Add a new bias detector

```python
# src/judge_kappa/bias/length_normalization.py
from judge_kappa.bias.base import BiasDetector
from judge_kappa.models import BiasResult, JudgeVerdict

class LengthNormalizationBiasDetector(BiasDetector):
    """Detects whether scores change significantly when outputs are length-normalized."""

    def detect(self, **kwargs) -> BiasResult:
        verdicts: list[JudgeVerdict] = kwargs["verdicts"]
        # truncate all outputs to median length, re-evaluate, compare
        ...
```

---

## Data flow

```
Input
  │
  ├─ SkillAdapter.load_with_context(skill_dir)   → (cases, skill_md)
  ├─ DatasetAdapter.load(data)                   → cases
  └─ Hand-built EvalCase list
  │
  ▼
JuryEvaluator.evaluate(cases, control, treatment)
  │
  ├─ _run_variant(variant, case, backend)        → output string
  │   priority: predict_fn → variant.generation_backend → shared backend
  │
  ├─ panel.evaluate(case, output, variant_name)  → list[JudgeVerdict]
  │   AssertionJudge / RubricJudge / RankJudge → backend.complete() → parse JSON
  │
  ├─ panel.aggregate_score(verdicts)             → float
  │
  ├─ KrippendorffAlpha.compute(verdicts)         → AgreementResult  (+ bootstrap CI, v0.2)
  ├─ CohenKappa.compute(verdicts)                → AgreementResult
  ├─ enrich_agreement_with_icc(result, verdicts) → AgreementResult  (+ ICC(2,k), v0.2)
  ├─ PositionalBiasDetector.test_case(...)       → PositionalBiasReport
  ├─ VerbosityBiasDetector.detect(verdicts=...)  → BiasResult
  ├─ _mcnemar_and_ci(case_results)               → UpliftSignificance (v0.2)
  └─ PersonFitAnalyzer.analyze(all_verdicts)     → list[JudgeFitResult] (v0.2)
  │
  ▼
EvalReport (Pydantic) → model_dump_json() → file / stdout / MLflow

─── Separate psychometric tools (not in JuryEvaluator pipeline) ────────────

IRTJudgeWeighter.fit(judge_scores, human_scores)
  └─ .weights() → panel weights for JudgePanel (v0.2)

DifferentialItemFunctioningDetector.analyze(verdicts)
  └─ DIFReport  (v0.2)

BehavioralAlignmentMetric.compute(condition_verdicts)
  └─ AgreementResult (cross-condition α) (v0.2)
```
