# Architecture and Extension Points

JuryEval is built on abstract base classes (ABCs) and Protocols. Every component can be replaced or extended without modifying existing code.

---

## Package structure

```
src/jury_eval/
├── models.py           # Pydantic domain models — stable schema contract
├── llm/
│   ├── base.py         # LLMBackend Protocol (structural — duck typing)
│   ├── openai_backend.py
│   └── anthropic_backend.py
├── adapters/
│   ├── base.py         # InputAdapter ABC
│   ├── skill.py        # SKILL.md + evals.json → EvalCase list
│   └── dataset.py      # list[dict] + predict_fn → EvalCase list
├── judges/
│   ├── base.py         # LLMJudge ABC + ICL alignment mixin
│   ├── assertion.py    # AssertionJudge — PASS/FAIL per assertion
│   ├── rubric.py       # RubricJudge — 0.0–1.0 per named dimension
│   └── pairwise.py     # PairwiseJudge — score A and B in one prompt
├── panel/
│   ├── base.py         # EvaluationPanel ABC
│   ├── panel.py        # JudgePanel — homogeneous rubric, measure agreement
│   └── jury.py         # JudgeJury — diverse rubrics, weighted aggregation
├── agreement/
│   ├── base.py         # AgreementMetric ABC
│   ├── alpha.py        # KrippendorffAlpha (default)
│   └── kappa.py        # CohenKappa + expected chance agreement
├── bias/
│   ├── base.py         # BiasDetector ABC
│   ├── positional.py   # PositionalBiasDetector — A/B swap test
│   └── verbosity.py    # VerbosityBiasDetector — Spearman ρ(length, score)
├── cli/
│   ├── config_schema.py # Pydantic config models (YAML schema)
│   ├── builder.py       # Config → runtime objects
│   └── main.py          # CLI entry point (no external deps)
├── tournament.py        # TournamentEvaluator — round-robin pairwise
└── evaluator.py         # JuryEvaluator — main orchestrator
```

---

## Key design decisions

### `LLMBackend` is a Protocol (structural typing)

Any object with a `complete(system, user, temperature) -> str` method works as a backend. No inheritance required.

```python
from jury_eval.llm.base import LLMBackend

class MyBackend:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return call_my_model(system, user)

# Works as a drop-in — no import from jury_eval needed
judge = AssertionJudge("my-judge", MyBackend())
```

### All abstractions are ABCs

Every module has an ABC that defines the contract. Subclass it to add a new implementation:

| ABC | What to subclass | To add |
|---|---|---|
| `LLMBackend` (Protocol) | Implement `complete()` | New LLM provider |
| `InputAdapter` | Subclass + implement `load()` | New input format |
| `LLMJudge` | Subclass + implement `judge()` | New judge type |
| `EvaluationPanel` | Subclass `evaluate()` + `aggregate_score()` | New aggregation strategy |
| `AgreementMetric` | Subclass + implement `compute()` | New agreement statistic |
| `BiasDetector` | Subclass + implement `detect()` | New bias type |

### Pydantic models are the schema contract

`models.py` is the stable interface. All cross-module data passes through Pydantic models. Adding a field is backward compatible (use `Optional` with a default). Removing or renaming a field is breaking.

### The CLI is intentionally thin

`cli/main.py` has no business logic. It parses arguments, calls `cmd_run()`, `cmd_validate()`, or `cmd_schema()`. All logic is in `builder.py` (config → objects) and `evaluator.py` (evaluation). This makes the Python API and CLI equivalent — the CLI is just a thin wrapper over the Python API.

---

## Extension examples

### Add a new LLM provider

```python
# src/jury_eval/llm/bedrock_backend.py
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
# src/jury_eval/judges/checklist.py
from jury_eval.judges.base import LLMJudge
from jury_eval.models import EvalCase, JudgeVerdict

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
# src/jury_eval/agreement/gwet.py
from jury_eval.agreement.base import AgreementMetric
from jury_eval.models import AgreementResult, JudgeVerdict, ScaleType

class GwetAC1(AgreementMetric):
    """Gwet's AC1 — more robust than κ when rater agreement is very high."""

    def compute(self, verdicts: list[JudgeVerdict], scale_type: ScaleType) -> AgreementResult:
        ...
        return AgreementResult(kappa=ac1, n_judges=n_judges, n_cases=n_cases)
```

Inject it into `JuryEvaluator`:
```python
from jury_eval.evaluator import JuryEvaluator

evaluator = JuryEvaluator(panel=panel, generation_backend=backend)
evaluator._kappa_metric = GwetAC1()   # override default CohenKappa
```

### Add a new bias detector

```python
# src/jury_eval/bias/length_normalization.py
from jury_eval.bias.base import BiasDetector
from jury_eval.models import BiasResult, JudgeVerdict

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
  │   AssertionJudge / RubricJudge → backend.complete(system, user) → parse JSON
  │
  ├─ panel.aggregate_score(verdicts)             → float
  │
  ├─ KrippendorffAlpha.compute(verdicts)         → AgreementResult
  ├─ CohenKappa.compute(verdicts)                → AgreementResult
  ├─ PositionalBiasDetector.test_case(...)       → PositionalBiasReport
  └─ VerbosityBiasDetector.detect(verdicts=...)  → BiasResult
  │
  ▼
EvalReport (Pydantic) → model_dump_json() → file / stdout / MLflow
```
