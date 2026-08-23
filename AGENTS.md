# AGENTS.md — judge-kappa

This file is written for AI agents, agentic harnesses, and automated tooling that need to understand, use, extend, or contribute to **judge-kappa**. It complements `README.md` (user-facing) and `docs/` (practitioner reference) with machine-navigable structure and explicit contribution rules.

> **Convention**: `AGENTS.md` follows the emerging standard for agent-oriented repository documentation. Agents and harnesses should read this file first when exploring the repository.

---

## 1 — Repository identity

| Field | Value |
|---|---|
| **Package name** | `judge-kappa` (PyPI), `judge_kappa` (Python import) |
| **Version** | 0.2.1 |
| **Repository** | https://github.com/williamcaban/judge-kappa-eval |
| **Language** | Python 3.12+ |
| **License** | Apache 2.0 |
| **Primary author** | William Caban \<william.caban@gmail.com\> |

Install:
```bash
pip install "judge-kappa[all]"
uv add "judge-kappa[all]"
```

---

## 2 — Purpose and philosophy

judge-kappa is a **statistically rigorous, scholarly-grounded** evaluation framework for LLM-as-judge pipelines. It bridges two widely-used but statistically incomplete frameworks — agent-skills-eval (A/B uplift) and MLflow LLMaJ (rubric scoring) — while adding the psychometric and statistical infrastructure needed for scientific and regulatory contexts.

**Core thesis**: LLM evaluation is a measurement problem. The same statistical methods that psychometricians use to validate educational assessments — inter-rater reliability, item response theory, differential item functioning, person-fit statistics — apply directly to LLM judge panels. Every metric in this library has a peer-reviewed scholarly basis.

**Non-goals**: This library does not implement prompt engineering heuristics, chain-of-thought tricks, or model-specific optimizations. Every capability must be grounded in established measurement science.

---

## 3 — Science foundations

The following table maps each implemented capability to its primary scholarly source. This is the authoritative reference for agents and contributors; the full bibliography is in `docs/11-scholarly-references.md`.

| Capability | Class / Function | Primary reference |
|---|---|---|
| Krippendorff's α inter-rater reliability | `KrippendorffAlpha` | Krippendorff (2004) *Human Communication Research* 30(3) |
| Bootstrap 95% CI for α | `KrippendorffAlpha(bootstrap_ci=True)` | Efron (1979) *Annals of Statistics* 7(1) |
| Cohen's κ pairwise agreement | `CohenKappa` | Cohen (1960) *Educational and Psychological Measurement* 20(1) |
| ICC(2,k) variance decomposition | `compute_icc` | Shrout & Fleiss (1979) *Psychological Bulletin* 86(2) |
| Outfit MNSQ person-fit | `PersonFitAnalyzer` | Wright & Masters (1982) *Rating Scale Analysis* (MESA Press) |
| McNemar significance test | `JuryEvaluator(compute_significance=True)` | McNemar (1947) *Psychometrika* 12(2) |
| 2PL IRT judge weighting | `IRTJudgeWeighter` | Birnbaum (1968); Fonseca Rivera et al. (2026) arXiv:2608.05086 |
| Differential item functioning | `DifferentialItemFunctioningDetector` | Swaminathan & Rogers (1990) *J. Educational Measurement* 27(4) |
| Behavioral alignment metric | `BehavioralAlignmentMetric` | Krippendorff (2004) — α repurposed cross-condition |
| Positional bias detection | `PositionalBiasDetector` | Ko et al. (2020) *EMNLP*; Zheng et al. (2023) arXiv:2306.05685 |
| Verbosity bias detection | `VerbosityBiasDetector` | Saito et al. (2023) arXiv:2310.10076 |
| Elo rating (tournament) | `TournamentEvaluator` | Elo (1978) *The Rating of Chessplayers*; Zheng et al. (2023) |
| ICL calibration alignment | `LLMJudge._icl_block()` | Brown et al. (2020) *NeurIPS* 33; Kim et al. (2024) *ICLR* |
| LLM-as-judge paradigm | All `LLMJudge` subclasses | Zheng et al. (2023) arXiv:2306.05685 |
| Listwise ranking | `RankJudge` | Cao et al. (2007) *ICML*; Liu et al. (2023) *EMNLP* |
| Spearman rank correlation | `VerbosityBiasDetector` | Spearman (1904) *Am. J. Psychology* 15(1) |
| Softmax panel weighting | `IRTJudgeWeighter.weights()` | Bridle (1990) in *Neurocomputing* |
| Classical test theory (background) | Multi-judge panel design | Lord & Novick (1968) *Statistical Theories of Mental Test Scores* |

---

## 4 — Quick-start for agents

### Discover the API

```python
import judge_kappa
help(judge_kappa)                        # full public API
print(judge_kappa.__all__)               # all exported symbols
```

### Minimal A/B evaluation (no API key — mock backend)

```python
from judge_kappa import (
    JuryEvaluator, JudgePanel, AssertionJudge,
    EvalCase, Assertion, Variant, ScaleType,
)

class MockBackend:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        import json
        return json.dumps({"assertion_scores": {"quality": 0.8}, "rationale": "ok"})

backend = MockBackend()
panel   = JudgePanel(judges=[AssertionJudge("j1", backend)])
ev      = JuryEvaluator(panel=panel, generation_backend=backend)

cases = [EvalCase(id="c1", prompt="Q", assertions=[Assertion(text="quality")])]
ctrl  = Variant(name="control",   predict_fn=lambda _: "baseline answer")
trt   = Variant(name="treatment", predict_fn=lambda _: "improved answer")

report = ev.evaluate(cases, ctrl, trt)
print(report.mean_uplift, report.agreement.alpha, report.significance.p_value)
```

### Key entry points

| Goal | Entry point | Returns |
|---|---|---|
| A/B uplift from skill directory | `ev.evaluate_skill(skill_dir)` | `EvalReport` |
| A/B uplift from callable pipeline | `ev.evaluate_dataset(data, predict_fn)` | `EvalReport` |
| A/B uplift, two live endpoints | `ev.evaluate_endpoints(cases, ctrl, trt)` | `EvalReport` |
| A/B uplift, pre-recorded outputs | `ev.evaluate_prerecorded(data)` | `EvalReport` |
| Pairwise preference rates | `ev.evaluate_pairwise_dataset(data, pairwise_judge)` | `PairwiseReport` |
| N-system Elo leaderboard (N ≤ 6) | `TournamentEvaluator.run_dataset(data, systems)` | `TournamentReport` |
| Listwise ranking (N ≥ 7) | `RankJudge.rank(case, system_outputs)` | `dict[str, JudgeVerdict]` |
| IRT-derived panel weights | `IRTJudgeWeighter().fit(...).weights()` | `dict[str, float]` |
| Cross-condition consistency | `BehavioralAlignmentMetric().compute(verdicts)` | `AgreementResult` |
| DIF case-level bias | `DifferentialItemFunctioningDetector().analyze(verdicts)` | `DIFReport` |

### Reading an EvalReport

```python
# Corpus-level
report.mean_uplift                        # float: treatment − control
report.agreement.alpha                    # Krippendorff's α
report.agreement.alpha_ci_low             # bootstrap 95% CI lower bound
report.agreement.alpha_ci_high            # bootstrap 95% CI upper bound
report.agreement.icc                      # ICC(2,k) absolute agreement
report.significance.p_value              # McNemar p-value
report.significance.significant          # bool: p < 0.05
report.significance.uplift_ci_low        # bootstrap CI for mean_uplift
report.judge_fit                          # list[JudgeFitResult] — per-judge MNSQ t-stat
report.bias.positional_bias_rate          # fraction of cases with A/B flip
report.bias.verbosity_bias_rho            # Spearman ρ(length, score)

# Serialize
report.model_dump_json(indent=2)
```

### Documentation index

| Doc | What it covers |
|---|---|
| `README.md` | Feature matrix, install, quick-start, example index |
| `docs/01-decision-guide.md` | Decision tree: which mode/panel/metric to use |
| `docs/02-input-types.md` | All entry points and their dataset formats |
| `docs/03-pairwise-tournament.md` | Pairwise, Tournament, RankJudge, champion-challenger; cost tables |
| `docs/04-examples.md` | 12 worked examples (9 YAML + 3 Python demos) |
| `docs/05-key-management.md` | API key resolution (three-case rule) |
| `docs/06-cli-reference.md` | CLI: `run`, `validate`, `schema`; output formats |
| `docs/07-python-api.md` | 13 usage patterns; all report fields |
| `docs/08-config-reference.md` | Every YAML config field |
| `docs/09-architecture.md` | Package structure, ABCs, data flow diagram |
| `docs/10-psychometric-methods.md` | Deep guide to all 9 psychometric capabilities |
| `docs/11-scholarly-references.md` | Full bibliography: 18 techniques × papers/books |
| `examples/demo_psychometric.py` | Runnable: CI, ICC, PersonFit, DIF, McNemar (no API key) |
| `examples/demo_irt_weighting.py` | Runnable: IRT judge weighting (no API key) |
| `examples/demo_rank_judge.py` | Runnable: RankJudge leaderboard (no API key) |

---

## 5 — Package structure (agent navigation)

```
src/judge_kappa/
├── models.py            # All Pydantic domain models — start here to understand data contracts
├── evaluator.py         # JuryEvaluator — main orchestrator; all EvalReport fields computed here
├── tournament.py        # TournamentEvaluator — Elo + positional bias
├── adapters/            # Input: SKILL.md→EvalCase, list[dict]→EvalCase
├── judges/
│   ├── base.py          # LLMJudge ABC + ICL alignment mixin (_parse_json, _icl_block)
│   ├── assertion.py     # PASS/FAIL per assertion
│   ├── rubric.py        # 0–1 per rubric dimension
│   ├── pairwise.py      # A vs B in one prompt
│   └── rank.py          # Listwise ranking (N ≥ 7)
├── panel/
│   ├── panel.py         # JudgePanel — homogeneous rubric; measures α
│   └── jury.py          # JudgeJury — diverse rubrics; weighted aggregation
├── agreement/
│   ├── alpha.py         # KrippendorffAlpha + bootstrap CI
│   ├── kappa.py         # CohenKappa
│   ├── icc.py           # ICC(2,k)
│   ├── personfit.py     # PersonFitAnalyzer (outfit MNSQ)
│   └── behavioral.py    # BehavioralAlignmentMetric
├── bias/
│   ├── positional.py    # A/B swap test
│   ├── verbosity.py     # Spearman ρ(length, score)
│   └── dif.py           # Differential Item Functioning
├── calibration/
│   └── irt.py           # IRTJudgeWeighter (2PL MLE)
├── llm/
│   ├── base.py          # LLMBackend Protocol — duck typing, any .complete() works
│   ├── openai_backend.py   # OpenAI-compatible (OpenAI, vLLM, Ollama, Groq, Together)
│   └── anthropic_backend.py
└── cli/
    ├── config_schema.py # Pydantic YAML config models
    ├── builder.py       # Config → runtime objects
    └── main.py          # CLI entry point
```

**Extension points** (subclass to add new capabilities):

| ABC / Protocol | To add |
|---|---|
| `LLMBackend` (Protocol — implement `.complete()`) | New LLM provider |
| `LLMJudge` (subclass + implement `.judge()`) | New judge type |
| `EvaluationPanel` (subclass `evaluate()` + `aggregate_score()`) | New aggregation |
| `AgreementMetric` (subclass + implement `.compute()`) | New reliability statistic |
| `BiasDetector` (subclass + implement `.detect()`) | New bias type |

---

## 6 — Contributing guidelines

These rules are **mandatory** for all contributions — whether authored by a human, an AI agent, or a human-AI pair. They enforce the scholarly rigor that is the library's defining property.

### 6.1 — Branch and PR workflow

All work goes on a branch. **Never commit directly to `main`.**

```bash
git checkout -b feat/<capability-name>   # new feature
git checkout -b fix/<issue-description>  # bug fix
git checkout -b docs/<topic>             # documentation only
git checkout -b release/vX.Y.Z          # version bump
```

Open a PR to `main`. Branch protection requires all CI jobs to pass before merge:
- `Test Python 3.12`
- `Test Python 3.13`
- `Validate example configs`

### 6.2 — Mandatory quality gates (must pass before opening a PR)

Run the following locally from the repo root. All must be clean:

```bash
uv run ruff check src tests     # zero errors — "All checks passed!"
uv run mypy src                 # zero errors — "Success: no issues found"
uv run pytest                   # zero failures
uv run judge-kappa validate examples/config_*.yaml   # all configs valid
```

The pre-push hook (`.git/hooks/pre-push`) runs these automatically.

**CI will reject PRs that fail any of these checks.** Do not open a PR until all gates pass locally.

### 6.3 — Mandatory scholarly citation for every new method

**This is the most important rule.** Every new statistical technique, psychometric method, or algorithmic approach must be grounded in peer-reviewed literature.

When adding a new capability, you must:

1. **Identify the primary scholarly source** — the original paper or authoritative textbook that defines the method. Preprints (arXiv) are acceptable for recent work; Wikipedia and blog posts are not.

2. **Add an entry to `docs/11-scholarly-references.md`** with the structure:

   ```markdown
   ## N — [Method Name]

   **What it is.** 2–3 sentences defining the method in plain language.

   **In judge-kappa.** Which class/function implements it and how it is used.

   **Primary sources.**
   - Author, A. (Year). Title. *Journal/Venue*, vol(issue), pages. DOI or arXiv link.
   - [Additional sources as needed]
   ```

3. **Add the method to the consolidated bibliography table** at the end of `docs/11-scholarly-references.md`.

4. **Reference the paper in the module docstring** of the implementing file.

No PR that adds a new method will be merged without a citation in `docs/11-scholarly-references.md`.

### 6.4 — Mandatory documentation updates

Every PR that adds a new capability must update **all** of the following that apply:

| File | What to update |
|---|---|
| `src/judge_kappa/__init__.py` | Export the new symbol; add to `__all__` |
| `docs/11-scholarly-references.md` | Citation entry + bibliography row (§6.3) |
| `docs/09-architecture.md` | Package structure tree; ABCs table if applicable; data flow if wired into JuryEvaluator |
| `docs/07-python-api.md` | New import in Core objects; new Pattern (#N) with code example |
| `docs/10-psychometric-methods.md` | If psychometric: add section with API, interpretation, references |
| `docs/01-decision-guide.md` | Add to decision tree and quick reference table |
| `docs/02-input-types.md` | If new entry point: add to entry point table and section |
| `docs/03-pairwise-tournament.md` | If ranking-related |
| `docs/04-examples.md` | Add example entry (§N) with run command and expected output |
| `README.md` | Add to feature matrix table; update docs table if adding a new doc |
| `CHANGELOG.md` | Document under `## [Unreleased]` |

### 6.5 — Mandatory example

Every new public capability must have a runnable example. Add a file `examples/demo_<capability>.py` that:

- Has a module docstring beginning with a description and `Run: python examples/demo_<capability>.py`
- Requires **no API key** wherever possible (use `MockBackend` or synthetic data)
- Demonstrates the primary use case with realistic synthetic data
- Prints output that illustrates what the user or agent would see in practice
- Is referenced in `docs/04-examples.md`

### 6.6 — PR sign-off requirements

Every PR description must include:

**Human sign-off** (mandatory):
```
Signed-off-by: Full Name <email@example.com>
```

**Agent / model attribution** (mandatory when any AI system generated or significantly modified code):
```
Generated-by: <model-name> via <harness-or-tool>
# Examples:
Generated-by: claude-sonnet-4-6 via Claude Code
Generated-by: gpt-4o via Cursor
Generated-by: gemini-1.5-pro via Aider
```

Both lines must appear in the PR body. A PR with AI-generated code and no `Generated-by` line will not be merged. This is not punitive — it is a transparency and provenance requirement consistent with the library's scholarly rigor.

Commit messages follow the same convention:

```
feat: add BivariateNormalAgreement metric

Implements a bivariate normal model for agreement between two continuous
judges, grounded in Krippendorff (2013) §12.

Refs: Krippendorff, K. (2013). Content Analysis (3rd ed.). SAGE.

Signed-off-by: Jane Smith <jane@example.com>
Generated-by: claude-sonnet-4-6 via Claude Code
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

### 6.7 — What belongs in this library vs. what does not

**In scope:**
- Statistical reliability and agreement metrics with peer-reviewed grounding
- Bias detection methods with documented theoretical basis
- Psychometric methods applicable to LLM judge calibration and evaluation
- Judge types that implement well-defined scoring protocols
- Adapters for standard evaluation dataset formats

**Out of scope:**
- Prompt engineering heuristics without measurement-theoretic grounding
- Model-specific optimizations or provider-specific features
- Evaluation metrics without a primary scholarly citation
- Anything that cannot be tested without real LLM API calls (all new code must be testable with `MockBackend`)

If you are uncertain whether a contribution is in scope, open an issue before writing code.

### 6.8 — Test requirements

All new code must be covered by tests in `tests/`. Requirements:

- Zero real LLM calls — use `MockLLMBackend` from `tests/conftest.py`
- Tests must pass under both `Python 3.12` and `Python 3.13`
- Coverage must not decrease (check with `uv run pytest --cov --cov-report=term-missing`)
- For new agreement metrics: test the boundary conditions (α=1.0 on perfect agreement, α≈0 on random)
- For new bias detectors: test both the flagged and clean cases

### 6.9 — Release process (for maintainers)

```bash
# 1. Create release branch
git checkout -b release/vX.Y.Z

# 2. Bump version in pyproject.toml; update CHANGELOG.md

# 3. Commit, push, open PR
git push -u origin release/vX.Y.Z
gh pr create --base main --title "chore: release vX.Y.Z"

# 4. After PR merges:
git checkout main && git pull
git tag vX.Y.Z
git push origin vX.Y.Z   # → triggers Publish to PyPI workflow automatically
```

Delete the release branch after tagging:
```bash
git branch -d release/vX.Y.Z
git push origin --delete release/vX.Y.Z
```

---

## 7 — Common agent tasks

### Find which file implements a given capability

```bash
grep -r "class KrippendorffAlpha" src/
grep -r "def compute_icc" src/
grep -r "IRTJudgeWeighter" src/
```

### Check current test count and coverage

```bash
uv run pytest -q               # fast run
uv run pytest --cov --cov-report=term-missing
```

### Validate a config before running

```bash
uv run judge-kappa validate examples/config_skill.yaml
```

### Run a single demo (no API key)

```bash
uv run python examples/demo_psychometric.py
uv run python examples/demo_irt_weighting.py
uv run python examples/demo_rank_judge.py
```

### Check that a new import is properly exported

```python
from judge_kappa import MyNewClass   # must not raise ImportError
assert "MyNewClass" in dir(__import__("judge_kappa"))
```

### Verify the scholarly reference doc is updated

```bash
grep -c "##" docs/11-scholarly-references.md   # should increase by 1 per new method
grep "MyNewMethod" docs/11-scholarly-references.md   # must exist
```
