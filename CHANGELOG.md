# Changelog

All notable changes to judge-kappa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2026-06-07

### Added

**Core evaluation engine**
- `JuryEvaluator` with five entry points:
  - `evaluate_skill` — agent-skills-eval-compatible A/B uplift via SKILL.md + evals.json
  - `evaluate_dataset` — MLflow LLMaJ-compatible callable/pipeline evaluation
  - `evaluate_endpoints` — per-variant generation backends for live endpoint comparison
  - `evaluate_prerecorded` — score pre-recorded outputs without any generation calls
  - `evaluate_pairwise_dataset` — preference rates from pre-recorded A/B pairs
- `TournamentEvaluator` — round-robin N-system Elo leaderboard

**Judges**
- `AssertionJudge` — PASS/FAIL per assertion, weighted mean score
- `RubricJudge` — 0.0–1.0 per named rubric dimension, weighted mean score
- `PairwiseJudge` — scores A and B in a single prompt; returns both scores and preference

**Judge alignment**
- ICL (In-Context Learning) calibration examples on every judge — grounds all judges on the same human-validated reference scale

**Panels and juries**
- `JudgePanel` — homogeneous rubric; Krippendorff's α measures inter-rater agreement
- `JudgeJury` — diverse rubrics; weighted aggregation across judges with different perspectives
- Aggregation strategies: `mean`, `weighted_mean`, `majority_vote`, `median`, `trimmed_mean`

**Statistical agreement metrics**
- `KrippendorffAlpha` — handles missing data, all scale types (nominal/ordinal/interval/ratio); default metric
- `CohenKappa` — pairwise κ with linear weighting for ordinal scales; reports expected chance agreement P(e)

**Bias detection**
- `PositionalBiasDetector` — A/B swap test; per-case positional flip detection
- `VerbosityBiasDetector` — Spearman ρ between output token count and judge score

**LLM backends**
- `OpenAIBackend` — covers OpenAI, vLLM, Ollama, OpenRouter, Groq, Together (any `/v1/chat/completions`)
- `AnthropicBackend` — native Anthropic client
- `api_key_env` three-case rule for safe key management (no keys in config files)

**Input adapters**
- `SkillAdapter` — reads SKILL.md + evals/evals.json
- `DatasetAdapter` — reads list[dict] in MLflow LLMaJ format

**CLI**
- `judge-kappa run` — run evaluation from YAML config; `--format text|json|jsonl`; `--verdicts`
- `judge-kappa validate` — validate config without running evaluation
- `judge-kappa schema` — print full JSON Schema for both config modes

**Output models**
- `EvalReport` — A/B uplift, Krippendorff's α, Cohen's κ, positional bias rate, verbosity ρ
- `PairwiseReport` — preference rates, tie rate, per-case scores
- `TournamentReport` — Elo standings, win matrix, per-pair preference rates

**Documentation**
- 9 modular docs in `docs/`
- 9 annotated example configs in `examples/`
- Decision guide, input types reference, pairwise/tournament cost tables

**Tests**
- 8 test modules, 60+ tests, zero real LLM calls (MockLLMBackend)
- Coverage: models, adapters, judges (scoring math, ICL, malformed JSON), panels, agreement metrics, bias detectors, evaluator integration, CLI schema validation, key resolution

[Unreleased]: https://github.com/williamcaban/judge-kappa/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/williamcaban/judge-kappa/releases/tag/v0.1.0
