# Changelog

All notable changes to judge-kappa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Reference-value and synthetic-recovery tests for eight agreement/bias/
  calibration capabilities, closing a gap where 5 (ICC, PersonFit, DIF, IRT,
  BehavioralAlignment) had zero test coverage and the remaining 3
  (Krippendorff's α, Cohen's κ, McNemar) were tested only against internal
  invariants rather than literature or hand-derived reference values.
  - `KrippendorffAlpha` / `BehavioralAlignmentMetric`: validated against the
    canonical Hayes/Krippendorff (2011) worked example reproduced on
    Wikipedia (α=0.691 nominal, α=0.811 interval).
  - `CohenKappa`: validated against a hand-derived contingency table
    (κ=0.5455), cross-checked against `sklearn.metrics.cohen_kappa_score`.
  - `compute_icc` (ICC(2,k)): validated against the canonical Shrout &
    Fleiss (1979) Table 2 dataset, cross-checked against an independent
    implementation (`pingouin`'s `ICC(A,k)` = 0.6201).
  - McNemar (`_mcnemar_and_ci`): validated against hand-computed
    continuity-corrected chi-square values for significant and
    non-significant cases.
  - `IRTJudgeWeighter`: validated via synthetic reliability-ranking
    recovery (a known judge-reliability ordering is correctly recovered
    from synthetic data). This does not validate 2PL item-parameter
    (discrimination/difficulty) or theta-magnitude recovery — see the
    `TestIRTJudgeWeighter` docstring in `tests/test_psychometrics.py` for
    what a full item-parameter-recovery test would additionally need.
  - `PersonFitAnalyzer`, `DifferentialItemFunctioningDetector`: unit tests
    plus documented findings — both modules have a design property
    (detailed in test docstrings) that currently prevents a clean
    "inject known effect, verify detection" test; flagged for maintainer
    follow-up rather than silently patched.

---

## [0.2.1] — 2026-08-23

### Added

- `AGENTS.md` — agent-oriented repository guide covering: repo identity, science
  foundations table (18 capabilities × primary citations), quick-start for agents,
  package structure navigation, and rigorous contributing guidelines (scholarly
  citation requirement, documentation checklist, PR sign-off rules).
- `AGENTS.md` included in sdist package distribution (`pyproject.toml`).

---

## [0.2.0] — 2026-08-23

### Fixed

- **Bug: Tournament positional bias inverted** (`tournament.py`). The detector was counting
  cases where system A won in *both* orderings (a quality signal) as positional flips.
  Corrected to count first-positioned output winning in both rounds — the same semantics as
  `PositionalBiasDetector` in `bias/positional.py`.

### Added

**Psychometric agreement metrics**
- `KrippendorffAlpha`: bootstrap 95% CI for α (`alpha_ci_low`, `alpha_ci_high` in
  `AgreementResult`). Configurable `n_bootstrap` (default 2000) and `seed`.
- `agreement/icc.py`: ICC(2,k) absolute agreement. Decomposes variance into between-cases
  (signal), between-judges (systematic bias), and residual — reveals *why* judges disagree.
  `icc` and `icc_interpretation` added to `AgreementResult`.
- `agreement/personfit.py`: `PersonFitAnalyzer` — outfit MNSQ t-statistic per judge.
  Flags judges that score inconsistently relative to the panel (|t| > 1.96 → `flagged_inconsistent`).
  Model-free approximation of the IRT lz statistic (Wright & Masters 1982).
- `agreement/behavioral.py`: `BehavioralAlignmentMetric` — Krippendorff α repurposed for
  cross-condition behavioral consistency. `judge_id` encodes the condition name; α < 0.80
  indicates the model is sensitive to prompt framing (DISC paper connection).

**IRT-based judge weighting**
- `calibration/irt.py`: `IRTJudgeWeighter` — 2PL MLE with log-normal prior on discrimination
  and normal prior on difficulty (Fonseca Rivera et al. 2026 methodology). Fits a judge ×
  calibration-example response matrix; converts latent θ to panel weights via softmax.
  Item discrimination/difficulty parameters available via `item_parameters()`.

**DIF analysis**
- `bias/dif.py`: `DifferentialItemFunctioningDetector` — logistic regression (Mantel-Haenszel
  protocol) flags eval cases where verdict probability differs systematically across judge model
  families after controlling for overall score level. Returns `DIFReport` with per-case p-values.

**RankJudge**
- `judges/rank.py`: `RankJudge` — listwise ranking of N systems in a single judge call.
  O(N) alternative to the O(N²) pairwise tournament for N ≥ 7. Rank converted to 0–1 score
  (rank 1 → 1.0, rank N → 0.0). Configurable `max_systems` cap.

**Uplift significance**
- `evaluator.py`: `_mcnemar_and_ci()` — continuity-corrected McNemar test on binarised per-case
  verdicts + bootstrap 95% CI for mean uplift. Exposed in `EvalReport.significance`
  (`UpliftSignificance` model).

**Models**
- `AgreementResult`: `alpha_ci_low`, `alpha_ci_high`, `icc`, `icc_interpretation`
- `JudgeFitResult`: `judge_id`, `lz_statistic`, `flagged_inconsistent`
- `UpliftSignificance`: `n_treatment_wins`, `n_control_wins`, `n_ties`,
  `mcnemar_statistic`, `p_value`, `significant`, `uplift_ci_low`, `uplift_ci_high`
- `EvalReport`: `significance: Optional[UpliftSignificance]`, `judge_fit: list[JudgeFitResult]`

**JuryEvaluator constructor params** (all default to enabled):
  `compute_significance`, `compute_icc`, `compute_judge_fit`, `bootstrap_ci`, `n_bootstrap`

**Optional dependency**: `tiktoken` extra (`pip install judge-kappa[tiktoken]`) for accurate
  token counts in `VerbosityBiasDetector`.

### Fixed (pre-existing test failures)

- `KrippendorffAlpha.compute()` now handles single-value domains (all judges agree) gracefully
  by returning α=1.0 by convention instead of raising `ValueError`.
- Verbosity threshold test data corrected to give Spearman ρ≈0.67 (not ρ=1.0 as before).
- Prerecorded evaluator tests now use `RubricJudge` (cases loaded from datasets have no assertions).
- Key resolution test: `anthropic-hosted` scenario passes `provider="anthropic"` correctly.
- Test count: 151 passed (up from 37 passing / 15 failing in v0.1.0).

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

[Unreleased]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/williamcaban/judge-kappa-eval/releases/tag/v0.1.0
