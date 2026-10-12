# Changelog

All notable changes to judge-kappa are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- **Verbosity bias no longer falls back to the judge's rationale length.** A verdict with
  `output_token_count == 0` used to be measured by the word count of its *rationale*, which is
  the judge's own wordiness, not the length of the output being judged, and it fed
  `verbosity_bias_rho` silently. Such verdicts are now left out of the correlation. Built-in
  judges always record the output's word count, so their numbers do not move (an empty output has
  length 0 and is now reported as unmeasured). Custom judges that never set `output_token_count`
  previously got a rationale-based figure and now get no estimate plus a warning.
- **`verbosity_bias_rho` / `verbosity_bias_p` are `None`, not `NaN`, when the correlation is
  undefined** (all scores identical, all lengths identical, or fewer than 3 measured verdicts).
  `spearmanr` used to be called on the constant vector, returned `NaN` and emitted a SciPy
  `ConstantInputWarning` that never reached the report; `model_dump()` held `nan` while the JSON
  held `null`. The call is now skipped. `verbosity_biased` is `False` in these cases, so check
  `verbosity_bias_rho is None` first. The CLI text report prints `not estimated (<reason>)`.

### Added

- `BiasResult.verbosity_n_measured` and `BiasResult.verbosity_n_unmeasured`, and a
  `data_quality.warnings` entry when verdicts were left out of the verbosity correlation.
- `BiasResult.verbosity_not_estimated_reason` (`too_few_measured`, `constant_scores`,
  `constant_lengths`): says why `verbosity_bias_rho` is `None`.

### Deprecated

- `VerbosityBiasDetector(tiktoken_encoding=...)`: it was accepted but never used. It now emits a
  `DeprecationWarning` and is ignored.
- The `tiktoken` extra is a no-op and is no longer part of `[all]`. `tiktoken` was never imported by
  the library, so nothing changes for code; `pip install "judge-kappa[tiktoken]"` still resolves.

### Fixed

- Documentation claimed that tiktoken gave "accurate token counts" for verbosity bias
  (README, `docs/10` §9, the detector's module docstring). Lengths are whitespace word counts of
  the evaluated output; `docs/10` §9 now says so, and explains how to supply your own measure for
  CJK text and code.

---

## [0.3.1] — 2026-10-12

> **Heads-up:** Anthropic judges now actually run at the configured temperature (0.0 was previously
> dropped, so the API default applied). Scores for Anthropic judges can differ from 0.3.0; see *Changed*.

### Added

- **Explicit, inspectable request parameters.** `max_tokens`, `max_tokens_param`,
  `omit_temperature`, `timeout_s` and `max_retries` on `OpenAIBackend`,
  `AnthropicBackend` and the YAML `backend:` block. Replies that hit the output-token
  limit are detected from the finish reason.
- **Usage capture.** Optional `complete_with_usage()` returns an `LLMResponse` (text,
  input/output tokens, finish reason, model, latency); `call_backend()` falls back to
  `complete()` so the public `LLMBackend` Protocol is unchanged. Built-in backends keep a
  thread-safe `UsageStats`.
- **Parse-failure accounting.** Judges validate replies and set `parse_failed`,
  `parse_error` and `truncated` on verdicts; reports carry `data_quality` (counts by judge
  and code, truncations, warnings). `on_parse_failure="raise"` raises `JudgeParseError`.
  `TournamentEvaluator(exclude_parse_failures=True)` drops unusable rounds.
- **Run provenance.** `EvalReport.run`, `PairwiseReport.run` and `TournamentReport.run`
  record library/SDK versions, evaluator settings, per-judge and per-backend
  configuration, and per-backend usage (scoped to the run, no secrets).
- **`CachedBackend` / `backend.cache_path`**: append-only JSONL response cache. An
  interrupted run resumes by re-running with the same file; re-runs are free and exactly
  reproducible.
- **`max_workers`** on `JuryEvaluator`, `TournamentEvaluator`, `evaluate_pairwise_dataset`
  and YAML configs. Default 1 (sequential, no threads); results are identical for any value.
- CLI text report prints data-quality warnings and a usage line. YAML judges accept
  `on_parse_failure: flag | raise`; a `raise` stops the CLI with a one-line error.
- `docs/12-reliability-and-reproducibility.md` and `examples/demo_reliability_controls.py`.
- Docstrings on every module, class, function and method in `src/judge_kappa` (previously
  about 40% of callables), enforced from now on by ruff's missing-docstring rules (`D1`).
  Documentation only: the code is unchanged.
- Guard tests (`tests/test_naming.py`): no stale package name in shipped text, every
  `from judge_kappa import ...` in `examples/` resolves, and CLI usage names the installed
  command.
- CI: AI review of pull requests through a shared PR Agent workflow, pinned by commit SHA with minimal
  permissions. The review is posted as comments and leaves the PR body (and its sign-off) untouched.

### Changed

- **Anthropic backend now sends `temperature` explicitly, including `0.0`.** Earlier
  releases dropped it when it was `0.0`, so the API default (documented as 1.0) applied and
  judges configured as deterministic were not. Scores for Anthropic judges can differ from
  0.3.x, and agreement usually improves. Re-baseline stored comparisons. Models that reject
  the parameter: set `omit_temperature: true`.
- **Unusable judge replies are no longer silent.** Valid replies score exactly as before.
  The historical fallback scores are kept by default (so numbers do not move) but are
  flagged, logged (first 5 per judge) and counted. A non-dict JSON reply no longer raises
  `AttributeError`, and a non-numeric `PairwiseJudge` score no longer raises `ValueError`.
- Anthropic replies are assembled from text blocks, so a leading thinking block no longer
  breaks extraction.

### Fixed

- Stale package name `jury-eval` / `jury_eval` removed from user-facing text:
  CLI usage and unknown-command message, CLI and config-schema docstrings, the
  `pip install` hints in the Anthropic and OpenAI backend `ImportError`
  messages (they pointed users at the wrong package), and the two scripts in
  `examples/` (`from jury_eval import ...` raised `ModuleNotFoundError`).
- CI coverage upload: the Codecov step passed the removed input `file`, so the
  uploader never ran; it now uses `files` and reads `CODECOV_TOKEN`. Uploads are
  still rejected until that repository secret is set (`fail_ci_if_error: false`
  keeps CI green; the rejection is visible in the step log).

---

## [0.3.0] — 2026-10-11

### Added

- **Order-aware Bradley-Terry ranking** (`judge_kappa.ranking`): `fit_btl`,
  `bootstrap_btl`, `ComparisonGame`, `order_effect_identifiable`. Estimates item
  scores and a single judge-level display-order effect by ridge-penalised maximum
  likelihood, with identifiability checks (rank / cycle criterion), a separation
  flag, Wald standard errors, and case-level bootstrap intervals.
  Validated against a hand-derived two-item closed form and an independent
  scikit-learn logistic-regression oracle (`tests/test_ranking_btl.py`).
- `PositionalBiasDetector.estimate_order_effect()` and additive
  `BiasResult.order_effect` / `order_effect_ci_low` / `order_effect_ci_high`,
  populated by `JuryEvaluator` when a positional judge is configured.
- `TournamentEvaluator(rating_method=, n_bootstrap=, bootstrap_seed=, ci_level=)`;
  `TournamentStanding.btl_score` / `btl_ci_low` / `btl_ci_high`;
  `TournamentReport.order_effect` / `order_effect_ci` / `order_identifiable` /
  `possibly_separated` / `rating_method` / `ci_level`. Leaderboard prints the
  interval column and the judge order effect.
- `examples/demo_order_aware_ranking.py` (no API key) and docs: 03, 04, 07, 09, 10 §10, 11 §19-21.
- Design inspired by the issue #5 review of DIAL (arXiv:2609.31215). Only the
  single-judge order-effect stage is implemented; no DIAL code is used.

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

### Changed

- **Tournament scoring is now order-balanced.** Previously the swapped round fed
  only the positional flip rate; win/loss/tie counts, `win_matrix` and Elo came
  from round 1 alone, so detected position bias was never corrected (and the docs
  claimed otherwise). With `detect_positional_bias=True` (default), per-case scores
  are averaged over both display orders before win/loss/tie counting. Rankings
  from position-biased judges can change. `detect_positional_bias=False` is
  unchanged (round 1 only; order effect reported as not identifiable).
- **Default tournament ratings are order-aware BTL scores on the Elo scale**
  (`rating_method="btl"`), not sequential Elo updates. Use `rating_method="elo"`
  for the previous computation (applied to the now order-balanced win matrix).
  The `elo` field and sort order are kept; `_compute_elo` is unchanged.
- `tests/test_tournament.py::test_dominant_system_ranked_first` used a mock that
  returned `(0.9, 0.1)` for every call (a purely position-biased judge) and
  expected the first-listed system to win. It now uses a content-aware mock.
  New tests cover pure-position judges (ties, large order effect) and an
  adversarial listing order.

### Security

- CI hardening: dependency audit (`pip-audit`, PR/push/weekly), Dependabot for
  Actions and Python, Actions pinned to commit SHAs with read-only token
  permissions, commit-hygiene check, TestPyPI gate before PyPI releases,
  committed `uv.lock`, and a `SECURITY.md` policy.

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

[Unreleased]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/williamcaban/judge-kappa-eval/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/williamcaban/judge-kappa-eval/releases/tag/v0.1.0
