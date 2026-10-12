"""
Stable domain models — the schema contract for the entire package.

All cross-module data passes through these Pydantic models.
Adding fields here is additive (backward compatible); removing or renaming is breaking.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ScaleType(StrEnum):
    """Measurement scale — governs which Krippendorff distance function is used."""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class AggregationStrategy(StrEnum):
    """How per-judge scores are collapsed into one score per output."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAJORITY_VOTE = "majority_vote"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


# ── Input models ──────────────────────────────────────────────────────────────


class Assertion(BaseModel):
    """A pass/fail statement a judge checks about an output, with a weight."""
    text: str
    weight: float = 1.0


class RubricDimension(BaseModel):
    """One rubric dimension: name, description, weight and optional low/mid/high anchors."""
    name: str
    description: str
    weight: float = 1.0
    anchors: list[str] = Field(default_factory=list)  # low / mid / high examples


class CalibrationExample(BaseModel):
    """ICL anchor: a human-validated (prompt, output, score) triple."""
    prompt: str
    output: str
    score: float
    rationale: str


class EvalCase(BaseModel):
    """One evaluation case: prompt, optional assertions, expected output, files and metadata."""
    id: str
    prompt: str
    assertions: list[Assertion] = Field(default_factory=list)
    expected_output: str | None = None
    files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Variant(BaseModel):
    """
    One side of an A/B evaluation.

    Output generation priority (first match wins):
      1. predict_fn       — call this Python callable with the case inputs
      2. generation_backend — call this specific LLMBackend (per-variant endpoint)
      3. JuryEvaluator.generation_backend — shared fallback backend
    """
    name: str
    system_prompt: str | None = None
    skill_context: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.0
    predict_fn: Callable[..., str] | None = None
    generation_backend: Any | None = None  # LLMBackend; typed as Any to avoid circular import

    class Config:
        """Pydantic settings: allow arbitrary callables and backends as fields."""
        arbitrary_types_allowed = True


# ── Output models ─────────────────────────────────────────────────────────────


class JudgeVerdict(BaseModel):
    """One judge's score for one output, with per-assertion or per-dimension scores.

    ``parse_failed`` marks a score that is a fallback value, not a real judgement.
    """
    judge_id: str
    eval_case_id: str
    variant_name: str
    score: float                                            # 0.0–1.0
    rationale: str
    assertion_scores: dict[str, float] = Field(default_factory=dict)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    output_token_count: int = 0
    # True when the judge reply was not usable JSON or did not match the expected schema.
    # `score` then holds the historical fallback value (0.0 for assertion/rubric, 0.5 for
    # pairwise, worst rank for ranking) and must not be read as a real judgement.
    parse_failed: bool = False
    parse_error: str | None = None    # short code, e.g. "invalid_json", "missing_assertions:1/3"
    truncated: bool = False           # reply hit the output-token limit


class DataQuality(BaseModel):
    """How much of a run's judge output was actually usable.

    Fallback scores from unusable replies are still included in the aggregate metrics;
    this block tells you how many there were so you can decide whether to trust them.
    """
    n_judge_calls: int = 0
    n_parse_failures: int = 0
    parse_failure_rate: float = 0.0
    parse_failures_by_judge: dict[str, int] = Field(default_factory=dict)
    parse_errors: dict[str, int] = Field(default_factory=dict)   # error code → count
    n_truncated_replies: int = 0
    warnings: list[str] = Field(default_factory=list)


class UsageSummary(BaseModel):
    """Token and call accounting for one backend over one run."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: int = 0              # replies that hit the output-token limit
    cache_hits: int = 0             # calls served from a CachedBackend (cost nothing)
    errors: int = 0
    latency_s: float = 0.0          # summed over calls (not wall-clock when concurrent)
    calls_without_usage: int = 0    # replies for which the provider reported no token counts


class BackendRecord(BaseModel):
    """One LLM backend used in a run, as it was configured. Never contains secrets."""
    role: str                       # "judge" | "generation"
    label: str                      # unique within the report, e.g. "anthropic:claude-x"
    judge_ids: list[str] = Field(default_factory=list)   # judges that used this backend
    description: dict[str, Any] = Field(default_factory=dict)  # provider, model, max_tokens, ...
    usage: UsageSummary | None = None                    # None if the backend tracks no usage


class JudgeRecord(BaseModel):
    """Provenance record for one judge used in a run."""
    judge_id: str
    judge_class: str
    temperature: float
    on_parse_failure: str
    n_calibration_examples: int = 0
    backend_label: str | None = None


class RunMetadata(BaseModel):
    """What produced this report: enough to rerun it or explain a difference."""
    judge_kappa_version: str
    python_version: str
    packages: dict[str, str] = Field(default_factory=dict)   # installed SDK / numeric libs
    started_at: str                                          # ISO 8601, UTC
    finished_at: str
    duration_s: float
    judges: list[JudgeRecord] = Field(default_factory=list)
    backends: list[BackendRecord] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)   # evaluator-level settings


class VariantResult(BaseModel):
    """Generated output and aggregated score for one variant of one case."""
    variant_name: str
    output: str
    score: float
    token_count: int


class AgreementResult(BaseModel):
    """Inter-rater agreement statistics: alpha with interval, kappa, ICC, and counts."""
    kappa: float | None = None
    alpha: float | None = None
    alpha_ci_low: float | None = None   # bootstrap 95% CI lower bound
    alpha_ci_high: float | None = None  # bootstrap 95% CI upper bound
    icc: float | None = None            # ICC(2,k) absolute agreement
    icc_interpretation: str | None = None
    expected_chance_agreement: float | None = None
    alpha_interpretation: str | None = None
    n_judges: int = 0
    n_cases: int = 0
    # Why alpha / kappa are None. None here means the statistic was estimated. Krippendorff's alpha
    # needs at least one case scored by 2 or more judges; Cohen's kappa needs a judge pair sharing
    # 2 cases and is undefined when every compared pair used one and the same single category.
    alpha_not_estimated_reason: Literal["single_judge", "no_pairable_values"] | None = None
    kappa_not_estimated_reason: Literal["no_judge_pair", "no_score_variation"] | None = None


class JudgeFitResult(BaseModel):
    """Per-judge person-fit statistic — flags inconsistent judges."""
    judge_id: str
    lz_statistic: float          # standardised log-likelihood; |lz| > 1.96 → p < 0.05
    flagged_inconsistent: bool   # True when |lz| > 1.96


class BiasResult(BaseModel):
    """Bias diagnostics: positional flip rate and order effect, and verbosity correlation."""
    positional_bias_rate: float = 0.0          # fraction of cases with position flip
    # Order-aware BTL estimate of the judge's display-order effect (log-odds; > 0 favours
    # the first-shown response). None when no positional reports were available.
    order_effect: float | None = None
    order_effect_ci_low: float | None = None   # bootstrap interval; None for a single case
    order_effect_ci_high: float | None = None
    verbosity_bias_rho: float | None = None  # Spearman ρ(length, score)
    verbosity_bias_p: float | None = None
    verbosity_biased: bool = False
    # Verdicts with a measured output length (output_token_count > 0) used in the correlation,
    # and verdicts left out because no length was recorded.
    verbosity_n_measured: int = 0
    verbosity_n_unmeasured: int = 0
    # Why verbosity_bias_rho is None. None here means it was estimated. The correlation is
    # undefined when every score (or every length) is identical, and too noisy to report with
    # fewer than 3 measured verdicts. ``verbosity_biased`` is False in all of these cases:
    # check ``verbosity_bias_rho is None`` before reading it as "no bias".
    verbosity_not_estimated_reason: (
        Literal["too_few_measured", "constant_scores", "constant_lengths"] | None
    ) = None


class UpliftSignificance(BaseModel):
    """McNemar test result for treatment vs. control significance."""
    n_treatment_wins: int    # cases where treatment > control
    n_control_wins: int      # cases where control > treatment
    n_ties: int
    mcnemar_statistic: float
    p_value: float
    significant: bool        # p < 0.05
    uplift_ci_low: float     # bootstrap 95% CI lower bound for mean_uplift
    uplift_ci_high: float    # bootstrap 95% CI upper bound for mean_uplift


class CaseResult(BaseModel):
    """Everything measured for one case: both variant results, uplift, verdicts, agreement and
    positional flip.
    """
    case_id: str
    control: VariantResult
    treatment: VariantResult
    uplift: float
    verdicts: list[JudgeVerdict]
    agreement: AgreementResult = Field(default_factory=AgreementResult)
    positional_flip: bool = False


class EvalReport(BaseModel):
    """Result of an A/B evaluation.

    Per-case results, uplift, agreement, significance, judge fit, bias diagnostics, data quality
    and run provenance.
    """
    cases: list[CaseResult]
    mean_uplift: float
    mean_control_score: float
    mean_treatment_score: float
    agreement: AgreementResult
    significance: UpliftSignificance | None = None   # McNemar + bootstrap CI
    judge_fit: list[JudgeFitResult] = Field(default_factory=list)  # per-judge l_z
    bias: BiasResult
    judge_ids: list[str]
    scale_type: ScaleType
    data_quality: DataQuality = Field(default_factory=DataQuality)
    run: RunMetadata | None = None


# ── Pairwise pre-recorded evaluation output ───────────────────────────────────


class PairwiseCaseResult(BaseModel):
    """Result of judging one pre-recorded A/B pair."""
    case_id: str
    prompt: str
    output_a: str
    output_b: str
    label_a: str
    label_b: str
    score_a: float
    score_b: float
    preferred: str     # "A", "B", or "tie"
    rationale: str
    parse_failed: bool = False       # judge reply unusable; scores are the 0.5/0.5 fallback
    parse_error: str | None = None
    truncated: bool = False          # judge reply hit the output-token limit


class PairwiseReport(BaseModel):
    """Result of pairwise-dataset evaluation: preference rates, data quality and provenance."""
    cases: list[PairwiseCaseResult]
    mean_score_a: float
    mean_score_b: float
    preference_rate_a: float   # fraction of cases where A was preferred
    preference_rate_b: float
    tie_rate: float
    label_a: str
    label_b: str
    judge_id: str
    data_quality: DataQuality = Field(default_factory=DataQuality)
    run: RunMetadata | None = None
