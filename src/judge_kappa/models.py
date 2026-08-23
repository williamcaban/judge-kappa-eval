"""
Stable domain models — the schema contract for the entire package.

All cross-module data passes through these Pydantic models.
Adding fields here is additive (backward compatible); removing or renaming is breaking.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ScaleType(StrEnum):
    """Measurement scale — governs which Krippendorff distance function is used."""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class AggregationStrategy(StrEnum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAJORITY_VOTE = "majority_vote"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


# ── Input models ──────────────────────────────────────────────────────────────


class Assertion(BaseModel):
    text: str
    weight: float = 1.0


class RubricDimension(BaseModel):
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
        arbitrary_types_allowed = True


# ── Output models ─────────────────────────────────────────────────────────────


class JudgeVerdict(BaseModel):
    judge_id: str
    eval_case_id: str
    variant_name: str
    score: float                                            # 0.0–1.0
    rationale: str
    assertion_scores: dict[str, float] = Field(default_factory=dict)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    output_token_count: int = 0


class VariantResult(BaseModel):
    variant_name: str
    output: str
    score: float
    token_count: int


class AgreementResult(BaseModel):
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


class JudgeFitResult(BaseModel):
    """Per-judge person-fit statistic — flags inconsistent judges."""
    judge_id: str
    lz_statistic: float          # standardised log-likelihood; |lz| > 1.96 → p < 0.05
    flagged_inconsistent: bool   # True when |lz| > 1.96


class BiasResult(BaseModel):
    positional_bias_rate: float = 0.0          # fraction of cases with position flip
    verbosity_bias_rho: float | None = None  # Spearman ρ(length, score)
    verbosity_bias_p: float | None = None
    verbosity_biased: bool = False


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
    case_id: str
    control: VariantResult
    treatment: VariantResult
    uplift: float
    verdicts: list[JudgeVerdict]
    agreement: AgreementResult = Field(default_factory=AgreementResult)
    positional_flip: bool = False


class EvalReport(BaseModel):
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


# ── Pairwise pre-recorded evaluation output ───────────────────────────────────


class PairwiseCaseResult(BaseModel):
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


class PairwiseReport(BaseModel):
    cases: list[PairwiseCaseResult]
    mean_score_a: float
    mean_score_b: float
    preference_rate_a: float   # fraction of cases where A was preferred
    preference_rate_b: float
    tie_rate: float
    label_a: str
    label_b: str
    judge_id: str
