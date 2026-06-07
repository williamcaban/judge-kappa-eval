"""
Stable domain models — the schema contract for the entire package.

All cross-module data passes through these Pydantic models.
Adding fields here is additive (backward compatible); removing or renaming is breaking.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class ScaleType(str, Enum):
    """Measurement scale — governs which Krippendorff distance function is used."""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class AggregationStrategy(str, Enum):
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
    expected_output: Optional[str] = None
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
    system_prompt: Optional[str] = None
    skill_context: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.0
    predict_fn: Optional[Callable[..., str]] = None
    generation_backend: Optional[Any] = None  # LLMBackend; typed as Any to avoid circular import

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
    kappa: Optional[float] = None
    alpha: Optional[float] = None
    expected_chance_agreement: Optional[float] = None
    alpha_interpretation: Optional[str] = None
    n_judges: int = 0
    n_cases: int = 0


class BiasResult(BaseModel):
    positional_bias_rate: float = 0.0          # fraction of cases with position flip
    verbosity_bias_rho: Optional[float] = None  # Spearman ρ(length, score)
    verbosity_bias_p: Optional[float] = None
    verbosity_biased: bool = False


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
