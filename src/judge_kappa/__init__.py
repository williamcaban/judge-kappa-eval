"""
judge_kappa — Unified LLM evaluation bridging agent-skills-eval and MLflow LLMaJ.

Quick start:

  from judge_kappa import JuryEvaluator, JudgePanel, AssertionJudge, AnthropicBackend
  from judge_kappa.models import CalibrationExample, ScaleType, Variant

  backend = AnthropicBackend("claude-sonnet-4-6")
  panel   = JudgePanel(judges=[AssertionJudge("j1", backend)])
  ev      = JuryEvaluator(panel=panel, generation_backend=backend)
  report  = ev.evaluate_skill("./my-skill")
  print(report.mean_uplift, report.agreement.alpha)
"""

from judge_kappa.adapters import DatasetAdapter, InputAdapter, SkillAdapter
from judge_kappa.agreement import (
    AgreementMetric,
    BehavioralAlignmentMetric,
    CohenKappa,
    KrippendorffAlpha,
    PersonFitAnalyzer,
)
from judge_kappa.bias import (
    BiasDetector,
    DifferentialItemFunctioningDetector,
    OrderEffectEstimate,
    PositionalBiasDetector,
    PositionalBiasReport,
    VerbosityBiasDetector,
)
from judge_kappa.calibration import IRTJudgeWeighter
from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges import AssertionJudge, LLMJudge, PairwiseJudge, RankJudge, RubricJudge
from judge_kappa.judges.base import JudgeParseError
from judge_kappa.llm import (
    AnthropicBackend,
    CachedBackend,
    LLMBackend,
    LLMResponse,
    OpenAIBackend,
)
from judge_kappa.models import (
    AggregationStrategy,
    AgreementResult,
    Assertion,
    BackendRecord,
    BiasResult,
    CalibrationExample,
    CaseResult,
    DataQuality,
    EvalCase,
    EvalReport,
    JudgeFitResult,
    JudgeRecord,
    JudgeVerdict,
    PairwiseCaseResult,
    PairwiseReport,
    RubricDimension,
    RunMetadata,
    ScaleType,
    UpliftSignificance,
    UsageSummary,
    Variant,
    VariantResult,
)
from judge_kappa.panel import EvaluationPanel, JudgeJury, JudgePanel
from judge_kappa.ranking import BTLBootstrap, BTLFit, ComparisonGame, bootstrap_btl, fit_btl
from judge_kappa.tournament import (
    PairResult,
    TournamentEvaluator,
    TournamentReport,
    TournamentStanding,
)

__all__ = [
    # Evaluator
    "JuryEvaluator",
    # Models
    "AggregationStrategy", "AgreementResult", "Assertion", "BackendRecord", "BiasResult",
    "CalibrationExample", "CaseResult", "DataQuality", "EvalCase", "EvalReport",
    "JudgeFitResult", "JudgeRecord", "JudgeVerdict", "PairwiseCaseResult", "PairwiseReport",
    "RubricDimension", "RunMetadata", "ScaleType", "UpliftSignificance", "UsageSummary",
    "Variant", "VariantResult",
    # Adapters
    "InputAdapter", "SkillAdapter", "DatasetAdapter",
    # Judges
    "LLMJudge", "AssertionJudge", "RubricJudge", "PairwiseJudge", "RankJudge", "JudgeParseError",
    # LLM backends
    "LLMBackend", "OpenAIBackend", "AnthropicBackend", "CachedBackend", "LLMResponse",
    # Panels
    "EvaluationPanel", "JudgePanel", "JudgeJury",
    # Agreement
    "AgreementMetric", "BehavioralAlignmentMetric", "CohenKappa",
    "KrippendorffAlpha", "PersonFitAnalyzer",
    # Bias
    "BiasDetector", "DifferentialItemFunctioningDetector",
    "OrderEffectEstimate", "PositionalBiasDetector", "PositionalBiasReport",
    "VerbosityBiasDetector",
    # Calibration
    "IRTJudgeWeighter",
    # Ranking
    "BTLBootstrap", "BTLFit", "ComparisonGame", "bootstrap_btl", "fit_btl",
    # Tournament
    "TournamentEvaluator", "TournamentReport", "TournamentStanding", "PairResult",
]
