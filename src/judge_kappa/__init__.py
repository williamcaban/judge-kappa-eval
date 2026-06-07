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

from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.tournament import TournamentEvaluator, TournamentReport, TournamentStanding, PairResult
from judge_kappa.models import (
    AggregationStrategy,
    AgreementResult,
    Assertion,
    BiasResult,
    CalibrationExample,
    CaseResult,
    EvalCase,
    EvalReport,
    JudgeVerdict,
    PairwiseCaseResult,
    PairwiseReport,
    RubricDimension,
    ScaleType,
    Variant,
    VariantResult,
)
from judge_kappa.adapters import DatasetAdapter, InputAdapter, SkillAdapter
from judge_kappa.judges import AssertionJudge, LLMJudge, PairwiseJudge, RubricJudge
from judge_kappa.llm import AnthropicBackend, LLMBackend, OpenAIBackend
from judge_kappa.panel import EvaluationPanel, JudgeJury, JudgePanel
from judge_kappa.agreement import AgreementMetric, CohenKappa, KrippendorffAlpha
from judge_kappa.bias import (
    BiasDetector,
    PositionalBiasDetector,
    PositionalBiasReport,
    VerbosityBiasDetector,
)

__all__ = [
    # Evaluator
    "JuryEvaluator",
    # Models
    "AggregationStrategy", "AgreementResult", "Assertion", "BiasResult",
    "CalibrationExample", "CaseResult", "EvalCase", "EvalReport",
    "JudgeVerdict", "PairwiseCaseResult", "PairwiseReport",
    "RubricDimension", "ScaleType", "Variant", "VariantResult",
    # Adapters
    "InputAdapter", "SkillAdapter", "DatasetAdapter",
    # Judges
    "LLMJudge", "AssertionJudge", "RubricJudge", "PairwiseJudge",
    # LLM backends
    "LLMBackend", "OpenAIBackend", "AnthropicBackend",
    # Panels
    "EvaluationPanel", "JudgePanel", "JudgeJury",
    # Agreement
    "AgreementMetric", "CohenKappa", "KrippendorffAlpha",
    # Bias
    "BiasDetector", "PositionalBiasDetector", "PositionalBiasReport",
    "VerbosityBiasDetector",
]
