"""
jury_eval — Unified LLM evaluation bridging agent-skills-eval and MLflow LLMaJ.

Quick start:

  from jury_eval import JuryEvaluator, JudgePanel, AssertionJudge, AnthropicBackend
  from jury_eval.models import CalibrationExample, ScaleType, Variant

  backend = AnthropicBackend("claude-sonnet-4-6")
  panel   = JudgePanel(judges=[AssertionJudge("j1", backend)])
  ev      = JuryEvaluator(panel=panel, generation_backend=backend)
  report  = ev.evaluate_skill("./my-skill")
  print(report.mean_uplift, report.agreement.alpha)
"""

from jury_eval.evaluator import JuryEvaluator
from jury_eval.tournament import TournamentEvaluator, TournamentReport, TournamentStanding, PairResult
from jury_eval.models import (
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
from jury_eval.adapters import DatasetAdapter, InputAdapter, SkillAdapter
from jury_eval.judges import AssertionJudge, LLMJudge, PairwiseJudge, RubricJudge
from jury_eval.llm import AnthropicBackend, LLMBackend, OpenAIBackend
from jury_eval.panel import EvaluationPanel, JudgeJury, JudgePanel
from jury_eval.agreement import AgreementMetric, CohenKappa, KrippendorffAlpha
from jury_eval.bias import (
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
