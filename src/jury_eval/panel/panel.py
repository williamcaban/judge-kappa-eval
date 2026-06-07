"""
JudgePanel — homogeneous rubric, multiple judges.

Use when you want to measure inter-rater agreement: all judges share the same
rubric/assertions. High Krippendorff's α means the rubric is unambiguous.
Low α is actionable: add more ICL calibration examples or tighten the rubric.

Aggregation strategies:
  MEAN          — equal weight (baseline)
  WEIGHTED_MEAN — weight judges by a calibration quality score you provide
  MEDIAN        — robust to one outlier judge
  TRIMMED_MEAN  — drops top/bottom 10% of scores; handles systematic outliers
  MAJORITY_VOTE — binary pass/fail (score ≥ 0.5 → pass)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from jury_eval.judges.base import LLMJudge
from jury_eval.models import AggregationStrategy, EvalCase, JudgeVerdict
from jury_eval.panel.base import EvaluationPanel


class JudgePanel(EvaluationPanel):
    def __init__(
        self,
        judges: list[LLMJudge],
        strategy: AggregationStrategy = AggregationStrategy.MEAN,
        weights: list[float] | None = None,
    ) -> None:
        if not judges:
            raise ValueError("JudgePanel requires at least one judge.")
        self._judges = judges
        self._strategy = strategy
        self._weights = weights or [1.0] * len(judges)

    @property
    def judge_ids(self) -> list[str]:
        return [j.judge_id for j in self._judges]

    def evaluate(self, case: EvalCase, output: str, variant_name: str) -> list[JudgeVerdict]:
        return [j.judge(case, output, variant_name) for j in self._judges]

    def aggregate_score(self, verdicts: list[JudgeVerdict]) -> float:
        scores = [v.score for v in verdicts]
        w = self._weights[: len(scores)]
        match self._strategy:
            case AggregationStrategy.MEAN:
                return float(np.mean(scores))
            case AggregationStrategy.WEIGHTED_MEAN:
                return float(np.average(scores, weights=w))
            case AggregationStrategy.MEDIAN:
                return float(np.median(scores))
            case AggregationStrategy.TRIMMED_MEAN:
                return float(stats.trim_mean(scores, proportiontocut=0.1))
            case AggregationStrategy.MAJORITY_VOTE:
                passes = sum(1 for s in scores if s >= 0.5)
                return 1.0 if passes > len(scores) / 2 else 0.0
            case _:
                return float(np.mean(scores))
