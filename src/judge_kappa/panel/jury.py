"""
JudgeJury — diverse judges with different rubrics, prompts, and models.

Design philosophy: reduce single-judge bias by assembling judges with different
perspectives. Each juror is a (judge, weight) pair. Low Krippendorff's α here
is expected and healthy — diverse rubrics should disagree. The statistic serves
as a jury-diversity diagnostic, not a quality gate.

Aggregation: WEIGHTED_MEAN by default, giving more influence to judges you
trust more (e.g., a specialized safety judge can carry higher weight).
"""

from __future__ import annotations

import numpy as np

from judge_kappa.judges.base import LLMJudge
from judge_kappa.models import AggregationStrategy, EvalCase, JudgeVerdict
from judge_kappa.panel.base import EvaluationPanel


class JudgeJury(EvaluationPanel):
    def __init__(
        self,
        jurors: list[tuple[LLMJudge, float]],
        strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN,
    ) -> None:
        if not jurors:
            raise ValueError("JudgeJury requires at least one juror.")
        self._jurors = jurors
        self._strategy = strategy

    @property
    def judge_ids(self) -> list[str]:
        return [j.judge_id for j, _ in self._jurors]

    def evaluate(self, case: EvalCase, output: str, variant_name: str) -> list[JudgeVerdict]:
        return [j.judge(case, output, variant_name) for j, _ in self._jurors]

    def aggregate_score(self, verdicts: list[JudgeVerdict]) -> float:
        scores  = [v.score for v in verdicts]
        weights = [w for _, w in self._jurors[: len(scores)]]
        match self._strategy:
            case AggregationStrategy.WEIGHTED_MEAN:
                return float(np.average(scores, weights=weights))
            case AggregationStrategy.MEAN:
                return float(np.mean(scores))
            case AggregationStrategy.MEDIAN:
                return float(np.median(scores))
            case _:
                return float(np.average(scores, weights=weights))
