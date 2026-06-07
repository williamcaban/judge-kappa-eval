"""AgreementMetric ABC — add new metrics without touching the panel or evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jury_eval.models import AgreementResult, JudgeVerdict, ScaleType


class AgreementMetric(ABC):
    @abstractmethod
    def compute(
        self,
        verdicts: list[JudgeVerdict],
        scale_type: ScaleType = ScaleType.ORDINAL,
    ) -> AgreementResult:
        ...
