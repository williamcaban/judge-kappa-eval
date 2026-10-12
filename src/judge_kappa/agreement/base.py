"""AgreementMetric ABC — add new metrics without touching the panel or evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod

from judge_kappa.models import AgreementResult, JudgeVerdict, ScaleType


class AgreementMetric(ABC):
    """Abstract base for inter-rater reliability statistics over judge verdicts.

    Implement ``compute()`` to add a new statistic.
    """
    @abstractmethod
    def compute(
        self,
        verdicts: list[JudgeVerdict],
        scale_type: ScaleType = ScaleType.ORDINAL,
    ) -> AgreementResult:
        """Compute the agreement statistic.

        Args:
            verdicts: One verdict per (judge, case).
            scale_type: Measurement scale of the scores.

        Returns:
            An ``AgreementResult`` with the fields this metric fills in.
        """
        ...
