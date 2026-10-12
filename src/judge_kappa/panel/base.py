"""EvaluationPanel ABC — swap between Panel and Jury without changing the evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod

from judge_kappa.judges.base import LLMJudge
from judge_kappa.models import EvalCase, JudgeVerdict


class EvaluationPanel(ABC):
    """Abstract base for combining judges; swap a panel for a jury without changing the evaluator.
    """
    @abstractmethod
    def evaluate(
        self, case: EvalCase, output: str, variant_name: str
    ) -> list[JudgeVerdict]:
        """Run all judges and return their individual verdicts."""
        ...

    @abstractmethod
    def aggregate_score(self, verdicts: list[JudgeVerdict]) -> float:
        """Collapse per-judge verdicts to a single score."""
        ...

    @property
    @abstractmethod
    def judge_ids(self) -> list[str]:
        """Ids of the panel's judges, in evaluation order."""
        ...

    @property
    def judges(self) -> list[LLMJudge]:
        """The panel's judge objects (used for run provenance). Custom panels may override."""
        return []
