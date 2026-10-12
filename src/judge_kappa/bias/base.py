"""BiasDetector ABC — extend for new bias types without touching the evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from judge_kappa.models import BiasResult


class BiasDetector(ABC):
    """Abstract base for bias detectors.

    Implement ``detect()`` and return a ``BiasResult``; the evaluator needs no changes to use a
    new detector.
    """
    @abstractmethod
    def detect(self, **kwargs: Any) -> BiasResult:
        """Run the detector.

        Keyword arguments are detector-specific (for example ``verdicts=`` or ``reports=``).

        Returns:
            A ``BiasResult`` with the fields this detector fills in.
        """
        ...
