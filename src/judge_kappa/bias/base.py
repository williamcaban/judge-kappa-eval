"""BiasDetector ABC — extend for new bias types without touching the evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from judge_kappa.models import BiasResult


class BiasDetector(ABC):
    @abstractmethod
    def detect(self, **kwargs: Any) -> BiasResult:
        ...
