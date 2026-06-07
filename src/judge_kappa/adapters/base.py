"""InputAdapter ABC — extend to support new input formats without touching judges."""

from __future__ import annotations

from abc import ABC, abstractmethod

from judge_kappa.models import EvalCase


class InputAdapter(ABC):
    @abstractmethod
    def load(self, source: object) -> list[EvalCase]:
        """Convert a source (file path, list, dict, …) into EvalCase list."""
        ...
