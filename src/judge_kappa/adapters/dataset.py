"""
DatasetAdapter — MLflow LLMaJ-compatible input.
Accepts list[{"inputs": {...}, "expectations": {...}}].
"""

from __future__ import annotations

import json
from typing import Any

from judge_kappa.adapters.base import InputAdapter
from judge_kappa.models import EvalCase


class DatasetAdapter(InputAdapter):
    """Adapter for MLflow-LLMaJ-style datasets: a list of
    ``{"id", "inputs", "expectations"}`` rows.
    """
    def load(self, source: object) -> list[EvalCase]:
        """Convert dataset rows into ``EvalCase`` objects.

        The prompt comes from ``inputs["question"]``, then ``inputs["prompt"]``, otherwise the
        JSON-serialised inputs. ``expectations["answer"]`` (or ``expected_answer``)
        becomes ``expected_output``. Rows without an ``id`` get ``case-NNNN``.

        Args:
            source: A list of row dicts.

        Returns:
            One ``EvalCase`` per row; the raw inputs and expectations are kept in ``metadata``.
        """
        data: list[dict[str, Any]] = source  # type: ignore[assignment]
        cases: list[EvalCase] = []
        for i, row in enumerate(data):
            inputs = row.get("inputs", {})
            exp = row.get("expectations", {})
            prompt = (
                inputs.get("question")
                or inputs.get("prompt")
                or json.dumps(inputs)
            )
            cases.append(
                EvalCase(
                    id=row.get("id", f"case-{i:04d}"),
                    prompt=str(prompt),
                    expected_output=(
                        exp.get("answer") or exp.get("expected_answer")
                    ),
                    metadata={"inputs": inputs, "expectations": exp},
                )
            )
        return cases
