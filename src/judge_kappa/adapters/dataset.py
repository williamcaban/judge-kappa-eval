"""
DatasetAdapter — MLflow LLMaJ-compatible input.
Accepts list[{"inputs": {...}, "expectations": {...}}].
"""

from __future__ import annotations

import json

from judge_kappa.adapters.base import InputAdapter
from judge_kappa.models import EvalCase


class DatasetAdapter(InputAdapter):
    def load(self, source: object) -> list[EvalCase]:
        data: list[dict] = source  # type: ignore[assignment]
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
