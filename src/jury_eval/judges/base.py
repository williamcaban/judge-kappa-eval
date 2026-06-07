"""
LLMJudge ABC — defines the judge contract and ICL alignment mixin.

ICL (In-Context Learning) alignment: calibration_examples are injected into
every judge prompt as few-shot anchors. This reduces inter-judge score drift
by grounding all judges on the same human-validated reference points, which
directly improves Krippendorff's α and Cohen's κ in panel evaluations.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from jury_eval.llm.base import LLMBackend
from jury_eval.models import CalibrationExample, EvalCase, JudgeVerdict


class LLMJudge(ABC):
    def __init__(
        self,
        judge_id: str,
        backend: LLMBackend,
        calibration_examples: list[CalibrationExample] | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.judge_id = judge_id
        self._backend = backend
        self._calibration = calibration_examples or []
        self._temperature = temperature

    def _icl_block(self) -> str:
        """Render calibration examples as an ICL section for prompt injection."""
        if not self._calibration:
            return ""
        lines = ["## Calibration Anchors\n"
                 "Use these human-validated examples to anchor your scoring scale.\n"]
        for i, ex in enumerate(self._calibration, 1):
            lines.append(
                f"Example {i} | score={ex.score:.2f}\n"
                f"  Prompt: {ex.prompt}\n"
                f"  Output: {ex.output}\n"
                f"  Why: {ex.rationale}\n"
            )
        return "\n".join(lines)

    def _call(self, system: str, user: str) -> str:
        return self._backend.complete(system, user, self._temperature)

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = match.group(0) if match else text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": text}

    @abstractmethod
    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        """Score `output` for `case` and return a structured verdict."""
        ...
