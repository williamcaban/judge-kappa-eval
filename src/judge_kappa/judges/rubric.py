"""
RubricJudge — MLflow LLMaJ-compatible scoring.

Each RubricDimension is scored 0.0–1.0. Final score = weighted mean.
Expected output (when provided) is included in the prompt for grounding.
"""

from __future__ import annotations

import textwrap
from typing import Any

from judge_kappa.judges.base import LLMJudge, numeric_scores, validate_score_map
from judge_kappa.models import EvalCase, JudgeVerdict, RubricDimension

_SYSTEM = textwrap.dedent("""\
    You are a rigorous evaluation judge. Score the AI output on each rubric
    dimension from 0.0 (completely fails the criterion) to 1.0 (fully satisfies it).
    Use the full 0.0–1.0 range; do not cluster scores near 0.5.

    {icl_block}

    Reply ONLY as valid JSON — no prose, no markdown:
    {{
      "dimension_scores": {{"<dimension name>": <0.0–1.0>, ...}},
      "rationale": "<one sentence explaining your overall verdict>"
    }}
""")


class RubricJudge(LLMJudge):
    """Scores each rubric dimension 0.0-1.0 (MLflow LLMaJ style); the verdict is the weight-averaged
    score.
    """
    def __init__(self, *args: Any, rubric: list[RubricDimension], **kwargs: Any) -> None:
        """Create a rubric judge.

        Args:
            rubric: The dimensions to score; other arguments are those of ``LLMJudge``.
        """
        super().__init__(*args, **kwargs)
        self._rubric = rubric

    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        """Judge one output against the rubric.

        A missing or invalid dimension score is flagged on the verdict and scored 0.0 (or
        raises, per ``on_parse_failure``).
        """
        system = _SYSTEM.format(icl_block=self._icl_block())

        dims_text = "\n".join(
            f"- {d.name} (weight={d.weight}): {d.description}"
            + (f"\n  anchors → {' | '.join(d.anchors)}" if d.anchors else "")
            for d in self._rubric
        )
        user = (
            f"## Prompt Given to the AI\n{case.prompt}\n\n"
            f"## AI Output\n{output}\n\n"
            f"## Rubric Dimensions\n{dims_text}"
            + (f"\n\n## Reference / Expected Output\n{case.expected_output}"
               if case.expected_output else "")
        )
        expected = [d.name for d in self._rubric]
        reply = self._parse_reply(
            self._call_full(system, user),
            case.id,
            required=("dimension_scores",),
            validate=lambda d: validate_score_map(d["dimension_scores"], expected, "dimension"),
        )
        parsed = reply.data
        d_scores = numeric_scores(parsed.get("dimension_scores"))

        total_weight = sum(d.weight for d in self._rubric)
        score = (
            sum(d_scores.get(d.name, 0.0) * d.weight for d in self._rubric)
            / total_weight
            if total_weight > 0 else 0.0
        )

        return JudgeVerdict(
            judge_id=self.judge_id,
            eval_case_id=case.id,
            variant_name=variant_name,
            score=round(score, 4),
            rationale=str(parsed.get("rationale", parsed.get("_raw", ""))),
            dimension_scores=d_scores,
            output_token_count=len(output.split()),
            parse_failed=reply.failed,
            parse_error=reply.error,
            truncated=reply.response.truncated,
        )
