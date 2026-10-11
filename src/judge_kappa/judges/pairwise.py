"""
PairwiseJudge — compares two outputs (A vs B) in a single judge call.

Primary use: positional bias detection. Call judge_pair() twice with A/B swapped
to check whether the judge always prefers the first-presented output.

This judge does NOT inherit the standard judge() method because its output
is inherently dual — it scores both A and B in one prompt.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Any

from judge_kappa.judges.base import LLMJudge
from judge_kappa.models import EvalCase, JudgeVerdict

_SYSTEM = textwrap.dedent("""\
    You are a rigorous evaluation judge comparing two AI outputs (A and B) for
    the same prompt. Score each output independently from 0.0 to 1.0.
    Do NOT let the order in which they are presented influence your scores —
    evaluate content quality only.

    {icl_block}

    Reply ONLY as valid JSON — no prose, no markdown:
    {{
      "score_a": <0.0–1.0>,
      "score_b": <0.0–1.0>,
      "preferred": "A" | "B" | "tie",
      "rationale": "<one sentence>"
    }}
""")


def _as_score(value: object, default: float) -> float:
    """A usable numeric score, or ``default`` when the reply gave something else."""
    if isinstance(value, bool) or not isinstance(value, int | float) or math.isnan(value):
        return default
    return float(value)


def _validate_pair(data: dict[str, Any]) -> str | None:
    for key in ("score_a", "score_b"):
        v = data[key]
        if isinstance(v, bool) or not isinstance(v, int | float) or math.isnan(v):
            return "non_numeric_score"
        if v < 0.0 or v > 1.0:
            return "out_of_range_score"
    return None


@dataclass
class PairwiseScore:
    score_a: float
    score_b: float
    preferred: str  # "A", "B", or "tie"
    rationale: str


class PairwiseJudge(LLMJudge):
    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        raise NotImplementedError(
            "PairwiseJudge uses judge_pair(). Use AssertionJudge or RubricJudge "
            "for single-output scoring."
        )

    def judge_pair(
        self,
        case: EvalCase,
        output_a: str,
        output_b: str,
        label_a: str = "A",
        label_b: str = "B",
    ) -> tuple[JudgeVerdict, JudgeVerdict]:
        system = _SYSTEM.format(icl_block=self._icl_block())
        user = (
            f"## Prompt\n{case.prompt}\n\n"
            f"## Output A\n{output_a}\n\n"
            f"## Output B\n{output_b}"
        )
        reply = self._parse_reply(
            self._call_full(system, user),
            case.id,
            required=("score_a", "score_b"),
            validate=_validate_pair,
        )
        parsed = reply.data
        pairwise = PairwiseScore(
            score_a=_as_score(parsed.get("score_a"), 0.5),
            score_b=_as_score(parsed.get("score_b"), 0.5),
            preferred=str(parsed.get("preferred", "tie")),
            rationale=str(parsed.get("rationale", parsed.get("_raw", ""))),
        )
        flags: dict[str, Any] = {
            "parse_failed": reply.failed,
            "parse_error": reply.error,
            "truncated": reply.response.truncated,
        }

        verdict_a = JudgeVerdict(
            judge_id=self.judge_id,
            eval_case_id=case.id,
            variant_name=label_a,
            score=pairwise.score_a,
            rationale=pairwise.rationale,
            output_token_count=len(output_a.split()),
            **flags,
        )
        verdict_b = JudgeVerdict(
            judge_id=self.judge_id,
            eval_case_id=case.id,
            variant_name=label_b,
            score=pairwise.score_b,
            rationale=pairwise.rationale,
            output_token_count=len(output_b.split()),
            **flags,
        )
        return verdict_a, verdict_b
