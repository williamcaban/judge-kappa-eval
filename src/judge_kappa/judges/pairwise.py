"""
PairwiseJudge — compares two outputs (A vs B) in a single judge call.

Primary use: positional bias detection. Call judge_pair() twice with A/B swapped
to check whether the judge always prefers the first-presented output.

This judge does NOT inherit the standard judge() method because its output
is inherently dual — it scores both A and B in one prompt.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

from judge_kappa.judges.base import LLMJudge, is_finite_number
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
    return float(value) if is_finite_number(value) else default


def _validate_pair(data: dict[str, Any]) -> str | None:
    """Error code if ``score_a`` / ``score_b`` are not finite numbers in [0, 1], else None."""
    for key in ("score_a", "score_b"):
        v = data[key]
        if not is_finite_number(v):
            return "non_numeric_score"
        if v < 0.0 or v > 1.0:
            return "out_of_range_score"
    return None


@dataclass
class PairwiseScore:
    """Parsed pairwise reply: scores for A and B, preferred label, rationale."""
    score_a: float
    score_b: float
    preferred: str  # "A", "B", or "tie"
    rationale: str


class PairwiseJudge(LLMJudge):
    """Compares two outputs in a single prompt; use ``judge_pair()``.

    The basis of the positional-bias swap test and of tournaments.
    """
    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        """Not supported: raises ``NotImplementedError``; use ``judge_pair``."""
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
        """Judge outputs A and B together.

        Args:
            case: The evaluation case.
            output_a: First output.
            output_b: Second output.
            label_a: Variant name recorded on A's verdict.
            label_b: Variant name recorded on B's verdict.

        Returns:
            ``(verdict_a, verdict_b)`` with scores in [0, 1]. An unusable reply yields a flagged
            0.5 / 0.5 pair.
        """
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
