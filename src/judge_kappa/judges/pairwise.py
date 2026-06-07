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
        parsed = self._parse_json(self._call(system, user))
        pairwise = PairwiseScore(
            score_a=float(parsed.get("score_a", 0.5)),
            score_b=float(parsed.get("score_b", 0.5)),
            preferred=parsed.get("preferred", "tie"),
            rationale=parsed.get("rationale", parsed.get("_raw", "")),
        )

        verdict_a = JudgeVerdict(
            judge_id=self.judge_id,
            eval_case_id=case.id,
            variant_name=label_a,
            score=pairwise.score_a,
            rationale=pairwise.rationale,
            output_token_count=len(output_a.split()),
        )
        verdict_b = JudgeVerdict(
            judge_id=self.judge_id,
            eval_case_id=case.id,
            variant_name=label_b,
            score=pairwise.score_b,
            rationale=pairwise.rationale,
            output_token_count=len(output_b.split()),
        )
        return verdict_a, verdict_b
