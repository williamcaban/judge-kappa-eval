"""
AssertionJudge — agent-skills-eval-compatible scoring.

Each assertion in the EvalCase is scored PASS(1.0)/FAIL(0.0) by the judge.
Final score = weighted mean across assertions.
"""

from __future__ import annotations

import textwrap

from judge_kappa.judges.base import LLMJudge, numeric_scores, validate_score_map
from judge_kappa.models import EvalCase, JudgeVerdict

_SYSTEM = textwrap.dedent("""\
    You are a rigorous evaluation judge. You will be given an AI output and a list
    of assertions. For each assertion, decide whether the output SATISFIES it
    (score: 1.0) or FAILS it (score: 0.0). Be strict: partial satisfaction = 0.0.

    {icl_block}

    Reply ONLY as valid JSON — no prose, no markdown:
    {{
      "assertion_scores": {{"<assertion text>": <1.0 or 0.0>, ...}},
      "rationale": "<one sentence explaining your overall verdict>"
    }}
""")


class AssertionJudge(LLMJudge):
    """Scores each assertion PASS (1.0) or FAIL (0.0); the verdict is the weight-averaged result."""
    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        """Judge one output against the case's assertions.

        Unusable replies are flagged on the verdict (``parse_failed``) and handled per the judge's
        ``on_parse_failure`` policy.

        Raises:
            ValueError: If the case has no assertions.
        """
        if not case.assertions:
            raise ValueError(f"Case '{case.id}' has no assertions for AssertionJudge.")

        system = _SYSTEM.format(icl_block=self._icl_block())
        user = (
            f"## Prompt Given to the AI\n{case.prompt}\n\n"
            f"## AI Output\n{output}\n\n"
            f"## Assertions to Evaluate\n"
            + "\n".join(f"- {a.text}" for a in case.assertions)
        )
        expected = [a.text for a in case.assertions]
        reply = self._parse_reply(
            self._call_full(system, user),
            case.id,
            required=("assertion_scores",),
            validate=lambda d: validate_score_map(d["assertion_scores"], expected, "assertion"),
        )
        parsed = reply.data
        a_scores = numeric_scores(parsed.get("assertion_scores"))

        total_weight = sum(a.weight for a in case.assertions)
        score = (
            sum(a_scores.get(a.text, 0.0) * a.weight for a in case.assertions)
            / total_weight
            if total_weight > 0 else 0.0
        )

        return JudgeVerdict(
            judge_id=self.judge_id,
            eval_case_id=case.id,
            variant_name=variant_name,
            score=round(score, 4),
            rationale=str(parsed.get("rationale", parsed.get("_raw", ""))),
            assertion_scores=a_scores,
            output_token_count=len(output.split()),
            parse_failed=reply.failed,
            parse_error=reply.error,
            truncated=reply.response.truncated,
        )
