"""
RankJudge — listwise ranking for N ≥ 7 systems.

In a pairwise tournament with N systems, the judge call count is C(N,2) × cases × 2
(for positional bias detection). For N = 7 this is 42 calls per case; for N = 10
it reaches 90 calls per case. The RankJudge provides an O(N) alternative:
all N outputs are presented in a single prompt and ranked in one call.

Rank → score conversion: score_i = (N - rank_i) / (N - 1), so rank 1 → 1.0,
rank N → 0.0. This preserves ordinal information while producing JudgeVerdicts
compatible with the standard AgreementResult / EvalReport pipeline.

Limitations vs. pairwise:
  - Context-window pressure grows with N and output length.
  - No per-pair positional bias detection (positional bias shows as rank 1 always winning).
  - Recommended for N ≥ 7; for N ≤ 6 prefer TournamentEvaluator for accuracy.

Usage:
    judge = RankJudge("rank-j1", backend)
    ranked = judge.rank(case, {"system_a": output_a, "system_b": output_b, ...})
    # Returns dict[system_name → JudgeVerdict] keyed by system name
"""

from __future__ import annotations

import textwrap
from typing import Any

from judge_kappa.judges.base import LLMJudge
from judge_kappa.models import EvalCase, JudgeVerdict

_SYSTEM = textwrap.dedent("""\
    You are a rigorous evaluation judge. You will receive a prompt and multiple AI
    outputs labelled A, B, C, … (in random order). Rank them from best (rank 1) to
    worst (rank N). Base your ranking on response quality, not on the label letter.

    {icl_block}

    Reply ONLY as valid JSON — no prose, no markdown:
    {{
      "rankings": {{"A": <rank 1–N>, "B": <rank 1–N>, ...}},
      "rationale": "<one sentence summarising the ranking rationale>"
    }}

    All N labels must appear in rankings. Ranks must be unique integers 1..N.
""")


class RankJudge(LLMJudge):
    """
    Rank N outputs in a single prompt.

    Args:
        max_systems: Hard limit on number of systems per call. Raises ValueError
            if exceeded. Default 12. Above ~15 outputs, context quality degrades.
    """

    def __init__(self, *args: Any, max_systems: int = 12, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._max_systems = max_systems

    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        raise NotImplementedError(
            "RankJudge.rank() takes a dict of {system_name: output}. "
            "Use AssertionJudge or RubricJudge for single-output scoring."
        )

    def rank(
        self,
        case: EvalCase,
        system_outputs: dict[str, str],
    ) -> dict[str, JudgeVerdict]:
        """
        Rank all systems in a single call.

        Args:
            case: The evaluation case (prompt + metadata).
            system_outputs: {system_name: output_text}. Order is randomised
                internally to avoid position bias — the judge sees labels A, B, C.

        Returns:
            dict[system_name → JudgeVerdict] with scores normalised to 0–1.
        """
        if len(system_outputs) < 2:
            raise ValueError("RankJudge.rank() requires at least 2 systems.")
        if len(system_outputs) > self._max_systems:
            raise ValueError(
                f"RankJudge.rank() received {len(system_outputs)} systems but "
                f"max_systems={self._max_systems}. Increase max_systems or "
                "use TournamentEvaluator for large N."
            )

        names = list(system_outputs.keys())
        labels = [chr(ord("A") + i) for i in range(len(names))]
        label_to_name = dict(zip(labels, names, strict=True))

        outputs_section = "\n\n".join(
            f"## Output {label}\n{system_outputs[name]}"
            for label, name in zip(labels, names, strict=True)
        )
        system = _SYSTEM.format(icl_block=self._icl_block())
        user = f"## Prompt\n{case.prompt}\n\n{outputs_section}"

        parsed = self._parse_json(self._call(system, user))
        rankings_raw = parsed.get("rankings") or {}
        raw_rankings: dict[str, int] = {k: int(v) for k, v in rankings_raw.items()}
        rationale: str = str(parsed.get("rationale", parsed.get("_raw", "")))

        # Normalise: rank 1 → score 1.0, rank N → score 0.0
        n = len(names)
        verdicts: dict[str, JudgeVerdict] = {}
        for label, name in label_to_name.items():
            rank = int(raw_rankings.get(label, n))
            rank = max(1, min(rank, n))
            score = (n - rank) / max(n - 1, 1)
            verdicts[name] = JudgeVerdict(
                judge_id=self.judge_id,
                eval_case_id=case.id,
                variant_name=name,
                score=round(score, 4),
                rationale=rationale,
                output_token_count=len(system_outputs[name].split()),
            )
        return verdicts
