"""
Demo: order-aware tournament ranking and display-order effect estimation.

Shows why scoring only the first display order is unsafe, and what the
order-aware BTL ratings report instead. No API key is needed: a deterministic
mock judge scores each output from a quality table and adds a bonus to
whichever output is displayed first (a position-biased but informative judge).

Run: python examples/demo_order_aware_ranking.py
"""

from __future__ import annotations

import json
import random
import zlib

from judge_kappa import EvalCase, PairwiseJudge, TournamentEvaluator

QUALITY = {"alpha": 0.9, "beta": 0.6, "gamma": 0.3}   # true ordering: alpha > beta > gamma
FIRST_POSITION_BONUS = 0.35                             # judge favours the first-shown output
SCORE_NOISE = 0.2                                       # +/- uniform noise per score (real judges are noisy)


class PositionBiasedMockJudge:
    """Implements the LLMBackend protocol without any network call.

    Noise is seeded from the prompt text, so runs are reproducible.
    """

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:  # noqa: ARG002
        a = user.split("## Output A\n", 1)[1].split("\n\n## Output B\n", 1)[0]
        b = user.split("\n\n## Output B\n", 1)[1]
        rng = random.Random(zlib.crc32(user.encode()))
        score_a = min(1.0, max(0.0, QUALITY[a] + FIRST_POSITION_BONUS + rng.uniform(-SCORE_NOISE, SCORE_NOISE)))
        score_b = min(1.0, max(0.0, QUALITY[b] + rng.uniform(-SCORE_NOISE, SCORE_NOISE)))
        return json.dumps({"score_a": score_a, "score_b": score_b, "preferred": "A" if score_a > score_b else "B",
                           "rationale": "mock"})


def main() -> None:
    cases = [EvalCase(id=f"case-{i}", prompt=f"Prompt {i}") for i in range(30)]
    # Deliberately listed worst-first: with one display order only, listing order decides ties.
    systems = {"gamma": lambda _: "gamma", "beta": lambda _: "beta", "alpha": lambda _: "alpha"}
    judge = PairwiseJudge("mock-judge", PositionBiasedMockJudge())  # type: ignore[arg-type]

    print("True quality order: alpha > beta > gamma\n")

    print("1) One display order only (detect_positional_bias=False):")
    single = TournamentEvaluator(judge, detect_positional_bias=False, n_bootstrap=0).run(cases, systems)
    single.print_leaderboard()

    print("\n2) Both display orders, order-aware BTL ratings (default):")
    both = TournamentEvaluator(judge, detect_positional_bias=True, n_bootstrap=300).run(cases, systems)
    both.print_leaderboard()

    print("\nOrder effect > 0 means the judge favours whichever response is shown first.")
    print("Its bootstrap interval excludes 0 here, so rankings from a single display order should not be trusted.")


if __name__ == "__main__":
    main()
