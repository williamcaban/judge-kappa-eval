"""
Demo: reliability and reproducibility controls of the LLM layer.

Shows, with no API key, that
  1. an unusable judge reply is counted and visible instead of silently becoming a score,
  2. every report records which backends and request parameters produced it,
  3. an interrupted tournament resumes from a response cache and matches the clean run,
  4. concurrency changes the speed, not the result.

Run: python examples/demo_reliability_controls.py
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from judge_kappa import (
    Assertion,
    AssertionJudge,
    CachedBackend,
    EvalCase,
    JudgePanel,
    JuryEvaluator,
    PairwiseJudge,
    TournamentEvaluator,
    Variant,
)

QUALITY = {"alpha": 0.9, "beta": 0.6, "gamma": 0.3}


class MockPairwise:
    """Deterministic pairwise judge: scores each displayed output from a quality table."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self._fail_on_call = fail_on_call
        self._lock = threading.Lock()

    def describe(self) -> dict[str, object]:
        return {"provider": "mock", "model": "pairwise-quality-v1"}

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self._fail_on_call == n:
            raise RuntimeError("simulated provider outage")
        a = user.split("## Output A\n", 1)[1].split("\n\n## Output B\n", 1)[0]
        b = user.split("\n\n## Output B\n", 1)[1]
        return json.dumps({"score_a": min(1.0, QUALITY[a] + 0.1), "score_b": QUALITY[b],
                           "preferred": "A", "rationale": "mock"})


class MockAssertionJudge:
    """Answers with valid JSON, except every third reply, which is prose."""

    def __init__(self) -> None:
        self.n = 0

    def describe(self) -> dict[str, object]:
        return {"provider": "mock", "model": "assertion-v1"}

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.n += 1
        if self.n % 3 == 0:
            return "I think the answer looks fine overall."
        return json.dumps({"assertion_scores": {"is correct": 1.0}, "rationale": "ok"})


def cases(n: int) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", prompt=f"Prompt {i}") for i in range(n)]


def systems() -> dict[str, object]:
    return {name: (lambda _x, name=name: name) for name in QUALITY}


def main() -> None:
    print("1) Unusable judge replies are counted, not silent")
    judge = AssertionJudge("flaky-judge", MockAssertionJudge())
    ev = JuryEvaluator(
        panel=JudgePanel([judge]), generation_backend=MockAssertionJudge(),
        compute_significance=False, bootstrap_ci=False, compute_judge_fit=False,
    )
    data = [EvalCase(id=f"c{i}", prompt="Q", assertions=[Assertion(text="is correct")]) for i in range(6)]
    report = ev.evaluate(
        data,
        Variant(name="control", predict_fn=lambda _i: "plain"),
        Variant(name="treatment", predict_fn=lambda _i: "better"),
    )
    dq = report.data_quality
    print(f"   judge calls: {dq.n_judge_calls}  unusable: {dq.n_parse_failures}"
          f"  ({dq.parse_failure_rate:.0%})  by code: {dq.parse_errors}")
    for w in dq.warnings:
        print(f"   warning: {w[:110]}...")

    print("\n2) The report records what produced it (no secrets)")
    run = report.run
    assert run is not None
    for j in run.judges:
        print(f"   judge {j.judge_id}: {j.judge_class}, temperature={j.temperature}, "
              f"on_parse_failure={j.on_parse_failure}")
    for b in run.backends:
        calls = b.usage.calls if b.usage else "n/a"
        print(f"   backend {b.label} ({b.role}): usage calls={calls}")
    print(f"   judge-kappa {run.judge_kappa_version}, python {run.python_version}")

    print("\n3) An interrupted tournament resumes from the response cache")
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "tournament-cache.jsonl"
        reference_backend = MockPairwise()
        reference = TournamentEvaluator(PairwiseJudge("pj", reference_backend), n_bootstrap=0).run(
            cases(4), systems()
        )
        total = reference_backend.calls

        dying = MockPairwise(fail_on_call=10)
        try:
            TournamentEvaluator(PairwiseJudge("pj", CachedBackend(dying, cache)), n_bootstrap=0).run(
                cases(4), systems()
            )
        except RuntimeError as exc:
            print(f"   run 1 stopped after {dying.calls - 1} good calls: {exc}")

        healthy = MockPairwise()
        resumed = TournamentEvaluator(
            PairwiseJudge("pj", CachedBackend(healthy, cache)), n_bootstrap=0
        ).run(cases(4), systems())
        print(f"   run 2 made {healthy.calls} calls instead of {total} (the rest came from the cache)")
        same = [(s.system, s.wins, s.losses) for s in reference.standings] == [
            (s.system, s.wins, s.losses) for s in resumed.standings
        ]
        print(f"   resumed result identical to the uninterrupted run: {same}")

    print("\n4) Concurrency changes the speed, not the result")
    seq = TournamentEvaluator(PairwiseJudge("pj", MockPairwise()), n_bootstrap=0, max_workers=1).run(
        cases(6), systems()
    )
    con = TournamentEvaluator(PairwiseJudge("pj", MockPairwise()), n_bootstrap=0, max_workers=6).run(
        cases(6), systems()
    )
    print(f"   identical standings and win matrix: "
          f"{seq.standings == con.standings and seq.win_matrix == con.win_matrix}")


if __name__ == "__main__":
    main()
