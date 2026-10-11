"""
Regression tests for the findings of the automated review of the LLM-layer PR.

Each class is one finding. Every finding has at least one test that failed against the
pre-fix source (verified by running these tests with the fixes stashed: 19 of 26 failed);
the rest are controls that pin behaviour which must not change.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from judge_kappa.bias.positional import PositionalBiasDetector
from judge_kappa.concurrency import ordered_map
from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.judges.rank import RankJudge
from judge_kappa.llm.cache import CachedBackend
from judge_kappa.llm.types import LLMResponse, sanitize_url
from judge_kappa.models import Assertion, EvalCase
from judge_kappa.panel.panel import JudgePanel
from judge_kappa.tournament import TournamentEvaluator

ROOT = Path(__file__).resolve().parent.parent


class _Plain:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return "x"


# ── tournament: exclusion must not break the report or the means ─────────────────────────


class _FailsOnQ1:
    """Valid pairwise reply except for prompt 'Q1' (unusable)."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        prompt = user.split("## Prompt\n", 1)[1].split("\n\n## Output A", 1)[0]
        if prompt == "Q1":
            return "unparseable"
        return json.dumps({"score_a": 1.0, "score_b": 0.0, "preferred": "A", "rationale": "r"})


def _two_cases() -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", prompt=f"Q{i}") for i in range(2)]


class TestTournamentExclusion:
    def _run(self, **kw: Any):
        return TournamentEvaluator(
            PairwiseJudge("p", _FailsOnQ1()), exclude_parse_failures=True, **kw
        ).run(_two_cases(), {"a": lambda _: "a", "b": lambda _: "b"})

    def test_one_usable_case_no_longer_crashes_the_bootstrap(self):
        report = self._run(n_bootstrap=50)        # used to raise: bootstrap needs 2 cases
        assert report.ci_level is None
        assert all(s.btl_ci_low is None and s.btl_ci_high is None for s in report.standings)
        assert any("intervals omitted" in w for w in report.data_quality.warnings)
        assert report.data_quality.n_parse_failures == 2

    def test_two_usable_cases_still_get_intervals(self):
        cases = [EvalCase(id=f"c{i}", prompt=f"Q{i}") for i in (0, 2, 3)]
        report = TournamentEvaluator(
            PairwiseJudge("p", _FailsOnQ1()), exclude_parse_failures=True, n_bootstrap=50
        ).run(cases, {"a": lambda _: "a", "b": lambda _: "b"})
        assert report.ci_level is not None
        assert all(s.btl_ci_low is not None for s in report.standings)

    def test_pair_means_use_the_usable_cases_only(self):
        report = self._run(n_bootstrap=0)
        pr = report.pair_results[0]
        # One usable case, two display orders (a first: 1.0 / 0.0; b first: a gets 0.0):
        # the mean over that case is 0.5, not 0.5 / 2 cases = 0.25.
        assert (pr.mean_score_a, pr.mean_score_b) == (0.5, 0.5)

    def test_means_unchanged_when_nothing_is_excluded(self):
        cases = [EvalCase(id=f"c{i}", prompt=f"Q{i}") for i in (0, 2)]
        report = TournamentEvaluator(PairwiseJudge("p", _FailsOnQ1()), n_bootstrap=0).run(
            cases, {"a": lambda _: "a", "b": lambda _: "b"}
        )
        pr = report.pair_results[0]
        assert (pr.mean_score_a, pr.mean_score_b) == (0.5, 0.5)


# ── cache: one model call per key under concurrency ──────────────────────────────────────


class _SlowCounting:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0
        self._fail_first = fail_first
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {"provider": "slow", "model": "m"}

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        with self._lock:
            self.calls += 1
            n = self.calls
        time.sleep(0.25)
        if self._fail_first and n == 1:
            raise RuntimeError("provider outage")
        return f"reply-{n}"


class TestCacheInflightDedup:
    def test_concurrent_identical_requests_make_one_call_and_one_line(self, tmp_path: Path):
        inner = _SlowCounting()
        path = tmp_path / "c.jsonl"
        c = CachedBackend(inner, path)
        replies = ordered_map(lambda _: c.complete("s", "same"), range(8), max_workers=8)
        assert inner.calls == 1
        assert len(set(replies)) == 1                       # every caller saw the same reply
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1
        snap = c.stats.snapshot()
        assert snap.calls == 8 and snap.cache_hits == 7     # one paid, seven free

    def test_distinct_keys_still_run_in_parallel(self, tmp_path: Path):
        inner = _SlowCounting()
        c = CachedBackend(inner, tmp_path / "c.jsonl")
        start = time.perf_counter()
        ordered_map(lambda i: c.complete("s", f"u{i}"), range(6), max_workers=6)
        assert inner.calls == 6
        assert time.perf_counter() - start < 1.0            # 6 x 0.25s serialised would be 1.5s

    def test_a_failure_reaches_every_waiter_and_is_not_cached(self, tmp_path: Path):
        inner = _SlowCounting(fail_first=True)
        c = CachedBackend(inner, tmp_path / "c.jsonl")

        def attempt(_: int) -> str:
            try:
                return c.complete("s", "same")
            except RuntimeError as exc:
                return f"ERR:{exc}"

        results = ordered_map(attempt, range(4), max_workers=4)
        assert all(r == "ERR:provider outage" for r in results)
        assert inner.calls == 1
        assert len(c) == 0
        assert c.complete("s", "same") == "reply-2"         # the key was not poisoned


# ── ordered_map surfaces a failure without waiting for slower earlier items ───────────────


class TestOrderedMapFailsFast:
    def test_failure_is_not_delayed_by_a_slow_earlier_item(self):
        started: list[int] = []
        lock = threading.Lock()

        def fn(i: int) -> int:
            with lock:
                started.append(i)
            if i == 0:
                time.sleep(0.8)
            elif i == 1:
                raise RuntimeError("boom")
            else:
                time.sleep(0.05)
            return i

        t0 = time.perf_counter()
        with pytest.raises(RuntimeError, match="boom"):
            ordered_map(fn, range(12), max_workers=2)
        assert time.perf_counter() - t0 < 0.5               # not the 0.8s of item 0
        assert len(started) < 12                            # queued work was cancelled

    def test_success_path_keeps_input_order(self):
        assert ordered_map(lambda i: (time.sleep(0.02 * (5 - i)), i)[1], range(5), max_workers=5) == [
            0, 1, 2, 3, 4
        ]


# ── truncation is counted for pairwise and positional judges ──────────────────────────────


class _TruncatedPair:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        # A reply that is valid JSON but stopped at the token limit.
        return LLMResponse(
            text='{"score_a": 0.9, "score_b": 0.1, "preferred": "A", "rationale": "r"}',
            finish_reason="length",
        )


def _evaluator(positional: PairwiseJudge | None = None) -> JuryEvaluator:
    return JuryEvaluator(
        panel=JudgePanel([AssertionJudge("j", _Plain())]),
        generation_backend=_Plain(),
        positional_judge=positional,
        compute_significance=False, bootstrap_ci=False, compute_judge_fit=False,
    )


class TestTruncationCounting:
    def test_pairwise_dataset_counts_truncated_replies(self):
        rows = [{"id": f"r{i}", "prompt": "Q", "output_a": "a", "output_b": "b"} for i in range(3)]
        report = _evaluator().evaluate_pairwise_dataset(rows, PairwiseJudge("p", _TruncatedPair()))
        assert report.data_quality.n_truncated_replies == 3
        assert all(c.truncated for c in report.cases)
        assert any("output-token limit" in w for w in report.data_quality.warnings)

    def test_positional_report_carries_the_flag(self):
        r = PositionalBiasDetector(PairwiseJudge("pos", _TruncatedPair())).test_case(
            EvalCase(id="c", prompt="Q"), "ctrl", "trt"
        )
        assert r.truncated is True and r.parse_error is None

    def test_positional_judge_truncations_reach_the_evaluation_report(self):
        from judge_kappa.models import Variant

        ev = _evaluator(positional=PairwiseJudge("pos", _TruncatedPair()))
        cases = [EvalCase(id=f"c{i}", prompt="Q", assertions=[Assertion(text="t")]) for i in range(2)]
        report = ev.evaluate(
            cases,
            Variant(name="control", predict_fn=lambda _i: "c"),
            Variant(name="treatment", predict_fn=lambda _i: "t"),
        )
        assert report.data_quality.n_truncated_replies == 2


# ── non-finite and enormous numbers in judge replies never crash ──────────────────────────


class _Raw:
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self._text


class TestNonFiniteNumbers:
    def test_is_finite_number(self):
        from judge_kappa.judges.base import is_finite_number

        assert is_finite_number(0.5) and is_finite_number(3)
        for bad in (float("inf"), float("-inf"), float("nan"), True, "1", None, 10**400):
            assert not is_finite_number(bad)

    def test_rank_judge_survives_infinity(self):
        reply = '{"rankings": {"A": Infinity, "B": 2, "C": 3}, "rationale": "x"}'
        verdicts = RankJudge("k", _Raw(reply)).rank(
            EvalCase(id="c", prompt="Q"), {"s1": "a", "s2": "b", "s3": "c"}
        )
        assert all(v.parse_failed and v.parse_error == "non_integer_rank" for v in verdicts.values())
        assert verdicts["s1"].score == 0.0                  # historical: unusable rank → worst

    @pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN", "1e999", str(10**400)])
    def test_assertion_judge_survives_non_finite_scores(self, token: str):
        reply = '{"assertion_scores": {"is correct": ' + token + '}, "rationale": "r"}'
        case = EvalCase(id="c", prompt="Q", assertions=[Assertion(text="is correct")])
        v = AssertionJudge("j", _Raw(reply)).judge(case, "out", "control")
        assert v.parse_failed and v.parse_error == "non_numeric_assertion_score"
        assert v.score == 0.0

    @pytest.mark.parametrize("token", ["Infinity", "NaN", str(10**400)])
    def test_pairwise_judge_survives_non_finite_scores(self, token: str):
        reply = '{"score_a": ' + token + ', "score_b": 0.2}'
        va, vb = PairwiseJudge("p", _Raw(reply)).judge_pair(EvalCase(id="c", prompt="Q"), "a", "b")
        assert va.parse_failed and va.parse_error == "non_numeric_score"
        assert (va.score, vb.score) == (0.5, 0.2)


# ── sanitize_url keeps IPv6 brackets ─────────────────────────────────────────────────────


class TestSanitizeUrlIPv6:
    def test_brackets_are_kept(self):
        assert sanitize_url("http://[::1]:8000/v1") == "http://[::1]:8000/v1"

    def test_credentials_query_and_fragment_still_stripped(self):
        got = sanitize_url("https://user:pw@[2001:db8::1]:9000/x?token=abc#frag")
        assert got == "https://[2001:db8::1]:9000/x"
        assert "pw" not in got and "abc" not in got

    def test_plain_hosts_unchanged(self):
        assert sanitize_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
        assert sanitize_url("https://api.example.com/v1") == "https://api.example.com/v1"


# ── CHANGELOG: Keep a Changelog allows one heading per change type per release ───────────


def test_changelog_has_no_duplicate_change_type_headings_in_recent_releases():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## \[", text)[1:]           # drop the preamble
    for section in sections[:3]:                           # Unreleased + the latest releases
        name = section.split("]", 1)[0]
        headings = re.findall(r"(?m)^### (.+)$", section.split("\n---\n", 1)[0])
        dupes = {h for h in headings if headings.count(h) > 1}
        assert not dupes, f"[{name}] repeats change-type headings: {sorted(dupes)}"
