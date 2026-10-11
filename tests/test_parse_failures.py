"""
Judge parse failures must be visible, countable, and optionally fatal.

Before this, an unusable judge reply silently became a score: 0.0 for assertion and
rubric judges, a 0.5/0.5 tie for pairwise, "worst rank" for ranking. Valid replies must
produce exactly the numbers they always did; only the failure path changed.
"""

from __future__ import annotations

import json
import logging

import pytest

from judge_kappa.bias.positional import PositionalBiasDetector
from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.judges.base import JudgeParseError
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.judges.rank import RankJudge
from judge_kappa.judges.rubric import RubricJudge
from judge_kappa.llm.types import LLMResponse
from judge_kappa.models import Assertion, EvalCase, RubricDimension, Variant
from judge_kappa.panel.panel import JudgePanel
from judge_kappa.quality import compute_data_quality
from judge_kappa.tournament import TournamentEvaluator


class _Scripted:
    """Backend that returns fixed text (optionally with a finish reason)."""

    def __init__(self, text: str, finish_reason: str | None = None) -> None:
        self._resp = LLMResponse(text=text, finish_reason=finish_reason)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self._resp.text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        return self._resp


def _case() -> EvalCase:
    return EvalCase(
        id="c1", prompt="Q",
        assertions=[Assertion(text="is correct"), Assertion(text="is concise")],
    )


def _assertion_judge(text: str, **kw) -> AssertionJudge:
    return AssertionJudge("j", _Scripted(text, kw.pop("finish_reason", None)), **kw)


_GOOD = json.dumps({"assertion_scores": {"is correct": 1.0, "is concise": 0.0}, "rationale": "ok"})


# ── AssertionJudge ────────────────────────────────────────────────────────────


class TestAssertionJudge:
    def test_valid_reply_is_unflagged_and_scored_as_before(self):
        v = _assertion_judge(_GOOD).judge(_case(), "out", "control")
        assert not v.parse_failed and v.parse_error is None and not v.truncated
        assert v.score == 0.5
        assert v.assertion_scores == {"is correct": 1.0, "is concise": 0.0}

    @pytest.mark.parametrize(
        ("reply", "code"),
        [
            ("Looks good to me!", "invalid_json"),
            (json.dumps({"rationale": "no scores"}), "missing_key:assertion_scores"),
            (json.dumps([1, 2, 3]), "not_an_object"),
            (json.dumps({"assertion_scores": {"correct": 1.0, "concise": 1.0}}), "missing_assertions:2/2"),
            (json.dumps({"assertion_scores": {"is correct": 1.0}}), "missing_assertions:1/2"),
            (json.dumps({"assertion_scores": {"is correct": "yes", "is concise": 1.0}}), "non_numeric_assertion_score"),
            (json.dumps({"assertion_scores": {"is correct": 7, "is concise": 1.0}}), "out_of_range_assertion_score"),
            (json.dumps({"assertion_scores": ["a"]}), "assertion_scores_not_an_object"),
        ],
    )
    def test_unusable_replies_are_flagged_with_a_code(self, reply, code):
        v = _assertion_judge(reply).judge(_case(), "out", "control")
        assert v.parse_failed is True
        assert v.parse_error == code

    def test_historical_fallback_score_is_preserved_but_flagged(self):
        v = _assertion_judge("not json").judge(_case(), "out", "control")
        assert v.score == 0.0 and v.parse_failed
        assert v.rationale == "not json"

    def test_truncated_reply_is_reported_as_truncation(self):
        v = _assertion_judge('{"assertion_scores": {"is corr', finish_reason="length").judge(
            _case(), "out", "control"
        )
        assert v.parse_error == "truncated_output"
        assert v.truncated is True

    def test_raise_policy(self):
        judge = _assertion_judge("garbage", on_parse_failure="raise")
        with pytest.raises(JudgeParseError) as err:
            judge.judge(_case(), "out", "control")
        assert err.value.judge_id == "j" and err.value.case_id == "c1"
        assert err.value.error == "invalid_json"

    def test_raise_policy_does_not_fire_on_valid_reply(self):
        _assertion_judge(_GOOD, on_parse_failure="raise").judge(_case(), "out", "control")

    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError, match="on_parse_failure"):
            _assertion_judge(_GOOD, on_parse_failure="ignore")

    def test_failure_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="judge_kappa.judges"):
            _assertion_judge("nope").judge(_case(), "out", "control")
        assert any("unusable reply" in r.message for r in caplog.records)

    def test_plain_backend_without_usage_still_works(self):
        class _Plain:
            def complete(self, system, user, temperature=0.0):
                return _GOOD

        v = AssertionJudge("j", _Plain()).judge(_case(), "out", "control")
        assert not v.parse_failed and v.score == 0.5


# ── RubricJudge ───────────────────────────────────────────────────────────────


class TestRubricJudge:
    _RUBRIC = [RubricDimension(name="faithfulness", description="d"), RubricDimension(name="relevance", description="d")]

    def _judge(self, text: str) -> RubricJudge:
        return RubricJudge("r", _Scripted(text), rubric=self._RUBRIC)

    def test_valid(self):
        reply = json.dumps({"dimension_scores": {"faithfulness": 1.0, "relevance": 0.5}, "rationale": "r"})
        v = self._judge(reply).judge(_case(), "out", "control")
        assert not v.parse_failed and v.score == 0.75

    def test_missing_dimension_is_flagged(self):
        reply = json.dumps({"dimension_scores": {"faithfulness": 1.0}})
        v = self._judge(reply).judge(_case(), "out", "control")
        assert v.parse_error == "missing_dimensions:1/2"
        assert v.score == 0.5       # historical: missing dimension counted as 0.0

    def test_invalid_json(self):
        assert self._judge("{{").judge(_case(), "out", "control").parse_error == "invalid_json"


# ── PairwiseJudge ─────────────────────────────────────────────────────────────


class TestPairwiseJudge:
    def _pair(self, text: str):
        return PairwiseJudge("p", _Scripted(text)).judge_pair(_case(), "A out", "B out", "A", "B")

    def test_valid(self):
        va, vb = self._pair(json.dumps({"score_a": 0.8, "score_b": 0.2, "preferred": "A", "rationale": "r"}))
        assert (va.score, vb.score) == (0.8, 0.2)
        assert not va.parse_failed and not vb.parse_failed

    def test_unusable_reply_falls_back_to_a_flagged_tie(self):
        va, vb = self._pair("I prefer B")
        assert (va.score, vb.score) == (0.5, 0.5)
        assert va.parse_failed and vb.parse_failed
        assert va.parse_error == "invalid_json"

    def test_non_numeric_score_no_longer_crashes(self):
        va, vb = self._pair(json.dumps({"score_a": "high", "score_b": 0.2}))
        assert va.parse_error == "non_numeric_score"
        assert (va.score, vb.score) == (0.5, 0.2)

    def test_out_of_range_is_flagged_not_clamped(self):
        va, _ = self._pair(json.dumps({"score_a": 9, "score_b": 2}))
        assert va.parse_error == "out_of_range_score"
        assert va.score == 9.0


# ── RankJudge ─────────────────────────────────────────────────────────────────


class TestRankJudge:
    def _rank(self, text: str):
        return RankJudge("k", _Scripted(text)).rank(_case(), {"s1": "o1", "s2": "o2", "s3": "o3"})

    def test_valid(self):
        v = self._rank(json.dumps({"rankings": {"A": 1, "B": 2, "C": 3}, "rationale": "r"}))
        assert [v[s].score for s in ("s1", "s2", "s3")] == [1.0, 0.5, 0.0]
        assert not any(x.parse_failed for x in v.values())

    @pytest.mark.parametrize(
        ("reply", "code"),
        [
            (json.dumps({"rankings": {"A": 1, "B": 2}}), "missing_rankings:1/3"),
            (json.dumps({"rankings": {"A": 1, "B": 1, "C": 2}}), "invalid_ranks"),
            (json.dumps({"rankings": {"A": 1, "B": "two", "C": 3}}), "non_integer_rank"),
            ("plain prose", "invalid_json"),
        ],
    )
    def test_unusable_rankings_are_flagged(self, reply, code):
        v = self._rank(reply)
        assert all(x.parse_failed and x.parse_error == code for x in v.values())

    def test_missing_label_keeps_historical_worst_rank(self):
        v = self._rank(json.dumps({"rankings": {"A": 1, "B": 2}}))
        assert v["s3"].score == 0.0


# ── Report-level data quality ──────────────────────────────────────────────────


class _Mixed:
    """Valid replies for one judge id, garbage for another (selected by system prompt)."""

    def __init__(self, bad: bool) -> None:
        self._bad = bad

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return "no json here" if self._bad else _GOOD


class TestEvaluatorDataQuality:
    def _evaluate(self, bad_judge: bool):
        panel = JudgePanel([
            AssertionJudge("good", _Mixed(False)),
            AssertionJudge("flaky", _Mixed(bad_judge)),
        ])
        ev = JuryEvaluator(
            panel=panel, generation_backend=_Mixed(False),
            compute_significance=False, bootstrap_ci=False,
        )
        cases = [EvalCase(id=f"c{i}", prompt="Q", assertions=[Assertion(text="is correct"), Assertion(text="is concise")])
                 for i in range(3)]
        return ev.evaluate(cases, Variant(name="control"), Variant(name="treatment"))

    def test_clean_run_has_empty_quality_block(self):
        dq = self._evaluate(False).data_quality
        assert dq.n_judge_calls == 12 and dq.n_parse_failures == 0
        assert dq.parse_failure_rate == 0.0 and dq.warnings == []

    def test_failures_are_counted_per_judge_and_code(self):
        report = self._evaluate(True)
        dq = report.data_quality
        assert dq.n_judge_calls == 12 and dq.n_parse_failures == 6
        assert dq.parse_failure_rate == 0.5
        assert dq.parse_failures_by_judge == {"flaky": 6}
        assert dq.parse_errors == {"invalid_json": 6}
        assert len(dq.warnings) == 1 and "50.0%" in dq.warnings[0]
        flagged = [v for c in report.cases for v in c.verdicts if v.parse_failed]
        assert len(flagged) == 6 and all(v.judge_id == "flaky" for v in flagged)

    def test_report_json_carries_the_block(self):
        payload = json.loads(self._evaluate(True).model_dump_json())
        assert payload["data_quality"]["n_parse_failures"] == 6
        assert payload["cases"][0]["verdicts"][0]["parse_failed"] in (True, False)

    def test_pairwise_dataset_counts_failures(self):
        ev = JuryEvaluator(
            panel=JudgePanel([AssertionJudge("j", _Mixed(False))]),
            generation_backend=_Mixed(False), compute_significance=False, bootstrap_ci=False,
        )
        judge = PairwiseJudge("p", _Mixed(True))
        report = ev.evaluate_pairwise_dataset(
            [{"id": f"r{i}", "prompt": "Q", "output_a": "a", "output_b": "b"} for i in range(4)], judge
        )
        assert report.data_quality.n_parse_failures == 4
        assert all(c.parse_failed for c in report.cases)
        assert report.tie_rate == 1.0       # historical fallback, now visible

    def test_positional_judge_failures_are_counted(self):
        pos = PairwiseJudge("pos", _Mixed(True))
        r = PositionalBiasDetector(pos).test_case(_case(), "ctrl", "trt")
        assert r.parse_error == "invalid_json"
        dq = compute_data_quality([("pos", r.parse_error, False)])
        assert dq.parse_failures_by_judge == {"pos": 1}


class TestComputeDataQuality:
    def test_empty(self):
        dq = compute_data_quality([])
        assert dq.n_judge_calls == 0 and dq.parse_failure_rate == 0.0 and dq.warnings == []

    def test_code_histogram_drops_the_detail_suffix(self):
        dq = compute_data_quality([("j", "missing_assertions:1/3", False), ("j", "missing_assertions:2/3", False)])
        assert dq.parse_errors == {"missing_assertions": 2}

    def test_truncation_warning(self):
        dq = compute_data_quality([("j", None, True)], extra_truncated=2)
        assert dq.n_truncated_replies == 3
        assert any("output-token limit" in w for w in dq.warnings)


# ── Tournament ────────────────────────────────────────────────────────────────


class _TournamentBackend:
    """Content-aware pairwise mock. Replies with prose for pairs involving 'bad' on the
    cases listed in ``fail_prompts`` (all cases when it is None)."""

    def __init__(self, quality: dict[str, float], fail_prompts: set[str] | None = None) -> None:
        self._q = quality
        self._fail = fail_prompts

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        a = user.split("## Output A\n", 1)[1].split("\n\n## Output B\n", 1)[0]
        b = user.split("\n\n## Output B\n", 1)[1]
        prompt = user.split("## Prompt\n", 1)[1].split("\n\n## Output A", 1)[0]
        if "bad" in (a, b) and (self._fail is None or prompt in self._fail):
            return "cannot decide"
        return json.dumps({"score_a": self._q[a], "score_b": self._q[b], "preferred": "A", "rationale": "r"})


def _systems() -> dict:
    return {"good": lambda _: "good", "ok": lambda _: "ok", "bad": lambda _: "bad"}


def _cases(n: int = 4) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", prompt=f"Q{i}") for i in range(n)]


class TestTournamentParseFailures:
    _Q = {"good": 0.9, "ok": 0.5, "bad": 0.1}

    def _run(self, fail_prompts: set[str] | None = None, **kw):
        judge = PairwiseJudge("tj", _TournamentBackend(self._Q, fail_prompts))
        return TournamentEvaluator(judge, n_bootstrap=0, **kw).run(_cases(), _systems())

    def test_failures_counted_and_historical_ties_kept_by_default(self):
        report = self._run()
        dq = report.data_quality
        assert dq.n_parse_failures == 16            # 2 pairs involve 'bad' x 4 cases x 2 rounds
        assert dq.parse_errors == {"invalid_json": 16}
        by_pair = {(p.system_a, p.system_b): p for p in report.pair_results}
        assert by_pair[("good", "bad")].tie_rate == 1.0   # fake ties, but now flagged
        assert any("unusable" in w for w in dq.warnings)

    def test_exclude_parse_failures_removes_only_the_unusable_games(self):
        # Replies fail on Q0 and Q1 only; Q2 and Q3 are usable for every pair.
        report = self._run(fail_prompts={"Q0", "Q1"}, exclude_parse_failures=True)
        assert report.data_quality.n_parse_failures == 8   # still reported
        s = {x.system: x for x in report.standings}
        # 'bad' loses its 2 usable cases against each of 'good' and 'ok': 4 losses, no fake ties.
        assert (s["bad"].wins, s["bad"].losses, s["bad"].ties) == (0, 4, 0)
        assert [x.system for x in report.standings] == ["good", "ok", "bad"]

    def test_default_policy_would_have_counted_fake_ties_for_the_same_run(self):
        report = self._run(fail_prompts={"Q0", "Q1"})
        s = {x.system: x for x in report.standings}
        assert s["bad"].ties == 4 and s["bad"].losses == 4     # 2 fake ties per pair

    def test_a_system_with_no_usable_game_is_named_in_the_error(self):
        with pytest.raises(ValueError, match="no usable game for: bad"):
            self._run(exclude_parse_failures=True)

    def test_all_replies_unusable_with_exclusion_gives_a_clear_error(self):
        judge = PairwiseJudge("tj", _Scripted("nothing usable"))
        ev = TournamentEvaluator(judge, n_bootstrap=0, exclude_parse_failures=True)
        with pytest.raises(ValueError, match="cannot rate the systems.*unusable and excluded"):
            ev.run([EvalCase(id="c0", prompt="Q")], {"a": lambda _: "a", "b": lambda _: "b"})

    def test_leaderboard_prints_the_warning(self, capsys):
        self._run().print_leaderboard()
        assert "Warning:" in capsys.readouterr().out
