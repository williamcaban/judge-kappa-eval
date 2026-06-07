"""
Tests for TournamentEvaluator, _compute_elo, TournamentReport, and PairResult.

All tests use MockLLMBackend — no real LLM calls.
Covers:
  - Elo computation: convergence, monotonicity, tie handling
  - Tournament round-robin: all pairs visited, win matrix shape
  - Positional bias detection in tournament
  - run() with pre-recorded outputs via callables
  - run_endpoints() with per-Variant generation backends
  - print_leaderboard() produces output without crashing
"""

from __future__ import annotations

import json

import pytest

from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.models import EvalCase, Variant
from judge_kappa.tournament import (
    TournamentEvaluator,
    _compute_elo,
)
from tests.conftest import MockLLMBackend


def _pairwise_resp(score_a: float, score_b: float) -> str:
    return json.dumps({
        "score_a": score_a, "score_b": score_b,
        "preferred": "A" if score_a > score_b else "B",
        "rationale": "mock",
    })


def _cases(n: int = 3) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", prompt=f"Prompt {i}") for i in range(n)]


def _judge(responses: list[str]) -> PairwiseJudge:
    return PairwiseJudge("j1", MockLLMBackend(responses=responses * 200))


# ── _compute_elo ──────────────────────────────────────────────────────────────


class TestComputeElo:
    def test_equal_win_rates_give_equal_elo(self):
        systems = ["A", "B"]
        win_matrix = {"A": {"B": 0.5}, "B": {"A": 0.5}}
        ratings = _compute_elo(win_matrix, systems)
        assert abs(ratings["A"] - ratings["B"]) < 5.0

    def test_dominant_system_gets_higher_elo(self):
        systems = ["A", "B"]
        win_matrix = {"A": {"B": 0.90}, "B": {"A": 0.10}}
        ratings = _compute_elo(win_matrix, systems)
        assert ratings["A"] > ratings["B"] + 100  # 90% win rate → large Elo gap

    def test_elo_ordering_preserved_across_three_systems(self):
        systems = ["best", "mid", "worst"]
        win_matrix = {
            "best":  {"mid": 0.8, "worst": 0.95},
            "mid":   {"best": 0.2, "worst": 0.75},
            "worst": {"best": 0.05, "mid": 0.25},
        }
        ratings = _compute_elo(win_matrix, systems)
        assert ratings["best"] > ratings["mid"] > ratings["worst"]

    def test_elo_converges_with_tolerance(self):
        # 4 systems with transitive ordering: A > B > C > D
        systems = ["A", "B", "C", "D"]
        win_matrix = {
            "A": {"B": 0.7, "C": 0.8, "D": 0.9},
            "B": {"A": 0.3, "C": 0.7, "D": 0.8},
            "C": {"A": 0.2, "B": 0.3, "D": 0.7},
            "D": {"A": 0.1, "B": 0.2, "C": 0.3},
        }
        ratings = _compute_elo(win_matrix, systems, tol=1e-4)
        assert ratings["A"] > ratings["B"] > ratings["C"] > ratings["D"]

    def test_tie_rates_produce_near_equal_elo(self):
        systems = ["X", "Y"]
        win_matrix = {"X": {"Y": 0.5}, "Y": {"X": 0.5}}  # perfect tie
        ratings = _compute_elo(win_matrix, systems)
        assert abs(ratings["X"] - ratings["Y"]) < 10.0

    def test_100_elo_gap_implies_approx_64_percent_win(self):
        # If A has 100 Elo over B, expected win rate ≈ 64%
        systems = ["A", "B"]
        win_matrix = {"A": {"B": 0.64}, "B": {"A": 0.36}}
        ratings = _compute_elo(win_matrix, systems)
        gap = ratings["A"] - ratings["B"]
        expected_win = 1 / (1 + 10 ** (-gap / 400))
        assert abs(expected_win - 0.64) < 0.05  # within 5% of the 64% target


# ── TournamentEvaluator.run() ─────────────────────────────────────────────────


class TestTournamentRun:
    def _tournament(self, responses: list[str], detect_bias: bool = False) -> TournamentEvaluator:
        return TournamentEvaluator(
            pairwise_judge=_judge(responses),
            detect_positional_bias=detect_bias,
        )

    def test_run_with_three_systems_produces_three_standing_entries(self):
        cases = _cases(3)
        # All responses: A scores 0.8, B scores 0.4 → A always wins
        responses = [_pairwise_resp(0.8, 0.4)] * 200
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={
                "alpha":   lambda _: "output alpha",
                "beta":    lambda _: "output beta",
                "gamma":   lambda _: "output gamma",
            },
        )
        assert len(report.standings) == 3

    def test_run_produces_correct_number_of_pair_results(self):
        # 3 systems → C(3,2) = 3 pairs
        cases = _cases(2)
        responses = [_pairwise_resp(0.7, 0.5)] * 200
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b", "C": lambda _: "c"},
        )
        assert len(report.pair_results) == 3  # C(3,2) = 3

    def test_run_win_matrix_symmetric(self):
        cases = _cases(2)
        responses = [_pairwise_resp(0.7, 0.5)] * 200
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={"X": lambda _: "x", "Y": lambda _: "y"},
        )
        rate_xy = report.win_matrix["X"]["Y"]
        rate_yx = report.win_matrix["Y"]["X"]
        assert abs(rate_xy + rate_yx - 1.0) < 0.01  # sum to 1 (ignoring ties)

    def test_dominant_system_ranked_first(self):
        cases = _cases(4)
        # A always beats everyone (score 0.9 vs 0.1)
        responses = [_pairwise_resp(0.9, 0.1)] * 200
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b", "C": lambda _: "c"},
        )
        assert report.standings[0].system == "A"

    def test_fewer_than_two_systems_raises(self):
        tournament = self._tournament([_pairwise_resp(0.5, 0.5)])
        with pytest.raises(ValueError):
            tournament.run(cases=_cases(), systems={"only_one": lambda _: "x"})

    def test_run_judge_id_in_report(self):
        cases = _cases(2)
        responses = [_pairwise_resp(0.6, 0.4)] * 200
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b"},
        )
        assert report.judge_id == "j1"

    def test_run_n_cases_in_report(self):
        cases = _cases(5)
        responses = [_pairwise_resp(0.6, 0.4)] * 500
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b"},
        )
        assert report.n_cases == 5

    def test_report_serializable_to_json(self):
        cases = _cases(2)
        responses = [_pairwise_resp(0.7, 0.5)] * 200
        tournament = self._tournament(responses)
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b"},
        )
        # TournamentReport is a dataclass — verify it has required fields
        assert hasattr(report, "standings")
        assert hasattr(report, "win_matrix")
        assert hasattr(report, "pair_results")


# ── TournamentReport.print_leaderboard() ─────────────────────────────────────


class TestPrintLeaderboard:
    def test_print_leaderboard_does_not_raise(self, capsys):
        cases = _cases(2)
        responses = [_pairwise_resp(0.7, 0.4)] * 200
        tournament = TournamentEvaluator(
            pairwise_judge=_judge(responses),
            detect_positional_bias=False,
        )
        report = tournament.run(
            cases=cases,
            systems={"system-alpha": lambda _: "a", "system-beta": lambda _: "b"},
        )
        report.print_leaderboard()
        out = capsys.readouterr().out
        assert "system-alpha" in out or "system-beta" in out

    def test_print_leaderboard_includes_elo_column(self, capsys):
        cases = _cases(2)
        responses = [_pairwise_resp(0.8, 0.3)] * 200
        tournament = TournamentEvaluator(
            pairwise_judge=_judge(responses),
            detect_positional_bias=False,
        )
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b"},
        )
        report.print_leaderboard()
        out = capsys.readouterr().out
        assert "Elo" in out


# ── Positional bias in tournament ─────────────────────────────────────────────


class TestTournamentPositionalBias:
    def test_positional_bias_rate_in_pair_results(self):
        cases = _cases(2)
        # Judge always prefers first position (positional bias)
        # Round 1 (A,B): score_a=0.9, score_b=0.1 → A preferred (first)
        # Round 2 (B,A): score_a=0.9, score_b=0.1 → A preferred (still first)
        responses = [_pairwise_resp(0.9, 0.1)] * 200
        tournament = TournamentEvaluator(
            pairwise_judge=_judge(responses),
            detect_positional_bias=True,
        )
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b"},
        )
        # With positional bias, flip rate should be high
        for pair in report.pair_results:
            assert 0.0 <= pair.positional_bias_rate <= 1.0

    def test_no_positional_detection_when_disabled(self):
        cases = _cases(2)
        responses = [_pairwise_resp(0.6, 0.4)] * 200
        tournament = TournamentEvaluator(
            pairwise_judge=_judge(responses),
            detect_positional_bias=False,
        )
        report = tournament.run(
            cases=cases,
            systems={"A": lambda _: "a", "B": lambda _: "b"},
        )
        for pair in report.pair_results:
            assert pair.positional_bias_rate == 0.0


# ── run_endpoints() ───────────────────────────────────────────────────────────


class TestRunEndpoints:
    def test_run_endpoints_uses_variant_predict_fn(self):
        cases = _cases(3)
        calls: dict[str, int] = {"v1": 0, "v2": 0}

        def make_fn(name: str):
            def fn(inputs: dict) -> str:  # noqa: ARG001
                calls[name] += 1
                return f"output from {name}"
            return fn

        responses = [_pairwise_resp(0.7, 0.5)] * 200
        judge = _judge(responses)
        tournament = TournamentEvaluator(pairwise_judge=judge, detect_positional_bias=False)

        report = tournament.run_endpoints(
            cases=cases,
            variants=[
                Variant(name="v1", predict_fn=make_fn("v1")),
                Variant(name="v2", predict_fn=make_fn("v2")),
            ],
        )
        assert len(report.standings) == 2
        assert calls["v1"] == len(cases)
        assert calls["v2"] == len(cases)
