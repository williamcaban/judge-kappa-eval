"""
Tests for JudgePanel and JudgeJury aggregation strategies.

All judges use MockLLMBackend with injected scores.
"""

from __future__ import annotations

import json

import pytest

from jury_eval.judges.assertion import AssertionJudge
from jury_eval.models import AggregationStrategy, Assertion, EvalCase, JudgeVerdict
from jury_eval.panel.jury import JudgeJury
from jury_eval.panel.panel import JudgePanel
from tests.conftest import MockLLMBackend, make_verdict


# ── JudgePanel ────────────────────────────────────────────────────────────────


class TestJudgePanelAggregation:
    """Test aggregate_score() with pre-built verdicts — no LLM calls needed."""

    def _verdicts(self, scores: list[float]) -> list[JudgeVerdict]:
        return [make_verdict(f"j{i}", "c1", s) for i, s in enumerate(scores)]

    def test_mean(self):
        panel = JudgePanel(
            judges=[AssertionJudge("j1", MockLLMBackend())],
            strategy=AggregationStrategy.MEAN,
        )
        assert panel.aggregate_score(self._verdicts([0.2, 0.8, 1.0])) == pytest.approx(2 / 3, abs=1e-4)

    def test_weighted_mean(self):
        panel = JudgePanel(
            judges=[AssertionJudge("j1", MockLLMBackend())],
            strategy=AggregationStrategy.WEIGHTED_MEAN,
            weights=[3.0, 1.0],
        )
        # (0.0*3 + 1.0*1) / (3+1) = 0.25
        assert panel.aggregate_score(self._verdicts([0.0, 1.0])) == pytest.approx(0.25)

    def test_median_with_odd_count(self):
        panel = JudgePanel(
            judges=[AssertionJudge("j1", MockLLMBackend())],
            strategy=AggregationStrategy.MEDIAN,
        )
        assert panel.aggregate_score(self._verdicts([0.1, 0.5, 0.9])) == pytest.approx(0.5)

    def test_majority_vote_passes(self):
        panel = JudgePanel(
            judges=[AssertionJudge("j1", MockLLMBackend())],
            strategy=AggregationStrategy.MAJORITY_VOTE,
        )
        # 2 pass (≥0.5), 1 fail → majority pass = 1.0
        assert panel.aggregate_score(self._verdicts([0.8, 0.9, 0.3])) == pytest.approx(1.0)

    def test_majority_vote_fails(self):
        panel = JudgePanel(
            judges=[AssertionJudge("j1", MockLLMBackend())],
            strategy=AggregationStrategy.MAJORITY_VOTE,
        )
        # 2 fail, 1 pass → majority fail = 0.0
        assert panel.aggregate_score(self._verdicts([0.1, 0.2, 0.9])) == pytest.approx(0.0)

    def test_trimmed_mean_drops_outliers(self):
        panel = JudgePanel(
            judges=[AssertionJudge("j1", MockLLMBackend())],
            strategy=AggregationStrategy.TRIMMED_MEAN,
        )
        # With 10 scores, 10% trim removes one from each end
        scores = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
        result = panel.aggregate_score(self._verdicts(scores))
        assert result == pytest.approx(0.5, abs=0.05)

    def test_empty_judges_raises(self):
        with pytest.raises(ValueError, match="at least one judge"):
            JudgePanel(judges=[])

    def test_judge_ids_property(self):
        j1 = AssertionJudge("alpha", MockLLMBackend())
        j2 = AssertionJudge("beta",  MockLLMBackend())
        panel = JudgePanel(judges=[j1, j2])
        assert panel.judge_ids == ["alpha", "beta"]

    def test_evaluate_calls_all_judges(self):
        case = EvalCase(
            id="c1", prompt="P",
            assertions=[Assertion(text="A", weight=1.0)],
        )
        response = json.dumps({"assertion_scores": {"A": 0.9}, "rationale": "ok"})
        j1 = AssertionJudge("j1", MockLLMBackend(responses=[response]))
        j2 = AssertionJudge("j2", MockLLMBackend(responses=[response]))
        panel = JudgePanel(judges=[j1, j2])
        verdicts = panel.evaluate(case, "output text", "control")
        assert len(verdicts) == 2
        assert {v.judge_id for v in verdicts} == {"j1", "j2"}


# ── JudgeJury ─────────────────────────────────────────────────────────────────


class TestJudgeJury:
    def test_weighted_mean_respects_juror_weights(self):
        jury = JudgeJury(
            jurors=[
                (AssertionJudge("j1", MockLLMBackend()), 3.0),
                (AssertionJudge("j2", MockLLMBackend()), 1.0),
            ],
            strategy=AggregationStrategy.WEIGHTED_MEAN,
        )
        verdicts = [make_verdict("j1", "c1", 1.0), make_verdict("j2", "c1", 0.0)]
        # (1.0*3 + 0.0*1) / 4 = 0.75
        assert jury.aggregate_score(verdicts) == pytest.approx(0.75)

    def test_empty_jurors_raises(self):
        with pytest.raises(ValueError, match="at least one juror"):
            JudgeJury(jurors=[])

    def test_judge_ids_property(self):
        j1 = AssertionJudge("safety",  MockLLMBackend())
        j2 = AssertionJudge("quality", MockLLMBackend())
        jury = JudgeJury(jurors=[(j1, 2.0), (j2, 1.0)])
        assert jury.judge_ids == ["safety", "quality"]
