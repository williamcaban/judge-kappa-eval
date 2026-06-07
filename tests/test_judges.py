"""
Tests for AssertionJudge, RubricJudge, and PairwiseJudge.

All tests use MockLLMBackend — no real LLM calls.
Tests focus on:
  - Weighted score computation
  - Missing key handling (graceful 0.0 fallback)
  - Malformed JSON handling
  - ICL block rendered in system prompt
  - Correct verdict fields
"""

from __future__ import annotations

import json

import pytest

from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.judges.rubric import RubricJudge
from judge_kappa.models import Assertion, EvalCase
from tests.conftest import MockLLMBackend


def _assertion_response(scores: dict[str, float], rationale: str = "ok") -> str:
    return json.dumps({"assertion_scores": scores, "rationale": rationale})


def _rubric_response(scores: dict[str, float], rationale: str = "ok") -> str:
    return json.dumps({"dimension_scores": scores, "rationale": rationale})


def _pairwise_response(score_a: float, score_b: float, preferred: str = "A") -> str:
    return json.dumps({"score_a": score_a, "score_b": score_b, "preferred": preferred, "rationale": "ok"})


# ── AssertionJudge ─────────────────────────────────────────────────────────────


class TestAssertionJudge:
    def test_weighted_mean_computed_correctly(self):
        assertions = [
            Assertion(text="A1", weight=2.0),
            Assertion(text="A2", weight=1.0),
        ]
        case = EvalCase(id="c1", prompt="P", assertions=assertions)
        # A1=PASS(1.0), A2=FAIL(0.0)  → (1.0*2 + 0.0*1) / 3 = 0.6667
        backend = MockLLMBackend(responses=[_assertion_response({"A1": 1.0, "A2": 0.0})])
        verdict = AssertionJudge("j1", backend).judge(case, "output", "treatment")
        assert verdict.score == pytest.approx(2 / 3, abs=1e-4)

    def test_all_pass_gives_one(self):
        assertions = [Assertion(text="A1"), Assertion(text="A2")]
        case = EvalCase(id="c1", prompt="P", assertions=assertions)
        backend = MockLLMBackend(responses=[_assertion_response({"A1": 1.0, "A2": 1.0})])
        verdict = AssertionJudge("j1", backend).judge(case, "output", "treatment")
        assert verdict.score == pytest.approx(1.0)

    def test_all_fail_gives_zero(self):
        assertions = [Assertion(text="A1"), Assertion(text="A2")]
        case = EvalCase(id="c1", prompt="P", assertions=assertions)
        backend = MockLLMBackend(responses=[_assertion_response({"A1": 0.0, "A2": 0.0})])
        verdict = AssertionJudge("j1", backend).judge(case, "output", "treatment")
        assert verdict.score == pytest.approx(0.0)

    def test_missing_assertion_in_response_treated_as_zero(self):
        assertions = [Assertion(text="A1", weight=1.0), Assertion(text="A2", weight=1.0)]
        case = EvalCase(id="c1", prompt="P", assertions=assertions)
        # Backend only returns A1; A2 defaults to 0.0
        backend = MockLLMBackend(responses=[_assertion_response({"A1": 1.0})])
        verdict = AssertionJudge("j1", backend).judge(case, "output", "treatment")
        assert verdict.score == pytest.approx(0.5)

    def test_malformed_json_does_not_raise(self):
        assertions = [Assertion(text="A1")]
        case = EvalCase(id="c1", prompt="P", assertions=assertions)
        backend = MockLLMBackend(responses=["not valid json at all"])
        verdict = AssertionJudge("j1", backend).judge(case, "output", "treatment")
        assert 0.0 <= verdict.score <= 1.0  # no crash, score defaults to 0.0

    def test_raises_when_no_assertions(self):
        case = EvalCase(id="c1", prompt="P")  # empty assertions
        backend = MockLLMBackend()
        with pytest.raises(ValueError, match="no assertions"):
            AssertionJudge("j1", backend).judge(case, "output", "treatment")

    def test_verdict_fields_populated(self):
        assertions = [Assertion(text="A1")]
        case = EvalCase(id="case-99", prompt="P", assertions=assertions)
        backend = MockLLMBackend(responses=[_assertion_response({"A1": 1.0}, "Great.")])
        verdict = AssertionJudge("judge-x", backend).judge(case, "some long output text", "control")
        assert verdict.judge_id == "judge-x"
        assert verdict.eval_case_id == "case-99"
        assert verdict.variant_name == "control"
        assert verdict.rationale == "Great."
        assert verdict.output_token_count == 4  # "some long output text"

    def test_icl_block_appears_in_system_prompt(self, calibration):
        assertions = [Assertion(text="A1")]
        case = EvalCase(id="c1", prompt="P", assertions=assertions)
        backend = MockLLMBackend(responses=[_assertion_response({"A1": 1.0})])
        AssertionJudge("j1", backend, calibration_examples=calibration).judge(case, "output", "t")
        system_prompt = backend.calls[0]["system"]
        assert "Calibration Anchors" in system_prompt
        assert "0.90" in system_prompt  # first calibration example score


# ── RubricJudge ───────────────────────────────────────────────────────────────


class TestRubricJudge:
    def test_weighted_mean_computed_correctly(self, sample_rubric):
        # faithfulness(w=2)=0.9, relevance(w=1)=0.6  → (0.9*2 + 0.6*1) / 3 = 0.8
        case = EvalCase(id="c1", prompt="P")
        backend = MockLLMBackend(responses=[_rubric_response({"faithfulness": 0.9, "relevance": 0.6})])
        verdict = RubricJudge("j1", backend, rubric=sample_rubric).judge(case, "output", "t")
        assert verdict.score == pytest.approx(0.8, abs=1e-4)

    def test_missing_dimension_treated_as_zero(self, sample_rubric):
        case = EvalCase(id="c1", prompt="P")
        # Only returns faithfulness; relevance defaults to 0.0
        backend = MockLLMBackend(responses=[_rubric_response({"faithfulness": 1.0})])
        verdict = RubricJudge("j1", backend, rubric=sample_rubric).judge(case, "output", "t")
        # (1.0*2 + 0.0*1) / 3 = 0.6667
        assert verdict.score == pytest.approx(2 / 3, abs=1e-4)

    def test_dimension_scores_stored_in_verdict(self, sample_rubric):
        case = EvalCase(id="c1", prompt="P")
        backend = MockLLMBackend(responses=[_rubric_response({"faithfulness": 0.7, "relevance": 0.5})])
        verdict = RubricJudge("j1", backend, rubric=sample_rubric).judge(case, "output", "t")
        assert verdict.dimension_scores["faithfulness"] == pytest.approx(0.7)
        assert verdict.dimension_scores["relevance"] == pytest.approx(0.5)

    def test_expected_output_included_in_user_prompt(self, sample_rubric):
        case = EvalCase(id="c1", prompt="P", expected_output="The answer is 42.")
        backend = MockLLMBackend(responses=[_rubric_response({"faithfulness": 0.5, "relevance": 0.5})])
        RubricJudge("j1", backend, rubric=sample_rubric).judge(case, "output", "t")
        user_prompt = backend.calls[0]["user"]
        assert "The answer is 42." in user_prompt

    def test_rubric_required_at_construction(self):
        with pytest.raises(TypeError):
            RubricJudge("j1", MockLLMBackend())  # type: ignore[call-arg]


# ── PairwiseJudge ─────────────────────────────────────────────────────────────


class TestPairwiseJudge:
    def test_judge_pair_returns_two_verdicts(self):
        case = EvalCase(id="c1", prompt="P")
        backend = MockLLMBackend(responses=[_pairwise_response(0.8, 0.5)])
        va, vb = PairwiseJudge("j1", backend).judge_pair(case, "output A", "output B")
        assert va.score == pytest.approx(0.8)
        assert vb.score == pytest.approx(0.5)

    def test_labels_assigned_correctly(self):
        case = EvalCase(id="c1", prompt="P")
        backend = MockLLMBackend(responses=[_pairwise_response(0.7, 0.6)])
        va, vb = PairwiseJudge("j1", backend).judge_pair(
            case, "ctrl output", "trt output", label_a="control", label_b="treatment"
        )
        assert va.variant_name == "control"
        assert vb.variant_name == "treatment"

    def test_judge_raises_not_implemented(self):
        case = EvalCase(id="c1", prompt="P")
        backend = MockLLMBackend()
        with pytest.raises(NotImplementedError):
            PairwiseJudge("j1", backend).judge(case, "output", "t")

    def test_token_count_reflects_output_length(self):
        case = EvalCase(id="c1", prompt="P")
        backend = MockLLMBackend(responses=[_pairwise_response(0.7, 0.6)])
        va, vb = PairwiseJudge("j1", backend).judge_pair(case, "one two three", "alpha beta")
        assert va.output_token_count == 3
        assert vb.output_token_count == 2
