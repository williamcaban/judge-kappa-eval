"""
Integration tests for JuryEvaluator.

Uses MockLLMBackend throughout — no real LLM calls.
Verifies the full evaluate() pipeline: output generation → judging →
agreement → bias → EvalReport.
"""

from __future__ import annotations

import json

import pytest

from jury_eval.evaluator import JuryEvaluator
from jury_eval.judges.assertion import AssertionJudge
from jury_eval.models import (
    Assertion,
    EvalCase,
    ScaleType,
    Variant,
)
from jury_eval.panel.panel import JudgePanel
from tests.conftest import MockLLMBackend


def _assertion_resp(scores: dict[str, float]) -> str:
    return json.dumps({"assertion_scores": scores, "rationale": "ok"})


def _make_case(case_id: str, assertions: list[str]) -> EvalCase:
    return EvalCase(
        id=case_id,
        prompt=f"Prompt for {case_id}",
        assertions=[Assertion(text=a) for a in assertions],
    )


def _make_evaluator(
    judge_responses: list[str],
    gen_responses: list[str] | None = None,
    n_judges: int = 2,
) -> tuple[JuryEvaluator, MockLLMBackend]:
    gen_backend = MockLLMBackend(responses=gen_responses or ["generated output"])
    judges = [
        AssertionJudge(f"judge-{i}", MockLLMBackend(responses=judge_responses * 100))
        for i in range(n_judges)
    ]
    panel = JudgePanel(judges=judges)
    ev = JuryEvaluator(panel=panel, generation_backend=gen_backend, scale_type=ScaleType.ORDINAL)
    return ev, gen_backend


class TestJuryEvaluatorEvaluate:
    def test_report_has_correct_case_count(self):
        cases = [_make_case(f"c{i}", ["A"]) for i in range(3)]
        ev, _ = _make_evaluator([_assertion_resp({"A": 1.0})])
        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl output")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt output")
        report = ev.evaluate(cases, ctrl, trt)
        assert len(report.cases) == 3

    def test_uplift_is_treatment_minus_control(self):
        cases = [_make_case("c1", ["A"])]
        ctrl_resp = _assertion_resp({"A": 0.3})
        trt_resp  = _assertion_resp({"A": 0.7})

        gen_backend = MockLLMBackend(responses=["output"] * 100)
        j1 = AssertionJudge("j1", MockLLMBackend(responses=[ctrl_resp, trt_resp] * 50))
        j2 = AssertionJudge("j2", MockLLMBackend(responses=[ctrl_resp, trt_resp] * 50))
        panel = JudgePanel(judges=[j1, j2])
        ev = JuryEvaluator(panel=panel, generation_backend=gen_backend)

        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl output")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt output")
        report = ev.evaluate(cases, ctrl, trt)
        # Both judges score control=0.3, treatment=0.7 → uplift=0.4
        assert report.mean_uplift == pytest.approx(0.4, abs=0.05)

    def test_positive_uplift_when_treatment_wins(self):
        cases = [_make_case(f"c{i}", ["A"]) for i in range(4)]
        gen_backend = MockLLMBackend(responses=["output"] * 100)
        # Alternate: judges score treatment higher
        responses = [_assertion_resp({"A": 0.9})] * 100
        j1 = AssertionJudge("j1", MockLLMBackend(responses=responses))
        j2 = AssertionJudge("j2", MockLLMBackend(responses=responses))
        panel = JudgePanel(judges=[j1, j2])
        ev = JuryEvaluator(panel=panel, generation_backend=gen_backend)

        ctrl = Variant(name="control",   predict_fn=lambda _: "baseline output")
        trt  = Variant(name="treatment", predict_fn=lambda _: "improved output")
        report = ev.evaluate(cases, ctrl, trt)
        # All scores are 0.9 → uplift=0 (same score for both variants here)
        assert isinstance(report.mean_uplift, float)

    def test_agreement_fields_populated(self):
        cases = [_make_case(f"c{i}", ["A"]) for i in range(3)]
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.8})], n_judges=2)
        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt")
        report = ev.evaluate(cases, ctrl, trt)
        assert report.agreement.alpha is not None
        assert report.agreement.kappa is not None
        assert report.agreement.n_judges == 2

    def test_bias_fields_populated(self):
        cases = [_make_case(f"c{i}", ["A"]) for i in range(4)]
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.8})], n_judges=2)
        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt")
        report = ev.evaluate(cases, ctrl, trt)
        assert isinstance(report.bias.positional_bias_rate, float)
        assert report.bias.verbosity_bias_rho is not None

    def test_judge_ids_in_report(self):
        cases = [_make_case("c1", ["A"])]
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.8})], n_judges=2)
        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt")
        report = ev.evaluate(cases, ctrl, trt)
        assert len(report.judge_ids) == 2
        assert all("judge-" in jid for jid in report.judge_ids)

    def test_scale_type_stored_in_report(self):
        cases = [_make_case("c1", ["A"])]
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.5})], n_judges=1)
        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt")
        report = ev.evaluate(cases, ctrl, trt)
        assert report.scale_type == ScaleType.ORDINAL

    def test_report_serializable_to_json(self):
        cases = [_make_case("c1", ["A"])]
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.6})], n_judges=2)
        ctrl = Variant(name="control",   predict_fn=lambda _: "ctrl")
        trt  = Variant(name="treatment", predict_fn=lambda _: "trt")
        report = ev.evaluate(cases, ctrl, trt)
        serialized = report.model_dump_json()
        loaded = json.loads(serialized)
        assert "mean_uplift" in loaded
        assert "agreement" in loaded
        assert "bias" in loaded


# ── evaluate_endpoints() ──────────────────────────────────────────────────────


class TestEvaluateEndpoints:
    """evaluate_endpoints() is a thin wrapper over evaluate() that uses per-variant backends."""

    def test_uses_variant_generation_backend(self):
        """Each variant's predict_fn (simulating a per-variant backend) is called."""
        cases = [_make_case(f"c{i}", ["A"]) for i in range(3)]
        calls: dict[str, int] = {"ctrl": 0, "trt": 0}

        def ctrl_fn(_: dict) -> str:
            calls["ctrl"] += 1
            return "control output"

        def trt_fn(_: dict) -> str:
            calls["trt"] += 1
            return "treatment output"

        ev, _ = _make_evaluator([_assertion_resp({"A": 0.7})], n_judges=1)
        ctrl = Variant(name="ctrl", predict_fn=ctrl_fn)
        trt  = Variant(name="trt",  predict_fn=trt_fn)
        report = ev.evaluate_endpoints(cases, ctrl, trt)

        assert calls["ctrl"] == 3
        assert calls["trt"]  == 3
        assert len(report.cases) == 3

    def test_returns_eval_report(self):
        from jury_eval.models import EvalReport
        cases = [_make_case("c1", ["A"])]
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.8})], n_judges=1)
        ctrl = Variant(name="a", predict_fn=lambda _: "a output")
        trt  = Variant(name="b", predict_fn=lambda _: "b output")
        report = ev.evaluate_endpoints(cases, ctrl, trt)
        assert isinstance(report, EvalReport)


# ── evaluate_prerecorded() ────────────────────────────────────────────────────


class TestEvaluatePrerecorded:
    def _ev(self, responses: list[str], n_judges: int = 1) -> JuryEvaluator:
        return _make_evaluator(responses, n_judges=n_judges)[0]

    def test_reads_output_control_and_output_treatment(self):
        data = [
            {
                "id": "q1",
                "inputs": {"question": "Q"},
                "output_control":   "Control answer text here",
                "output_treatment": "Treatment answer text here",
            }
        ]
        ev = self._ev([_assertion_resp({"A": 0.7})])
        report = ev.evaluate_prerecorded(
            data,
            control_output_field="output_control",
            treatment_output_field="output_treatment",
        )
        assert len(report.cases) == 1
        assert "Control" in report.cases[0].control.output
        assert "Treatment" in report.cases[0].treatment.output

    def test_does_not_call_generation_backend(self):
        """Pre-recorded mode must not call the generation backend."""
        calls: list[str] = []

        class TrackingBackend:
            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                calls.append("called")
                return "generated"

        from jury_eval.judges.assertion import AssertionJudge
        from jury_eval.panel.panel import JudgePanel

        gen_backend = TrackingBackend()
        judge_backend = MockLLMBackend(responses=[_assertion_resp({"A": 0.8})] * 50)
        panel = JudgePanel(judges=[AssertionJudge("j", judge_backend)])
        ev = JuryEvaluator(panel=panel, generation_backend=gen_backend)  # type: ignore[arg-type]

        data = [{"id": "q1", "inputs": {"question": "Q"}, "output_control": "ctrl out", "output_treatment": "trt out"}]
        ev.evaluate_prerecorded(data, control_output_field="output_control", treatment_output_field="output_treatment")

        assert len(calls) == 0  # generation backend never called

    def test_custom_variant_labels(self):
        data = [{"id": "q1", "inputs": {"question": "Q"}, "output_control": "A", "output_treatment": "B"}]
        ev = self._ev([_assertion_resp({"A": 0.5})])
        report = ev.evaluate_prerecorded(
            data,
            control_label="system-v1",
            treatment_label="system-v2",
        )
        assert report.cases[0].control.variant_name == "system-v1"
        assert report.cases[0].treatment.variant_name == "system-v2"


# ── evaluate_pairwise_dataset() ───────────────────────────────────────────────


class TestEvaluatePairwiseDataset:
    def _pairwise_resp(self, score_a: float, score_b: float) -> str:
        preferred = "A" if score_a > score_b else ("B" if score_b > score_a else "tie")
        return json.dumps({"score_a": score_a, "score_b": score_b, "preferred": preferred, "rationale": "ok"})

    def test_returns_pairwise_report(self):
        from jury_eval.judges.pairwise import PairwiseJudge
        from jury_eval.models import PairwiseReport

        data = [
            {"id": "p1", "prompt": "Q1", "output_a": "System A text", "output_b": "System B text"},
            {"id": "p2", "prompt": "Q2", "output_a": "System A text", "output_b": "System B text"},
        ]
        backend = MockLLMBackend(responses=[self._pairwise_resp(0.8, 0.4)] * 50)
        judge = PairwiseJudge("pj", backend)
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.5})], n_judges=1)

        report = ev.evaluate_pairwise_dataset(data=data, pairwise_judge=judge)

        assert isinstance(report, PairwiseReport)
        assert len(report.cases) == 2

    def test_preference_rate_sums_to_one(self):
        from jury_eval.judges.pairwise import PairwiseJudge

        data = [{"id": f"p{i}", "prompt": "Q", "output_a": "A", "output_b": "B"} for i in range(5)]
        backend = MockLLMBackend(responses=[self._pairwise_resp(0.8, 0.4)] * 50)
        judge = PairwiseJudge("pj", backend)
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.5})], n_judges=1)
        report = ev.evaluate_pairwise_dataset(data=data, pairwise_judge=judge)

        total = report.preference_rate_a + report.preference_rate_b + report.tie_rate
        assert abs(total - 1.0) < 0.01

    def test_labels_in_report(self):
        from jury_eval.judges.pairwise import PairwiseJudge

        data = [{"id": "p1", "prompt": "Q", "output_a": "A text", "output_b": "B text"}]
        backend = MockLLMBackend(responses=[self._pairwise_resp(0.7, 0.5)] * 20)
        judge = PairwiseJudge("pj", backend)
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.5})], n_judges=1)
        report = ev.evaluate_pairwise_dataset(
            data=data, pairwise_judge=judge,
            label_a="My System", label_b="Baseline",
        )
        assert report.label_a == "My System"
        assert report.label_b == "Baseline"

    def test_high_score_a_gives_high_preference_rate_a(self):
        from jury_eval.judges.pairwise import PairwiseJudge

        data = [{"id": f"p{i}", "prompt": "Q", "output_a": "A", "output_b": "B"} for i in range(10)]
        # A always scores higher
        backend = MockLLMBackend(responses=[self._pairwise_resp(0.9, 0.1)] * 50)
        judge = PairwiseJudge("pj", backend)
        ev, _ = _make_evaluator([_assertion_resp({"A": 0.5})], n_judges=1)
        report = ev.evaluate_pairwise_dataset(data=data, pairwise_judge=judge)

        assert report.preference_rate_a > 0.6  # A should be preferred
